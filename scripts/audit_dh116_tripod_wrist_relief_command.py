"""Compile and screen the archived Tripod wrist relief against issued targets."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
from go2_pvcnn.control.dh116.tabletop_command_kernels import (
    finger_pre_guard, surface_final_guard, wrist_requested_targets,
)
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import (
    CLEARANCE_M, DT, project_policy_wrist_targets,
)
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import BACKEND_SOURCE, WRIST_ANGULAR, WRIST_LINEAR
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_palm_mount import local_translation
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import IDENTITY_FIELDS, measure_supported_distal_contact
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT, validate_actions
from go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance import WireClearance
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from scripts.audit_dh116_tripod_index_contact_loss import _contact_row, _validated_pose
from scripts.audit_dh116_tripod_wrist_relief import LINKS, _check_readback_clock, _opposed
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.solve_dh116_right_pinch_owned_approach import grid_lipschitz
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery


SIDES = ("left", "right")
LINEAR = slice(12, 15)
ANGULAR = slice(18, 21)
RATES = dict(wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
             rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
SCOPE = "fixed_archive_target_geometry_no_physics"


def _rotation(wxyz):
    return Rotation.from_quat(np.roll(wxyz, -1, axis=-1))


def _require_pose_array(value, shape):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != shape or not np.isfinite(pose).all()
            or not np.allclose(np.linalg.norm(pose[..., 3:], axis=-1), 1., atol=1e-6, rtol=0.)):
        raise ValueError("finite normalized WXYZ pose required")
    return pose


def remaining_control_steps(substep):
    if type(substep) is not int or not 0 <= substep < 8:
        raise ValueError("completed SDK substep index must be in 0..7")
    return 7 - substep if substep < 7 else 8


def compile_left_wrist(*, seed, targets, angles, measured, clearances, goal, steps):
    """Find one constant canonical action and replay the exact native target arithmetic."""
    seed, targets, measured = (_require_pose_array(x, (2, 7)) for x in (seed, targets, measured))
    goal = _require_pose_array(goal, (7,))
    angles = np.asarray(angles, dtype=float)
    clearance = np.asarray(clearances, dtype=float)
    if (type(steps) is not int or not 1 <= steps <= 8 or angles.shape != (2, 3)
            or not np.isfinite(angles).all() or clearance.shape != (2,) or not np.isfinite(clearance).all()):
        raise ValueError("finite actual mount target state and one owned control interval required")

    root, base, clearance_w, state_angles = (
        torch.as_tensor(x, dtype=torch.float64)[None] for x in (seed, measured, clearance, angles))
    reproduced = compose_rotation(root[..., 3:], state_angles).numpy()[0]
    if np.max((_rotation(reproduced).inv() * _rotation(targets[:, 3:])).magnitude()) > 1e-12:
        raise ValueError("archived target does not agree with Reset seed and intrinsic XYZ")
    target_angles = (_rotation(seed[0, 3:]).inv() * _rotation(goal[3:])).as_euler("XYZ")
    duration = steps * DT
    action = np.zeros(24)
    action[LINEAR] = (goal[:3] - targets[0, :3]) / (RATES["wrist_rate_m_s"] * duration)
    action[ANGULAR] = (target_angles - angles[0]) / (RATES["rotation_rate_rad_s"] * duration)
    if any(np.linalg.norm(action[slot]) > 1. for slot in (LINEAR, ANGULAR)):
        raise ValueError("goal exceeds the original wrist rate for the available substeps")

    command = torch.as_tensor(action)[None]
    validate_actions(command)
    offsets = torch.as_tensor(targets[:, :3] - seed[:, :3], dtype=torch.float64)[None]
    mask = torch.ones(1, dtype=torch.bool)
    poses, angle_path = [targets], [angles]
    for _ in range(steps):
        requested_p, requested_a = wrist_requested_targets(offsets, state_angles, command, RATES, DT)
        projected = project_policy_wrist_targets(
            root_seed_w=root, measured_base_w=base, common_lift_offset_m=root.new_zeros(1),
            measured_clearance_m=clearance_w, palm_offsets_m=offsets, requested_palm_offsets_m=requested_p,
            angles_rad=state_angles, requested_angles_rad=requested_a, allowed=mask, live=mask)
        delta_p = action[LINEAR] * DT * RATES["wrist_rate_m_s"]
        delta_a = action[ANGULAR] * DT * RATES["rotation_rate_rad_s"]
        next_p, next_a = projected["palm_offsets_m"], projected["projected_angles"]
        if (projected["projection_reasons"].any()
                or not np.allclose((next_p-offsets).numpy()[0], [delta_p, np.zeros(3)], atol=1e-12, rtol=0.)
                or not np.allclose((next_a-state_angles).numpy()[0], [delta_a, np.zeros(3)], atol=1e-12, rtol=0.)):
            raise ValueError("original wrist projection clipped or altered the archived action")
        offsets, state_angles = next_p, next_a
        poses.append(projected["projected_root_poses"].numpy()[0])
        angle_path.append(state_angles.numpy()[0])
    poses = np.stack(poses)
    if (not np.allclose(poses[-1, 0, :3], goal[:3], atol=1e-12, rtol=0.)
            or (_rotation(poses[-1, 0, 3:]).inv() * _rotation(goal[3:])).magnitude() > 1e-12
            or not np.allclose(poses[:, 1], targets[1], atol=1e-12, rtol=0.)):
        raise ValueError("original projection did not reach the left goal with right target held")
    return dict(action24=action, target_poses=poses, intrinsic_angles_rad=np.stack(angle_path))



def _path(plan, seed, measured, fractions, *, carry_readback_offset):
    poses = plan["target_poses"][:, 0]
    endpoints = plan["intrinsic_angles_rad"][[0, -1], 0]
    angles = endpoints[0] + fractions[:, None] * (endpoints[1] - endpoints[0])
    orientation = _rotation(seed[0, 3:]) * Rotation.from_euler("XYZ", angles)
    shift = fractions[:, None] * (poses[-1, :3] - poses[0, :3])
    if carry_readback_offset:
        orientation = orientation * (_rotation(poses[0, 3:]).inv() * _rotation(measured[0, 3:]))
        return measured[0, :3] + shift, orientation
    return poses[0, :3] + shift, orientation


def certify_pad_grid(grid, local, reserve, witness, radius, max_depth):
    """Bound a trilinear grid between samples, rejecting unobserved lattice entry."""
    low = np.asarray(grid["origin"])
    high = low + (np.asarray(grid["values"].shape) - 1) * grid["spacing"]
    values, gradients, valid = sample_grid(grid, local.reshape(-1, 3), dtype=np.float64)
    values = values.reshape(local.shape[:-1])
    gradients = gradients.reshape(local.shape)
    valid = valid.reshape(local.shape[:-1])
    outside = np.maximum(np.maximum(low - local, local - high), 0.)
    if np.any(~valid & (np.linalg.norm(outside, axis=-1) <= reserve)):
        return dict(passed_depth=False, lattice_covered=False)
    inside = np.minimum(local - low, high - local).min(axis=-1)
    if not np.all(valid[:, witness] & (inside[:, witness] > reserve[witness])):
        return dict(passed_depth=False, lattice_covered=False)
    error = grid_lipschitz(grid) * reserve
    lower = float(np.min(np.where(valid, values - radius - error, np.inf)))
    return dict(lattice_covered=True, passed_depth=lower >= -max_depth, lower_gap_m=lower,
                witness_upper_gap_m=float(np.max(values[:, witness] - radius + error[witness])),
                start_gap_m=float(values[0, witness] - radius), end_gap_m=float(values[-1, witness] - radius),
                final_normal_local=gradients[-1, witness])


def _geometry_screen(plan, seed, measured, after, *, grids, witness, old_index_gap, model, radius, depth,
                     boxes, table_top, carry_readback_offset):
    # A rotation driven by independent Euler axes has angular path length at
    # most the sum of absolute intrinsic increments, regardless of their frame.
    path_angles = plan["intrinsic_angles_rad"][:, 0]
    angular = float(np.abs(path_angles[-1] - path_angles[0]).sum())
    target = plan["target_poses"][:, 0, :3]
    linear = float(np.linalg.norm(target[-1] - target[0]))
    nodes = after["rod"]["node_positions"][0]
    palm = measured[0, :3]
    measured_r = _rotation(measured[0, 3:])
    links = after["sdk_hand_geometry"]["poses"]
    args = (plan, seed, measured)

    hull_intervals = 4 * (len(target) - 1)
    palms, rotations = _path(*args, np.linspace(0., 1., hull_intervals + 1),
                             carry_readback_offset=carry_readback_offset)
    envelope = WireClearance(nodes[None], radius_m=radius)
    shapes = [(group, name, entry["vertices"]) for group in ("source_colliders", "cooked_pads")
              for name, entry in model[group].items()]
    if len(shapes) != 22:
        raise ValueError("all 22 original left source/cooked shapes are required")
    hand = after["action_readback"]["post"]["hands"]["left"]
    active = hand["active_indices"]
    if [hand["joint_names"][i] for i in active] != list(ACTIVE):
        raise ValueError("original measured active joint order required")
    reconstructed = fk_poses(model, measured[0], hand["q"][0, active])
    hull_rows = []
    for group, name, vertices in shapes:
        link = f"left.{name}"
        if link in links:
            pos, rot = _validated_pose(links[link])
            source = "measured_link_pose"
        else:
            transform = reconstructed[name]
            pos, rot = transform[:3, 3], Rotation.from_matrix(transform[:3, :3])
            source = "locked_fk_from_measured_active_q"
        palm_local = measured_r.inv().apply(pos + rot.apply(vertices) - palm)
        # Table and support distance are also 1-Lipschitz in hull motion.
        reserve = (linear + angular * np.linalg.norm(palm_local, axis=-1).max()) / (2 * hull_intervals)
        values = []
        first_failure = None
        for i, (p, r) in enumerate(zip(palms, rotations, strict=True)):
            world = p + r.apply(palm_local)
            check = envelope.certify(world, selected=f"left.{name}" in LINKS, reserve_m=float(reserve))
            if not check["passed"] and first_failure is None:
                first_failure = dict(sample_index=i, failed_segment=check["failed_segment"],
                                     threshold_m=check["threshold_m"])
            values.append((
                bool(check["passed"]), check["lower_m"] - check["threshold_m"] - reserve,
                world[:, 2].min() - table_top - CLEARANCE_M - reserve,
                min(support_clearance_lower(world, lo, hi) for lo, hi in boxes) - reserve,
            ))
        values = np.asarray(values)
        margins = values[:, 1:].min(axis=0)
        hull_rows.append(dict(group=group, link=name, pose_source=source, sample_reserve_m=float(reserve),
                              passed=bool(values[:, 0].all() and (margins[1:] > 0.).all()),
                              first_rod_failure=first_failure,
                              rod_margin_m=float(margins[0]), table_margin_m=float(margins[1]),
                              support_margin_m=float(margins[2])))

    pad_intervals = 1024
    palms, rotations = _path(*args, np.linspace(0., 1., pad_intervals + 1),
                             carry_readback_offset=carry_readback_offset)
    lever = np.maximum(np.linalg.norm(nodes - palms[0], axis=1), np.linalg.norm(nodes - palms[-1], axis=1))
    node_reserve = (linear + angular * lever) / (2 * pad_intervals)
    nodes_palm_local = np.einsum("tni,tij->tnj", nodes[None] - palms[:, None], rotations.as_matrix())
    pads, normals = {}, []
    for name in LINKS:
        position, orientation = _validated_pose(links[name])
        relative_p = measured_r.inv().apply(position - palm)
        relative_r = measured_r.inv() * orientation
        local = relative_r.inv().apply((nodes_palm_local - relative_p).reshape(-1, 3)).reshape(-1, len(nodes), 3)
        check = certify_pad_grid(grids[name], local, node_reserve, witness[name], radius, depth)
        if check["lattice_covered"]:
            normals.append(rotations[-1].apply(relative_r.apply(check.pop("final_normal_local"))))
        pads[name] = check
    opposed = bool(len(normals) == len(LINKS) and np.all(np.linalg.norm(normals, axis=1) > 0.)
                   and _opposed(np.asarray(normals)))
    protected = all(pads[name].get("witness_upper_gap_m", np.inf) < 0. for name in (LINKS[0], LINKS[2]))
    index_recovered = pads[LINKS[1]].get("end_gap_m", np.inf) <= old_index_gap
    hull_passed = all(row["passed"] for row in hull_rows)
    depth_passed = all(row["passed_depth"] for row in pads.values())
    return dict(
        passed=bool(hull_passed and depth_passed and protected and opposed and index_recovered),
        hull_passed=hull_passed, depth_passed=depth_passed, protected_pads_continuous=protected,
        final_sdf_opposes=opposed, final_index_gap_restored=index_recovered,
        hull_samples=hull_intervals + 1, pad_samples=pad_intervals + 1, hulls=hull_rows, pads=pads)


def _read_bound_wrist(row):
    rod = row["rod"]
    _check_readback_clock(row, "post", rod)
    proj = row["wrist_projection"]
    target = proj["projected_root_poses"]
    angles = proj["projected_angles"]
    action = row["action_readback"]
    if (not np.array_equal(action["submission"]["projected_root_poses"], target)
            or not np.array_equal(action["submission"]["projected_angles"], angles)):
        raise ValueError("same-clock original submission and projection disagree")
    measured, errors = [], {}
    for i, side in enumerate(SIDES):
        anchor = action["post"]["hands"][side]
        ids = [anchor["joint_names"].index(name) for name in (*WRIST_LINEAR, *WRIST_ANGULAR)]
        mount = torch.as_tensor(anchor["native_anchor_pose_w"], dtype=torch.float64)
        pose = torch.as_tensor(target[:, i], dtype=torch.float64)
        xyz = local_translation(pose, mount).numpy()
        reconstructed = np.concatenate([xyz, angles[:, i]], axis=1).astype(np.float32)
        if (anchor["backend_source"] != BACKEND_SOURCE or not np.allclose(
                anchor["backend_targets"][:, ids], reconstructed, atol=1e-7, rtol=0.)):
            raise ValueError("original local carriage backend targets differ from the archived projection")
        p, r = _validated_pose(row["sdk_hand_geometry"]["poses"][f"{side}.base_link"])
        measured.append(np.r_[p, np.roll(r.as_quat(), 1)])
        delta = target[0, i, :3] - p
        errors[side] = dict(target_minus_measured_world_m=delta.tolist(),
                            translation_norm_m=float(np.linalg.norm(delta)),
                            orientation_error_rad=float((_rotation(target[0, i, 3:]) * r.inv()).magnitude()))
    return target[0], angles[0], np.stack(measured), errors


def _backend_interval(plan, row):
    """Reproduce mounted wrist Float32 writes and check that finger guards retain targets."""
    hands = row["action_readback"]["post"]["hands"]
    right_targets, wrist_left = None, None
    qs, commands = [], []
    for i, side in enumerate(SIDES):
        hand = hands[side]
        active = hand["active_indices"]
        if [hand["joint_names"][j] for j in active] != list(ACTIVE):
            raise ValueError("original active finger order required")
        qs.append(hand["q"][:, active])
        commands.append(hand["backend_targets"][:, active])
        poses = torch.as_tensor(plan["target_poses"][:, i], dtype=torch.float64)
        native_anchor = torch.as_tensor(hand["native_anchor_pose_w"], dtype=torch.float64)
        local = local_translation(poses, native_anchor.expand(len(poses), 7)).numpy()
        wrist = np.column_stack((local, plan["intrinsic_angles_rad"][:, i])).astype(np.float32)
        if side == "left":
            wrist_left = wrist
        else:
            right_targets = wrist
    if not np.array_equal(right_targets, np.repeat(right_targets[:1], len(right_targets), axis=0)):
        raise ValueError("right backend wrist targets must be held on every step")
    measured = torch.as_tensor(np.concatenate(qs, axis=1), dtype=torch.float64)
    current = torch.as_tensor(np.concatenate(commands, axis=1), dtype=torch.float64)
    low, high = (torch.as_tensor(np.tile(x, 2), dtype=torch.float64) for x in active_limits())
    action = torch.as_tensor(plan["action24"])[None]
    finger = finger_pre_guard(current, action, measured, current.new_zeros((1, 10)), low, high, DT,
                              {"joint_rate_rad_s": ACTION_CONTRACT["joint_rate_rad_s"]}, policy_owned=True)
    _, finger = surface_final_guard(current, finger, low, high, DT)
    if not torch.equal(finger.float(), current.float()):
        raise ValueError("holding finger inputs did not retain original backend targets under the joint guard")
    return dict(left_wrist_targets_float32=wrist_left.tolist(), right_wrist_held=True, finger_targets_held=True)


def _goal(pose, seed):
    pos = pose[:3] + seed["world_translation_m"]
    rotation = Rotation.from_rotvec(seed["world_rotation_vector_rad"]) * _rotation(pose[3:])
    return np.r_[pos, np.roll(rotation.as_quat(), 1)]


def replay(run, *, receipt_sha256, pose_result, pose_result_sha256, source_sdk, project):
    outer = json.loads(checked_file(run/"outer.json", receipt_sha256))
    relief = json.loads(checked_file(pose_result, pose_result_sha256))
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or not outer["debugger_status_verified"] or outer["debugger_inferior_signal"] is not None
            or outer["forced_exit"] or outer["exceeded_budget"] or relief["receipt_sha256"] != receipt_sha256):
        raise ValueError("original natural-exit receipt and same-run relief proof required")
    if not any(r["sdk"] == source_sdk and r["sampled_hull_screen_passed"] for r in relief["hull_screen"]):
        raise ValueError("the selected sampled wrist relief seed is required")
    candidate, = (r for r in relief["rows"] if r["sdk"] == source_sdk and r["status"] == "sampled_pose_candidate")
    inventory = outer["artifact_sha256"]

    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name, inventory[name])))

    close = read("app_close_returned.json")
    if close["app_close_returned"] is not True or close["close_mode"] != "normal":
        raise ValueError("sealed normal app.close required")
    reset = read("tripod_reset_source_2.json")["evidence"]
    original_seed = np.asarray(read("tripod_reset_source_1.json")["reset"]["seed"]["reset_root_pose_w"])
    radius = reset["rod_radius_m"]
    depth = read("frozen_plan.json")["right_tripod_distal_contact"]["max_pad_penetration_m"]
    settings = read("tripod_support_scene.json")["supports"]
    boxes = [(np.asarray(x["center_m"]) - np.asarray(x["size_m"]) / 2,
              np.asarray(x["center_m"]) + np.asarray(x["size_m"]) / 2) for x in settings]
    table_top = reset["table_center_m"][2] + .5 * reset["table_size_m"][2]
    start = _contact_row(read, source_sdk)
    end_sdk = source_sdk + 7 - start["substep"]
    if end_sdk == source_sdk:
        raise ValueError("this playback requires the recorded mid-window loss")
    prior = _contact_row(read, source_sdk - 1)
    rows = [prior, start, *(_contact_row(read, tick) for tick in range(source_sdk + 1, end_sdk + 1))]
    seed_identity = tuple(start["rod"][key] for key in IDENTITY_FIELDS[:-2])
    if any(tuple(row["rod"][key] for key in IDENTITY_FIELDS[:-2]) != seed_identity for row in rows):
        raise ValueError("the relief window cannot cross run, episode, reset, or rod")
    for older, newer in zip(rows[:-1], rows[1:], strict=True):
        if not check_reaction_delivery(older, newer):
            raise ValueError("the original previous World reaction was not delivered")
    final = rows[-1]
    if (any(row["policy_step"] != start["policy_step"]
            or row["substep"] != start["substep"] + row["physics_step"] - source_sdk for row in rows[1:])
            or final["substep"] != 7):
        raise ValueError("archived loss did not close into the original control boundary")

    grids = {}
    bindings = start["rod"]["mesh_bindings"]
    for pad in LINKS:
        binding = bindings[pad]
        if (binding["frame"] != "PhysX_link_origin" or binding["unit"] != "m"
                or relief["sdf_sha256"][pad] != binding["sdf_sha256"]
                or any(row["rod"]["mesh_bindings"][pad] != binding for row in rows)):
            raise ValueError("same locked metric pad SDF binding required")
        with np.load(io.BytesIO(checked_file(Path(binding["sdf_path"]), binding["sdf_sha256"])),
                     allow_pickle=False) as source:
            grids[pad] = {key: source[key] for key in ("values", "spacing", "origin")}
    model = load_geometry(project, side="left")
    if any(model["cooked_pads"][pad.split(".")[1]]["capture_sha256"] != bindings[pad]["capture_sha256"]
           for pad in LINKS):
        raise ValueError("original cooked geometry does not match the pad binding")

    cases = []
    for row in (start, final):
        steps = remaining_control_steps(row["substep"])
        target, angles, measured, errors = _read_bound_wrist(row)
        loads = measure_supported_distal_contact(row["rod"], grasp="tripod")
        case = dict(sdk=row["physics_step"], substep=row["substep"], policy_step=row["policy_step"],
                    steps=steps, next_policy_step=row["policy_step"] + int(row["substep"] == 7),
                    normal_control_boundary=row["substep"] == 7,
                    prospective_sdk_interval=[row["physics_step"] + 1, row["physics_step"] + steps],
                    target_vs_measured=errors,
                    original_right_loaded=loads["per_hand"]["right"]["loaded"],
                    original_minimum_pad_load_n={
                        side: {pad: min(s["per_pad"][pad]["normal_load_n"]
                                        for s in hand["per_substep"]) for pad in hand["required_pads"]}
                        for side, hand in loads["per_hand"].items()}, plans={})
        for mode, reference in (("target_relative", target), ("measured_absolute_goal", measured)):
            try:
                plan = compile_left_wrist(seed=original_seed, targets=target, angles=angles, measured=measured,
                    clearances=row["sdk_hand_geometry"]["table_clearance_m"][0],
                    goal=_goal(reference[0], candidate), steps=steps)
            except ValueError as exc:
                case["plans"][mode] = dict(status="compiler_rejected", reason=str(exc))
                continue
            backend = _backend_interval(plan, row)
            moves = plan["action24"]
            evidence = dict(status="compiled_projector_unclipped", action24=moves.tolist(),
                            peak_normalized_group_norms={name: float(np.linalg.norm(moves[part]))
                                                         for name, part in (("linear", LINEAR), ("angular", ANGULAR))},
                            peak_abs_action=float(np.abs(moves).max()),
                            peak_linear_velocity_m_s=float(np.linalg.norm(moves[LINEAR]) * RATES["wrist_rate_m_s"]),
                            peak_intrinsic_angular_rate_rad_s=float(
                                np.linalg.norm(moves[ANGULAR]) * RATES["rotation_rate_rad_s"]),
                            backend=backend, target_start=plan["target_poses"][0].tolist(),
                            target_end=plan["target_poses"][-1].tolist())
            for tracked, name in ((True, "readback_offset_held"), (False, "target_pose_as_geometry")):
                evidence[name] = _geometry_screen(plan, original_seed, measured, row, grids=grids,
                    witness=candidate["witness_nodes"], old_index_gap=candidate["old_index_gap_m"], model=model,
                    radius=radius, depth=depth, boxes=boxes, table_top=table_top, carry_readback_offset=tracked)
            case["plans"][mode] = evidence
        cases.append(case)

    return dict(version="dh116-tripod-wrist-relief-command-v1", evidence_scope=SCOPE,
                source_head=outer["head"], receipt_sha256=receipt_sha256, pose_result_sha256=pose_result_sha256,
                source_rows_sha256={str(r["physics_step"]): inventory[f"tripod_control_{r['physics_step']}.json"]
                                    for r in rows},
                cases=cases, native_steps_executed=0, teacher_admission=False, grasp_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("run", "pose-result", "output", "project"):
        parser.add_argument(f"--{path}", required=True, type=Path)
    for digest in ("receipt-sha256", "pose-result-sha256"):
        parser.add_argument(f"--{digest}", required=True)
    parser.add_argument("--source-sdk", type=int, required=True)
    args = parser.parse_args()
    result = replay(args.run, receipt_sha256=args.receipt_sha256, pose_result=args.pose_result,
                    pose_result_sha256=args.pose_result_sha256, source_sdk=args.source_sdk, project=args.project)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, allow_nan=False, indent=2)
        f.write(chr(10))


if __name__ == "__main__":
    main()
