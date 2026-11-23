"""Screen complete archived Tripod windows before the loaded hold becomes blocked."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_command_kernels import (
    finger_pre_guard, surface_final_guard, wrist_requested_targets,
)
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import (
    CLEARANCE_M, DT, project_policy_wrist_targets,
)
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import (
    IDENTITY_FIELDS, MIN_LINK_LOAD_N, measure_supported_distal_contact,
)
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT, validate_actions
from go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance import WireClearance
from scripts.audit_dh116_pinch_initial_pose import load_geometry, path_displacement_bound
from scripts.audit_dh116_tripod_index_contact_loss import _contact_row, _validated_pose
from scripts.audit_dh116_tripod_wrist_relief_command import _read_bound_wrist, _rotation
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.solve_dh116_right_pinch_owned_approach import grid_lipschitz
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery


SIDES = ("left", "right")
GROUPS = ("source_colliders", "cooked_pads")
PADS = ("finger14_link", "finger23_link", "finger33_link")
RATES = dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],
             wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
             rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
WINDOW = 8


def require_boundary(row, *, sdk, start_sdk):
    elapsed = sdk - start_sdk
    if (elapsed <= 0 or elapsed % WINDOW or row["physics_step"] != sdk or row["substep"] != WINDOW - 1
            or row["policy_step"] != elapsed // WINDOW - 1):
        raise ValueError("expected complete authored control boundary")
    return elapsed // WINDOW


def select_precontact_advance(authored, certify):
    """Keep the largest proven fraction of one already authored 24D ray."""
    authored = np.asarray(authored, dtype=float)
    validate_actions(torch.as_tensor(authored)[None])
    if authored.shape != (24,):
        raise ValueError("one original 24D control action required")
    if not certify(np.zeros_like(authored)):
        return dict(status="zero_action_proof_failed", fraction=0.)
    if certify(authored):
        return dict(status="authored_action_certified", fraction=1., action24=authored)
    low, high = 0., 1.
    for _ in range(12):
        mid = (low + high) / 2.
        if certify(mid * authored):
            low = mid
        else:
            high = mid
    if low == 0.:
        return dict(status="no_positive_fraction_certified", fraction=0.)
    return dict(status="clearance_limited_advance", fraction=low, action24=low * authored)


def _hand_geometry(row, models, target, measured):
    """Use actual link poses where captured and retain the measured-q FK residual."""
    shapes, finger_q, finger_target = {}, [], []
    for i, (side, model) in enumerate(zip(SIDES, models, strict=True)):
        hand = row["action_readback"]["post"]["hands"][side]
        ids = hand["active_indices"]
        if [hand["joint_names"][j] for j in ids] != list(ACTIVE):
            raise ValueError("named original active joints required")
        q = hand["q"][0, ids]
        cmd = hand["backend_targets"][0, ids]
        finger_q.append(q)
        finger_target.append(cmd)
        pose = measured[i]
        move = _rotation(target[i, 3:]) * _rotation(pose[3:]).inv()
        fk = fk_poses(model, pose, q)
        for group in GROUPS:
            for name, entry in model[group].items():
                link = f"{side}.{name}"
                if link in row["sdk_hand_geometry"]["poses"]:
                    p, r = _validated_pose(row["sdk_hand_geometry"]["poses"][link])
                    source = "measured"
                else:
                    p, r = fk[name][:3, 3], Rotation.from_matrix(fk[name][:3, :3])
                    source = "measured_q_fk"
                local = entry["vertices"]
                world = p + r.apply(local)
                shapes[side, group, name] = dict(
                    source=source, measured=world, target_wrist=target[i, :3] + move.apply(world - pose[:3]),
                    position=p, orientation=r, vertices=local, fk_at_measured=fk[name])
    if len(shapes) != 44:
        raise ValueError("original 22 shapes per hand are required")
    return shapes, np.concatenate(finger_q), np.concatenate(finger_target)


def _shape_proof(wire, vertices, *, selected, boxes, table_top, hand_motion=0., rod_motion=0.):
    proof = wire.certify(vertices, selected=selected, reserve_m=float(hand_motion + rod_motion))
    table = float(vertices[:, 2].min() - table_top - CLEARANCE_M - hand_motion)
    support = float(min(support_clearance_lower(vertices, lo, hi) for lo, hi in boxes) - hand_motion)
    return dict(passed=bool(proof["passed"] and table > 0. and support > 0.),
                rod=proof, table_margin_m=table, support_margin_m=support)


def _initial_boundary(row, previous, models, *, radius, boxes, table_top):
    target, angles, measured, errors = _read_bound_wrist(row)
    loads = measure_supported_distal_contact(row["rod"], grasp="tripod")
    if loads["wrong_side_contacts"] or loads["unexpected_hand_contacts"]:
        raise ValueError("exact supported Tripod mesh attribution required")
    shape, q, cmd = _hand_geometry(row, models, target, measured)
    wire = WireClearance(row["rod"]["node_positions"], radius_m=radius)
    thumb = {}
    for source in ("measured", "target_wrist"):
        for group in GROUPS:
            vertices = shape["left", group, PADS[0]][source]
            thumb[f"{source}_{group}"] = _shape_proof(wire, vertices, selected=True, boxes=boxes, table_top=table_top)
    first_scene_failure = None
    for (side, group, name), body in shape.items():
        for source in ("measured", "target_wrist"):
            proof = _shape_proof(wire, body[source], selected=name in PADS, boxes=boxes, table_top=table_top)
            if not proof["passed"]:
                first_scene_failure = dict(side=side, group=group, link=name, pose_source=source, **proof)
                break
        if first_scene_failure is not None:
            break
    pad_loads = {side: {pad: min(x["per_pad"][pad]["normal_load_n"] for x in hand["per_substep"])
                        for pad in hand["required_pads"]} for side, hand in loads["per_hand"].items()}
    left = loads["per_hand"]["left"]
    any_thumb = any(x["per_pad"]["thumb"]["normal_load_n"] > 0. for x in left["per_substep"])
    no_hand = not any(
        s["total_normal_load_n"] > 0. for h in loads["per_hand"].values() for s in h["per_substep"])
    protected = (loads["per_hand"]["right"]["loaded"]
                 and min(pad_loads["left"]["thumb"], pad_loads["left"]["middle"]) > MIN_LINK_LOAD_N)
    supports = {name: min(loads["external_support_force_norm_n"][name]) for name in ("support_0", "support_1")}
    return dict(
        sdk=row["physics_step"], policy_step=row["policy_step"], thumb_proof=thumb,
        thumb_admissible=all(p["passed"] for p in thumb.values()),
        zero_reserve_scene_passed=first_scene_failure is None, first_scene_failure=first_scene_failure,
        any_left_thumb_load=any_thumb, original_minimum_pad_load_n=pad_loads, no_hand_load=no_hand,
        protected_loaded_hold=protected and min(supports.values()) > 0., support_minimums_n=supports,
        target_vs_measured=errors,
        preceding_max_rod_displacement_m=float(np.linalg.norm(
            row["rod"]["node_positions"] - previous["rod"]["node_positions"], axis=-1).max())), (
                target, angles, measured, shape, q, cmd)


def _controller_endpoint(action, row, seed, state):
    """Repeat the same 24D action for eight original target updates."""
    target, angles, measured, _, q, cmd = state
    root, base, clearance, intr, offset, q0, q_target = (torch.as_tensor(x, dtype=torch.float64)[None] for x in (
        seed, measured, row["sdk_hand_geometry"]["table_clearance_m"][0],
        angles, target[:, :3] - seed[:, :3], q, cmd))
    low, high = (torch.as_tensor(np.tile(x, 2), dtype=torch.float64) for x in active_limits())
    mask = torch.ones(1, dtype=torch.bool)
    action = torch.as_tensor(action, dtype=torch.float64)[None]
    validate_actions(action)
    angular = np.zeros((2, 3))
    wrist_move = np.zeros((2, 3))
    finger_move = np.zeros(12)
    for _ in range(WINDOW):
        req_pos, req_ang = wrist_requested_targets(offset, intr, action, RATES, DT)
        proj = project_policy_wrist_targets(
            root_seed_w=root, measured_base_w=base, common_lift_offset_m=root.new_zeros(1),
            measured_clearance_m=clearance, palm_offsets_m=offset, requested_palm_offsets_m=req_pos,
            angles_rad=intr, requested_angles_rad=req_ang, allowed=mask, live=mask)
        req_q = finger_pre_guard(q_target, action, q0, q0.new_zeros((1, 2 * len(MIMIC))),
                                 low, high, DT, RATES, policy_owned=True)
        _, req_q = surface_final_guard(q_target, req_q, low, high, DT)
        d_pos, d_ang = proj["palm_offsets_m"] - offset, proj["projected_angles"] - intr
        if (proj["projection_reasons"].any()
                or not np.allclose(d_pos.numpy()[0], action.numpy()[0, 12:18].reshape(2, 3) * RATES["wrist_rate_m_s"] * DT,
                                   atol=1e-12, rtol=0.)
                or not np.allclose(d_ang.numpy()[0], action.numpy()[0, 18:24].reshape(2, 3) * RATES["rotation_rate_rad_s"] * DT,
                                   atol=1e-12, rtol=0.)
                or not np.allclose((req_q-q_target).numpy()[0], action.numpy()[0, :12] * RATES["joint_rate_rad_s"] * DT,
                                   atol=1e-12, rtol=0.)):
            return None
        wrist_move += d_pos.numpy()[0]
        angular += d_ang.numpy()[0]
        finger_move += (req_q-q_target).numpy()[0]
        offset, intr, q_target = proj["palm_offsets_m"], proj["projected_angles"], req_q
    poses = proj["projected_root_poses"].numpy()[0]
    return dict(target_poses_wxyz=poses.tolist(), intrinsic_angles_rad=intr.numpy()[0].tolist(),
                finger_targets_rad=q_target.numpy()[0].tolist(),
                wrist_move=wrist_move, angular=angular, finger_move=finger_move)


def _pad_depth(grid, local, reserve, radius, depth):
    values, _, inside = sample_grid(grid, local, dtype=np.float64)
    origin = np.asarray(grid["origin"])
    high = origin + (np.asarray(grid["values"].shape) - 1) * grid["spacing"]
    distance_outside = np.linalg.norm(np.maximum(np.maximum(origin-local, local-high),0.), axis=1)
    if np.any(~inside & (distance_outside <= reserve)):
        return dict(passed=False, reason="uncertified_lattice_entry")
    if not inside.any():
        return dict(passed=True, lower_gap_m=None)
    lower = float(np.min(values[inside] - radius - grid_lipschitz(grid) * reserve[inside]))
    return dict(passed=lower >= -depth, lower_gap_m=lower)


def _window_proof(action, *, row, seed, state, measured_previous, models, grids, radius, depth,
                  boxes, table_top, observed_rod_m):
    compiled = _controller_endpoint(action, row, seed, state)
    if compiled is None:
        return dict(passed=False, reason="original_projector_or_joint_guard")
    target, _, measured, shape, q, cmd = state
    nodes = row["rod"]["node_positions"][0]
    wire = WireClearance(nodes[None], radius_m=radius)
    rod_budget = WINDOW * observed_rod_m
    proofs = []
    for i, (side, model) in enumerate(zip(SIDES, models, strict=True)):
        fingers = slice(6*i, 6*i+6)
        finger_delta = compiled["finger_move"][fingers]
        actual_q, backend_q = q[fingers], cmd[fingers]
        rigid_move = float(np.linalg.norm(compiled["wrist_move"][i]))
        rigid_rotation = float(np.abs(compiled["angular"][i]).sum())
        bend_rotation = float(np.abs(expand(actual_q + finger_delta)-expand(actual_q)).sum())
        absolute_fk = fk_poses(model, target[i], backend_q)
        corrected = dict(model)
        for group in GROUPS:
            cooked = group == "cooked_pads"
            frames = {}
            corrected[group] = {}
            for name, item in model[group].items():
                body = shape[side, group, name]
                frame = np.linalg.solve(body["fk_at_measured"], _matrix(body["position"], body["orientation"]))
                frames[name] = frame
                # Include the verified residual and the SDF origin in the joint lever-arm bound.
                points = np.vstack((body["vertices"], np.zeros(3))) if cooked and name in PADS else body["vertices"]
                corrected[group][name] = dict(item, vertices=points @ frame[:3, :3].T + frame[:3, 3])
            joint_travel = path_displacement_bound(corrected, actual_q, finger_delta, cooked=cooked)
            for name in model[group]:
                key = side, group, name
                body = shape[key]
                absolute = absolute_fk[name] @ frames[name]
                for mode in ("measured", "absolute_backend"):
                    if mode == "measured":
                        p, r, palm = body["position"], body["orientation"], measured[i, :3]
                    else:
                        p, r, palm = absolute[:3, 3], Rotation.from_matrix(absolute[:3, :3]), target[i, :3]
                    vertices = p + r.apply(body["vertices"])
                    lever = np.linalg.norm(vertices - palm, axis=1).max() + joint_travel[name]
                    point_motion = rigid_move + rigid_rotation * lever + joint_travel[name]
                    # The previous same-source SDK displacement is an explicit
                    # scenario allowance, not a prediction of future joint/rod dynamics.
                    observed = WINDOW * np.linalg.norm(body["measured"] - measured_previous[key]["measured"], axis=1).max()
                    hand_budget = float(point_motion + observed)
                    hull = _shape_proof(wire, vertices, selected=name in PADS, boxes=boxes,
                                        table_top=table_top, hand_motion=hand_budget, rod_motion=rod_budget)
                    record = dict(side=side, group=group, link=name, mode=mode, hand_reserve_m=hand_budget, hull=hull)
                    if cooked and name in PADS:
                        local = r.inv().apply(nodes - p)
                        reserve = (hand_budget + rod_budget + (rigid_rotation + bend_rotation)
                                   * (np.linalg.norm(nodes-p, axis=1) + hand_budget + rod_budget))
                        record["pad"] = _pad_depth(grids[f"{side}.{name}"], local, reserve, radius, depth)
                    proofs.append(record)
    passed = all(row["hull"]["passed"] and row.get("pad", {}).get("passed", True) for row in proofs)
    for key in ("wrist_move", "angular", "finger_move"):
        compiled[key] = compiled[key].tolist()
    return dict(passed=passed, scenario_rod_reserve_m=rod_budget,
                geometry=proofs, compiled=compiled)


def _matrix(position, rotation):
    value = np.eye(4)
    value[:3, :3] = rotation.as_matrix()
    value[:3, 3] = position
    return value


def replay(run, *, receipt_sha256, project, start_sdk, end_sdk):
    outer = json.loads(checked_file(run/"outer.json", receipt_sha256))
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or not outer["debugger_status_verified"] or outer["debugger_inferior_signal"] is not None
            or outer["forced_exit"] or outer["exceeded_budget"]):
        raise ValueError("sealed natural-exit supported Tripod run required")
    inventory = outer["artifact_sha256"]

    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name, inventory[name])))

    close = read("app_close_returned.json")
    if close["app_close_returned"] is not True or close["close_mode"] != "normal":
        raise ValueError("sealed normal app.close required")
    reset = read("tripod_reset_source_2.json")["evidence"]
    seed = np.asarray(read("tripod_reset_source_1.json")["reset"]["seed"]["reset_root_pose_w"])
    radius = reset["rod_radius_m"]
    table_top = reset["table_center_m"][2] + .5 * reset["table_size_m"][2]
    boxes = [(np.asarray(z["center_m"]) - .5*np.asarray(z["size_m"]),
              np.asarray(z["center_m"]) + .5*np.asarray(z["size_m"])) for z in read("tripod_support_scene.json")["supports"]]
    frozen = read("frozen_plan.json")
    schedule = np.asarray(frozen["tripod_approach"])
    depth = frozen["right_tripod_distal_contact"]["max_pad_penetration_m"]
    models = [load_geometry(project, side=side) for side in SIDES]
    if not start_sdk+WINDOW <= end_sdk or (end_sdk-start_sdk) % WINDOW:
        raise ValueError("finite whole-window archived scan span required")
    boundaries, eligible = [], []
    episode = None
    bindings = None
    for sdk in range(start_sdk+WINDOW, end_sdk+1, WINDOW):
        row, prior = (_contact_row(read, tick) for tick in (sdk, sdk-1))
        next_policy = require_boundary(row, sdk=sdk, start_sdk=start_sdk)
        identity = tuple(row["rod"][k] for k in IDENTITY_FIELDS[:-2])
        if episode is None:
            episode = identity
        if identity != episode or any(row["rod"][k] != prior["rod"][k] for k in IDENTITY_FIELDS[:-2]):
            raise ValueError("previous delivered reaction must belong to the same full-window source")
        check_reaction_delivery(prior, row)
        current_bindings = {f"{side}.{pad}":row["rod"]["mesh_bindings"][f"{side}.{pad}"]
                            for side in SIDES for pad in PADS}
        if bindings is None:
            bindings = current_bindings
        if current_bindings != bindings:
            raise ValueError("locked source changed across the scan")
        result, state = _initial_boundary(row, prior, models, radius=radius, boxes=boxes, table_top=table_top)
        boundaries.append(result)
        if result["zero_reserve_scene_passed"] and result["no_hand_load"] and min(result["support_minimums_n"].values()) > 0.:
            eligible.append((row, prior, next_policy, result, state))
    grids = {}
    for name, binding in bindings.items():
        side, pad = name.split(".")
        capture = models[SIDES.index(side)]["cooked_pads"][pad]["capture_sha256"]
        if (binding["frame"] != "PhysX_link_origin" or binding["unit"] != "m"
                or capture != binding["capture_sha256"]):
            raise ValueError("locked metric pad SDF and matching cooked capture required")
        with np.load(io.BytesIO(checked_file(Path(binding["sdf_path"]), binding["sdf_sha256"])),
                     allow_pickle=False) as sdf:
            grids[name] = {k:sdf[k] for k in ("values","origin","spacing")}

    attempts = []
    for row, prior, index, result, state in reversed(eligible):
        if not schedule[index].any():
            continue
        older_target, _, older_measured, _ = _read_bound_wrist(prior)
        old_shape, _, _ = _hand_geometry(prior, models, older_target, older_measured)

        def proof(move):
            return _window_proof(move, row=row, seed=seed, state=state, measured_previous=old_shape,
                                 models=models, grids=grids, radius=radius, depth=depth, boxes=boxes,
                                 table_top=table_top, observed_rod_m=result["preceding_max_rod_displacement_m"])

        choice = select_precontact_advance(schedule[index], lambda move: proof(move)["passed"])
        item = dict(sdk=row["physics_step"], next_policy=index, authored=schedule[index].tolist(),
                    status=choice["status"], fraction=choice["fraction"])
        if "action24" in choice:
            item["action24"] = choice["action24"].tolist()
            item["proof"] = proof(choice["action24"])
            item["full_authored_proof"] = proof(schedule[index])
        else:
            item["zero_action_proof"] = proof(np.zeros(24))
        attempts.append(item)
        if "action24" in choice:
            break
    return dict(version="dh116-tripod-early-boundary-v1", receipt_sha256=receipt_sha256,
                source_head=outer["head"], start_sdk=start_sdk, end_sdk=end_sdk,
                boundaries=boundaries, precontact_attempts=attempts,
                source_rows_sha256={str(tick):inventory[f"tripod_control_{tick}.json"]
                    for sdk in range(start_sdk+WINDOW,end_sdk+1,WINDOW) for tick in (sdk-1,sdk)},
                native_steps_executed=0, teacher_admission=False, grasp_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("run", "project", "output"):
        parser.add_argument(f"--{path}", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--start-sdk", required=True, type=int)
    parser.add_argument("--end-sdk", required=True, type=int)
    args = parser.parse_args()
    result = replay(args.run, receipt_sha256=args.receipt_sha256, project=args.project,
                    start_sdk=args.start_sdk, end_sdk=args.end_sdk)
    with args.output.open("x", encoding="utf-8") as dst:
        json.dump(result,dst,allow_nan=False,indent=2)
        dst.write(chr(10))


if __name__ == "__main__":
    main()
