"""Certify the compiled partial Power route on its SHA-bound archived rod."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import JOINTS, coupling_matrix, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import RADIUS
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.audit_dh116_pinch_initial_pose import checked_json, joint_rate_displacement_bound, path_displacement_bound
from scripts.audit_dh116_power_fixed_wrist_digits import (
    align_measured_wrist, load_geometry, load_measured_palm_boundary, measured_palm_boundary,
)
from scripts.audit_dh116_wire_contact_witness import endpoint_gap, shapes_world
from scripts.certify_dh116_tripod_tracking_envelope import certify_box, verify_box_certificate
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.prepare_dh116_power_palm_seed import thumb_clearance
from scripts.prepare_dh116_wire_pose_candidate import _pad_gap_interval
from scripts.screen_dh116_power_measured_reseat import _locked_grids, _support_clearance_lower
from scripts.screen_dh116_power_partial_opposition import THUMB, endpoint, rod_clearance_required, staged_joints
from scripts.screen_dh116_power_thumb_clearance import thumb_path
from scripts.solve_dh116_right_pinch_owned_approach import grid_lipschitz
from scripts.verify_dh116_power_staged_palm_witness import MAX_PALM_PENETRATION_M

READINESS = "docs/contracts/dh116_teacher_power_thumb_clearance_v1.json"
CONTROL_HZ = 60
SIDES = ("left", "right")
STAGES = ("wrist_reseat", "thumb_first", "two_pad_closure")


def compile_stage(start, end, *, retreat_world_m, flex_exponents, joint_detour_rad=None):
    """Return 60 Hz target chords and their normalized 24D command rows."""
    delta = end - start
    translation = ACTION_CONTRACT["translation_rate_m_s"]
    rotation = ACTION_CONTRACT["rotation_rate_rad_s"]
    joint = ACTION_CONTRACT["joint_rate_rad_s"]
    xyz_speed = np.linalg.norm(delta[:, :3], axis=1) + np.pi * np.linalg.norm(retreat_world_m, axis=1)
    rpy_speed = np.linalg.norm(delta[:, 3:6], axis=1)
    joint_speed = np.abs(delta[:, 6:])
    joint_speed[:, 1] *= flex_exponents
    if joint_detour_rad is not None:
        joint_detour_rad = np.asarray(joint_detour_rad, dtype=float)
        if joint_detour_rad.shape != (2, 6) or not np.isfinite(joint_detour_rad).all():
            raise ValueError("finite per-hand joint detour required")
        joint_speed += np.pi * np.abs(joint_detour_rad)
    duration = max(xyz_speed.max() / translation, rpy_speed.max() / rotation, joint_speed.max() / joint)
    # Leave one control window at an exact rate boundary instead of
    # rounding the discrete normalized command just above its hard limit.
    steps = max(1, math.ceil(CONTROL_HZ * duration + 1e-10))
    f = np.linspace(0., 1., steps + 1)
    states = start + f[:, None, None] * delta
    states[:, :, :3] -= np.sin(np.pi * f)[:, None, None] * retreat_world_m
    for hand, exponent in enumerate(flex_exponents):
        states[:, hand, 6:] = thumb_path(start[hand, 6:], end[hand, 6:], f, flex_exponent=exponent)
    if joint_detour_rad is not None:
        states[:, :, 6:] += np.sin(np.pi * f)[:, None, None] * joint_detour_rad
    # Set the exact poses at the transitions between compiled stages.
    states[0] = start
    states[-1] = end
    increments = np.diff(states, axis=0) * CONTROL_HZ
    actions = np.concatenate((increments[:, :, 6:].reshape(steps, 12) / joint,
                              increments[:, :, :3].reshape(steps, 6) / translation,
                              increments[:, :, 3:6].reshape(steps, 6) / rotation), axis=1)
    if (np.abs(actions[:, :12]).max() > 1.
            or np.linalg.norm(actions[:, 12:].reshape(steps, 4, 3), axis=-1).max() > 1.):
        raise ValueError("timed Power route exceeds the original 24D rate limits")
    return states, actions


def state_at(states, fraction):
    """Evaluate the actual piecewise-linear target curve between control samples."""
    t = fraction * (len(states) - 1)
    index = min(int(t), len(states) - 2)
    return states[index] + (t - index) * (states[index + 1] - states[index])


def validate_palm_lattice(grid, *, radius):
    values = np.asarray(grid["values"])
    minimum = min(float(np.take(values, edge, axis=axis).min()) for axis in range(3) for edge in (0, -1))
    if not math.isfinite(minimum) or minimum <= radius:
        raise ValueError("locked palm lattice face does not clear the documented wire radius")
    return minimum


def palm_grid_margins(grid, centers, *, radius, lipschitz, weights=1.):
    """Prove safety of nodes that can enter the SDF and continued lattice coverage."""
    low = np.asarray(grid["origin"], dtype=float)
    high = low + float(grid["spacing"]) * (np.array(grid["values"].shape) - 1)
    # The high boundary is excluded by sample_grid. Use an interior argument,
    # and account for its distance from the true projection in the SDF bound.
    edge_guard = 64. * np.finfo(float).eps * max(np.abs(np.r_[low, high]).max(), float(grid["spacing"]))
    inner = np.clip(centers, low + edge_guard, high - edge_guard)
    boundary = np.clip(centers, low, high)
    phi, _, inside = sample_grid(grid, inner, dtype=np.float64)
    if not inside.all():
        raise ValueError("could not evaluate the inner boundary of the locked palm lattice")
    clearance = (phi - radius + MAX_PALM_PENETRATION_M) / max(1., lipschitz)
    clearance -= np.linalg.norm(inner - boundary, axis=1) + edge_guard
    # Projection onto the box is nonexpansive. For each node either the
    # outside margin keeps it excluded, or its clamped SDF value bounds entry.
    outside = np.maximum(low - centers, centers - high).max(axis=1)
    return dict(palm_clearance=float((weights * np.maximum(outside, clearance)).min()),
                palm_domain=float((weights * (-outside - edge_guard)).max()))


def read_hand(project, row, nodes, frozen, selected, side):
    model = load_geometry(project, side=side)
    root, active, _, _ = align_measured_wrist(model, side, row, apply_to_geometry=True)
    grids, hashes = _locked_grids(row["rod"], side, frozen)
    target = np.asarray(selected["wrist_wxyz_m"])
    projection = row["wrist_projection"]
    index = SIDES.index(side)
    reference = projection["projected_root_poses"][0, index]
    origin = reference[:3] - projection["palm_offsets_m"][0, index]
    seed = Rotation.from_quat(reference[[4, 5, 6, 3]]) * Rotation.from_euler(
        "XYZ", projection["projected_angles"][0, index]).inv()
    angles = [(seed.inv() * Rotation.from_quat(p[[4, 5, 6, 3]])).as_euler("XYZ") for p in (root, target)]
    partner = selected["pair"]["partner"]
    _, close = staged_joints(active, selected["pair"])
    first = active.copy()
    first[:2] = close[:2]
    start = np.r_[root[:3], angles[0], active]
    seated = np.r_[target[:3], angles[1], active]
    waypoints = np.stack((start, seated, np.r_[seated[:6], first], np.r_[seated[:6], close]))
    palm_grid = grids["base_link"]
    detour = selected["detour"]
    retreat = detour["retreat_world_m"]
    return dict(model=model, seed=seed, origin=origin, waypoints=waypoints, palm_grid=palm_grid,
                lipschitz=grid_lipschitz(palm_grid), palm_grid_sha256=hashes["base_link"],
                targets=((), (THUMB,), (THUMB, partner)), separating_sign=1. if side == "left" else -1.,
                retreat=np.zeros(3) if retreat is None else np.asarray(retreat),
                flex_exponent=detour["flex_exponent"])


def root_pose(state, seed):
    q = (seed * Rotation.from_euler("XYZ", state[3:6])).as_quat()
    return np.r_[state[:3], q[[3, 0, 1, 2]]]


def shape_speed_bounds(shapes, root, *, linear_speed, angular_speed, joint_motion):
    """Bound point speeds, allowing joints to extend the wrist lever arm."""
    return {
        name: linear_speed + joint_motion[name] + angular_speed * (
            np.linalg.norm(points - root, axis=1).max() + joint_motion[name])
        for name, points in shapes.items()
    }


def maximum_joint_rates(states):
    """Absolute slopes of every full joint over the normalized command interval."""
    slopes = np.diff(states[:, 6:], axis=0) * (len(states) - 1)
    full_slopes = slopes @ coupling_matrix().T
    return dict(zip(JOINTS, np.abs(full_slopes).max(axis=0), strict=True))


def prove_stage(hand, states, targets, nodes, scene, *, allow_concurrent=False,
                allow_joint_reversal=False, require_palm_reach=True, max_samples=4096, max_seconds=150.):
    model, seed, origin = hand["model"], hand["seed"], hand["origin"]
    grid = hand["palm_grid"]
    offsets = states[:, :3] - origin
    if (np.any(offsets < WRIST["palm_lower_m"]) or np.any(offsets > WRIST["palm_upper_m"])
            or np.any(np.abs(states[:, 3:6]) > WRIST["rotation_limit_rad"])):
        raise ValueError("compiled wrist targets exceed the original workspace")
    for state in states:
        expand(state[6:])
    start, end = states[[0, -1]]
    start_pose = root_pose(start, seed)
    endpoints = [endpoint(model, root_pose(state, seed), state[6:], nodes, scene, grid, targets=targets,
                          require_palm_reach=require_palm_reach) for state in (start, end)]
    delta = np.diff(states, axis=0) * (len(states) - 1)
    xyz_speed = float(np.linalg.norm(delta[:, :3], axis=1).max())
    angular_speed = float(np.abs(delta[:, 3:6]).sum(axis=1).max())
    # Bound each joint's rate by the maximum piecewise slope relative to
    # its total travel. The ordinary stage mode holds wrist or joints fixed.
    q_delta = end[6:] - start[6:]
    moving = np.abs(q_delta) > 0.
    concurrent = (xyz_speed > 0. or angular_speed > 0.) and np.any(delta[:, 6:] != 0.)
    reversals = np.any(delta[:, 6:][:, ~moving] != 0.) or np.any(delta[:, 6:] * q_delta < 0.)
    if (concurrent and not allow_concurrent) or (reversals and not allow_joint_reversal):
        raise ValueError("Power swept bounds require explicit concurrent and joint reversal modes")
    factor = float(np.max(np.abs(delta[:, 6:][:, moving] / q_delta[moving]))) if moving.any() else 0.
    joint_motion = {}
    for kind in ("source", "cooked"):
        if allow_joint_reversal:
            speeds = joint_rate_displacement_bound(model, maximum_joint_rates(states), cooked=kind == "cooked")
        else:
            travel = path_displacement_bound(model, start[6:], q_delta, cooked=kind == "cooked")
            speeds = {name: metres * factor for name, metres in travel.items()}
        joint_motion.update({f"{kind}:{name}": metres for name, metres in speeds.items()})
    shapes = shapes_world(model, start_pose, start[6:])
    speeds = shape_speed_bounds(shapes, start[:3], linear_speed=xyz_speed,
                                angular_speed=angular_speed, joint_motion=joint_motion)
    if joint_motion["source:base_link"] != 0.:
        raise ValueError("Power palm lattice must be fixed to the wrist")
    node_speeds = xyz_speed + angular_speed * (np.linalg.norm(nodes - start[:3], axis=1) + xyz_speed)
    # Scale both margins and rates so that every node uses the same
    # Lipschitz coefficient without inheriting the farthest node's speed.
    palm_loc_speed = 0.
    weights = np.ones(len(nodes))
    if np.any(node_speeds > 0.):
        node_speeds = np.maximum(node_speeds, 1e-12)
        palm_loc_speed = float(node_speeds.min())
        weights = palm_loc_speed / node_speeds
    thumb_speed = max(speed for name, speed in speeds.items() if name.split(":")[1].startswith("finger1"))
    digit_speed = max(speed for name, speed in speeds.items() if name.split(":")[1].startswith(
        ("finger2", "finger3", "finger4", "finger5")))
    motion = dict(palm_clearance=palm_loc_speed, palm_domain=palm_loc_speed,
                  thumb_separation=thumb_speed + digit_speed)
    for name, speed in speeds.items():
        for key in ("table", "support", "hand_side"):
            motion[f"{key}:{name}"] = speed
        if name != "source:base_link":
            motion[f"rod:{name}"] = speed
    radius = scene["wire_radius_m"]
    if radius != RADIUS:
        raise ValueError("archived Power proof uses the original 10 mm wire distance model")

    def margins(point):
        state = state_at(states, float(point[0]))
        pose = root_pose(state, seed)
        world = shapes_world(model, pose, state[6:])
        palm = fk_poses(model, pose, state[6:])["base_link"]
        local = (nodes - palm[:3, 3]) @ palm[:3, :3]
        result = palm_grid_margins(grid, local, radius=radius, lipschitz=hand["lipschitz"], weights=weights)
        result["thumb_separation"] = thumb_clearance(world) - 1e-12
        for name, vertices in world.items():
            result[f"table:{name}"] = float(vertices[:, 2].min() - scene["table_top_m"] - CLEARANCE_M)
            result[f"support:{name}"] = _support_clearance_lower({name: vertices}, scene["supports"]) - CLEARANCE_M
            result[f"hand_side:{name}"] = float((vertices[:, 1] * hand["separating_sign"]).min() - 1e-12)
            if name != "source:base_link":
                required = rod_clearance_required(name, targets)
                gap = rod_support_lower(vertices, nodes)
                if gap <= required + .001:
                    gap = max(gap, _pad_gap_interval(vertices, nodes, radius)["lower_m"])
                if gap <= required + .0001:
                    gap = endpoint_gap(vertices, nodes, radius)["lower_m"]
                result[f"rod:{name}"] = gap - required - 1e-12
        return result

    bounds = {key: [value] for key, value in motion.items()}
    proof = certify_box(margins, bounds, [0.], [1.], max_samples=max_samples, max_seconds=max_seconds)
    verified = verify_box_certificate(proof, margins, bounds, [0.], [1.]) if proof["proved"] else None
    return dict(endpoints_passed=all(record["passed"] for record in endpoints), proof=proof, verified=verified,
                travel_bounds_m=motion, endpoint_palm_gap_m=[record["palm_gap_m"] for record in endpoints])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("palm-receipt", "thumb-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--left-retreat-m", type=float,
                        help="Replay a CPU candidate with the archived left retreat direction and a new magnitude.")
    args = parser.parse_args(argv)
    if args.left_retreat_m is not None and (not math.isfinite(args.left_retreat_m) or args.left_retreat_m <= 0.):
        parser.error("a positive finite left retreat is required")
    project = Path(__file__).resolve().parents[1]
    ready = json.loads((project / READINESS).read_bytes())
    archive_sha = ready["audit"]["artifact_sha256"]
    audit = checked_json(args.thumb_audit, archive_sha)
    row, scene, archive = load_measured_palm_boundary(args.palm_receipt, project=project)
    clock, nodes, _ = measured_palm_boundary(row)
    if audit["archive"] != archive or audit["clock"] != clock:
        raise ValueError("partial Power paths require the original wrist, rod and support exchange")
    outer = checked_json(args.palm_receipt, archive["outer_sha256"])
    plan = json.loads(args.palm_receipt.with_name("plan.json").read_bytes())
    if digest(plan) != outer["frozen_sha256"]:
        raise ValueError("SDF sources must come from the original frozen plan")
    hands = []
    for side in SIDES:
        selected = ready["audit"]["hands"][side]["selected"]
        candidate, = [p for p in audit["hand_results"][side]["proposals"]
                      if p["candidate_index"] == selected["candidate_index"]
                      and p["wrist_wxyz_m"] == selected["wrist_wxyz_m"]
                      and any(a["pair"] == selected["pair"] and a.get("thumb_first_detour") == selected["detour"]
                              for a in p["attempts"])]
        if not selected["detour"]["sampled_route_passed"] or candidate["wrist_wxyz_m"] != selected["wrist_wxyz_m"]:
            raise ValueError("one pinned sampled partial Power route per side required")
        hands.append(read_hand(project, row, nodes, plan["frozen_files"], selected, side))
    lattice_faces = {side: validate_palm_lattice(hand["palm_grid"], radius=scene["wire_radius_m"])
                     for side, hand in zip(SIDES, hands, strict=True)}
    waypoints = np.stack([hand["waypoints"] for hand in hands], axis=1)
    retreat = np.stack([hand["retreat"] for hand in hands])
    if args.left_retreat_m is not None:
        retreat[0] *= args.left_retreat_m / np.linalg.norm(retreat[0])
    flex = tuple(hand["flex_exponent"] for hand in hands)
    stages = []
    for index, name in enumerate(STAGES):
        states, actions = compile_stage(waypoints[index], waypoints[index + 1],
                                        retreat_world_m=retreat if index == 0 else np.zeros_like(retreat),
                                        flex_exponents=flex if index == 1 else (1., 1.))

        records = {side: prove_stage(hand, states[:, side_index], hand["targets"][index], nodes, scene)
                   for side_index, (side, hand) in enumerate(zip(SIDES, hands, strict=True))}
        stages.append(dict(stage=name, control_steps=len(actions), planned_24d_commands=actions.tolist(),
                           target_waypoints=states.tolist(), hands=records))
        print(name, len(actions),
              {side: (record["proof"]["proved"], len(record["proof"]["samples"]))
               for side, record in records.items()}, flush=True)
    passed = all(record["verified"] and record["endpoints_passed"]
                 for stage in stages for record in stage["hands"].values())
    output = dict(version="dh116-power-thumb-route-v1", archive=archive, clock=clock,
                  thumb_audit_sha256=archive_sha, control_hz=CONTROL_HZ, fixed_rod_only=True,
                  wrist_retreat_world_m=retreat.tolist(), thumb_first_flex_exponents=flex,
                  minimum_six_lattice_faces_m=lattice_faces,
                  plan_control_steps=sum(stage["control_steps"] for stage in stages),
                  continuous_target_geometry_proved=passed, candidate_for_native_contact_only=passed,
                  native_execution_ready=False, native_runs=0, teacher_admission=False,
                  grasp_qualified=False, generalization_qualified=False,
                  palm_sdf_sha256=dict(zip(SIDES, [h["palm_grid_sha256"] for h in hands], strict=True)),
                  stages=stages)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
