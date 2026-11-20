#!/usr/bin/env python3
"""Prove a no-contact, right-active Tripod reference on the grounded cable."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays
from scripts.audit_dh116_pinch_initial_pose import checked_json, load_geometry, path_displacement_bound
from scripts.audit_dh116_tripod_fixed_root_path import certify_path
from scripts.audit_dh116_tripod_sdk_path import compile_actions, runtime_reference
from scripts.audit_dh116_wire_contact_witness import endpoint_gap, shapes_world
from scripts.prepare_dh116_wire_pose_candidate import _pad_gap_interval


ENDPOINT_SHA = "16fcae86fcdf1a10f914d99d2cbc487dca1a44b359cf7584259cf7adb4a6f146"
REFERENCE_SHA = "fa6910beac936c05d809b878ff4763d7f47e64736ae592df7d302ff43de9814c"
SDK_MOTION_STEPS = 160
TABLE_PROOF_M = CLEARANCE_M + .0008
ROD_PROOF_M = .002
SIDE_PROOF_M = .02


def _geometry_query(geometry, nodes):
    layout = geometry["reference"]["reference_layout"]
    table = layout["table_center"][2] + layout["table_size"][2] / 2
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    sign = 1 if geometry["side"] == "left" else -1
    if (geometry["old_candidate"]["rod"]["pinned_node_ids"] != []
            or nodes.shape != (41, 3)
            or not np.isfinite(nodes).all()
            or not np.allclose(nodes[:, 2], table + radius, atol=1e-8, rtol=0.)):
        raise ValueError("same finite free cable initially at the table required")

    def query(pose, active):
        results = {}
        for key, hull in shapes_world(geometry, pose, active).items():
            gap = _pad_gap_interval(hull, nodes, radius)["lower_m"]
            if gap < .0025:
                gap = endpoint_gap(hull, nodes, radius)["lower_m"]
            results.update({"table:" + key: float(hull[:, 2].min() - table),
                            "side:" + key: float((sign * hull[:, 1]).min()),
                            "rod:" + key: gap})
        return results

    return query


def _limits(values):
    return {axis: min(value for name, value in values.items() if name.startswith(axis + ":"))
            for axis in ("table", "rod", "side")}


def _right_sweep(plan, geometry, query):
    samples = runtime_reference(plan, geometry)["rows"]
    if any(row["projection_reasons"] for row in samples[1:]):
        raise ValueError("right reference uses a controller projection")
    poses = np.asarray([row["pose_wxyz"] for row in samples])
    active = np.asarray([row["q_rad"] for row in samples])
    if not np.allclose(poses[:, 3:], poses[0, 3:], atol=1e-12, rtol=0.):
        raise ValueError("right fixed-orientation reference pose required")
    steps = len(samples) - 1
    translation_rate = steps * np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).max()
    joint_rates = steps * abs(np.diff(active, axis=0)).max(axis=0)
    displacement = {prefix + key: translation_rate + bounds
                    for prefix, cooked in (("source:", False), ("cooked:", True))
                    for key, bounds in path_displacement_bound(geometry, active[0], joint_rates,
                                                              cooked=cooked).items()}
    thresholds = {"table": TABLE_PROOF_M, "rod": ROD_PROOF_M, "side": SIDE_PROOF_M}
    travel = {axis + ":" + key: bounds for axis in thresholds for key, bounds in displacement.items()}
    floors = {key: thresholds[key.split(":")[0]] for key in travel}

    def at(fraction):
        index = min(int(fraction * steps), steps - 1)
        blend = fraction * steps - index
        pose = poses[index].copy()
        pose[:3] = (1 - blend) * pose[:3] + blend * poses[index + 1, :3]
        q = (1 - blend) * active[index] + blend * active[index + 1]
        return query(pose, q)

    proof = certify_path(at, travel, floors)
    if not proof["proved"]:
        raise ValueError("right continuous table, rod or side clearance failed")
    margins = {key: min(interval["margin_lower_m"][key] for interval in proof["certified_intervals"])
               + floor for key, floor in floors.items()}
    return proof, _limits(margins)


def build_tracking_plan(endpoint, reference, *, project):
    if (reference["input_sha256"] != ENDPOINT_SHA
            or reference["continuous_reference_path_proved"] is not True
            or reference["native_contact_verified"] is not False):
        raise ValueError("fixed grounded Tripod endpoint provenance required")
    nodes, _, _, _ = seed_arrays(endpoint["trial_seed"])
    geometry = {side: load_geometry(project, side=side) for side in ("left", "right")}
    poses, open_q, end_q, queries = {}, {}, {}, {}
    increment = np.array([0., .01, .01, .01, 0., 0.])
    for side, (x_backoff, z_backoff) in (("left", (-.002, .003)), ("right", (0., .006))):
        if geometry[side]["source_sha256"] != endpoint["hands"][side]["source_sha256"]:
            raise ValueError("side-specific hand geometry provenance differs")
        hand = reference["hands"][side]
        poses[side] = np.array(hand["executed_target_end_pose_wxyz"], dtype=float)
        poses[side][[0, 2]] += [x_backoff, z_backoff]
        end_q[side] = np.array(hand["executed_target_end_q_rad"], dtype=float)
        open_q[side] = end_q[side] - increment
        queries[side] = _geometry_query(geometry[side], nodes)

    left_parked, right_start = poses["left"].copy(), poses["right"].copy()
    left_parked[2] += .0007
    right_start[2] += .0007
    left = _limits(queries["left"](left_parked, open_q["left"]))
    if left["table"] < TABLE_PROOF_M or left["rod"] < ROD_PROOF_M or left["side"] < SIDE_PROOF_M:
        raise ValueError("left parked table, rod or side clearance failed")

    plan = compile_actions(right_start, poses["right"], open_q["right"], end_q["right"],
                           side="right", physics_steps=SDK_MOTION_STEPS, mode="intrinsic_linear")
    proof, right = _right_sweep(plan, geometry["right"], queries["right"])
    actions = np.vstack((np.zeros((1, 24)), plan["actions"], np.zeros((1, 24))))
    seed = prepare_reset_state(
        hand_roots={"left": left_parked.tolist(), "right": right_start.tolist()}, origin=[0.] * 3,
        root_xy=[[0., 0.], [0., 0.]], initial_offset_m=0.,
        joint_positions=[expand(open_q[side]).tolist() for side in ("left", "right")],
        joint_velocities=[[0.] * 11 for _ in range(2)],
        source_description="grounded cable with static left and independently cleared right Tripod tracking")
    return dict(version="dh116-right-tripod-noncontact-tracking-plan-v1", reset_seed=seed,
                rod_nodes_m=nodes.tolist(), actions24=actions.tolist(), right_reference=plan,
                measured_right_tracking=False, self_collision_verified=False,
                grasp_qualified=False, teacher_admission=False, native_qualification=False,
                original_clearance_floor_m=CLEARANCE_M, proved_right_sweep=proof,
                minimum_rod_gap_m=dict(left=left["rod"], right=right["rod"]),
                minimum_table_gap_m=dict(left=left["table"], right=right["table"]),
                bilateral_y_gap_m=left["side"] + right["side"],
                budget=dict(sdk_steps=2 + len(actions) * 8, setup=2, quiet=8, motion=160, quiet_after=8,
                            ppo_updates=0, instances=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint", "reference", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_tracking_plan(checked_json(args.endpoint, ENDPOINT_SHA),
                                 checked_json(args.reference, REFERENCE_SHA),
                                 project=Path(__file__).resolve().parents[1])
    result.update(endpoint_sha256=ENDPOINT_SHA, reference_sha256=REFERENCE_SHA)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
