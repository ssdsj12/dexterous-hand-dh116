#!/usr/bin/env python3
"""Replay one fixed bilateral Reset and limited close command against retained rod states."""
import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import DT
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_rod_separation_certificate import RodCapsuleSeparation
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.audit_dh116_pinch_early_approach import (
    MARGIN, OPPOSITION_COS, PADS, SIDES, _bind_pad_grids, _bound_model, _checked_bytes,
    _contact, _finite, _frames, _non_target_certificate, _same_clock, _windows,
)


def compile_close(plan):
    if set(plan) != set(SIDES):
        raise ValueError("one absolute candidate per original hand required")
    left, right = [plan[side]["steps"] for side in SIDES]
    if type(left) is not int or type(right) is not int or left != right or not 8 <= left <= 240 or left % 8:
        raise ValueError("one bounded synchronized 8-substep control horizon required")
    exposure = left * DT
    parts = []
    low, high = active_limits()
    for side in SIDES:
        hand = plan[side]
        root = _finite(hand["root_pose"], (7,))
        if not np.isclose(np.linalg.norm(root[3:]), 1., atol=1e-8, rtol=0.):
            raise ValueError("normalized absolute wrist quaternion required")
        q_open = _finite(hand["q_open"], (6,))
        q_close = _finite(hand["q_close"], (6,))
        if np.any(np.minimum(q_open, q_close) < low) or np.any(np.maximum(q_open, q_close) > high):
            raise ValueError("close path exceeds original joint limits")
        change = q_close - q_open
        if change[0] or np.any(change[3:]) or np.any(change[1:3] <= 0.):
            raise ValueError("only positive thumb and index closure is allowed")
        xyz = _finite(hand["wrist_delta"], (3,))
        euler = _finite(hand["euler_command"], (3,))
        if (np.any(abs(euler) > np.pi / 2) or np.any(xyz < WRIST["palm_lower_m"])
                or np.any(xyz > WRIST["palm_upper_m"])):
            raise ValueError("close path exceeds original wrist workspace")
        finger = change / (exposure * ACTION_CONTRACT["joint_rate_rad_s"])
        linear = xyz / (exposure * ACTION_CONTRACT["translation_rate_m_s"])
        angular = euler / (exposure * ACTION_CONTRACT["rotation_rate_rad_s"])
        if max(np.max(abs(finger)), np.linalg.norm(linear), np.linalg.norm(angular)) > 1.:
            raise ValueError("close path exceeds an original per-step rate")
        parts.append((finger, linear, angular))
    return np.concatenate([parts[side][group] for group in range(3) for side in range(2)])


def _retained_link_correction(model, row, side):
    raw = row["closure_geometry"]["raw"]["hands"][side]
    actual, xyz = _frames(model, raw["body_names"], raw["link_poses_xyzw"])
    post = row["action_readback"]["post"]["hands"][side]
    names = post["joint_names"]
    if [names[i] for i in post["active_indices"]] != list(ACTIVE):
        raise ValueError("original active joint readback required")
    q, = post["q"]
    q_full = q[[names.index(name) for name in JOINTS]]
    angles = q[[names.index(name) for name in WRIST_ANGULAR]]
    rot = Rotation.from_quat(np.roll(post["native_anchor_pose_w"][0][3:], -1)).as_matrix()
    reference = model._frames(rot, xyz, angles, q_full)
    return {name: actual[name] @ np.linalg.inv(reference[path]) for name, path in model.paths.items()}


def _pad_geometry(grids, poses, nodes, radius):
    samples = [_contact(grids[name], poses[name], nodes, radius) for name in PADS]
    gaps = [gap for gap, _, _ in samples]
    normal = (float(np.clip(samples[0][2] @ samples[1][2], -1., 1.))
              if all(n is not None for _, _, n in samples) else None)
    return dict(gaps_m=gaps, node_ids=[node for _, node, _ in samples], opposition_cos=normal,
                passes_nominal_gate=(normal is not None and normal <= OPPOSITION_COS
                    and all(gap is not None and gap <= -MARGIN for gap in gaps)))


def _evaluate(model, hand, correction, grids, nodes, radius, drift):
    root = np.array(hand["root_pose"])
    rotation = Rotation.from_quat(np.roll(root[3:], -1)).as_matrix()
    xyz, angles = np.asarray(hand["wrist_delta"]), np.asarray(hand["euler_command"])
    q0, q1 = [expand(np.asarray(hand[k]).astype(np.float32)) for k in ("q_open", "q_close")]

    def poses(t):
        state = model._frames(rotation, root[:3] + t * xyz, t * angles, q0 + t * (q1 - q0))
        return {name: correction[name] @ state[path] for name, path in model.paths.items()}

    initial, closed = poses(0.), poses(1.)
    travel = {name: float(np.linalg.norm(xyz) + abs(angles).sum() * model.wrist_radii[name]
                         + model.joint_radii[name] @ abs(q1 - q0)) for name in model.paths}
    boundary = dict(measured_link_frames=initial, anchor_rotation=rotation,
                    start_xyz=root[:3], start_angles=np.zeros(3), start_q=q0, final_q=q1)
    table = model.boundary(**boundary).prove(root[:3] + xyz, angles)
    table_reserve = copy(model)
    table_reserve.table_top += drift
    reserve = table_reserve.boundary(**boundary).prove(root[:3] + xyz, angles)
    node_bounds = np.full(len(nodes), drift)
    rod = RodCapsuleSeparation(nodes, radius, node_motion_bound_m=node_bounds)
    quiet = {}
    y_min, y_max = [], []
    for name, frame in initial.items():
        vertices = model.vertices[name] @ frame[:3, :3].T + frame[:3, 3]
        quiet[name] = rod.gap_lower_m(vertices)
        y_min.append(vertices[:, 1].min() - travel[name])
        y_max.append(vertices[:, 1].max() + travel[name])
    proximal = _non_target_certificate(model, poses, travel, nodes, radius,
                                       node_motion_bound_m=node_bounds)
    return dict(table=table, table_reserve=reserve, quiet_minimum_gap_m=min(quiet.values()),
                non_target_rod=proximal, closed=_pad_geometry(grids, closed, nodes, radius),
                swept_y_bounds_m=[float(min(y_min)), float(max(y_max))])


def audit(project, journal, digest, candidate):
    plan = candidate["hands"]
    action = compile_close(plan)
    samples = candidate["sdk_steps"]
    drift = candidate["sensitivity_bound_m"]
    if (not samples or any(type(tick) is not int for tick in samples)
            or samples != sorted(set(samples)) or not np.isfinite(drift) or drift < 0.):
        raise ValueError("ordered measured ticks and finite sensitivity setting required")
    models, sources = zip(*[_bound_model(project, side) for side in SIDES])
    events = [json.loads(line) for line in _checked_bytes(journal, digest).splitlines()]
    wanted = set(samples)
    rows = {packet["physics_step"]: decode_owned(packet)
            for window in _windows(events) for packet in window["physical"]
            if packet["physics_step"] in wanted}
    if set(rows) != wanted:
        raise ValueError("all requested owned SDK boundaries must be retained")
    grids = {side: {} for side in SIDES}
    locks = {}
    out = []
    for tick in samples:
        row = rows[tick]
        result = dict(sdk_step=tick, identity=_same_clock(row))
        nodes = _finite(row["rod"]["node_positions"][0], (41, 3))
        radius = row["closure_geometry"]["rod_radius_m"]
        for model, side, source in zip(models, SIDES, sources):
            _bind_pad_grids(row, side, source, grids, locks)
            correction = _retained_link_correction(model, row, side)
            result[side] = _evaluate(model, plan[side], correction, grids[side], nodes, radius, drift)
        left, right = [result[side]["swept_y_bounds_m"] for side in SIDES]
        between = max(left[0], right[0]) - min(left[1], right[1])
        result["interhand_gap_lower_m"] = between
        result["staged_geometry_passed"] = between > 0. and all(
            part["table"]["proved"] and part["non_target_rod"]["proved"]
            and part["quiet_minimum_gap_m"] > 0. and part["closed"]["passes_nominal_gate"]
            for part in (result[side] for side in SIDES))
        out.append(result)
    return dict(schema="dh116-absolute-pinch-close-replay-v1", source_journal_sha256=digest,
                candidate=candidate, close_action24=action.tolist(), rows=out,
                passed_retained_boundaries=sum(row["staged_geometry_passed"] for row in out),
                motion_is_hypothetical=True, new_native_steps=0, teacher_admission=False,
                native_qualification=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("project", "journal", "candidate", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--journal-sha256", required=True)
    p.add_argument("--candidate-sha256", required=True)
    args = p.parse_args(argv)
    candidate = json.loads(_checked_bytes(args.candidate, args.candidate_sha256))
    result = audit(args.project, args.journal, args.journal_sha256, candidate)
    result["candidate_sha256"] = args.candidate_sha256
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
