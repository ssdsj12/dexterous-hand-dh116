#!/usr/bin/env python3
"""Screen an earlier left-finger command against same-clock Pinch evidence on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M, DT
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_rod_separation_certificate import RodCapsuleSeparation
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.audit_dh116_pinch_early_approach import (
    REQUIRED, _bind_pad_grids, _bound_model, _finite, _frames, _same_clock,
)
from scripts.audit_dh116_pinch_staged_closure import _pad_geometry
from scripts.run_dh116_pinch_late_wrist import BASELINE_LEDGER_SHA, BASELINE_OUTER_SHA, base

FIRST_WINDOW = 4
FINGER_CHANNELS = (1, 2)
STEPS_PER_WINDOW = 8
RATE = ACTION_CONTRACT["joint_rate_rad_s"]
GAINS = (.8, 1.)
ROD_SENSITIVITY_M = .00005


def compile_variant(original, *, thumb_rad, index_rad):
    original = _finite(original, (9, 24))
    extra = _finite([thumb_rad, index_rad], (2,))
    if np.any(extra < 0.) or np.any(abs(original) > 1.):
        raise ValueError("original action range and nonnegative flexion increments required")
    amended = original.copy()
    duration = (len(amended) - FIRST_WINDOW) * STEPS_PER_WINDOW * DT
    amended[FIRST_WINDOW:, FINGER_CHANNELS] += extra / (duration * RATE)
    if np.any(abs(amended) > 1.):
        raise ValueError("original per-channel joint rate exceeded")
    return amended


def additional_target_rad(original, amended):
    delta = _finite(amended, (9, 24)) - _finite(original, (9, 24))
    if (np.any(delta[:FIRST_WINDOW]) or np.any(np.delete(delta, FINGER_CHANNELS, axis=1))
            or np.any(delta < 0.)):
        raise ValueError("only the later left thumb and index channels may change")
    ticks = np.repeat(delta[:, FINGER_CHANNELS] * DT * RATE, STEPS_PER_WINDOW, axis=0)
    return np.cumsum(ticks, axis=0)


def qualified_preflight(rows, *, terminal_row_count):
    return (terminal_row_count > 0 and len(rows) == terminal_row_count
            and all(r["non_target_lower_m"] > 0. and r["table_lower_m"] >= CLEARANCE_M for r in rows)
            and len(rows[-1]["terminal_cases"]) == 5
            and all(sample["passes_nominal_gate"] for sample in rows[-1]["terminal_cases"]))


def load_reference(case):
    native = (case / "native").resolve(strict=True)
    outer = base._checked(case / "outer.json", BASELINE_OUTER_SHA)
    ledger = native / "physical.jsonl"
    original = base._checked(native / "frozen_plan.json", outer["plan_sha256"])
    if (outer["output"] != str(native) or outer["exit_code"] != 0 or outer["forced_exit"] is not False
            or outer["artifact_sha256"]["physical.jsonl"] != BASELINE_LEDGER_SHA
            or base.lifecycle.native.sha(ledger) != BASELINE_LEDGER_SHA):
        raise ValueError("exact source-bound original Pinch native receipt required")
    audit = base.inspect_interval(ledger, np.asarray(original["action_schedule"]), sealed_archive=original)
    if audit["terminal_reason"] or audit["last_sdk_step"] != 74 or audit["measured_physics_rows"] != 72:
        raise ValueError("complete original quiet plus close evidence required")
    rows = []
    with ledger.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("operation") == "physical":
                row = base.decode_owned(item["payload"])["exchange"]
                _same_clock(row)
                rows.append(row)
    if [r["physics_step"] for r in rows] != list(range(3, 75)):
        raise ValueError("native SDK boundary order differs from original comparison")
    return np.asarray(original["action_schedule"]), rows


def measure_boundary(model, source, row, extra_rad, *, terminal=False):
    raw = row["closure_geometry"]["raw"]["hands"]["left"]
    frames, xyz = _frames(model, raw["body_names"], raw["link_poses_xyzw"])
    hand = row["action_readback"]["post"]["hands"]["left"]
    names = hand["joint_names"]
    if [names[index] for index in hand["active_indices"]] != list(ACTIVE):
        raise ValueError("canonical active joint identities required")
    readback, = hand["q"]
    q = readback[[names.index(n) for n in JOINTS]]
    active = readback[hand["active_indices"]]
    target = hand["backend_targets"][0, hand["active_indices"]].copy()
    target[1:3] += extra_rad
    low, high = active_limits()
    if np.any(target < low) or np.any(target > high):
        raise ValueError("issued hypothesis exceeds an original active joint limit")
    angles = readback[[names.index(n) for n in WRIST_ANGULAR]]
    anchor = Rotation.from_quat(np.roll(hand["native_anchor_pose_w"][0, 3:], -1)).as_matrix()
    reference = model._frames(anchor, xyz, angles, q)
    correction = {key: frame @ np.linalg.inv(reference[model.paths[key]]) for key, frame in frames.items()}

    def q_delta(increments):
        added = active.copy()
        added[1:3] += increments
        if np.any(added < low) or np.any(added > high):
            raise ValueError("geometry hypothesis exceeds an original active joint limit")
        return expand(added) - expand(active)

    def poses(dq):
        fk = model._frames(anchor, xyz, angles, q + dq)
        return {key: correction[key] @ fk[path] for key, path in model.paths.items()}

    center_delta = q_delta(extra_rad * np.mean(GAINS))
    half_span = abs(q_delta(extra_rad * GAINS[1]) - q_delta(extra_rad * GAINS[0])) / 2
    center = poses(center_delta)
    nodes = _finite(row["rod"]["node_positions"][0], (41, 3))
    radius = row["closure_geometry"]["rod_radius_m"]
    rod = RodCapsuleSeparation(nodes, radius, node_motion_bound_m=np.full(len(nodes), ROD_SENSITIVITY_M))
    table_lower, non_target_lower = {}, {}
    for key, transform in center.items():
        verts = model.vertices[key] @ transform[:3, :3].T + transform[:3, 3]
        travel_bound = float(model.joint_radii[key] @ half_span)
        table_lower[key] = float(verts[:, 2].min() - model.table_top - travel_bound)
        if key.split(":", 1)[1] not in REQUIRED:
            non_target_lower[key] = rod.gap_lower_m(verts) - travel_bound
    table_link = min(table_lower, key=table_lower.get)
    non_target_link = min(non_target_lower, key=non_target_lower.get)
    terminal_cases = []
    if terminal:
        grids = {"left": {}, "right": {}}
        _bind_pad_grids(row, "left", source, grids, {})
        for gains in ((.9, .9), (.8, .8), (.8, 1.), (1., .8), (1., 1.)):
            status = _pad_geometry(grids["left"], poses(q_delta(extra_rad * gains)), nodes, radius)
            terminal_cases.append(dict(gains=gains, **status))
    return dict(sdk_step=row["physics_step"], extra_target_rad=extra_rad.tolist(),
                table_lower_m=table_lower[table_link], table_link=table_link,
                non_target_lower_m=non_target_lower[non_target_link], non_target_link=non_target_link,
                terminal_cases=terminal_cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumb-rad", type=float, default=.0026)
    parser.add_argument("--index-rad", type=float, default=.007)
    args = parser.parse_args(argv)
    original, reference = load_reference(args.baseline.resolve(strict=True))
    schedule = compile_variant(original, thumb_rad=args.thumb_rad, index_rad=args.index_rad)
    progress = additional_target_rad(original, schedule)
    model, source = _bound_model(base.ROOT, "left")
    prefix = FIRST_WINDOW * STEPS_PER_WINDOW
    results = [
        measure_boundary(model, source, row, delta, terminal=row["physics_step"] == 74)
        for row, delta in zip(reference[prefix - 1:], progress[prefix - 1:])
    ]
    accepted = qualified_preflight(results, terminal_row_count=41)
    result = dict(schema="dh116-pinch-early-finger-cpu-v1", frozen_original_ledger_sha256=BASELINE_LEDGER_SHA,
                  geometry_source_sha256=source, action_schedule=schedule.tolist(), rows=results,
                  measured_joint_increment_gain_range=GAINS, rod_sensitivity_m=ROD_SENSITIVITY_M,
                  scope="static safety box per archived SDK boundary, five nominal terminal contact samples",
                  qualified_preflight=accepted, native_ready=False, teacher_admission=False,
                  physics_steps=0, ppo_updates=0)
    base.lifecycle.native.write_new(args.output, result)
    print(json.dumps(dict(qualified_preflight=accepted, boundaries=len(results),
                          minimum_non_target=min(results, key=lambda row: row["non_target_lower_m"]),
                          terminal=results[-1]), default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
