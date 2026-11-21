#!/usr/bin/env python3
"""Prepare a bounded, measured Pinch hold target and reject unsafe stale-drive catch-up."""
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
from scripts import run_dh116_pinch_wrist_relief as relief
from scripts.audit_dh116_pinch_early_approach import (
    PADS, REQUIRED, _bind_pad_grids, _bound_model, _contact, _finite, _frames, _same_clock,
)
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.certify_dh116_tripod_tracking_envelope import certify_box, verify_box_certificate

base = relief.base
SOURCE_OUTER_SHA = "8ee0198172bda45989acfce3743508bd7a7ecc8d52685442462819af9fed50b5"
SOURCE_LEDGER_SHA = "486b4b08ace112fa497ed7165600dd34790018f7f20f7dbb321ad3cbc1184ce0"
WINDOW_SCALE = 8 * DT * np.repeat([ACTION_CONTRACT["joint_rate_rad_s"],
                                  ACTION_CONTRACT["translation_rate_m_s"],
                                  ACTION_CONTRACT["rotation_rate_rad_s"]], [12, 6, 6])
VECTORS = (slice(12, 15), slice(15, 18), slice(18, 21), slice(21, 24))
NODE_RESERVE_M = .00005
CERTIFICATE_RADIUS_M = .005


def bounded_transition(command, target):
    remainder = (_finite(target, (24,)) - _finite(command, (24,))) / WINDOW_SCALE
    schedule = np.zeros((4, 24))
    for k in range(2):
        issued = np.clip(remainder, -1., 1.)
        for group in VECTORS:
            vector = remainder[group]
            issued[group] = vector / max(1., np.linalg.norm(vector))
        schedule[k] = issued
        remainder -= issued
    if np.any(abs(remainder) > 1e-12):
        raise ValueError("target cannot be reached in two original windows")
    return schedule


def archived_endpoint(folder):
    native = (folder / "native").resolve(strict=True)
    receipt = base._checked(folder / "outer.json", SOURCE_OUTER_SHA)
    ledger = native / "physical.jsonl"
    if (receipt["output"] != str(native) or receipt["exit_code"] != 0 or receipt["forced_exit"] is not False
            or receipt["artifact_sha256"]["physical.jsonl"] != SOURCE_LEDGER_SHA
            or base.lifecycle.native.sha(ledger) != SOURCE_LEDGER_SHA):
        raise ValueError("exact naturally closed simultaneous-contact source required")
    original = base._checked(native / "frozen_plan.json", receipt["plan_sha256"])
    summary = base.inspect_interval(ledger, np.asarray(original["action_schedule"]), sealed_archive=original)
    if summary["last_sdk_step"] != 74 or summary["terminal_reason"] or not summary["rows_with_simultaneous_loaded_opposition"]:
        raise ValueError("complete qualifying diagnostic, with measured bilateral contact required")
    last = admission = None
    with ledger.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("operation") == "physical":
                last = base.decode_owned(event["payload"])["exchange"]
                if last["physics_step"] == 58:
                    admission = last.get("wrist_relief_admission")
    if last is None or last["physics_step"] != 74 or not admission or not admission["passed"]:
        raise ValueError("last same-instance wrist-relief readback required")
    return last, _same_clock(last)


def joint_and_wrist(row, side):
    index = ("left", "right").index(side)
    hand = row["action_readback"]["post"]["hands"][side]
    names, measured = hand["joint_names"], hand["q"][0]
    indices = hand["active_indices"]
    if [names[j] for j in indices] != list(ACTIVE):
        raise ValueError("original mounted active identity required")
    bodies = row["closure_geometry"]["raw"]["hands"][side]
    xyz = bodies["link_poses_xyzw"][0, bodies["body_names"].index("base_link"), :3]
    angles = measured[[names.index(j) for j in WRIST_ANGULAR]]
    wrist = row["wrist_projection"]
    actual = np.concatenate((measured[indices], xyz, angles))
    command = np.concatenate((hand["backend_targets"][0, indices],
                              wrist["projected_root_poses"][0, index, :3], wrist["projected_angles"][0, index]))
    return actual, command


def global_channel_order(left, right):
    return np.concatenate((left[:6], right[:6], left[6:9], right[6:9], left[9:12], right[9:12]))


def retained_target(actual, command, side):
    retained = np.zeros(12)
    retained[1:3] = .12 if side == "left" else .10
    retained[9:12] = 0. if side == "left" else .25
    result = actual + (command - actual) * retained
    low, high = active_limits()
    if np.any(result[:6] < low) or np.any(result[:6] > high) or np.any(abs(result[-3:]) > np.pi / 2):
        raise ValueError("held target leaves original DH116 limits")
    return result


def assess_side(row, side, measured, issued, target):
    model, source = _bound_model(base.ROOT, side)
    raw = row["closure_geometry"]["raw"]["hands"][side]
    frames, xyz = _frames(model, raw["body_names"], raw["link_poses_xyzw"])
    hand = row["action_readback"]["post"]["hands"][side]
    q = hand["q"][0, [hand["joint_names"].index(name) for name in JOINTS]]
    anchor = Rotation.from_quat(np.roll(hand["native_anchor_pose_w"][0, 3:], -1)).as_matrix()
    angle = measured[9:12]
    posed = model._frames(anchor, xyz, angle, q)
    alignment = {name: fr @ np.linalg.inv(posed[model.paths[name]]) for name, fr in frames.items()}
    nodes = row["rod"]["node_positions"][0]
    rad = row["closure_geometry"]["rod_radius_m"]
    if abs(rad - CERTIFICATE_RADIUS_M) > 1e-8:
        raise ValueError("exact sealed 5 mm rod contract required")
    rod = RodCapsuleSeparation(nodes, rad, node_motion_bound_m=np.full(len(nodes), NODE_RESERVE_M))
    grids = {"left": {}, "right": {}}
    _bind_pad_grids(row, side, source, grids, {})

    # Axis 0/1: thumb and index reach. Axis 2/3: linear/angular wrist response
    # independently. Other fingers retain the actual measured passive relation.
    def make_frames(delta, progress):
        offset = np.zeros(6)
        offset[1:3] = delta[1:3] * progress[:2]
        state = model._frames(anchor, xyz + progress[2] * delta[6:9], angle + progress[3] * delta[9:12],
                              q + expand(measured[:6] + offset) - expand(measured[:6]))
        return {name: alignment[name] @ state[path] for name, path in model.paths.items()}

    def world_vertices(name, fr):
        return model.vertices[name] @ fr[:3, :3].T + fr[:3, 3]

    nominal = target - measured
    after = make_frames(nominal, np.ones(4))
    contact = [_contact(grids[side][key], after[key], nodes, rad) for key in PADS]
    pad_gaps = [s[0] for s in contact]

    def lower(name, v):
        quick = rod.gap_lower_m(v)
        if quick >= .00013:
            return quick
        tight = endpoint_gap(v, nodes, CERTIFICATE_RADIUS_M)["lower_m"] + CERTIFICATE_RADIUS_M - rad - NODE_RESERVE_M
        return max(quick, tight)

    def query(x):
        values = {}
        for key, fr in make_frames(nominal, x).items():
            verts = world_vertices(key, fr)
            values["table:" + key] = float(verts[:, 2].min() - model.table_top - CLEARANCE_M)
            if key.split(":", 1)[1] not in REQUIRED:
                values["rod:" + key] = lower(key, verts)
        return values

    group = []
    for index in (1, 2):
        difference = np.zeros(6)
        difference[index] = nominal[index]
        group.append(abs(expand(measured[:6] + difference) - expand(measured[:6])))
    bounds = {}
    for name in frames:
        values = [float(model.joint_radii[name] @ change) for change in group]
        values += [float(np.linalg.norm(nominal[6:9])), float(abs(nominal[9:]).sum() * model.wrist_radii[name])]
        bounds["table:" + name] = values
        if name.split(":", 1)[1] not in REQUIRED:
            bounds["rod:" + name] = values
    box0, box1 = np.zeros(4), np.ones(4)
    proof = certify_box(query, bounds, box0, box1, max_samples=768, max_seconds=45)
    independent = verify_box_certificate(proof, query, bounds, box0, box1) if proof["proved"] else None

    # Expose high finger catch-up at the original retained command while the
    # wrist remains where it was measured. This case stays outside the target tube.
    stale = make_frames(issued - measured, np.array([1., 1., 0., 0.]))
    old_keys = [key for key in stale if key.split(":", 1)[1] not in REQUIRED]
    gaps = {key: endpoint_gap(world_vertices(key, stale[key]), nodes, CERTIFICATE_RADIUS_M) for key in old_keys}
    for item in gaps.values():
        item["upper_m"] += CERTIFICATE_RADIUS_M - rad
    invalid = {key: value["upper_m"] for key, value in gaps.items() if value["upper_m"] < 0.}
    return dict(nominal_target=target.tolist(), nominal_pad_gaps_m=pad_gaps,
                terminal_contact_geometry=all(s is not None and s < 0. for s in pad_gaps),
                independent_tracking_target_box=proof, independently_checked=independent,
                full_old_finger_lead_intersections_upper_m=invalid)


def screen(folder):
    row, identity = archived_endpoint(folder)
    measured, issued = map(list, zip(*(joint_and_wrist(row, side) for side in ("left", "right"))))
    target = [retained_target(a, b, side) for side, a, b in zip(("left", "right"), measured, issued)]
    schedule = bounded_transition(global_channel_order(*issued), global_channel_order(*target))
    results = {side: assess_side(row, side, a, b, c)
               for side, a, b, c in zip(("left", "right"), measured, issued, target)}
    return dict(schema="dh116-pinch-prelift-target-study-v1", measurement_identity=identity,
                source_outer_sha256=SOURCE_OUTER_SHA, source_ledger_sha256=SOURCE_LEDGER_SHA,
                planned_rollback_windows=2, planned_zero_observation_windows=2,
                action24_schedule=schedule.tolist(), sides=results, node_reserve_m=NODE_RESERVE_M,
                scope="conditional target tube from a measured frame; old-drive transition not certified",
                native_ready=False, hold_verified=False, lift_tested=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = screen(args.archive.resolve(strict=True))
    base.lifecycle.native.write_new(args.output, result)
    print(json.dumps({side: {"target_tube_proved": item["independent_tracking_target_box"]["proved"],
                           "pad_gaps_m": item["nominal_pad_gaps_m"],
                           "old_finger_intersections_m": item["full_old_finger_lead_intersections_upper_m"]}
                      for side, item in result["sides"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
