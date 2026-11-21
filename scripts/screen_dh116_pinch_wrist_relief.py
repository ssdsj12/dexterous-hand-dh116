#!/usr/bin/env python3
"""Screen a source-bound Pinch wrist correction around the measured index contact."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M, DT
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_rod_separation_certificate import RodCapsuleSeparation
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts import run_dh116_pinch_early_finger as early
from scripts.audit_dh116_pinch_early_approach import (
    PADS, REQUIRED, _bind_pad_grids, _bound_model, _contact, _finite, _frames, _same_clock,
)

base = early.base
OUTER_SHA = "e641252a85320ee2e0df272e3e5011fc8a6d9a046dc12ab3adc7dd48ed49b119"
LEDGER_SHA = "277a665067c21084a0ea4939d69ae7fec1b925edf4df7f8a01dffa429bc02501"
FIRST_SDK = 58
ANGLE_RAD = .001
NODE_RESERVE_M = .00005
DURATION_S = (base.SDK_STEPS - FIRST_SDK) * DT


def relief_axis(thumb, index, measured_tangent):
    chord = _finite(thumb, (3,)) - _finite(index, (3,))
    axis = np.cross(chord, -_finite(measured_tangent, (3,)))
    size = np.linalg.norm(axis)
    if not np.isfinite(size) or size <= 1e-10:
        raise ValueError("nondegenerate pair witness and tangential velocity required")
    return axis / size


def target_difference(xyz, angles, anchor, pivot, axis):
    world_rotation = Rotation.from_rotvec(axis * ANGLE_RAD)
    moved = pivot + world_rotation.apply(xyz - pivot)
    intrinsic = anchor.inv() * world_rotation * anchor * Rotation.from_euler("XYZ", angles)
    return moved - xyz, intrinsic.as_euler("XYZ") - angles


def compile_schedule(original, dx, de):
    original = _finite(original, (9, 24))
    if np.any(abs(original) > 1.):
        raise ValueError("original action rate ball required")
    result = original.copy()
    result[-2:, 12:15] += _finite(dx, (3,)) / (ACTION_CONTRACT["translation_rate_m_s"] * DURATION_S)
    result[-2:, 18:21] += _finite(de, (3,)) / (ACTION_CONTRACT["rotation_rate_rad_s"] * DURATION_S)
    if any(np.any(np.linalg.norm(result[-2:, sl], axis=1) > 1.) for sl in (slice(12, 15), slice(18, 21))):
        raise ValueError("wrist relief exceeds an original rate ball")
    return result


def load_archive(case):
    native = (case / "native").resolve(strict=True)
    receipt = base._checked(case / "outer.json", OUTER_SHA)
    ledger = native / "physical.jsonl"
    if (receipt["exit_code"] != 0 or receipt["forced_exit"] is not False or receipt["output"] != str(native)
            or receipt["artifact_sha256"]["physical.jsonl"] != LEDGER_SHA
            or base.lifecycle.native.sha(ledger) != LEDGER_SHA):
        raise ValueError("exact naturally closed early-finger reference required")
    plan = base._checked(native / "frozen_plan.json", receipt["plan_sha256"])
    audit = base.inspect_interval(ledger, np.asarray(plan["action_schedule"]), sealed_archive=plan)
    if audit["terminal_reason"] or audit["last_sdk_step"] != base.SDK_STEPS:
        raise ValueError("uninterrupted reference interval required")
    rows = {}
    with ledger.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("operation") == "physical":
                row = base.decode_owned(event["payload"])["exchange"]
                _same_clock(row)
                rows[row["physics_step"]] = row
    if list(rows) != list(range(3, base.SDK_STEPS + 1)) or not rows[34]["early_finger_admission"]["passed"]:
        raise ValueError("complete measured same-clock prefix and close required")
    return plan, rows


def local_pad_pose(row, link):
    raw = row["closure_geometry"]["raw"]["hands"]["left"]
    pose = _finite(raw["link_poses_xyzw"][0], (23, 7))[raw["body_names"].index(link)]
    return pose[:3], Rotation.from_quat(pose[3:])


def decisive_witness(rows):
    previous, row = (rows[tick] for tick in (72, 73))
    bundle = row["rod"]["rod_mesh_contact_pairs"]["pairs"]
    entries = [p for p in bundle if p["substep"] == 9 and p["mesh_index"] in (0, 1)]
    if len(entries) != 2:
        raise ValueError("exact two source contact witnesses required")
    thumb, index = sorted(entries, key=lambda p: p["mesh_index"])
    if [thumb["node_index"], index["node_index"]] != [35, 34]:
        raise ValueError("measured node 35/34 witness required")
    pa = np.asarray(thumb["contact_point_world_m"])
    pb = np.asarray(index["contact_point_world_m"])
    now, rotation = local_pad_pose(row, "finger14_link")
    before, old_rotation = local_pad_pose(previous, "finger14_link")
    local = rotation.inv().apply(pa - now)
    delta = pa - (before + old_rotation.apply(local))
    dt = row["rod"]["timestamp_s"] - previous["rod"]["timestamp_s"]
    if abs(dt - DT) > 1e-8:
        raise ValueError("one 480 Hz boundary for pad velocity required")
    relative = delta / dt - row["rod"]["velocities"][0, 35]
    n = np.asarray(thumb["outward_normal_world"])
    tangent = relative - (relative @ n) * n
    return pa, pb, tangent


def mount_angles(row):
    hand = row["action_readback"]["post"]["hands"]["left"]
    q, = hand["q"]
    anchor = Rotation.from_quat(np.roll(hand["native_anchor_pose_w"][0, 3:], -1))
    angles = q[[hand["joint_names"].index(name) for name in WRIST_ANGULAR]]
    return anchor, angles


def boundary_geometry(model, grids, row, dx, de):
    fraction = (row["physics_step"] - FIRST_SDK) / (base.SDK_STEPS - FIRST_SDK)
    raw = row["closure_geometry"]["raw"]["hands"]["left"]
    frames, root = _frames(model, raw["body_names"], raw["link_poses_xyzw"])
    nodes = row["rod"]["node_positions"][0]
    radius = row["closure_geometry"]["rod_radius_m"]
    rod = RodCapsuleSeparation(nodes, radius, node_motion_bound_m=np.full(len(nodes), NODE_RESERVE_M))
    anchor, angles = mount_angles(row)
    native_rotation = anchor * Rotation.from_euler("XYZ", angles)
    dxyz, angles_increase = dx * fraction, de * fraction

    def rotation(gain):
        return anchor * Rotation.from_euler("XYZ", angles + gain * angles_increase) * native_rotation.inv()

    middle = rotation(.5)
    table, proximal = {}, {}
    for key, fr in frames.items():
        rel = model.vertices[key] @ fr[:3, :3].T + fr[:3, 3] - root
        vertices = root + middle.apply(rel) + .5 * dxyz
        travel = .5 * (np.linalg.norm(dxyz) + np.linalg.norm(rel, axis=1).max() * abs(angles_increase).sum())
        table[key] = float(vertices[:, 2].min() - model.table_top - travel)
        if key.split(":", 1)[1] not in REQUIRED:
            proximal[key] = float(rod.gap_lower_m(vertices) - travel)
    smallest = min(proximal, key=proximal.get)
    nominal = []
    full = rotation(1.)
    for key in PADS:
        fr = frames[key].copy()
        fr[:3, :3] = full.as_matrix() @ fr[:3, :3]
        fr[:3, 3] = root + full.apply(fr[:3, 3] - root) + dxyz
        nominal.append(_contact(grids[key], fr, nodes, radius)[0])
    return dict(sdk_step=row["physics_step"], minimum_table_m=min(table.values()),
                non_target_link=smallest, non_target_lower_m=proximal[smallest], nominal_pad_gap_m=nominal)


def screen(case):
    plan, rows = load_archive(case)
    thumb, pivot, tangent = decisive_witness(rows)
    axis = relief_axis(thumb, pivot, tangent)
    end = rows[base.SDK_STEPS]
    anchor, _ = mount_angles(end)
    issued_xyz = end["wrist_projection"]["projected_root_poses"][0, 0, :3]
    issued_angles = end["wrist_projection"]["projected_angles"][0, 0]
    dx, de = target_difference(issued_xyz, issued_angles, anchor, pivot, axis)
    schedule = compile_schedule(plan["action_schedule"], dx, de)
    model, source = _bound_model(base.ROOT, "left")
    grids = {"left": {}, "right": {}}
    _bind_pad_grids(end, "left", source, grids, {})
    measured = [boundary_geometry(model, grids["left"], rows[tick], dx, de)
                for tick in range(FIRST_SDK, base.SDK_STEPS + 1)]
    qualified = (all(x["minimum_table_m"] >= CLEARANCE_M and x["non_target_lower_m"] > 0. for x in measured)
                 and all(x is not None and x < 0. for x in measured[-1]["nominal_pad_gap_m"]))
    return dict(schema="dh116-pinch-wrist-relief-screen-v1", qualified=qualified, native_ready=False,
                source_ledger_sha256=LEDGER_SHA, scope="conditional per-boundary independent wrist translation/rotation",
                source_geometry=source, wrist_target_translation_delta_m=dx.tolist(), wrist_intrinsic_delta_rad=de.tolist(),
                pivot_world_m=pivot.tolist(), axis_world=axis.tolist(), thumb_tangent_m_s=tangent.tolist(),
                node_reserve_m=NODE_RESERVE_M, boundary_rows=measured, schedule=schedule.tolist(),
                physics_steps=0, ppo_updates=0, teacher_admission=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = screen(args.archive.resolve(strict=True))
    base.lifecycle.native.write_new(args.output, result)
    print(json.dumps(dict(qualified=result["qualified"], terminal=result["boundary_rows"][-1],
                          evidence_sha256=base.lifecycle.native.sha(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
