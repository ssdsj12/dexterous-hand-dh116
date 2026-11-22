#!/usr/bin/env python3
"""Audit each Power digit from a fixed pregrasp or measured palm boundary."""
import argparse
import json
from pathlib import Path

from scipy.spatial.transform import Rotation

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_power_contact
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, checked_json, digest, finite_array
from scripts.certify_dh116_tripod_tracking_envelope import certify_box, verify_box_certificate
from scripts.prepare_dh116_pinch_closure_margin import fk_poses
from scripts.screen_dh116_tripod_measured_precontact import as_matrix
from scripts.audit_dh116_power_material_reach import (
    ACTIVE, _pad_gap_interval, active_limits, endpoint_gap, load_geometry,
    path_displacement_bound, seated_geometry, shapes_world, summarize_middle_reach,
    support_clearance, supported_seed_plan,
)


PROBE_SHA = "bed01826298bfbb88ce573ee324a593341275e77523ff77a75a868feeb085c17"
SAMPLES = 65


def _digit_reach(geometry, wrist, open_q, index, pad, nodes, radius, supports, table, limits):
    lower, upper = limits
    q = open_q.copy()
    q[index] = lower[index]
    delta = np.zeros_like(q)
    delta[index] = upper[index] - lower[index]
    reserve = path_displacement_bound(geometry, q, delta, cooked=True)[pad] / (2 * (SAMPLES - 1))
    pad_key = "cooked:" + pad
    prefix = f"finger{index}"
    samples = []
    for angle in np.linspace(lower[index], upper[index], SAMPLES):
        q[index] = angle
        digit = {key: vertices for key, vertices in shapes_world(geometry, wrist, q).items()
                 if key.split(":")[1].startswith(prefix)}
        gap = _pad_gap_interval(digit[pad_key], nodes, radius)
        table_gap = min(float(vertices[:, 2].min() - table) for vertices in digit.values())
        hits, support_gap = support_clearance(digit, supports)
        rod_clear = True
        for vertices in digit.values():
            check = _pad_gap_interval(vertices, nodes, radius)
            if check["lower_m"] < 0 <= check["upper_m"]:
                check = endpoint_gap(vertices, nodes, radius)
            if check["lower_m"] < 0:
                rod_clear = False
                break
        samples.append(dict(angle_rad=float(angle), gap_lower_m=gap["lower_m"], gap_upper_m=gap["upper_m"],
                            clear=bool(rod_clear and table_gap >= CLEARANCE_M
                                       and not hits and support_gap >= CLEARANCE_M)))
    best = min(samples, key=lambda row: row["gap_upper_m"])
    q[index] = best["angle_rad"]
    tight = endpoint_gap(shapes_world(geometry, wrist, q)[pad_key], nodes, radius)
    return dict(link=pad, summary=summarize_middle_reach(samples, point_travel_m=reserve),
                travel_reserve_m=reserve, tight_best_sample=tight, samples=samples)


def audit_fixed_wrist(probe, *, project):
    candidate, = [item for item in probe["left"] if item["abduction_fraction"] == .15 and item["roll"] == 1.45]
    wrist = np.asarray(candidate["root"], dtype=float)
    open_q = np.asarray(candidate["open_q"], dtype=float)
    lower, upper = active_limits()
    expected_q = lower.copy()
    expected_q[0] += .15 * (upper[0] - lower[0])
    if (wrist.shape != (7,) or not np.isfinite(wrist).all()
            or not np.isclose(np.linalg.norm(wrist[3:]), 1., atol=1e-10, rtol=0.)
            or not np.array_equal(open_q, expected_q)):
        raise ValueError("original fixed left wrist and 15% thumb abduction required")
    plan, _ = supported_seed_plan(project)
    nodes = np.asarray(plan["seeds_by_style"]["power"]["nodes"], dtype=float)
    supports = plan["scene"]["supports"]
    geometry = load_geometry(project, side="left")
    layout = geometry["reference"]["reference_layout"]
    table = layout["table_center"][2] + layout["table_size"][2] / 2
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    hulls = shapes_world(geometry, wrist, open_q)
    palm = endpoint_gap(hulls["source:base_link"], nodes, radius)
    if not 0 <= palm["lower_m"] <= palm["upper_m"] <= .0005:
        raise ValueError("fixed wrist must retain the original palm witness")
    if not seated_geometry(hulls, nodes, supports, table, radius)["eligible"]:
        raise ValueError("fixed wrist must pass the original open geometry guards")
    digits = {
        ACTIVE[index]: _digit_reach(geometry, wrist, open_q, index, pad, nodes, radius, supports,
                                   table, (lower, upper))
        for index, pad in enumerate(DISTAL_PAD_LINKS, 1)
    }
    excluded = [key for key, row in digits.items() if row["summary"]["fixed_wrist_reach_excluded"]]
    return dict(joint_results=digits, fixed_wrist_excluded_by=excluded, palm_upper_m=palm["upper_m"],
                independent_digit_screen=True, side="left", native_contact_verified=False,
                teacher_admission=False, actual_isaac_steps=0)



PALM_WITNESS = "docs/contracts/dh116_teacher_power_staged_palm_witness_native_v1.json"
POWER_TASK = "configs/dh116_tasks/flexible_power_supported_contact_v1.json"
POWER_TASK_SHA = "5c54a8f741f02d3b2cda7110acb8b92a109fe212e549d08cd9b33678c8e07df0"


def load_measured_palm_boundary(receipt, *, project):
    """Bind one archived solver exchange and its independently measured support scene."""
    contract_path = project / PALM_WITNESS
    contract = json.loads(contract_path.read_bytes())
    outer = checked_json(receipt, contract["receipt_sha256"])
    native = receipt.parent / "native"
    artifacts = outer["artifact_sha256"]
    close = checked_json(native / "app_close_returned.json", artifacts["app_close_returned.json"])
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or outer["debugger_inferior_signal"] is not None or outer["forced_exit"]
            or outer["exceeded_budget"] or close["app_close_returned"] is not True
            or outer["frozen_sha256"] != contract["frozen_sha256"]):
        raise ValueError("verified natural exit of the pinned palm witness required")
    name = f"power_control_{contract['last_sdk']}.json"
    if artifacts[name] != contract["last_motion_row_sha256"]:
        raise ValueError("expected terminal motion row is not in the native receipt")
    row = decode_owned(checked_json(native / name, artifacts[name]))
    support_file = "power_support_scene.json"
    support = checked_json(native / support_file, artifacts[support_file])
    scene = checked_json(project / POWER_TASK, POWER_TASK_SHA)["support_scene"]
    if (support["supports"] != scene["supports"] or support["no_hidden_rod_pins"] is not True
            or support["authored_rod_force"] is not False):
        raise ValueError("pinned, free supported Power scene required")
    clock = _clock(lambda: row["rod"])
    reset = _clock(lambda: support["reset_clock"])
    epoch = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
    if (any(clock[key] != reset[key] for key in epoch) or reset["rod_tick"] != 0
            or clock["rod_tick"] != 12 * (clock["physics_step"] - reset["physics_step"])):
        raise ValueError("matching supported Power Reset and solver exchange required")
    return row, scene, dict(outer_sha256=contract["receipt_sha256"], support_sha256=artifacts[support_file],
                            motion_row_sha256=artifacts[name], contract_sha256=digest(contract_path),
                            config_sha256=POWER_TASK_SHA)


def _refine_digit_exclusion(geometry, wrist, q, index, pad, nodes, radius, limits):
    """Cover a separated finger's complete flexion interval with tighter bounds."""
    initial_q = q.copy()
    initial_q[index] = limits[0][index]
    delta = np.zeros_like(initial_q)
    delta[index] = limits[1][index] - limits[0][index]
    travel = path_displacement_bound(geometry, initial_q, delta, cooked=True)[pad]
    shell = .0005

    def margin(fraction):
        pose = fk_poses(geometry, wrist, initial_q + fraction[0] * delta)[pad]
        local = geometry["cooked_pads"][pad]["vertices"]
        vertices = local @ pose[:3, :3].T + pose[:3, 3]
        lower = _pad_gap_interval(vertices, nodes, radius)["lower_m"]
        if lower < 2 * shell:
            lower = endpoint_gap(vertices, nodes, radius)["lower_m"]
        return {pad: lower - shell}

    bounds = {pad: [travel]}
    proof = certify_box(margin, bounds, [0.], [1.], max_samples=128, max_seconds=90.)
    if proof["proved"]:
        verified = verify_box_certificate(proof, margin, bounds, [0.], [1.])
        proof["verified"] = verified
        proof["continuous_minimum_lower_m"] = shell + verified["minimum_margin_m"]
    return proof


def align_measured_wrist(geometry, side, row, *, apply_to_geometry):
    hand = row["action_readback"]["post"]["hands"][side]
    indices = hand["active_indices"]
    if ([hand["joint_names"][index] for index in indices] != list(ACTIVE)
            or hand["measured_source"] != "physx_backend_get_dof_positions/get_dof_velocities"):
        raise ValueError("actual PhysX active joint readback in DH116 order required")
    q = np.asarray(hand["q"][0, indices], dtype=float)
    expand(q)
    poses = row["sdk_hand_geometry"]["poses"]
    actual = {name: as_matrix(*poses[f"{side}.{name}"]) for name in ("base_link", *DISTAL_PAD_LINKS)}
    palm = actual["base_link"]
    wrist = np.r_[palm[:3, 3], Rotation.from_matrix(palm[:3, :3]).as_quat()[[3, 0, 1, 2]]]
    start_fk = fk_poses(geometry, wrist, q)
    alignment, transforms = {}, {}
    for name in DISTAL_PAD_LINKS:
        binding = row["rod"]["mesh_bindings"][f"{side}.{name}"]
        cooked = geometry["cooked_pads"][name]
        if (binding["capture_sha256"] != cooked["capture_sha256"]
                or binding["source_geometry_sha256"] != cooked["geometry_sha256"]):
            raise ValueError("native and offline cooked distal identities differ")
        offset = np.linalg.solve(start_fk[name], actual[name])
        translation = float(np.linalg.norm(offset[:3, 3]))
        angle = float(Rotation.from_matrix(offset[:3, :3]).magnitude())
        if translation > 1e-6 or angle > 5e-6:
            raise ValueError("original FK does not reproduce measured cooked pad poses")
        alignment[name] = dict(position_m=translation, rotation_rad=angle)
        transforms[name] = offset
        if apply_to_geometry:
            for group in ("source_colliders", "cooked_pads"):
                shape = geometry[group][name]
                shape["vertices"] = shape["vertices"] @ offset[:3, :3].T + offset[:3, 3]
    return wrist, q, alignment, transforms


def _measure_wrist(geometry, side, row, scene, nodes):
    wrist, q, alignment, _ = align_measured_wrist(geometry, side, row, apply_to_geometry=True)
    limits = active_limits()
    radius = scene["wire_radius_m"]
    results, excluded = {}, []
    for index, pad in enumerate(DISTAL_PAD_LINKS, 1):
        result = _digit_reach(geometry, wrist, q, index, pad, nodes, radius,
                              scene["supports"], scene["table_top_m"], limits)
        separated = result["summary"]
        if 0. <= separated["continuous_minimum_lower_m"] <= .0005 < separated["best_sample_upper_m"]:
            result["continuous_refinement"] = _refine_digit_exclusion(
                geometry, wrist, q, index, pad, nodes, radius, limits)
        if separated["fixed_wrist_reach_excluded"] or result.get("continuous_refinement", {}).get("proved"):
            excluded.append(pad)
        results[pad] = result
    return dict(wrist_wxyz_m=wrist.tolist(), active_q_rad=q.tolist(),
                thumb_abduction_fixed_rad=float(q[0]), cooked_pad_anchor_error=alignment,
                digits=results, excluded_pads=excluded, sources=geometry["source_sha256"])


def measured_palm_boundary(row):
    """Validate measured hand and rod exchanges against the same SDK clock."""
    rod = row["rod"]
    clock = _clock(lambda: rod)
    post = row["action_readback"]["post"]
    if (row["physics_step"] != clock["physics_step"] or row["guard_failure"] is not None
            or row["sdk_hand_geometry"]["physics_step"] != clock["physics_step"]
            or len(post["identity"]) != 1 or _clock(lambda: post["identity"][0]) != clock):
        raise ValueError("same physics step required for rod, joint readback and hand poses")
    nodes = finite_array(rod["node_positions"], (1, 41, 3))[0]
    contact = measure_power_contact(rod)
    hand_loads = {
        side: [sample["palm_normal_load_n"] for sample in hand["per_substep"]]
        for side, hand in contact["hands"].items()
    }
    if (contact["wrong_side_contacts"] or any(min(load) <= 0 for load in hand_loads.values())):
        raise ValueError("measured palm contact on both assigned wire halves required")
    return clock, nodes, hand_loads


def audit_measured_wrist(row, scene, *, project):
    """Exclude unreachable closing postures against the observed curved rod."""
    clock, nodes, hand_loads = measured_palm_boundary(row)
    hands = {side: _measure_wrist(load_geometry(project, side=side), side, row, scene, nodes)
             for side in hand_loads}
    return dict(scope="archived_palm_fixed_wrist_independent_flexion", clock=clock,
                native_palm_normal_load_n=hand_loads, wire_radius_m=scene["wire_radius_m"], hands=hands,
                fixed_wrist_excluded=any(hand["excluded_pads"] for hand in hands.values()),
                simultaneous_closure_verified=False, native_contact_verified=False,
                teacher_admission=False, actual_isaac_steps=0)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--probe", type=Path)
    inputs.add_argument("--palm-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    if args.palm_receipt:
        row, scene, sources = load_measured_palm_boundary(args.palm_receipt, project=project)
        result = audit_measured_wrist(row, scene, project=project)
        result["archive_sources"] = sources
    else:
        probe = checked_json(args.probe, PROBE_SHA)
        result = audit_fixed_wrist(probe, project=project)
        result["probe_sha256"] = PROBE_SHA
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    return 2 if result.get("fixed_wrist_excluded") else 0


if __name__ == "__main__":
    raise SystemExit(main())
