"""Recompute only the position-fixed, hands-clear opening phase of the R6 902 Pinch route."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.certify_dh116_902_pinch_relief_route import _frame_contact, _geometry, _ports, _pose_maps
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.fit_dh116_902_pinch_alignment import PADS, PAD_LIMIT_M, NONPAD_RESERVE_M, SIDES, _boundary_context
from scripts.screen_dh116_902_command_path import audit_boundary, command_step
from scripts.verify_dh116_902_flexible_reset import load_candidate, sha


R6_SHA = "4bb06240a9eef836b53503ef4782a947e4e93089c7bda7d44f1d6db208ddd1a1"
PAIRED_ACTION_SHA = "7357136f4ef4b348848895080d138c8102064f591e6dd3d49c8170c01a9f36f2"
PARKED_SDK = 240
ANCHOR_SDK = 432
WIDTH = 8
TAIL_WINDOWS = 8
END_SDK = ANCHOR_SDK + WIDTH * TAIL_WINDOWS
PHASE = "hands_clear_index_withdrawal"
NONPARTICIPATING = (3, 4, 5, 9, 10, 11)


def checked_withdrawal(action):
    """Retain the precise R6 double input, including wrist arithmetic residue."""
    a = np.asarray(action, dtype=float)
    if (a.shape != (24,) or not np.isfinite(a).all()
            or np.max(np.abs(a[[2, 8]] + .75)) > 2e-15
            or np.any(a[12:18]) or np.max(np.abs(a[18:24])) > 2e-15
            or np.any(np.delete(a[:12], [2, 8]))):
        raise ValueError("original bilateral index-only withdrawal with fixed wrist required")
    return a


def load_r6(r6_path, expected_source):
    if sha(r6_path) != R6_SHA:
        raise ValueError("exact complete staged R6 route evidence required")
    evidence = json.loads(r6_path.read_bytes())
    if (evidence["command_path_qualified"] is not True or evidence["native_contact_verified"] is not False
            or any(evidence[k] != expected_source[k] for k in ("outer_sha256", "archive_sha256", "last_row_sha256"))):
        raise ValueError("R6 route and original SDK432 inputs differ")
    attempt, = evidence["attempts"]
    phases = attempt["phases"]
    if (attempt["first_guard"] is not None or attempt["path_passed"] is not True
            or attempt["checked_substeps"] != 1264 or attempt["index_offsets_rad"] != {"left": -.03, "right": -.03}
            or [(stage["name"], stage["windows"]) for stage in phases] != [
                ("withdraw_index", 8), ("aim_with_index_reserve", 142), ("seat_at_fixed_wrist", 8)]
            or digest(phases) != PAIRED_ACTION_SHA or attempt["action_sha256"] != PAIRED_ACTION_SHA):
        raise ValueError("sealed complete paired command phases required")
    return checked_withdrawal(phases[0]["action"]), attempt["start"]


def _ordinary_clearance(margins):
    # The hands-clear leg does not use any selected-pad penetration exemption.
    values = []
    for key, value in margins.items():
        if key.startswith("rod:"):
            pad = key.split(":")[-1] in PADS
            values.append(value / 1000. + (-PAD_LIMIT_M if pad else NONPAD_RESERVE_M))
    return min(values)


def screen(receipt, r6_path):
    context, models, source = _boundary_context(receipt, sides=SIDES)
    action, expected_start = load_r6(r6_path, source)
    checked_prefix = audit_boundary(receipt, "pinch")
    if not checked_prefix["all_passed"] or checked_prefix["boundary_sdk"] != ANCHOR_SDK:
        raise ValueError("original eight binary32 action transitions did not reproduce")
    _, config, _ = load_candidate("pinch")
    row = context["row"]
    projection = row["wrist_projection"]
    hands = row["action_readback"]["post"]["hands"]
    poses = row["sdk_hand_geometry"]["poses"]
    roots = np.array([[np.r_[poses[s + ".base_link"][0], poses[s + ".base_link"][1][[3, 0, 1, 2]]] for s in SIDES]])
    backend = np.concatenate([hands[s]["backend_targets"][:, hands[s]["active_indices"]] for s in SIDES], 1)
    measured = np.concatenate([hands[s]["q"][:, hands[s]["active_indices"]] for s in SIDES], 1)
    recorded = dict(projected_roots=projection["projected_root_poses"].tolist(),
                    active_fingers=backend.tolist(), native_measured_roots=roots.tolist())
    if recorded != expected_start:
        raise ValueError("R6 withdrawal must start from the original issued and measured SDK432 state")
    anchors = np.stack([hands[s]["mount_anchor_pose_w"] for s in SIDES], axis=1)
    anchor_rotations = [Rotation.from_quat(anchors[0, i, [4, 5, 6, 3]]) for i in range(2)]
    clearance = np.asarray(row["sdk_hand_geometry"]["table_clearance_m"]).reshape(1, 2).copy()
    seed = np.array([[config["hands"][s]["pose_wxyz"] for s in SIDES]])
    palms, angles = [projection[k].copy() for k in ("palm_offsets_m", "projected_angles")]
    ports = _ports(context, models)
    grids = {s: {p: context["grids"][f"{s}.{p}"] for p in PADS} for s in SIDES}
    stations = dict(left=38, right=4)  # Used only to inspect the SDF; all node contacts are rejected here.
    held = backend[:, NONPARTICIPATING].astype("<f4")
    rows = []
    for sdk in range(ANCHOR_SDK + 1, END_SDK + 1):
        issued = command_step(action=action[None], backend_targets=backend, measured_q=measured, seed=seed,
                              measured_roots=roots, clearance=clearance, palms=palms, angles=angles,
                              mount_anchor_poses=anchors)
        submitted = issued["submitted_finger_targets"]
        if submitted[:, NONPARTICIPATING].astype("<f4").tobytes() != held.tobytes():
            raise ValueError(f"nonparticipating Pinch backend target changed at SDK {sdk}")
        backend = submitted.astype(float)
        measured = backend.copy()
        wrist = issued["submitted_wrist_targets"].astype(float)
        for i, basis in enumerate(anchor_rotations):
            target = wrist[0, i]
            roots[0, i, :3] = anchors[0, i, :3] + basis.apply(target[:3])
            roots[0, i, 3:] = (basis * Rotation.from_euler("XYZ", target[3:])).as_quat()[[3, 0, 1, 2]]
        palms, angles = [issued["projection"][k] for k in ("palm_offsets_m", "projected_angles")]
        world, clearance, cross, margins, failures = _geometry(context, ports, roots, measured, stations, models)
        minimum_rod = min(_ordinary_clearance(margins[s]) for s in SIDES)
        if minimum_rod <= NONPAD_RESERVE_M or failures:
            raise ValueError(f"hands-clear whole-hull guard could not qualify withdrawal SDK {sdk}")
        bodies = _pose_maps(world, models)
        if any(any(_frame_contact(nodes, grids[s], bodies[s], stations[s])["negative_nodes"].values())
               for nodes in context["nodes"] for s in SIDES):
            raise ValueError(f"withdrawal has a pad SDF contact at SDK {sdk}")
        rows.append(dict(sdk=sdk, submitted_fingers_f32=issued["submitted_finger_targets"].tolist(),
                         submitted_wrists_f32=issued["submitted_wrist_targets"].tolist(),
                         full_rod_clearance_m=minimum_rod, positive_y_separation_m=cross))
    return dict(version="dh116-902-pinch-withdrawal-cpu-v1", **source, r6_sha256=R6_SHA,
                command_sha256=PAIRED_ACTION_SHA, phase=PHASE, action24=action.tolist(), rows=rows,
                initial_projection_verified=True, archived_prefix_kernel_replay=checked_prefix,
                nonparticipating_slots=list(NONPARTICIPATING), nonparticipating_baseline_f32=held.tolist(),
                nonparticipating_targets_held_in_tail=True,
                minimum_full_rod_clearance_m=min(r["full_rod_clearance_m"] for r in rows),
                minimum_cross_hand_m=min(r["positive_y_separation_m"] for r in rows),
                later_phases_authorized=False, native_contact_verified=False, teacher_admission=False,
                scope="conditional_archived_rod_and_ideal_submission_tracking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--r6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = screen(args.receipt, args.r6)
    with args.output.open("x") as out:
        json.dump(proof, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
