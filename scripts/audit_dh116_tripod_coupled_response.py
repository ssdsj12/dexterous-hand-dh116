"""Compare both hands' observed response around one Tripod control change."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_supported_distal_contact


SIDES = ("left", "right")
PADS = ("thumb", "index", "middle")
SUBSTEPS = 12
PERCENTILES = (0, 5, 50, 95)
EPISODE = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")


def summarize_window(evidence):
    """Summarize sourced solver data without changing the existing success criteria."""
    if not evidence:
        raise ValueError("at least one physical Tripod sample required")
    hands = {}
    for side in SIDES:
        reasons = Counter()
        minimums, totals, vectors = [], [], []
        qualified = opposed = 0
        for sample in evidence:
            hand = sample["per_hand"][side]
            rows = hand["per_substep"]
            if [row["substep"] for row in rows] != list(range(SUBSTEPS)):
                raise ValueError("12 ordered solver substeps required")
            qualified += int(hand["loaded"])
            opposed += sum(int(row["all_required_pads_on_opposing_faces"]) for row in rows)
            reasons.update(reason for row in rows for reason in row["reasons"])
            minimums.append([min(row["per_pad"][pad]["normal_load_n"] for row in rows) for pad in PADS])
            totals.append(min(row["total_normal_load_n"] for row in rows))
            vectors.extend(np.sum([row["per_pad"][pad]["force_on_rod_world_n"] for pad in PADS], axis=0)
                           for row in rows)
        quantiles = np.percentile(minimums, PERCENTILES, axis=0)
        hands[side] = dict(qualified_ticks=qualified, opposing_geometry_substeps=opposed,
                           reason_counts=dict(reasons),
                           pad_minimum_quantiles_n=dict(zip(PADS, quantiles.T.tolist(), strict=True)),
                           total_minimum_quantiles_n=np.percentile(totals, PERCENTILES).tolist(),
                           mean_force_on_rod_world_n=np.mean(vectors, axis=0).tolist())
    return dict(physical_ticks=len(evidence), solver_substeps=len(evidence)*SUBSTEPS,
                percentiles=list(PERCENTILES), hands=hands,
                bilateral_qualified_ticks=sum(int(row["bilateral"]) for row in evidence))


def audit(run, *, receipt_sha256, boundary_sdk, transition_steps, window_steps):
    """Check sealed sample bytes, ordered clocks and one actual episode before comparing."""
    if not (type(boundary_sdk) is type(transition_steps) is type(window_steps) is int
            and 0 <= transition_steps <= 64 and 1 <= window_steps <= 480 and boundary_sdk-window_steps >= 2):
        raise ValueError("valid SDK boundary and finite comparison windows required")
    run = Path(run)
    receipt = json.loads(checked_file(run/"outer.json", receipt_sha256))
    if (receipt["exit_code"] != 0 or receipt["debugger_inferior_exit_code"] != 0
            or receipt["debugger_status_verified"] is not True or receipt["debugger_inferior_signal"] is not None
            or receipt["forced_exit"] or receipt["exceeded_budget"]):
        raise ValueError("natural outer and GDB exits required")
    inventory = receipt["artifact_sha256"]
    epoch = None
    output = {}
    for name, first in (("before", boundary_sdk-window_steps+1), ("after", boundary_sdk+transition_steps+1)):
        last = first + window_steps - 1
        measurements, centers = [], []
        for tick in range(first, last+1):
            filename = f"tripod_control_{tick}.json"
            row = decode_owned(json.loads(checked_file(run/"native"/filename, inventory[filename])))
            rod = row["rod"]
            if (row["physics_step"] != tick or rod["physics_step"] != tick
                    or row["sdk_hand_geometry"]["physics_step"] != tick or rod["rod_tick"] != (tick-2)*SUBSTEPS
                    or row["guard_failure"] is not None):
                raise ValueError("same-clock qualified geometry and solver source required")
            identity = tuple(rod[key] for key in EPISODE)
            if epoch is None:
                epoch = identity
            if identity != epoch:
                raise ValueError("an action comparison cannot cross episodes or resets")
            contact = measure_supported_distal_contact(rod, grasp="tripod")
            if contact["wrong_side_contacts"] or contact["unexpected_hand_contacts"]:
                raise ValueError("ambiguous hand identity in recorded contact pair")
            measurements.append(contact)
            centers.append(rod["node_positions"][0, 20])
        output[name] = dict(summarize_window(measurements), first_sdk=first, last_sdk=last,
                            mean_center_world_m=np.mean(centers, axis=0).tolist())
    return dict(output, receipt_sha256=receipt_sha256, source_head=receipt["head"],
                action_boundary_sdk=boundary_sdk, transition_steps=transition_steps,
                evidence_scope="read_only_native_archive", teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--boundary-sdk", type=int, required=True)
    parser.add_argument("--transition-steps", type=int, required=True)
    parser.add_argument("--window-steps", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, receipt_sha256=args.receipt_sha256, boundary_sdk=args.boundary_sdk,
                           transition_steps=args.transition_steps, window_steps=args.window_steps)))


if __name__ == "__main__":
    main()
