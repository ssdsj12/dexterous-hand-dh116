"""Replay recorded supported-rod exchanges against their fixed measured Reset."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_supported_lift_observation import (
    FixedLiftReference, observe_supported_lift,
)


def audit(archive, receipt_path, *, prefix, grasp, first, last, prelift_stable_s=None):
    """Recompute every endpoint in a contiguous, externally hashed cohort."""
    receipt_bytes = receipt_path.read_bytes()
    inventory = json.loads(receipt_bytes)["artifact_sha256"]

    def read(name):
        raw = (archive / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != inventory[name]:
            raise ValueError("archive file SHA differs: " + name)
        return decode_owned(json.loads(raw))

    event = read(f"{prefix}_reset_source_1.json")
    reset = event["reset"]
    rod_reset = reset["rod_reset"]
    if (event["kind"] != "reset" or reset["status"] != "complete"
            or any(rod_reset.get(key) != value for key, value in reset["after_clock"].items())
            or not np.array_equal(rod_reset["node_positions"],
                                  event["rod_boundary"]["rod_snapshot"]["node_positions"])):
        raise ValueError("same verified Reset clock and rod positions required")

    def geometry(index, side):
        event = read(f"{prefix}_reset_source_{index}.json")
        row = event["evidence"]
        identity = row["identity"]
        if (event["kind"] != "initial_geometry" or row["side"] != side
                or identity["sdk_step"] != rod_reset["physics_step"]
                or not math.isclose(identity["sdk_time_s"], rod_reset["timestamp_s"],
                                    abs_tol=2e-7, rel_tol=0.)):
            raise ValueError("same Reset table and radius configuration required")
        return row["rod_radius_m"], row["table_center_m"][2] + row["table_size_m"][2] / 2

    radius, table_top = geometry(2, "left")
    if geometry(3, "right") != (radius, table_top):
        raise ValueError("left/right Reset table and radius disagree")
    reference = FixedLiftReference.from_rod_reset(rod_reset, radius_m=radius, table_top_m=table_top)
    progress = None
    if prelift_stable_s is not None:
        if first != reference.physics_step + 1:
            raise ValueError("phase replay must begin at the first exchange after Reset")
        from go2_pvcnn.control.dh116.tabletop_teacher_lift_progress import SupportedLiftProgress
        progress = SupportedLiftProgress(reference, grasp=grasp, prelift_stable_s=prelift_stable_s)
    rows = []
    for tick in range(first, last + 1):
        control = f"{prefix}_control_{tick}.json"
        if control in inventory:
            event = read(control)
        elif progress is not None:
            event = read(f"{prefix}_exchange_{tick}.json")
            previous_tick = None if tick == reference.physics_step + 1 else tick - 1
            if event["preceding_reaction_sdk_step"] != previous_tick:
                raise ValueError("parked exchange reaction source clock differs")
            event.update(physics_step=tick, sdk_hand_geometry=dict(
                physics_step=tick, poses=event["hand_link_poses"]))
        else:
            raise ValueError("recorded control row missing: " + control)
        if event["physics_step"] != tick or event["rod"]["physics_step"] != tick:
            raise ValueError("stored control row physics identity differs")
        if progress is None:
            rows.append(asdict(observe_supported_lift(event["rod"], reference, grasp=grasp)))
        else:
            result = progress.observe(event)
            rows.append(dict(asdict(result.observation), phase=asdict(result.phase),
                palm_speed_m_s=result.palm_speed_m_s, relative_node_speed_m_s=result.relative_node_speed_m_s,
                lift_potential=result.lift_potential))
    valid_lifts = [row["off_support_lift_m"] for row in rows if row["off_support_lift_m"] is not None]
    phase_counts = None
    if progress is not None:
        from collections import Counter
        phase_counts = dict(Counter(row["phase"]["phase"] for row in rows))
    return dict(scope="recorded_native_exchanges_offline", grasp=grasp,
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(), reference=asdict(reference),
        first_physics_step=first, last_physics_step=last, row_count=len(rows),
        bilateral_rows=sum(row["bilateral"] for row in rows),
        registered_support_clear_rows=sum(row["registered_support_clear"] for row in rows),
        off_support_lift_rows=len(valid_lifts), maximum_off_support_lift_m=max(valid_lifts, default=None),
        largest_minimum_node_delta_m=max(row["minimum_delta_m"] for row in rows),
        rows_reaching_off_support_30mm=sum(value >= .03 for value in valid_lifts),
        phase_counts=phase_counts, prelift_stable_s=prelift_stable_s,
        new_physics_steps=0, ppo_updates=0, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--prefix", choices=("pinch", "tripod", "power"), required=True)
    parser.add_argument("--grasp", choices=("pinch", "tripod", "power"), required=True)
    parser.add_argument("--prelift-stable-s", type=float)
    parser.add_argument("--first", required=True, type=int)
    parser.add_argument("--last", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.first <= 0 or args.last < args.first:
        parser.error("a nonempty positive physics step interval is required")
    result = audit(args.archive, args.receipt, prefix=args.prefix, grasp=args.grasp,
                   first=args.first, last=args.last, prelift_stable_s=args.prelift_stable_s)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
