#!/usr/bin/env python3
"""Identify which right Tripod candidate axes lack matching historical motion."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts import audit_dh116_tripod_right_seed_corridor as corridor


WRIST = ("wrist_x", "wrist_y", "wrist_z", "wrist_roll", "wrist_pitch", "wrist_yaw")
AXES = (*ACTIVE, *WRIST)
CANDIDATE_SHA = "fcd4b86385507b3564002bea9d052b010779f06a4c361f21cd56aaad366190fa"
COMMAND_EPS = 1e-7


def axis_profile(target, actual):
    target, actual = np.asarray(target, dtype=float), np.asarray(actual, dtype=float)
    if (target.ndim != 1 or len(target) < 2 or actual.shape != target.shape
            or not np.isfinite([target, actual]).all()):
        raise ValueError("finite same-clock target and actual series required")
    delta = np.diff(target)
    excited = np.abs(delta) > COMMAND_EPS
    error = target - actual
    delay = error[1:][excited] / delta[excited]
    return dict(motion_steps=int(excited.sum()),
                positive_steps=int(np.count_nonzero(delta > COMMAND_EPS)),
                negative_steps=int(np.count_nonzero(delta < -COMMAND_EPS)),
                peak_signed_error=[float(error.min()), float(error.max())],
                apparent_delay_ticks=[float(delay.min()), float(delay.max())] if len(delay) else None)


def classify_direction(delta, profile):
    if abs(delta) <= COMMAND_EPS:
        return "stationary_in_candidate"
    if not profile["motion_steps"]:
        return "no_motion_observed"
    direction = "positive_steps" if delta > 0. else "negative_steps"
    return "historical_direction_only" if profile[direction] else "direction_not_observed"


def evaluate(baseline, candidate_path):
    candidate = corridor.checked_json(candidate_path, CANDIDATE_SHA)
    if candidate["source_ledger_sha256"] != corridor.LEDGER_SHA:
        raise ValueError("candidate and tracking history must share the right-hand source")
    row = corridor.terminal_readback(baseline)
    if row["physics_step"] != 178:
        raise ValueError("sealed right tracking boundary required")
    ledger = baseline / "native/physical.jsonl"
    actual, target, ticks = [], [], []
    with ledger.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("operation") != "physical":
                continue
            row = decode_owned(event["payload"])["exchange"]
            hand = row["action_readback"]["post"]["hands"]["right"]
            indices = [hand["joint_names"].index(key) for key in AXES]
            actual.append(hand["q"][0, indices])
            target.append(hand["backend_targets"][0, indices])
            ticks.append(row["physics_step"])
    if ticks != list(range(3, 179)):
        raise ValueError("complete 176-sample right Tripod SDK series required")
    plan = candidate["plan"]
    start = np.asarray(plan["seed_pose_wxyz"])
    end = np.asarray(plan["end_pose_wxyz"])
    mount = Rotation.from_quat(np.roll(start[3:], -1))
    local = mount.inv().apply(end[:3] - start[:3])
    wrist_angles = np.asarray(plan["intrinsic_knots_rad"][-1])
    requested = np.r_[np.asarray(plan["q1_rad"]) - plan["q0_rad"], local, wrist_angles]
    actual, target = np.array(actual), np.array(target)
    axes = {}
    for j, (key, movement) in enumerate(zip(AXES, requested)):
        stats = axis_profile(target[:, j], actual[:, j])
        axes[key] = dict(**stats, candidate_delta=float(movement), evidence=classify_direction(movement, stats))
    return dict(schema="dh116-right-tripod-axis-observability-v1", source_ledger_sha256=corridor.LEDGER_SHA,
                candidate_sha256=CANDIDATE_SHA, measured_ticks=len(ticks), axes=axes,
                native_lag_bound_qualified=False, candidate_ready=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.baseline, args.candidate)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({axis: item["evidence"] for axis, item in result["axes"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
