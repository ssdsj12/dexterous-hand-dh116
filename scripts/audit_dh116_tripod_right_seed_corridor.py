#!/usr/bin/env python3
"""Screen a lower right Tripod seed using the original controller and measured clearance stress."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, checked_json, digest, load_geometry
from scripts.audit_dh116_tripod_sdk_path import WRIST, compile_actions, runtime_reference
from scripts.audit_dh116_wire_contact_witness import audit_hand, shapes_world


ROOT = Path(__file__).resolve().parents[1]
LEDGER_SHA = "036fa9766ef10b48cec900b04b9dca7c9a660cc7c231af75edda4e91c73d00ec"
OUTER_SHA = "4390107124cc125ce14b754a66934c398ed39b0fa6b4537e927e3c180ee1e274"
ENDPOINT_SHA = "4acfbab6222ee54d827b614313405470549a3c9e37561ed248b0c9467e5c1fa4"
TRACKING_SHA = "485cc1fc9511b570d6aa432cc73f038400b724dd49d0b3b7dfb634ae30f0d916"
SEED_DROP_M = .0035
PADS = DISTAL_PAD_LINKS[:3]


def tracking_stress_margin(lower_bound_m, observed_peak_m):
    if not math.isfinite(lower_bound_m) or not math.isfinite(observed_peak_m) or observed_peak_m < 0.:
        raise ValueError("finite gap and finite nonnegative same-link tracking displacement required")
    return lower_bound_m - observed_peak_m


def terminal_readback(baseline):
    receipt = checked_json(baseline / "native-outer.json", OUTER_SHA)
    ledger = baseline / "native" / "physical.jsonl"
    if receipt["exit_code"] != 0 or receipt["artifact_sha256"]["physical.jsonl"] != LEDGER_SHA:
        raise ValueError("successful sealed right Tripod tracking source required")
    if digest(ledger) != LEDGER_SHA:
        raise ValueError("sealed right Tripod ledger SHA mismatch")
    terminal = None
    with ledger.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("operation") == "physical":
                terminal = event["payload"]
    if terminal is None:
        raise ValueError("sealed right Tripod tracking ledger has no physical row")
    row = decode_owned(terminal)["exchange"]
    if row["physics_step"] != 178 or row["guard_failure"] is not None:
        raise ValueError("sealed valid SDK178 boundary required")
    return row


def evaluate(baseline, endpoint_path, tracking_path):
    row = terminal_readback(baseline)
    endpoint_file = checked_json(endpoint_path, ENDPOINT_SHA)
    if endpoint_file["source_ledger_sha256"] != LEDGER_SHA or endpoint_file["sdk_step"] != row["physics_step"]:
        raise ValueError("right endpoint must reference the same native SDK178 rod")
    endpoint = endpoint_file["after"]
    tracking = checked_json(tracking_path, TRACKING_SHA)["limits"]
    nodes = row["rod"]["node_positions"][0]
    hand = row["action_readback"]["post"]["hands"]["right"]
    indices = [hand["joint_names"].index(name) for name in ACTIVE]
    q0 = hand["q"][0, indices]
    anchor = hand["native_anchor_pose_w"][0]
    anchor_rotation = Rotation.from_quat(np.roll(anchor[3:], -1))
    axes = ("wrist_x", "wrist_y", "wrist_z", "wrist_roll", "wrist_pitch", "wrist_yaw")
    measured = hand["q"][0, [hand["joint_names"].index(name) for name in axes]]
    wrist_rotation = anchor_rotation * Rotation.from_euler("XYZ", measured[3:])
    pose0 = np.r_[anchor[:3] + anchor_rotation.apply(measured[:3]), np.roll(wrist_rotation.as_quat(), 1)]
    goal_pose, q1 = endpoint["pose_wxyz"], endpoint["q_rad"]
    measured_delta_z = float(goal_pose[2] - pose0[2])
    original_relative_z = measured_delta_z + float(row["wrist_projection"]["palm_offsets_m"][0, 1, 2])
    pose0[2] -= SEED_DROP_M
    geometry = load_geometry(ROOT, side="right")
    layout = geometry["reference"]["reference_layout"]
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    plan = compile_actions(pose0, goal_pose, q0, q1, side="right", physics_steps=176, mode="intrinsic_linear")
    samples = []
    for state in runtime_reference(plan, geometry)["rows"][::8]:
        shapes = shapes_world(geometry, state["pose_wxyz"], state["q_rad"])
        audit = audit_hand(shapes, nodes, radius, layout, PADS)
        name, gap = min(audit["gaps"].items(), key=lambda item: item[1]["lower_m"])
        stressed = {key: tracking_stress_margin(value["lower_m"], tracking[key]["peak_displacement_m"])
                    for key, value in audit["gaps"].items()}
        limiting = min(stressed, key=stressed.get)
        samples.append(dict(tick=state["physics_index"], table_m=audit["minimum_table_clearance_m"],
                            nominal_shape=name, nominal_lower_m=gap["lower_m"], stress_shape=limiting,
                            stress_lower_m=stressed[limiting], approach_gate=audit["geometric_approach_passed"],
                            direction_gate=audit["direction_gate_passed"]))
    return dict(schema="dh116-teacher-right-tripod-seed-corridor-v1", source_sdk=178,
                source_ledger_sha256=LEDGER_SHA, endpoint_sha256=ENDPOINT_SHA, tracking_sha256=TRACKING_SHA,
                measured_wrist_delta_z_m=measured_delta_z, original_relative_goal_z_m=original_relative_z,
                original_wrist_z_lower_m=WRIST["palm_lower_m"][2], proposed_seed_drop_m=SEED_DROP_M,
                proposed_relative_goal_z_m=float(goal_pose[2] - pose0[2]), plan=plan, samples=samples,
                measured_stress_passed=all(item["stress_lower_m"] > 0. for item in samples),
                continuous_tracking_qualified=False, native_ready=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "endpoint", "tracking", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.baseline, args.endpoint, args.tracking)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"measured_stress_passed": result["measured_stress_passed"], "terminal": result["samples"][-1]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
