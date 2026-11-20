#!/usr/bin/env python3
"""Rotate and verify the measured right wrist before raised Tripod approach."""
import argparse
import functools
import json
from pathlib import Path
import time
from collections import deque
from uuid import uuid4

import numpy as np

from scripts import run_dh116_tripod_right_axis_calibration as cal


tracking, native, reset = cal.tracking, cal.native, cal.reset
VERSION = "dh116-right-tripod-staged-wrist-v1"
PROOF_SHA = "e41f555689f1840bb3c368fa0758c6b19ca2dd179e643afc32a46e2fd744a807"
ROTATION_WINDOWS = 8
HOLD_WINDOWS = 32
APPROACH_START = 1 + ROTATION_WINDOWS + HOLD_WINDOWS
CONTROL_WINDOWS = APPROACH_START + 20 + 1
GATE_SDK = 2 + 8 * APPROACH_START
SDK_STEPS = 2 + CONTROL_WINDOWS * 8
WALL_SECONDS = 280
BUDGET = dict(tracking.BUDGET, sdk_steps=SDK_STEPS, motion=(CONTROL_WINDOWS - 2) * 8)
ANGLES = ("wrist_roll", "wrist_pitch", "wrist_yaw")
ANGLE_ERROR_MAX_RAD = .0002
SPEED_MAX_RAD_S = .001


def staged_reference(reference):
    base = tracking.finite(reference["actions"], (20, 24))
    ramp = np.zeros((ROTATION_WINDOWS, 24))
    ramp[:, 21:24] = base[:, 21:24].sum(axis=0) / ROTATION_WINDOWS
    approach = base.copy()
    approach[:, 21:24] = 0.
    actions = np.vstack((np.zeros((1, 24)), ramp, np.zeros((HOLD_WINDOWS, 24)), approach, np.zeros((1, 24))))
    endpoint = tracking.finite(reference["intrinsic_knots_rad"], (21, 3))[-1]
    angles = np.vstack((np.zeros((1, 3)), np.linspace(np.zeros(3), endpoint, ROTATION_WINDOWS + 1),
                        np.tile(endpoint, (HOLD_WINDOWS + 21, 1))))
    offsets = tracking.finite(reference["world_offset_knots_m"], (21, 3))
    offsets = np.vstack((np.zeros((APPROACH_START, 3)), offsets, offsets[-1:]))
    if (np.any(np.delete(actions, cal.MOVING_CHANNELS, axis=1)) or np.any(np.abs(actions) > 1.)
            or any(np.any(np.linalg.norm(actions[:, part], axis=1) > 1.)
                   for part in (slice(15, 18), slice(21, 24)))):
        raise ValueError("staged actions exceed the original right-only contract")
    return dict(reference, physics_steps=SDK_STEPS-2, policy_steps=CONTROL_WINDOWS,
                actions=actions.tolist(), intrinsic_knots_rad=angles.tolist(), world_offset_knots_m=offsets.tolist())


def wrist_gate(rows, desired, *, expected_sdk):
    desired = tracking.finite(desired, (3,))
    if [row["sdk_step"] for row in rows] != list(range(expected_sdk - 7, expected_sdk + 1)):
        return dict(passed=False, reason="missing final eight same-instance wrist samples")
    target, q, qd = (tracking.finite([row[key] for row in rows], (8, 3)) for key in ("target", "q", "qd"))
    target_error = float(np.max(abs(target - desired)))
    tracking_error = float(np.max(abs(target - q)))
    speed = float(np.max(abs(qd)))
    return dict(passed=target_error <= 1e-6 and tracking_error <= ANGLE_ERROR_MAX_RAD and speed <= SPEED_MAX_RAD_S,
                sdk_step=expected_sdk, target_error_rad=target_error, tracking_error_rad=tracking_error,
                speed_rad_s=speed, required_angle_error_rad=ANGLE_ERROR_MAX_RAD,
                required_speed_rad_s=SPEED_MAX_RAD_S, rows=list(rows))


def prepare(args):
    original = cal.prepare(args)
    reference = staged_reference(original["tracking_plan"]["right_reference"])
    proof = tracking.checked_json(args.proof, PROOF_SHA)
    if (proof["reference"] != reference or proof["original_raised_study_sha256"] != cal.CANDIDATE_SHA
            or proof["native_ready"] is not False or proof["projections"] != 0
            or not proof["proof"]["proved"] or proof["proof"]["minimum_clearance_margin_m"] < .002):
        raise ValueError("independent original-geometry continuous proof required")
    packet = dict(original["tracking_plan"], actions24=reference["actions"], right_reference=reference,
                  budget=BUDGET, diagnostic_version=VERSION, proved_right_sweep=proof["proof"])
    original["budget"].update(mode="right_tripod_staged_wrist", sdk_steps=SDK_STEPS, control_steps=CONTROL_WINDOWS,
                              solver_steps=(SDK_STEPS-2)*12, soft_wall_seconds=WALL_SECONDS)
    original["frozen_files"].update({str(args.proof):PROOF_SHA,
                                     str(Path(__file__).resolve()):native.sha(Path(__file__))})
    original.update(version=VERSION, tracking_plan=packet,
                    preserved_scope="raised no-contact staged right wrist and approach readback")
    return original


def acquire(sim, owners, trace, output, *, plan, args, deadline, observe_boundary=None):
    samples = deque(maxlen=8)
    run_uuid = "tripod-staged-wrist-" + uuid4().hex
    packet = plan["tracking_plan"]
    actions = tracking.finite(packet["actions24"], (CONTROL_WINDOWS, 24))
    desired = packet["right_reference"]["intrinsic_knots_rad"][-1]

    def observe(exchange):
        if observe_boundary is not None:
            observe_boundary(exchange)
        hand = exchange["action_readback"]["post"]["hands"]["right"]
        ids = [hand["joint_names"].index(name) for name in ANGLES]
        samples.append(dict(sdk_step=exchange["physics_step"], target=hand["backend_targets"][0, ids].tolist(),
                            q=hand["q"][0, ids].tolist(), qd=hand["qd"][0, ids].tolist()))
        if exchange["physics_step"] == GATE_SDK:
            gate = wrist_gate(list(samples), desired, expected_sdk=GATE_SDK)
            native.write_new(output/"wrist_gate.json", gate)
            if not gate["passed"]:
                raise tracking.TrackingStopped("measured right Tripod wrist not settled")

    def read_plan(path, expected):
        if Path(path) != args.plan or native.sha(args.plan) != expected:
            raise ValueError("exact staged wrist plan required")
        return packet, actions

    try:
        with cal.bindings((tracking, "BUDGET", BUDGET), (tracking, "MOVING_CHANNELS", cal.MOVING_CHANNELS),
                          (tracking, "read_tracking_plan", read_plan), (tracking, "validate_schedule", lambda _: actions)):
            return tracking.run_owned_tracking(
                sim, owners, trace, project=cal.launcher.ROOT, plan_path=args.plan, plan_sha256=native.sha(args.plan),
                output=output, run_uuid=run_uuid, episode_uid=run_uuid + ":0", soft_deadline=deadline,
                observe=observe,
            )
    except tracking.TrackingStopped as error:
        return dict(failure=str(error), guarded_stop_sdk_step=native.clock(sim)["sdk_step"],
                    grasp_qualified=False, teacher_admission=False, ppo_updates=0)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("raised-study", "original", "proof", "output", "plan"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native.validate_execution_context(cal.launcher.ROOT)
    plan = prepare(args)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if args.approved_frozen_sha256 != digest or json.loads(args.plan.read_bytes()) != plan or args.output.exists():
            raise ValueError("exact approved staged wrist plan and fresh output required")
        deadline = time.monotonic() + WALL_SECONDS - cal.launcher.CLOSE_RESERVE_SECONDS
        reset.execute(plan, acquire_operation=functools.partial(acquire, plan=plan, args=args, deadline=deadline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
