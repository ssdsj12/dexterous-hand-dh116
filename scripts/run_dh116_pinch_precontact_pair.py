#!/usr/bin/env python3
"""Run the left precontact pause and early right angular Pinch contrast."""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

from scripts import prepare_dh116_pinch_precontact_hold as hold
from scripts import run_dh116_pinch_angular_gate as angular


base, prelift, late, wrist = angular.base, angular.prelift, angular.late, angular.wrist
LEFT = np.r_[0:6, 12:15, 18:21]
RIGHT = np.r_[6:12, 15:18, 21:24]
PAUSE_SDK, ANGULAR_SDK, LAST_SDK = 18, 50, 58
LEFT_SHA = "1e8cbc3522094799df6f116a37c288252e72f65660ca1bc952840f763d310254"
RIGHT_SHA = "8119489b49ef0e6935137a8ceb7bb325e0701b288fb5b8642052ee6805ef4834"


def pair_schedule(previous, hold_commands, rotation):
    actions = prelift._finite(previous, (9, 24))[:7].copy()
    stop = prelift._finite(hold_commands, (2, 24))
    delta = prelift._finite(rotation, (3,))
    if np.any(stop[:, RIGHT]) or np.linalg.norm(delta) > 1:
        raise ValueError("isolated left pause and bounded right angular action required")
    actions[2:, LEFT] = 0.
    actions[2:4, LEFT] = stop[:, LEFT]
    actions[-1, RIGHT] = np.r_[np.zeros(9), delta]
    return actions


def guard_hand_contact(row):
    if row["guard_failure"] is None and any(
            pair["mesh_name"] != "table" for pair in row["rod"]["rod_mesh_contact_pairs"]["pairs"]):
        row["guard_failure"] = "pinch_precontact_unexpected_hand_load"


def right_prefix(measured, reference):
    comparison = dict(reference)
    for key in ("left_q", "left_pose"):
        comparison[key] = measured[key]
    return late.compare_prefix(measured, comparison)


def prepare(args):
    previous = angular.prepare(args)
    source = base._checked(args.left_proof, LEFT_SHA)
    right_proof = base._checked(args.right_proof, RIGHT_SHA)["angle_first"]
    if (source["source_ledger_sha256"] != prelift.SOURCE_LEDGER_SHA
            or source["identity"]["physics_step"] != PAUSE_SDK
            or not source["proof"]["proved"] or not right_proof["proved"]):
        raise ValueError("exact earlier same-source conditional proofs required")
    previous_path = args.previous / "native" / "frozen_plan.json"
    receipt = base._checked(args.previous / "outer.json", prelift.SOURCE_OUTER_SHA)
    plan = base._checked(previous_path, receipt["plan_sha256"])
    rows = {}
    with (args.previous / "native" / "physical.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("operation") == "physical" and event["payload"]["exchange"]["physics_step"] in (PAUSE_SDK, ANGULAR_SDK):
                row = base.decode_owned(event["payload"])["exchange"]
                rows[row["physics_step"]] = row
    actual, target = prelift.joint_and_wrist(rows[PAUSE_SDK], "left")
    hold_actions, _ = hold.left_hold_windows(actual, target)
    if hold_actions.tolist() != source["requested_windows"]:
        raise ValueError("left pause source actions differ from conditional proof")
    _, issued = prelift.joint_and_wrist(rows[ANGULAR_SDK], "right")
    nominal = prelift._finite(previous["angular_gate"]["nominal"], (12,))
    angular_actions, _, _ = wrist.right_control_windows(issued, nominal)
    if np.any(angular_actions[1]):
        raise ValueError("right angular target requires more than one approved window")
    actions = pair_schedule(plan["action_schedule"], hold_actions, angular_actions[0, 21:])
    for path in (Path(__file__).resolve(), Path(hold.__file__).resolve(), Path(angular.__file__).resolve()):
        previous["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    previous["frozen_files"].update({str(args.left_proof): LEFT_SHA, str(args.right_proof): RIGHT_SHA})
    previous.update(staged_version="dh116-pinch-precontact-pair-v1", action_schedule=actions.tolist(),
                    left_gate=dict(prefix=late.prefix_measurements(rows[PAUSE_SDK], expected_sdk=PAUSE_SDK),
                                   actions=hold_actions.tolist()),
                    angular_gate=dict(prefix=late.prefix_measurements(rows[ANGULAR_SDK], expected_sdk=ANGULAR_SDK),
                                      nominal=nominal.tolist(), rotation=actions[-1, 21:].tolist()),
                    preserved_scope="one precontact left pause and one right angle window, no grasp or PPO")
    previous["budget"].update(control_steps=len(actions), sdk_steps=LAST_SDK,
                              solver_steps=(LAST_SDK - 2) * base.SUBSTEPS)
    return previous


def acquire(sim, owners, trace, output, *, plan):
    angular_gate = angular.AngularGate(plan["angular_gate"], sdk=ANGULAR_SDK, compare=right_prefix)
    left_nodes = None

    class ObservedPairClose(base.ObservedClose):
        def _step(self, action, substep):
            nonlocal left_nodes
            row = None
            try:
                row = super()._step(action, substep)
                guard_hand_contact(row)
                step = row["physics_step"]
                if row["guard_failure"] is not None or step < PAUSE_SDK:
                    return row
                try:
                    nodes = prelift._finite(row["rod"]["node_positions"][0], (41, 3))
                    if step == PAUSE_SDK:
                        cfg = plan["left_gate"]
                        prefix = late.compare_prefix(late.prefix_measurements(row, expected_sdk=step), cfg["prefix"])
                        actual, issued = prelift.joint_and_wrist(row, "left")
                        requests, _ = hold.left_hold_windows(actual, issued)
                        same_request = bool(np.allclose(requests, cfg["actions"], atol=1e-7, rtol=0.))
                        receipt = row["precontact_left_gate"] = dict(prefix=prefix, same_request=same_request, passed=False)
                        if prefix["passed"] and same_request:
                            receipt["proof"] = hold.certify_left_hold(row, actual, issued)
                            receipt["passed"] = receipt["proof"]["proved"]
                        if not receipt["passed"]:
                            row["guard_failure"] = "pinch_precontact_left_gate_rejected"
                            return row
                        left_nodes = nodes.copy()
                    else:
                        prelift._same_clock(row)
                        motion = float(np.linalg.norm(nodes - left_nodes, axis=1).max())
                        row["precontact_left_rod_motion_m"] = motion
                        if motion > prelift.NODE_RESERVE_M:
                            row["guard_failure"] = "pinch_precontact_left_rod_reserve_exceeded"
                            return row
                    angular_gate.observe(row)
                except Exception as error:
                    row["precontact_source_error"] = f"{type(error).__name__}: {error}"
                    row["guard_failure"] = "pinch_precontact_source_invalid"
                return row
            finally:
                self = action = row = None

    try:
        with base.bindings((base, "ObservedClose", ObservedPairClose)):
            return base.acquire(sim, owners, trace, output, plan=plan, sdk_steps=LAST_SDK)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "sealed-reset", "baseline", "preflight", "archive",
                 "wrist-preflight", "previous", "angular-proof", "left-proof", "right-proof"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native = base.lifecycle.native
    native.validate_execution_context(base.ROOT)
    plan = prepare(args)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if (args.approved_frozen_sha256 != digest or json.loads(args.plan.read_text()) != plan
                or args.output.exists()):
            raise ValueError("exact approved precontact plan and fresh output required")
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
