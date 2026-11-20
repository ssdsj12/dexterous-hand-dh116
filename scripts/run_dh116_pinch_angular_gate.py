#!/usr/bin/env python3
"""Compare one right-wrist recovery window against the sealed Pinch prefix."""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

from scripts import prepare_dh116_pinch_prelift as prelift
from scripts import prepare_dh116_pinch_wrist_first as wrist
from scripts import run_dh116_pinch_wrist_relief as relief


base, late = relief.base, relief.early.late
FIRST_SDK, LAST_SDK = 58, 66
PROOF_SHA = "0cbdb527190130101c055c3e46425b953010c178b9511d60045c7c0b4c1b8438"
VERSION = "dh116-pinch-live-angular-gate-v1"


def angular_schedule(original, rotation_request):
    prior = prelift._finite(original, (9, 24))
    request = prelift._finite(rotation_request, (24,))
    if np.any(request[:21]) or np.linalg.norm(request[21:]) > 1:
        raise ValueError("right wrist rotation only required")
    actions = prior[:8].copy()
    actions[-1, 6:12] = actions[-1, 15:18] = 0.
    actions[-1, 21:] = request[21:]
    return actions


class AngularGate:
    def __init__(self, config, prove=wrist.right_geometry):
        self.config = config
        self.prove = prove
        self.nodes = None

    def observe(self, row):
        step = row["physics_step"]
        if row["guard_failure"] is not None or step < FIRST_SDK:
            return
        if step == FIRST_SDK:
            self._admit(row)
            return
        prelift._same_clock(row)
        nodes = prelift._finite(row["rod"]["node_positions"][0], (41, 3))
        if self.nodes is None:
            raise ValueError("current-instance angular gate missing")
        motion = float(np.linalg.norm(nodes - self.nodes, axis=1).max())
        row["angular_rod_motion_m"] = motion
        if motion > prelift.NODE_RESERVE_M:
            row["guard_failure"] = "pinch_angular_rod_reserve_exceeded"

    def _admit(self, row):
        receipt = row["angular_gate"] = {"passed": False}
        try:
            observed = late.prefix_measurements(row, expected_sdk=FIRST_SDK)
            prefix = late.compare_prefix(observed, self.config["prefix"])
            receipt["prefix"] = prefix
            if not prefix["passed"]:
                row["guard_failure"] = "pinch_angular_prefix_mismatch"
                return
            receipt["identity"] = prelift._same_clock(row)
            nodes = prelift._finite(row["rod"]["node_positions"][0], (41, 3))
            actual, issued = prelift.joint_and_wrist(row, "right")
            nominal = prelift._finite(self.config["nominal"], (12,)).copy()
            rotation = prelift._finite(self.config["rotation"], (3,))
            nominal[9:] = issued[9:] + prelift.WINDOW_SCALE[21:] * rotation
            envelope = wrist.phase_envelope(actual, issued, nominal, release_progress=wrist.RELEASE_PROGRESS)
            proofs, _ = self.prove(row, actual, issued, nominal, envelope, nodes=nodes,
                                    phase_names=("angle_first",), phase_seconds=25.)
            receipt["phase_bounds"] = [a.tolist() for a in envelope["angle_first"]]
            receipt["right_angular_target_rad"] = nominal[9:].tolist()
            receipt["proof"] = proofs["angle_first"]
            if not receipt["proof"]["proved"]:
                row["guard_failure"] = "pinch_angular_live_certificate_rejected"
                return
            self.nodes = nodes.copy()
            receipt["passed"] = True
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            row["guard_failure"] = "pinch_angular_source_invalid"


def prepare(args):
    parent = relief.prepare(args)
    endpoint, _ = prelift.archived_endpoint(args.previous)
    source_path = args.previous / "native" / "frozen_plan.json"
    receipt = base._checked(args.previous / "outer.json", prelift.SOURCE_OUTER_SHA)
    previous = base._checked(source_path, receipt["plan_sha256"])
    proof = base._checked(args.angular_proof, PROOF_SHA)
    if (previous["action_schedule"] != parent["action_schedule"]
            or proof["source_ledger_sha256"] != prelift.SOURCE_LEDGER_SHA
            or proof["sdk_step"] != FIRST_SDK
            or not proof["conditioned_right_pose_proof"]["angle_first"]["proved"]):
        raise ValueError("sealed right angular contrast and unchanged left prefix required")
    nominal = prelift.retained_target(*prelift.joint_and_wrist(endpoint, "right"), "right")
    if nominal.tolist() != proof["full_linear_progress_reference"]:
        raise ValueError("right angular target differs from sealed endpoint")
    action = angular_schedule(parent["action_schedule"], proof["angle_only_requests"][0])
    parent["frozen_files"].update({
        str(args.angular_proof): PROOF_SHA,
        str(args.previous / "outer.json"): prelift.SOURCE_OUTER_SHA,
        str(source_path): receipt["plan_sha256"],
        str(args.previous / "native" / "physical.jsonl"): prelift.SOURCE_LEDGER_SHA,
    })
    for module in (prelift, wrist):
        path = Path(module.__file__).resolve()
        parent["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    path = Path(__file__).resolve()
    parent["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    parent.update(staged_version=VERSION, action_schedule=action.tolist(),
                  angular_gate=dict(prefix=parent["wrist_relief_prefix"], nominal=nominal.tolist(),
                                    rotation=action[-1, 21:].tolist()),
                  preserved_scope="one angle-only contrast, zero additional left windows, no lift or PPO")
    parent["budget"].update(control_steps=len(action), sdk_steps=LAST_SDK,
                            solver_steps=(LAST_SDK - 2) * base.SUBSTEPS)
    return parent


def acquire(sim, owners, trace, output, *, plan):
    gate = AngularGate(plan["angular_gate"])

    class ObservedAngularClose(base.ObservedClose):
        def _step(self, action, substep):
            row = None
            try:
                row = super()._step(action, substep)
                relief.early.guard_prefix(row, plan["early_finger_prefix"])
                if FIRST_SDK < row["physics_step"] and row["guard_failure"] is None:
                    try:
                        gate.observe(row)
                    except Exception as error:
                        row["angular_monitor_error"] = f"{type(error).__name__}: {error}"
                        row["guard_failure"] = "pinch_angular_monitor_source_invalid"
                else:
                    gate.observe(row)
                return row
            finally:
                self = action = row = None

    try:
        with base.bindings((base, "ObservedClose", ObservedAngularClose)):
            return base.acquire(sim, owners, trace, output, plan=plan, sdk_steps=LAST_SDK)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "sealed-reset", "baseline", "preflight", "archive",
                 "wrist-preflight", "previous", "angular-proof"):
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
            raise ValueError("exact approved angular plan and fresh output required")
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
