#!/usr/bin/env python3
"""Run one raised, right-only Tripod actuator calibration on the original free rod."""
import argparse
import functools
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from scripts import launch_dh116_tripod_right_tracking as launcher
from scripts import verify_dh116_tripod_right_tracking as tracking
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings


reset, native = launcher.reset, launcher.native
CANDIDATE_SHA = "f264d1dfdbaea83d2f9ebc7d6e451e7c5a58dd17cfdb8b2e7c86d269ac1581df"
MOVING_CHANNELS = (6, 7, 8, 9, 15, 16, 17, 21, 22, 23)
VERSION = "dh116-right-tripod-axis-calibration-v1"


def tracking_actions(motion):
    actions = tracking.finite(motion, (20, 24))
    if np.any(np.delete(actions, MOVING_CHANNELS, axis=1)) or np.any(np.abs(actions) > 1.):
        raise ValueError("original right-only action axes are required")
    if any(np.any(np.linalg.norm(actions[:, part], axis=1) > 1.) for part in (slice(15, 18), slice(21, 24))):
        raise ValueError("original normalized wrist vector limit exceeded")
    if not np.all(np.any(actions, axis=1)):
        raise ValueError("all calibration motion windows must be active")
    return np.vstack((np.zeros((1, 24)), actions, np.zeros((1, 24))))


def make_tracking_packet(original, right, seed, proof):
    packet = dict(original, diagnostic_version=VERSION, actions24=tracking_actions(right["actions"]).tolist(),
                  reset_seed=seed, right_reference=right, proved_right_sweep=proof,
                  raised_study_sha256=CANDIDATE_SHA)
    with bindings((tracking, "MOVING_CHANNELS", MOVING_CHANNELS)):
        tracking.validate_schedule(packet)
    return packet


def prepare(args):
    raised = tracking.checked_json(args.raised_study, CANDIDATE_SHA)
    original, _ = tracking.read_tracking_plan(args.original, launcher.STRAIGHT_TRACKING_SHA)
    if (raised["side"] != "right" or raised["offset_m"] != [0., 0., .016] or raised["projections"] != 0
            or raised["proof"]["proved"] is not True or raised["proof"]["minimum_clearance_margin_m"] < .002
            or raised["native_ready"] is not False):
        raise ValueError("raised right-only static proof on the original free rod required")
    right = raised["plan"]
    left_seed = original["reset_seed"]
    seed = prepare_reset_state(
        hand_roots={"left": left_seed["hand_roots"][0], "right": right["seed_pose_wxyz"]},
        origin=[0.] * 3, root_xy=[[0., 0.], [0., 0.]], initial_offset_m=0.,
        joint_positions=[left_seed["joint_positions"][0], expand(right["q0_rad"]).tolist()],
        joint_velocities=[[0.] * 11 for _ in range(2)],
        source_description="right mounted Tripod dynamics with original left parked and rod free on table",
    )
    packet = make_tracking_packet(original, right, seed, raised["proof"])
    plan = reset.freeze_plan(args.output)
    plan["budget"].update(mode="right_tripod_axis_calibration", sdk_steps=178, control_steps=22,
                          solver_steps=2112, soft_wall_seconds=launcher.WALL_SECONDS)
    plan["frozen_files"].update({str(args.raised_study): CANDIDATE_SHA, str(args.original): launcher.STRAIGHT_TRACKING_SHA,
                                 str(Path(__file__).resolve()): native.sha(Path(__file__))})
    plan.update(version=VERSION, tracking_plan=packet, preserved_scope="right actuator calibration with no hand contact")
    return plan


def acquire(sim, owners, trace, output, *, plan, args, deadline):
    run_uuid = "tripod-right-axis-calibration-" + uuid4().hex

    def read_plan(path, expected):
        expected_file = args.plan.resolve(strict=True)
        if Path(path) != expected_file or native.sha(expected_file) != expected:
            raise ValueError("exact frozen axis calibration plan required")
        source = json.loads(expected_file.read_bytes())["tracking_plan"]
        if source != plan["tracking_plan"]:
            raise ValueError("axis calibration plan differs from its frozen identity")
        return source, tracking_actions(source["right_reference"]["actions"])

    try:
        with bindings((tracking, "MOVING_CHANNELS", MOVING_CHANNELS), (tracking, "read_tracking_plan", read_plan)):
            return tracking.run_owned_tracking(
                sim, owners, trace, project=launcher.ROOT, plan_path=args.plan.resolve(strict=True),
                plan_sha256=native.sha(args.plan), output=output, run_uuid=run_uuid,
                episode_uid=run_uuid + ":0", soft_deadline=deadline,
            )
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raised-study", "original", "output", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native.validate_execution_context(launcher.ROOT)
    plan = prepare(args)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if (args.approved_frozen_sha256 != digest or json.loads(args.plan.read_bytes()) != plan
                or args.output.exists()):
            raise ValueError("exact approved axis diagnostic and fresh output required")
        deadline = time.monotonic() + launcher.WALL_SECONDS - launcher.CLOSE_RESERVE_SECONDS
        reset.execute(plan, acquire_operation=functools.partial(acquire, plan=plan, args=args, deadline=deadline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
