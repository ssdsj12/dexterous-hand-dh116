#!/usr/bin/env python3
"""Replay the right Tripod near-rod approach with measured clearance guards."""

import argparse
import functools
import json
from pathlib import Path
import time

import numpy as np

from go2_pvcnn.control.dh116 import tabletop_deformable_clearance_guard as clearance
from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from go2_pvcnn.control.dh116.tabletop_palm_mount import CONTRACT as WRIST, ROTATION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import (
    MeasuredClosureGeometry,
    rod_support_lower,
)
from scripts import run_dh116_tripod_staged_wrist as staged
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, pose_matrix, transform
from scripts.audit_dh116_wire_contact_witness import endpoint_gap


VERSION = "dh116-right-tripod-near-corridor-v1"
CANDIDATE_SHA = "715bd3286c49ec21ed2e84562c14f6814be38f13de240a4f517c5c0918a6a272"
STRESS_SHA = "4ac1af53639df017f6ced93535236f8bd57e9af52c84801d69f537f09eeea025"
CLEARANCE_FLOOR_M = .0021
TABLE_TOP_M = .30
WALL_SECONDS = 520
PADS = DISTAL_PAD_LINKS[:3]


def validate_stress_rows(rows):
    if [row["sdk_step"] for row in rows] != list(range(3, staged.SDK_STEPS + 1)):
        raise ValueError("exactly one tracking-error transfer per completed SDK step required")
    staged.tracking.finite([row["native_dof_tracking_delta"] for row in rows],
                           (staged.SDK_STEPS - 2, 12))
    table = staged.tracking.finite([row["min_table"]["table_gap_m"] for row in rows],
                                  (staged.SDK_STEPS - 2,))
    non_target = staged.tracking.finite([row["nearest_non_target"]["rod_gap_lower_m"] for row in rows],
                                       (staged.SDK_STEPS - 2,))
    if np.any(table < CLEARANCE_FLOOR_M) or np.any(non_target <= 0.):
        raise ValueError("transferred clearances must preserve the original floor and clear the rod")
    return float(table.min()), float(non_target.min())


def measure_clearance(geometry, poses, nodes):
    """Use actual link poses to bound every source and frozen cooked shape."""
    table, rod = {}, {}
    for kind, shapes in geometry.items():
        for name, vertices in shapes.items():
            key = kind + ":" + name
            world = transform(vertices, poses[name])
            table[key] = float(world[:, 2].min() - TABLE_TOP_M)
            bound = rod_support_lower(world, nodes)
            if bound < 0.:
                bound = endpoint_gap(world, nodes, .005)["lower_m"]
            rod[key] = bound
    return dict(table_m=table, rod_lower_m=rod,
                passed=min(table.values()) >= CLEARANCE_FLOOR_M and min(rod.values()) > 0.)


class NearCorridorGeometry(MeasuredClosureGeometry):
    def _evaluate(self, raw, exchange):
        # The inherited capture pins raw q/qd, the rod identity, and actual link buffers.
        measured = {}
        projection = exchange["wrist_projection"]
        targets = staged.tracking.finite(projection["projected_root_poses"], (1, 2, 7))
        for index, side in enumerate(("left", "right")):
            hand = raw["hands"][side]
            poses = {name: pose_matrix(pose[[0, 1, 2, 6, 3, 4, 5]])
                     for name, pose in zip(hand["body_names"], hand["link_poses_xyzw"][0], strict=True)}
            result = measure_clearance(self._geometry[side], poses, raw["rod_nodes_m"])
            joints = raw["joint_readback"]["hands"][side]
            wrist_q = joints["q"][0, [joints["joint_names"].index(axis) for axis in staged.ANGLES]]
            base = hand["body_names"].index("base_link")
            tracking_error = float(np.linalg.norm(poses["base_link"][:3, 3] - targets[0, index, :3]))
            speed = float(np.linalg.norm(hand["link_velocities_world"][0, base, :3]))
            result.update(wrist_tracking_error_m=tracking_error, wrist_speed_m_s=speed,
                          passed=result["passed"] and tracking_error <= WRIST["tracking_guard_m"]
                          and speed <= WRIST["linear_speed_guard_m_s"]
                          and bool(np.all(np.abs(wrist_q) <= ROTATION_CONTRACT["angle_limit_rad"])))
            measured[side] = result
        request = exchange["action_readback"]["request"]
        unprojected = not (np.asarray(projection["safe_z_clipped"]).any()
                           or np.asarray(projection["projection_reasons"]).any())
        for name, source in (("palm_offsets_m", "requested_palm_offsets_m"),
                             ("projected_angles", "requested_angles_rad")):
            measured_targets = staged.tracking.finite(projection[name], (1, 2, 3))
            unprojected = unprojected and not np.any(measured_targets[0, 0])
            unprojected = unprojected and np.allclose(
                measured_targets, staged.tracking.finite(request[source], (1, 2, 3)), atol=1e-8, rtol=0.
            )
        hand_pairs = sum(pair["mesh_name"] != "table"
                         for pair in exchange["rod"]["rod_mesh_contact_pairs"]["pairs"])
        return dict(identity=raw["identity"], hands=raw["hands"], sources=self.sources,
                    sides=measured, original_wrist_projection_inactive=bool(unprojected),
                    right_target_pad_gap_lower_m={
                        name: measured["right"]["rod_lower_m"]["cooked:" + name] for name in PADS
                    }, hand_pairs=hand_pairs, teacher_admission=False, native_qualification=False,
                    passed=bool(unprojected and not hand_pairs and all(hand["passed"] for hand in measured.values())))


class NearCorridorExchange(clearance.ClearanceGuardedExchange):
    geometry = None

    def _step(self, action, substep):
        row = None
        try:
            row = super()._step(action, substep)
            if row["guard_failure"] is None:
                try:
                    audit = self.geometry.capture(row, sim=self._sim, bindings=self._bindings,
                                                  read_clock=self._rod.clock)
                    row["tripod_near_geometry"] = audit
                except Exception as error:
                    row["tripod_near_geometry_error"] = f"{type(error).__name__}: {error}"
            return row
        finally:
            self = action = row = None


def prepare(args):
    previous = staged.prepare(args)
    packet = previous["tracking_plan"]
    candidate = staged.tracking.checked_json(args.near_proof, CANDIDATE_SHA)
    stress = staged.tracking.checked_json(args.stress, STRESS_SHA)
    proof, reference = candidate["proof"], candidate["reference"]
    if (candidate["native_ready"] is not False or candidate["projections"] != 0 or proof["proved"] is not True
            or not np.isfinite(proof["minimum_clearance_margin_m"]) or proof["minimum_clearance_margin_m"] <= 0.
            or reference["side"] != "right" or reference["native_ready"] is not False
            or reference["physics_steps"] != staged.SDK_STEPS - 2
            or reference["policy_steps"] != staged.CONTROL_WINDOWS
            or stress["candidate_sha256"] != CANDIDATE_SHA
            or stress["native_qualification"] is not False
            or stress["independent_geometry_not_closed_loop_proof"] is not True):
        raise ValueError("preserved static and tracking-error transfer evidence required")
    table, gap = validate_stress_rows(stress["sampled"])
    actions = staged.tracking.finite(reference["actions"], (staged.CONTROL_WINDOWS, 24))
    with staged.cal.bindings((staged.tracking, "MOVING_CHANNELS", staged.cal.MOVING_CHANNELS)):
        for row in actions:
            staged.tracking._validate_action(row)
    before_approach = staged.APPROACH_START
    if (np.any(actions[0]) or np.any(actions[-1])
            or np.any(actions[1:1 + staged.ROTATION_WINDOWS, :21])
            or np.any(actions[1 + staged.ROTATION_WINDOWS:before_approach])
            or np.any(actions[before_approach:, 21:24])
            or any(np.any(np.linalg.norm(actions[:, axes], axis=1) > 1.)
                   for axes in (slice(15, 18), slice(21, 24)))):
        raise ValueError("original right-only normalized motion and quiet windows required")
    seed = packet["reset_seed"]
    right = staged.tracking.finite(reference["seed_pose_wxyz"], (7,))
    raised = staged.tracking.finite(packet["right_reference"]["seed_pose_wxyz"], (7,))
    if not np.allclose(right, raised - [0., 0., .016, 0., 0., 0., 0.], atol=1e-12, rtol=0):
        raise ValueError("only the certified right base height differs from the raised seed")
    if reference["q0_rad"] != packet["right_reference"]["q0_rad"]:
        raise ValueError("original frozen right joint seed required")
    new_seed = prepare_reset_state(
        hand_roots={"left": seed["hand_roots"][0], "right": right.tolist()}, origin=[0.] * 3,
        root_xy=[[0., 0.], [0., 0.]], initial_offset_m=0.,
        joint_positions=[seed["joint_positions"][0], expand(reference["q0_rad"]).tolist()],
        joint_velocities=[[0.] * 11 for _ in range(2)],
        source_description="near-rod right staged Tripod with the same parked left hand and free rod",
    )
    packet.update(reset_seed=new_seed, right_reference=reference, actions24=actions.tolist(),
                  proved_right_sweep=proof, diagnostic_version=VERSION,
                  tracking_transfer_table_min_m=table, tracking_transfer_non_target_min_m=gap)
    previous["frozen_files"].update({
        str(args.near_proof): CANDIDATE_SHA,
        str(args.stress): STRESS_SHA,
        str(Path(__file__).resolve()): staged.native.sha(Path(__file__)),
        str(Path(staged.tracking.__file__).resolve()): staged.native.sha(Path(staged.tracking.__file__)),
    })
    previous["budget"].update(mode="right_tripod_near_corridor", soft_wall_seconds=WALL_SECONDS)
    previous.update(version=VERSION, preserved_scope="right-only no-contact approach with measured geometry")
    return previous


def verify_boundary(exchange):
    if "tripod_near_geometry_error" in exchange:
        raise staged.tracking.TrackingStopped(exchange["tripod_near_geometry_error"])
    if not exchange["tripod_near_geometry"]["passed"]:
        raise staged.tracking.TrackingStopped("Tripod near-rod boundary did not certify clearance")


def acquire(sim, owners, trace, output, *, plan, args, deadline, gate_sdk=staged.GATE_SDK):
    # Build once after App launch, retaining only owned CPU geometry arrays.
    geometry = NearCorridorGeometry(staged.cal.launcher.ROOT)
    try:
        with staged.cal.bindings((NearCorridorExchange, "geometry", geometry),
                                 (clearance, "ClearanceGuardedExchange", NearCorridorExchange)):
            return staged.acquire(sim, owners, trace, output, plan=plan, args=args, deadline=deadline,
                                  observe_boundary=verify_boundary, gate_sdk=gate_sdk)
    finally:
        sim = owners = trace = geometry = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raised-study", "original", "proof", "near-proof", "stress", "output", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native = staged.native
    native.validate_execution_context(staged.cal.launcher.ROOT)
    plan = prepare(args)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if digest != args.approved_frozen_sha256 or json.loads(args.plan.read_bytes()) != plan or args.output.exists():
            raise ValueError("exact frozen near-rod candidate and fresh output required")
        deadline = time.monotonic() + WALL_SECONDS - staged.cal.launcher.CLOSE_RESERVE_SECONDS
        staged.reset.execute(
            plan, acquire_operation=functools.partial(acquire, plan=plan, args=args, deadline=deadline)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
