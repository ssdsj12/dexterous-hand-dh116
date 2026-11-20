#!/usr/bin/env python3
"""Bounded comparison with a smaller left-wrist change in the last close window."""
import argparse
import functools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import run_dh116_staged_pinch_contact as base
from scripts.audit_dh116_pinch_early_approach import (
    _bind_pad_grids, _bound_model, _contact, _finite, _frames, _non_target_certificate,
    _same_clock, PADS, SIDES,
)
from go2_pvcnn.control.dh116.linear_coupling import JOINTS, expand
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

ROOT = base.ROOT
VERSION = "dh116-pinch-late-left-wrist-v1"
BASELINE_OUTER_SHA = "86e870cd65aa72e0bb79cc734258691bad43900dceba61c46368e6f388163b0e"
BASELINE_LEDGER_SHA = "2c3f9693ec9110c5dddf1e225585beb6b57b12f293dd9f86fc0414c767200202"
LAST_PREFIX_SDK = 66
WINDOW_S = 8 / 480
REMAINING_WRIST_FRACTION = .25
NODE_SENSITIVITY_M = .00005


def variant_schedule(actions, xyz_delta, euler_delta):
    result = _finite(actions, (9, 24)).copy()
    if np.any(abs(result) > 1.):
        raise ValueError("original normalized action range required")
    result[-1, 12:15] += _finite(xyz_delta, (3,)) / (WINDOW_S * ACTION_CONTRACT["translation_rate_m_s"])
    result[-1, 18:21] += _finite(euler_delta, (3,)) / (WINDOW_S * ACTION_CONTRACT["rotation_rate_rad_s"])
    if max(np.linalg.norm(result[-1, 12:15]), np.linalg.norm(result[-1, 18:21])) > 1.:
        raise ValueError("adjusted left wrist exceeds an original rate ball")
    return result


def prefix_measurements(row):
    _same_clock(row)
    if row["physics_step"] != LAST_PREFIX_SDK or row["guard_failure"] is not None:
        raise ValueError("same-clock safe SDK66 readback required")
    geometry = row["closure_geometry"]["raw"]["hands"]
    readback = row["action_readback"]["post"]["hands"]
    measured = {"rod_nodes_m": _finite(row["rod"]["node_positions"][0], (41, 3)).tolist()}
    for side in SIDES:
        names = geometry[side]["body_names"]
        pose = _finite(geometry[side]["link_poses_xyzw"][0], (23, 7))[names.index("base_link")]
        measured[side + "_pose"] = pose.tolist()
        measured[side + "_q"] = _finite(readback[side]["q"][0], (17,)).tolist()
    return measured


def compare_prefix(current, expected):
    rod = np.linalg.norm(_finite(current["rod_nodes_m"], (41, 3))
                         - _finite(expected["rod_nodes_m"], (41, 3)), axis=-1)
    metrics = {"max_rod_node_error_m": float(rod.max())}
    passed = metrics["max_rod_node_error_m"] <= .000005
    for side in SIDES:
        q, q0 = [_finite(p[side + "_q"], (17,)) for p in (current, expected)]
        x, x0 = [_finite(p[side + "_pose"], (7,)) for p in (current, expected)]
        if not np.allclose([np.linalg.norm(v[3:]) for v in (x, x0)], 1., rtol=0., atol=1e-5):
            raise ValueError("unit quaternion required for actual mounted wrist")
        angle = (Rotation.from_quat(x[3:]).inv() * Rotation.from_quat(x0[3:])).magnitude()
        values = (float(abs(q - q0).max()), float(np.linalg.norm(x[:3] - x0[:3])), float(angle))
        metrics[side] = dict(max_q_error=values[0], root_error_m=values[1], root_error_rad=values[2])
        passed = passed and values[0] <= .00005 and values[1] <= .000005 and values[2] <= .00005
    return dict(passed=passed, metrics=metrics)


def left_geometry(row, target, *, scenario_nodes):
    model, source = _bound_model(ROOT, "left")
    poses = row["closure_geometry"]["raw"]["hands"]["left"]
    readback = row["action_readback"]["post"]["hands"]["left"]
    measured, xyz = _frames(model, poses["body_names"], poses["link_poses_xyzw"])
    names, q_all = readback["joint_names"], readback["q"][0]
    q = q_all[[names.index(n) for n in JOINTS]]
    angles = q_all[[names.index(n) for n in WRIST_ANGULAR]]
    anchor = Rotation.from_quat(np.roll(readback["native_anchor_pose_w"][0, 3:], -1)).as_matrix()
    end_q = expand(np.asarray(target["active_fingers_rad"], dtype=np.float32))
    root = _finite(target["root_xyz_m"], (3,))
    turn = _finite(target["wrist_intrinsic_xyz_rad"], (3,))
    dx, da, dq = root - xyz, turn - angles, end_q - q
    reference = model._frames(anchor, xyz, angles, q)
    correction = {key: measured[key] @ np.linalg.inv(reference[path]) for key, path in model.paths.items()}

    def path(fraction):
        frames = model._frames(anchor, xyz + fraction*dx, angles + fraction*da, q + fraction*dq)
        return {key: correction[key] @ frames[p] for key, p in model.paths.items()}

    table = model.boundary(measured_link_frames=measured, anchor_rotation=anchor, start_xyz=xyz,
                           start_angles=angles, start_q=q, final_q=end_q).prove(root, turn)
    travel = {key: float(np.linalg.norm(dx) + abs(da).sum()*model.wrist_radii[key]
                          + model.joint_radii[key]@abs(dq)) for key in model.paths}
    grid = dict(left={}, right={})
    _bind_pad_grids(row, "left", source, grid, {})
    radius = row["closure_geometry"]["rod_radius_m"]
    final = path(1.)
    cases = {}
    for key, nodes in scenario_nodes.items():
        proof = _non_target_certificate(model, path, travel, nodes, radius,
                                         node_motion_bound_m=np.full(len(nodes), NODE_SENSITIVITY_M))
        contacts = [_contact(grid["left"][pad], final[pad], nodes, radius)[0] for pad in PADS]
        cases[key] = dict(geometry=proof, nominal_gaps_m=contacts)
    return dict(passed=table["proved"] and all(c["geometry"]["proved"] for c in cases.values()),
                table=table, rod_cases=cases,
                scope="conditional_left_path_model_no_live_contact_qualification")


def _load_variant(output, sealed_reset, baseline):
    plan = base.freeze(output, sealed_reset)
    native = (baseline / "native").resolve(strict=True)
    outer_path = baseline / "outer.json"
    outer = base._checked(outer_path, BASELINE_OUTER_SHA)
    ledger = native / "physical.jsonl"
    digest = base.lifecycle.native.sha
    if (outer["exit_code"] != 0 or outer["forced_exit"] is not False or outer["output"] != str(native)
            or outer["artifact_sha256"]["physical.jsonl"] != BASELINE_LEDGER_SHA
            or digest(ledger) != BASELINE_LEDGER_SHA):
        raise ValueError("exact naturally closed 74-step baseline ledger required")
    archived_plan = base._checked(native / "frozen_plan.json", outer["plan_sha256"])
    original = _finite(plan["action_schedule"], (9, 24))
    if archived_plan["action_schedule"] != plan["action_schedule"]:
        raise ValueError("baseline and current quiet/close commands differ")
    rows = {}
    with ledger.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("operation") == "physical":
                row = base.decode_owned(event["payload"])["exchange"]
                if row["physics_step"] >= LAST_PREFIX_SDK:
                    rows[row["physics_step"]] = row
    if set(rows) != set(range(LAST_PREFIX_SDK, base.SDK_STEPS + 1)):
        raise ValueError("complete baseline SDK66 through SDK74 evidence required")
    first, last = rows[LAST_PREFIX_SDK], rows[base.SDK_STEPS]
    prefix = prefix_measurements(first)
    _same_clock(last)
    left = last["action_readback"]["post"]["hands"]["left"]
    q = left["q"][0]
    angles = q[[left["joint_names"].index(n) for n in WRIST_ANGULAR]]
    raw = last["closure_geometry"]["raw"]["hands"]["left"]
    xyz = raw["link_poses_xyzw"][0, raw["body_names"].index("base_link"), :3]
    issued_xyz = last["wrist_projection"]["projected_root_poses"][0, 0, :3]
    issued_angles = last["wrist_projection"]["projected_angles"][0, 0]
    xyz_delta = (1.-REMAINING_WRIST_FRACTION) * (xyz - issued_xyz)
    euler_delta = (1.-REMAINING_WRIST_FRACTION) * (angles - issued_angles)
    target = dict(root_xyz_m=(issued_xyz+xyz_delta).tolist(),
                  wrist_intrinsic_xyz_rad=(issued_angles+euler_delta).tolist(),
                  active_fingers_rad=left["backend_targets"][0, left["active_indices"]].tolist())
    scenarios = {str(tick): np.asarray(row["rod"]["node_positions"])[0] for tick,row in rows.items()}
    certificate = left_geometry(first, target, scenario_nodes=scenarios)
    if not certificate["passed"] or max(certificate["rod_cases"]["74"]["nominal_gaps_m"]) >= -.0001:
        raise ValueError("new left candidate does not pass the retained collision and terminal two-pad screen")
    plan["frozen_files"].update({str(outer_path): BASELINE_OUTER_SHA, str(ledger): BASELINE_LEDGER_SHA,
                                 str(native/"frozen_plan.json"): outer["plan_sha256"]})
    plan.update(staged_version=VERSION, action_schedule=variant_schedule(original, xyz_delta, euler_delta).tolist(),
                preserved_scope="one changed left wrist window; original right request and no lift or PPO",
                variant=dict(prefix_measurements=prefix,left_terminal_target=target,
                             physical_proof=certificate, baseline_ledger_sha256=BASELINE_LEDGER_SHA))
    return plan


def acquire(sim, owners, trace, output, *, plan):
    inherited = base.ObservedClose

    class LateWristObservedClose(inherited):
        def _step(self, action, substep):
            row = None
            try:
                row = super()._step(action, substep)
                if row["physics_step"] == LAST_PREFIX_SDK and row["guard_failure"] is None:
                    try:
                        current = prefix_measurements(row)
                        bound = plan["variant"]
                        observed = compare_prefix(current, bound["prefix_measurements"])
                        if observed["passed"]:
                            proof = left_geometry(row, bound["left_terminal_target"], scenario_nodes={
                                "actual_sdk66": np.asarray(current["rod_nodes_m"])})
                            observed.update(left_current_geometry=proof, passed=proof["passed"])
                        row["late_wrist_admission"] = observed
                        if not observed["passed"]:
                            row["guard_failure"] = "late_left_wrist_prefix_mismatch"
                    except Exception as error:
                        row["late_wrist_admission"] = dict(passed=False, error=type(error).__name__ + ": " + str(error))
                        row["guard_failure"] = "late_left_wrist_source_invalid"
                return row
            finally:
                self = action = row = None

    try:
        with base.bindings((base, "ObservedClose", LateWristObservedClose)):
            return base.acquire(sim, owners, trace, output, plan=plan)
    finally:
        sim = owners = trace = None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "plan", "sealed-reset", "baseline"):
        p.add_argument("--"+field, type=Path, required=True)
    p.add_argument("--approved-frozen-sha256")
    args = p.parse_args(argv)
    base.lifecycle.native.validate_execution_context(ROOT)
    if args.approved_frozen_sha256 is None:
        plan = _load_variant(args.output, args.sealed_reset, args.baseline)
        digest = base.lifecycle.native.digest(plan)
        base.lifecycle.native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        # The FK/USD certificate imports pxr. Rebuild it in a separate CPU process
        # so the AppLauncher process reaches its existing clean import boundary.
        with tempfile.TemporaryDirectory(prefix="pinch-certificate-", dir=args.plan.parent) as work:
            rebuilt = Path(work) / "plan.json"
            subprocess.run([
                sys.executable, "-B", "-m", "scripts.run_dh116_pinch_late_wrist",
                "--output", str(args.output), "--plan", str(rebuilt),
                "--sealed-reset", str(args.sealed_reset), "--baseline", str(args.baseline),
            ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            plan = json.loads(rebuilt.read_text())
        if (args.approved_frozen_sha256 != base.lifecycle.native.digest(plan)
                or json.loads(args.plan.read_text()) != plan or args.output.exists()):
            raise ValueError("exact staged successor recipe rebuild and fresh output required")
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
