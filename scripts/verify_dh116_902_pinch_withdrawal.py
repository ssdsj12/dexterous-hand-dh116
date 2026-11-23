"""Freeze and verify the first hands-clear 902 Pinch withdrawal separately from contact."""

import argparse
from functools import partial
import inspect
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from scripts import verify_dh116_902_flexible_reset as reset_plan
from scripts import verify_dh116_power_supported_task as supported
from scripts.screen_dh116_902_flexible_contact import boundary
from scripts.screen_dh116_902_flexible_precontact import load_approach
from scripts.screen_dh116_902_pinch_withdrawal import (
    ANCHOR_SDK, END_SDK, NONPARTICIPATING, PARKED_SDK, PHASE, R6_SHA, TAIL_WINDOWS, WIDTH, checked_withdrawal, load_r6,
)


shared = supported.shared
native, sources, reset = shared.native, shared.sources, shared.reset
ROOT = reset_plan.ROOT
SIDES = ("left", "right")
FIRST_EXACT_PREFIX_SDK = ANCHOR_SDK - 7
END_SUPPORT = ("support_0", "support_1")


def _same_binary32(a, b):
    a, b = np.asarray(a, dtype="<f4"), np.asarray(b, dtype="<f4")
    return (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and a.tobytes() == b.tobytes())


def _exact_controls(actual, archived):
    shared.verify_waypoint_boundary(actual, archived, sdk_step=archived["physics_step"], exact_geometry=True)
    for key in ("rod_mesh_contact_pairs", "hand_wrenches_com_world", "timestamp_s", "solver_time_s", "clock_valid"):
        if actual["rod"][key] != archived["rod"][key]:
            raise ValueError(f"sealed prefix rod {key} differs")
    for key in ("issued_composer_wrench", "wrist_projection"):
        if actual[key] != archived[key]:
            raise ValueError(f"sealed prefix control {key} differs")
    if actual["action_readback"]["submission"] != archived["action_readback"]["submission"]:
        raise ValueError("sealed prefix target submission differs")


class PinchWithdrawalGate:
    """Withhold further windows if the real prefix, targets, or hands-clear ledger diverge."""

    task = "pinch"

    def __init__(self, parked, archived, recipe, *, output=None):
        self._parked = parked
        self._archived = archived
        self._recipe = recipe
        self._output = output
        self._last = None
        self._exact = []

    def check_parked(self, row):
        if self._last is not None or row["rod"]["physics_step"] != PARKED_SDK:
            raise ValueError("unexpected Pinch parked SDK240 row")
        shared.verify_proximal_action_boundary(row, self._parked)
        for key in ("preceding_reaction_sdk_step", "route"):
            if row[key] != self._parked[key]:
                raise ValueError(f"SDK240 parked {key} differs")
        for key in ("rod_mesh_contact_pairs", "hand_wrenches_com_world", "timestamp_s", "solver_time_s", "clock_valid"):
            if row["rod"][key] != self._parked["rod"][key]:
                raise ValueError(f"SDK240 parked rod {key} differs")
        self._last = PARKED_SDK

    def before_window(self, sdk):
        if self._last is None:
            raise ValueError("verified Pinch parked SDK240 boundary required")
        if self._last != sdk or not PARKED_SDK <= sdk < END_SDK or (sdk - PARKED_SDK) % WIDTH:
            raise ValueError("consecutive Pinch SDK window boundary required")
        if sdk >= ANCHOR_SDK and self._exact != list(range(FIRST_EXACT_PREFIX_SDK, ANCHOR_SDK + 1)):
            raise ValueError("exact sealed Pinch SDK432 action boundary required before withdrawal")

    def observe(self, row):
        from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
        from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR, WRIST_LINEAR

        sdk = row["physics_step"]
        rod = row["rod"]
        if (self._last is None or sdk != self._last + 1 or sdk > END_SDK
                or rod["physics_step"] != sdk or rod["rod_tick"] != (sdk - 2) * shared.SUBSTEPS
                or row["sdk_hand_geometry"]["physics_step"] != sdk
                or row["substep"] != (sdk - PARKED_SDK - 1) % WIDTH
                or row["reaction_source_physics_step"] != sdk - 1 or row["guard_failure"] is not None
                or rod.get("clock_valid") is not True or rod.get("pair_capture_status") != "validated_solver_samples"):
            raise ValueError(f"Pinch owned hand/rod/reaction clock or guard at SDK {sdk}")
        action = self._recipe["actions"][(sdk - PARKED_SDK - 1) // WIDTH]
        owned = decode_owned(row)
        if not np.array_equal(owned["action_readback"]["request"]["action24"], np.asarray(action)[None]):
            raise ValueError(f"Pinch issued 24D command differs at SDK {sdk}")
        pairs = rod["rod_mesh_contact_pairs"]["pairs"]
        loads = {(p["mesh_name"], p["substep"]) for p in pairs if p["normal_force_on_rod_n"] > 0.}
        expected = {(name, k) for name in END_SUPPORT for k in range(shared.SUBSTEPS)}
        if any(p["mesh_name"] not in END_SUPPORT for p in pairs) or loads != expected:
            raise ValueError(f"Pinch withdrawal must retain both supports and zero other contact at SDK {sdk}")
        if FIRST_EXACT_PREFIX_SDK <= sdk <= ANCHOR_SDK:
            _exact_controls(row, self._archived[sdk])
            self._exact.append(sdk)
        if sdk > ANCHOR_SDK:
            target = self._recipe["expected_tail_rows"][sdk - ANCHOR_SDK - 1]
            post = owned["action_readback"]["post"]["hands"]
            if any([post[s]["joint_names"][j] for j in post[s]["active_indices"]] != list(ACTIVE) for s in SIDES):
                raise ValueError("original left/right active finger slots required")
            fingers = np.concatenate([post[s]["backend_targets"][:, post[s]["active_indices"]] for s in SIDES], 1)
            wrists = np.stack([post[s]["backend_targets"][:, [post[s]["joint_names"].index(name)
                                for name in WRIST_LINEAR + WRIST_ANGULAR]] for s in SIDES], axis=1)
            if (target["sdk"] != sdk or not _same_binary32(fingers, target["submitted_fingers_f32"])
                    or not _same_binary32(wrists, target["submitted_wrists_f32"])):
                raise ValueError(f"Pinch binary32 backend target differs at SDK {sdk}")
        self._last = sdk

    def finish(self):
        if self._last != END_SDK or self._exact != list(range(FIRST_EXACT_PREFIX_SDK, ANCHOR_SDK + 1)):
            raise ValueError("the guarded original prefix and 64 withdrawal SDK steps did not complete")
        result = dict(phase=PHASE, verified_archive_anchor=ANCHOR_SDK, final_sdk=END_SDK,
                      verified_binary32_tail_steps=END_SDK - ANCHOR_SDK,
                      later_phases_authorized=False, native_contact_verified=False,
                      force_hold_verified=False, lift_verified=False, teacher_admission=False)
        if self._output is not None:
            native.write_new(self._output / "pinch_withdrawal_gate_result.json", result)
        return result


def _audit_in_child(receipt, r6):
    with TemporaryDirectory(prefix="dh116-902-pinch-withdrawal-") as folder:
        output = Path(folder) / "proof.json"
        subprocess.run([sys.executable, "-B", "-m", "scripts.screen_dh116_902_pinch_withdrawal",
                        "--receipt", str(receipt), "--r6", str(r6), "--output", str(output)],
                       cwd=ROOT, check=True)
        return json.loads(output.read_bytes())


def freeze(args):
    """Pin original acquisition, exact archive and one immutable hands-clear tail."""
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    config_report = reset_plan.run_screen("pinch")
    _, config, files = reset_plan.load_candidate("pinch")
    parked, prefix, prefix_files = load_approach(args.settled_receipt, args.precontact_proof, "pinch", config)
    if len(prefix) * WIDTH != ANCHOR_SDK - PARKED_SDK:
        raise ValueError("exact original 24-window Pinch prefix required")
    archived, _, anchor_files, sdk = boundary(args.approach_receipt, "pinch")
    if sdk != ANCHOR_SDK:
        raise ValueError("actual Pinch SDK432 anchor required")
    outer_digest = native.sha(args.approach_receipt)
    outer = sources.checked_json(args.approach_receipt, outer_digest)
    archive_dir = args.approach_receipt.parent / "native"
    expected_parked_path = archive_dir / "pinch_exchange_240.json"
    expected_parked = sources.checked_json(expected_parked_path, outer["artifact_sha256"][expected_parked_path.name])
    PinchWithdrawalGate(expected_parked, {}, None).check_parked(parked)
    sealed_plan_path = archive_dir / "frozen_plan.json"
    sealed_plan = sources.checked_json(sealed_plan_path, outer["artifact_sha256"][sealed_plan_path.name])
    if native.digest(sealed_plan) != outer["frozen_sha256"] or not np.array_equal(
            sealed_plan["action_schedule"], prefix):
        raise ValueError("original full Pinch prefix did not reproduce")
    # Audit in another process so standalone USD cannot enter the native import boundary.
    proof = _audit_in_child(args.approach_receipt, args.r6)
    if (json.loads(args.cpu_proof.read_bytes()) != proof or proof["phase"] != PHASE
            or proof.get("later_phases_authorized") is not False):
        raise ValueError("fresh hands-clear proof differs from the frozen CPU evidence")
    action, _ = load_r6(args.r6, proof)
    actions = shared.supported_action_schedule(np.concatenate((prefix, np.tile(action, (TAIL_WINDOWS, 1)))),
                                              END_SDK - PARKED_SDK)
    plan = reset.freeze_plan(args.output)
    _, proximal_inputs = shared.proximal_registration_inputs(args.proximal_capture, args.proximal_bake)
    scripts = ROOT / "scripts"
    files = [*files, *prefix_files, *anchor_files, expected_parked_path, sealed_plan_path,
             Path(__file__), Path(shared.__file__), Path(supported.__file__), args.rays, *proximal_inputs,
             args.r6, args.cpu_proof, *(scripts / name for name in (
                 "screen_dh116_902_pinch_withdrawal.py", "certify_dh116_902_pinch_index_clearance.py",
                 "certify_dh116_902_pinch_relief_route.py", "screen_dh116_902_command_path.py",
                 "fit_dh116_902_pinch_alignment.py", "audit_dh116_pinch_initial_pose.py",
                 "audit_dh116_certified_segment_distance.py"))]
    palms = sources.checked_json(args.rays, sources.RAYS_SHA)["palms"]
    for side, bake in (("left", args.left_bake), ("right", args.right_bake)):
        _, _, record, _ = sources.fixture(side, palms[side], bake)
        raw = args.rays.parent / f"{side}_base_link.json"
        sources.checked_json(raw, record["gpu_row_sha256"])
        files.extend((raw, bake / "result.json", bake / f"{side}.base_link.npz"))
    for path in files:
        path = str(path.resolve(strict=True))
        if plan["frozen_files"].setdefault(path, native.sha(path)) != native.sha(path):
            raise ValueError("conflicting frozen native source")
    prefix_other = np.abs(np.asarray(prefix)[:, NONPARTICIPATING])
    recipe = dict(phase=PHASE, actions=actions.tolist(), expected_tail_rows=proof["rows"],
                  nonparticipating_slots=list(NONPARTICIPATING),
                  nonparticipating_tail_baseline_f32=proof["nonparticipating_baseline_f32"],
                  nonparticipating_tail_targets_held=proof["nonparticipating_targets_held_in_tail"],
                  archived_prefix_nonparticipant_max_abs=prefix_other.max(axis=0).tolist(),
                  archived_prefix_strict_zero=not bool(np.any(prefix_other)),
                  original_archive_sha256=outer_digest, r6_route_sha256=R6_SHA, tail_cpu_sha256=native.sha(args.cpu_proof),
                  parked_sdk=PARKED_SDK, sealed_anchor_sdk=ANCHOR_SDK, terminal_sdk=END_SDK,
                  later_phases_authorized=False)
    plan.update(version="dh116-902-pinch-native-withdrawal-v1", screen=config_report,
                action_schedule=recipe["actions"], pinch_withdrawal=recipe,
                preserved_scope="hands-clear fixed-wrist Pinch index withdrawal after exact SDK432 replay")
    plan["budget"].update(mode="902_pinch_index_only_withdrawal", sdk_steps=END_SDK,
                          control_steps=len(actions), solver_steps=(END_SDK - 2) * shared.SUBSTEPS,
                          new_power_mesh_registrations=6, soft_wall_seconds=300)
    assert_clean_import_boundary()
    actual_archive = {r["physics_step"]: r for r in archived if r["physics_step"] >= FIRST_EXACT_PREFIX_SDK}
    return plan, config, expected_parked, actual_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "plan", "settled-receipt", "precontact-proof", "approach-receipt", "r6",
                 "cpu-proof", "rays", "left-bake", "right-bake", "proximal-capture", "proximal-bake"):
        parser.add_argument("--" + name, required=True, type=Path)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--verify-existing-plan", action="store_true")
    controls.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    args.verify_near_open_settling = True
    args.measure_contact_evidence = False
    native.validate_execution_context(ROOT)
    plan, config, parked, archived = freeze(args)
    fingerprint = native.digest(plan)
    if args.output.exists():
        raise ValueError("fresh native evidence output required")
    if args.verify_existing_plan:
        if json.loads(args.plan.read_bytes()) != plan:
            raise ValueError("fresh Pinch freeze differs from saved CPU plan")
        print(json.dumps(dict(frozen_sha256=fingerprint, files=len(plan["frozen_files"]),
                              budget=plan["budget"], cpu_only=True)))
        return 0
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=fingerprint, budget=plan["budget"], cpu_only=True)))
        return 0
    if args.approved_frozen_sha256 != fingerprint or json.loads(args.plan.read_bytes()) != plan:
        raise ValueError("exact approved Pinch withdrawal recipe required")
    if "replay_gate" not in inspect.signature(shared.run_shared).parameters:
        raise ValueError("native Pinch replay requires the integrated generic replay gate")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    gate = PinchWithdrawalGate(parked, archived, plan["pinch_withdrawal"], output=args.output)
    acquire = partial(supported.acquire_supported_reset, config=config, sdk_end=END_SDK, task="pinch")
    reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(ROOT),
        args=args, supported_acquire=acquire, sdk_end=END_SDK, task="pinch", require_no_hand_contact=True,
        live_support_scene=config["support_scene"], action_after_settle=plan["action_schedule"],
        pinch_archive=parked, replay_gate=gate))
    gate.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
