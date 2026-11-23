#!/usr/bin/env python3
"""Reproduce the sealed Tripod 902 approach through two guarded candidate windows."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from scripts import verify_dh116_tripod_supported_reset as original

shared, native, reset = original.shared, original.native, original.reset
OUTER_SHA = "5306afb0af3892b0da3c2edf8c3f5560e7bf5798578b26fe3060f681417f6b06"
DRAFT_SHA = "dbbd67046b5dd3b051c84b9ae24710b7ab123d7e115f57d17b9ccac311db196a"
SEED_SHA = "9d16393ecdcc02d007c7518d2d25e79b16d14b268e8e228765a86935b0799704"
PARKED_SDK = 240
ANCHOR_SDK = 552
END_SDK = 568
WIDTH = 8
END_SUPPORT_NAMES = {"support_0", "support_1"}


class TripodReplayGate:
    """Check archived physics boundaries before authorizing the next command."""

    def __init__(self, parked, anchor, *, output=None):
        self._parked = parked
        self._anchor = anchor
        self._output = output
        self.completed = ()
        self._last_candidate_sdk = ANCHOR_SDK

    def _record(self, sdk):
        if self._output is not None:
            native.write_new(self._output / f"tripod_902_interlock_{sdk}.json",
                             {"physics_step": sdk, "archive_match": sdk != END_SDK,
                              "candidate_precontact_complete": sdk == END_SDK, "teacher_admission": False})
        self.completed += (sdk,)

    def check_parked(self, row):
        try:
            if (self.completed or row["rod"]["physics_step"] != PARKED_SDK
                    or row["rod"]["rod_tick"] != (PARKED_SDK - 2) * shared.SUBSTEPS):
                raise ValueError("unexpected parked boundary")
            shared.verify_proximal_action_boundary(row, self._parked)
            for key in ("preceding_reaction_sdk_step", "route"):
                if row[key] != self._parked[key]:
                    raise ValueError(f"parked {key} differs")
            for key in ("rod_mesh_contact_pairs", "hand_wrenches_com_world", "timestamp_s",
                        "solver_time_s", "clock_valid"):
                if row["rod"][key] != self._parked["rod"][key]:
                    raise ValueError(f"parked rod {key} differs")
        except (KeyError, ValueError) as error:
            raise ValueError(f"Tripod 902 interlock SDK {PARKED_SDK}: {error}") from error
        self._record(PARKED_SDK)

    def before_window(self, sdk):
        if PARKED_SDK not in self.completed:
            raise ValueError(f"Tripod 902 interlock SDK {PARKED_SDK} is absent")
        if sdk >= ANCHOR_SDK and (ANCHOR_SDK not in self.completed or self._last_candidate_sdk != sdk):
            raise ValueError(f"Tripod 902 interlock SDK {ANCHOR_SDK} or subsequent SDK rows are absent")

    def observe(self, row):
        sdk = row["physics_step"]
        if sdk < ANCHOR_SDK:
            return
        if sdk == ANCHOR_SDK:
            try:
                if (self.completed != (PARKED_SDK,)
                        or row["substep"] != WIDTH - 1 or row["guard_failure"] is not None
                        or row["reaction_source_physics_step"] != sdk - 1):
                    raise ValueError("incomplete or guarded replay window")
                shared.verify_waypoint_boundary(row, self._anchor, sdk_step=sdk, exact_geometry=True)
                for key in ("rod_mesh_contact_pairs", "hand_wrenches_com_world", "timestamp_s",
                            "solver_time_s", "clock_valid"):
                    if row["rod"][key] != self._anchor["rod"][key]:
                        raise ValueError(f"rod {key} differs")
                for key in ("issued_composer_wrench", "wrist_projection"):
                    if row[key] != self._anchor[key]:
                        raise ValueError(f"control {key} differs")
            except (KeyError, ValueError) as error:
                raise ValueError(f"Tripod 902 interlock SDK {ANCHOR_SDK}: {error}") from error
            self._record(sdk)
            return
        if ANCHOR_SDK not in self.completed:
            raise ValueError(f"Tripod 902 interlock SDK {ANCHOR_SDK} is absent")
        rod = row["rod"]
        if (sdk != self._last_candidate_sdk + 1 or sdk > END_SDK
                or rod["physics_step"] != sdk or rod["rod_tick"] != (sdk - 2) * shared.SUBSTEPS
                or row["sdk_hand_geometry"]["physics_step"] != sdk
                or row["reaction_source_physics_step"] != sdk - 1 or row["guard_failure"] is not None):
            raise ValueError(f"Tripod 902 candidate guard or reaction clock at SDK {sdk}")
        pairs = rod["rod_mesh_contact_pairs"]["pairs"]
        if any(pair["mesh_name"] not in END_SUPPORT_NAMES for pair in pairs):
            raise ValueError(f"Tripod 902 precontact candidate loaded a hand or table at SDK {sdk}")
        loads = {(pair["mesh_name"], pair["substep"]) for pair in pairs if pair["normal_force_on_rod_n"] > 0.}
        expected = {(name, substep) for name in END_SUPPORT_NAMES for substep in range(shared.SUBSTEPS)}
        if loads != expected:
            raise ValueError(f"Tripod 902 precontact candidate lost support at SDK {sdk}")
        self._last_candidate_sdk = sdk
        if sdk == END_SDK:
            self._record(sdk)

    def finish(self):
        if self.completed != (PARKED_SDK, ANCHOR_SDK, END_SDK):
            raise ValueError("Tripod 902 replay did not complete both interlocks and the bounded tail")


def freeze(args):
    """Bind the original control prefix and matched archive to the new bounded tail."""
    plan, config, parked, waypoint, precontact, distal, _ = original.freeze(args)
    if plan["tripod_seed_sha256"] != SEED_SHA or distal is None:
        raise ValueError("exact 902 Reset seed and original first distal probe required")

    draft = original.sources.checked_json(args.draft, DRAFT_SHA)
    outer = original.sources.checked_json(args.historical_receipt, OUTER_SHA)
    if (outer["exit_code"] != 0 or outer["forced_exit"] or outer["exceeded_budget"]
            or outer["debugger_inferior_exit_code"] != 0 or not outer["debugger_status_verified"]
            or draft["historical_outer_sha256"] != OUTER_SHA or draft["historical_reset_seed_sha256"] != SEED_SHA):
        raise ValueError("complete sealed original Tripod run required")

    archive = args.historical_receipt.parent / "native"
    evidence = [args.historical_receipt, args.draft, Path(__file__).resolve()]
    archived_rows = {}
    for name in ("tripod_exchange_240.json", "tripod_control_552.json"):
        path = archive / name
        digest = draft["archived_anchor_files_sha256"][name]
        if outer["artifact_sha256"][name] != digest:
            raise ValueError(f"Tripod 902 archive digest disagreement for {name}")
        archived_rows[name] = original.sources.checked_json(path, digest)
        evidence.append(path)
    original_parked = archived_rows["tripod_exchange_240.json"]
    anchor = archived_rows["tripod_control_552.json"]
    shared.verify_proximal_action_boundary(parked, original_parked)
    TripodReplayGate(original_parked, anchor).check_parked(parked)

    actions = shared.supported_action_schedule(draft["action24_window_sequence"], END_SDK - PARKED_SDK)
    prefix_count = (ANCHOR_SDK - PARKED_SDK) // WIDTH
    if not np.array_equal(actions[:prefix_count], np.asarray(plan["tripod_approach"])[:prefix_count]):
        raise ValueError("original commands do not reproduce the sealed 39-window prefix")
    for path in evidence:
        resolved = str(path.resolve(strict=True))
        digest = native.sha(path)
        if plan["frozen_files"].setdefault(resolved, digest) != digest:
            raise ValueError("conflicting Tripod 902 source fingerprint")

    plan.update(version="dh116-tripod-902-guarded-approach-v1",
                preserved_scope="two new precontact controls after exact same-seed Tripod replay",
                tripod_approach=actions.tolist(),
                replay_interlocks={"parked_sdk": PARKED_SDK, "prefix_sdk": ANCHOR_SDK, "end_sdk": END_SDK,
                                   "archived_receipt_sha256": OUTER_SHA, "command_draft_sha256": DRAFT_SHA})
    plan["budget"].update(mode="tripod_902_precontact_interlocked", sdk_steps=END_SDK,
                          control_steps=len(actions), solver_steps=(END_SDK - 2) * shared.SUBSTEPS,
                          soft_wall_seconds=420)
    return plan, config, original_parked, waypoint, precontact, distal, anchor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("output", "plan", "draft", "historical-receipt", "geometry-audit",
                   "archived-settling-receipt", "rays", "left-bake", "right-bake",
                   "proximal-capture", "proximal-bake", "approach-reference", "waypoint-reference",
                   "waypoint-geometry", "precontact-reference", "precontact-screen",
                   "distal-reference", "distal-screen"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--verify-existing-plan", action="store_true")
    parser.set_defaults(bilateral_reference=None, bilateral_screen=None, verify_near_open_settling=True,
                        measure_contact_evidence=False)
    args = parser.parse_args(argv)
    if args.verify_existing_plan and args.approved_frozen_sha256 is not None:
        parser.error("CPU-only verification and native execution are exclusive")
    native.validate_execution_context(shared.ROOT)
    plan, config, parked, waypoint, precontact, distal, anchor = freeze(args)
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    assert_clean_import_boundary()
    digest = native.digest(plan)
    if args.verify_existing_plan:
        if json.loads(args.plan.read_bytes()) != plan or args.output.exists():
            raise ValueError("Tripod 902 verified plan or fresh output differs")
        print(json.dumps({"frozen_sha256": digest, "verified_frozen_files": len(plan["frozen_files"]),
                          "cpu_only": True}))
        return 0
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps({"frozen_sha256": digest, "budget": plan["budget"]}))
        return 0
    if (digest != args.approved_frozen_sha256 or json.loads(args.plan.read_bytes()) != plan
            or args.output.exists()):
        raise ValueError("the exact approved 902 fingerprint and exclusive output are required")

    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    gate = TripodReplayGate(parked, anchor, output=args.output)
    reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(shared.ROOT),
                  args=args, supported_acquire=partial(original.supported.acquire_supported_reset,
                      config=config, sdk_end=END_SDK, task="tripod"),
                  sdk_end=END_SDK, require_no_hand_contact=True, task="tripod",
                  tripod_archive=parked, action_after_settle=plan["tripod_approach"],
                  live_support_scene=config["support_scene"], waypoint_archive=waypoint,
                  precontact_archive=precontact,
                  distal_contact=dict(plan["right_tripod_distal_contact"], archive=distal),
                  tripod_replay_gate=gate))
    gate.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
