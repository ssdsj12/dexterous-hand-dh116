#!/usr/bin/env python3
"""Verify an independent Tripod Reset and hands-clear supported rod interval."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from scripts import screen_dh116_tripod_supported_reset as screen
from scripts import verify_dh116_power_supported_task as supported

shared = supported.shared
native, reset, sources = shared.native, shared.reset, shared.sources
SDK_END = 240
AUDIT_SHA = "d83baf7e07e2e60d461272b2cd8df22b773e23e02bee83461ac5253d391685ec"


APPROACH_CONTRACT = screen.ROOT/"docs/contracts/dh116_teacher_tripod_supported_reset_native_v1.json"


def approach_commands():
    command = np.zeros(24)
    command[14] = -.75
    command[15:18] = [.30, .45, .45]
    return np.repeat(command[None, :], 4, axis=0)


def approach_reference(receipt):
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES

    expected = json.loads(APPROACH_CONTRACT.read_bytes())["actual"]
    outer = sources.checked_json(receipt, expected["outer_sha256"])
    archive = receipt.parent/"native"
    sample, marker, summary = (archive/"tripod_exchange_240.json", archive/"app_close_returned.json",
                               archive/"tripod_shared_result.json")
    close = sources.checked_json(marker, outer["artifact_sha256"][marker.name])
    result = sources.checked_json(summary, expected["result_sha256"])
    state = sources.checked_json(sample, outer["artifact_sha256"][sample.name])
    if (outer["forced_exit"] or outer["exceeded_budget"]
            or outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or close["app_close_returned"] is not True or result["solver_substeps"] != 2856
            or result["scope"] != "measured_tripod_support_settling_without_hand"
            or state["rod"]["physics_step"] != SDK_END or state["rod"]["rod_tick"] != 2856
            or tuple(state["rod"]["mesh_names"]) != PROXIMAL_POWER_MESH_NAMES):
        raise ValueError("exact native Tripod support settling and row identity required")
    return state, (APPROACH_CONTRACT, receipt, sample, marker, summary)


def freeze(args):
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state

    request, config = screen.scene_request()
    if native.sha(args.archived_settling_receipt) != request["archived_settling_outer_sha256"]:
        raise ValueError("the exact supported rod source is required")
    audited = json.loads(args.geometry_audit.read_bytes())
    if native.sha(args.geometry_audit) != AUDIT_SHA:
        raise ValueError("the exact sealed Tripod geometry audit is required")
    if (audited["request_sha256"] != native.sha(screen.REQUEST) or audited["all_passed"] is not True
            or audited["scope"] != "tripod_supported_reset_archived_rod_envelope_only"
            or audited["init_and_archived_sdk_samples"] != 239):
        raise ValueError("both Tripod poses must pass the complete archived support interval")
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task

    task = load_flexible_grasp_task(screen.TASKS/"flexible_dual_hand_grasp_v1.json", screen.ROOT)
    table = dict(center_m=task["scene"]["table_center_m"], size_m=task["scene"]["table_size_m"])
    seed, _, _, _ = prepare_initial_state(config, table, expected_scope=screen.SCOPE)
    plan = reset.freeze_plan(args.output)
    plan.update(version="dh116-tripod-supported-seed-native-v1", preserved_scope=screen.SCOPE,
                tripod_seed_sha256=seed["sha256"])
    plan["budget"].update(mode="tripod_supported_seed_no_hand_contact", sdk_steps=SDK_END,
        control_steps=0, solver_steps=(SDK_END-2)*shared.SUBSTEPS, new_power_mesh_registrations=6)
    _, proximal_files = shared.proximal_registration_inputs(args.proximal_capture, args.proximal_bake)
    files=[screen.REQUEST, screen.BASE_SCENE, *screen.ROUTES.values(), Path(__file__), Path(screen.__file__),
           Path(supported.__file__), Path(shared.__file__), args.rays, args.geometry_audit,
           args.archived_settling_receipt, *proximal_files]
    rays = sources.checked_json(args.rays, sources.RAYS_SHA)["palms"]
    for side, folder in (("left", args.left_bake), ("right", args.right_bake)):
        _, _, record, _ = sources.fixture(side, rays[side], folder)
        archive = args.rays.parent/f"{side}_base_link.json"
        sources.checked_json(archive, record["gpu_row_sha256"])
        files.extend((archive, folder/"result.json", folder/f"{side}.base_link.npz"))
    for source in files:
        path = str(source.resolve(strict=True))
        digest = native.sha(source)
        if plan["frozen_files"].setdefault(path, digest) != digest:
            raise ValueError("inconsistent Tripod source fingerprints")
    archived = None
    if args.approach_reference is not None:
        archived, files = approach_reference(args.approach_reference)
        plan["frozen_files"].update({str(path.resolve(strict=True)):native.sha(path) for path in files})
        actions = approach_commands()
        from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

        rate = ACTION_CONTRACT["translation_rate_m_s"]
        travel = np.linalg.norm(actions[:, 12:18].reshape(-1, 2, 3), axis=-1).sum(axis=0) * rate * shared.DT * 8
        for side, distance in zip(shared.SIDES, travel, strict=True):
            bounds = audited["hands"][side]
            if (bounds["rod_gap_lower_m"] - distance <= .00025 or bounds["support_gap_m"] <= distance
                    or bounds["table_clearance_m"] - distance < .0021):
                raise ValueError("Tripod travel exceeds its archived no-contact geometry reserve")
        plan["tripod_travel_bound_m"] = dict(zip(shared.SIDES, travel.tolist(), strict=True))
        sdk_end = SDK_END + 8*len(actions)
        plan["tripod_approach"] = actions.tolist()
        plan["budget"].update(mode="tripod_supported_guarded_approach", sdk_steps=sdk_end,
                              control_steps=len(actions), solver_steps=(sdk_end-2)*shared.SUBSTEPS)
    return plan, config, archived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("output","plan","geometry-audit","archived-settling-receipt","rays","left-bake","right-bake",
                   "proximal-capture","proximal-bake"):
        parser.add_argument("--"+option, required=True, type=Path)
    parser.add_argument("--approach-reference", type=Path)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native.validate_execution_context(screen.ROOT)
    plan, config, archived = freeze(args)
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    assert_clean_import_boundary()
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
        return 0
    if (digest != args.approved_frozen_sha256 or json.loads(args.plan.read_bytes()) != plan or args.output.exists()):
        raise ValueError("exact approved Tripod recipe and fresh exclusive output required")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    args.verify_near_open_settling = True
    args.measure_contact_evidence = False
    sdk_end = plan["budget"]["sdk_steps"]
    reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(screen.ROOT),
        args=args, supported_acquire=partial(supported.acquire_supported_reset, config=config, sdk_end=sdk_end,
        task="tripod"), sdk_end=sdk_end, require_no_hand_contact=True, task="tripod", tripod_archive=archived,
        action_after_settle=plan.get("tripod_approach"), live_support_scene=config["support_scene"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
