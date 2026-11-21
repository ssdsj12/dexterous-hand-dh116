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


WAYPOINT_CONTRACT = screen.ROOT/"docs/contracts/dh116_teacher_tripod_guarded_approach_native_v1.json"
WAYPOINT_INDEX = {"left": 1, "right": 2}
WAYPOINT_WINDOWS = 20
DISTAL_CONTACT_PROOF_SHA = "bc2202bae667df2d2b55084569ee7b50d28af7873bf78ff743ac7815faa9a1f8"
PRECONTACT_PROOF_SHA = "a3f3436c87c43ad516666bc7b1ad6005ff50bdda5f4be7267ddda96b571ea8fe"
WAYPOINT_PROOF_SHA = "15c89406150e583bde244a827951ebfc702c69b1e48cd8e8b5729f9a72620c27"


def waypoint_controls(config, row, desired, *, windows=WAYPOINT_WINDOWS):
    from scipy.spatial.transform import Rotation
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

    wrist = row["wrist_projection"]
    dt = windows * 8 * shared.DT
    low, high = active_limits()
    joint_delta, position_delta, rotation_delta = [], [], []
    for i, side in enumerate(shared.SIDES):
        hand = row["action_readback"]["post"]["hands"][side]
        indices = hand["active_indices"]
        if [hand["joint_names"][j] for j in indices] != list(ACTIVE):
            raise ValueError("exact original active joint index order required for each Tripod hand")
        goal = np.asarray(desired[side]["active_q_rad"])
        if np.any(goal < low) or np.any(goal > high):
            raise ValueError("Tripod waypoint exceeds the original coupled limits")
        issued = np.asarray(hand["backend_targets"])[0, indices]
        joint_delta.extend(goal-issued)
        target = np.asarray(desired[side]["pose_wxyz"])
        position_delta.extend(target[:3]-wrist["projected_root_poses"][0,i,:3])
        seed = np.asarray(config["hands"][side]["pose_wxyz"])
        seed_rot, target_rot = Rotation.from_quat(np.stack((seed,target))[:,[4,5,6,3]])
        rotation_delta.extend((seed_rot.inv()*target_rot).as_euler("XYZ")-wrist["projected_angles"][0,i])
    command = np.concatenate([np.asarray(joint_delta)/(dt*ACTION_CONTRACT["joint_rate_rad_s"]),
                              np.asarray(position_delta)/(dt*ACTION_CONTRACT["translation_rate_m_s"]),
                              np.asarray(rotation_delta)/(dt*ACTION_CONTRACT["rotation_rate_rad_s"])])
    if (not np.isfinite(command).all() or np.max(np.abs(command))>1.
            or any(np.linalg.norm(command[k:k+3])>1. for k in (12,15,18,21))):
        raise ValueError("Tripod waypoint exceeds the original 24D action rate bounds")
    return np.repeat(command[None], windows, axis=0)


def load_waypoint(receipt, proof, config):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

    evidence = json.loads(WAYPOINT_CONTRACT.read_bytes())["actual"]
    outer = sources.checked_json(receipt,evidence["outer_sha256"])
    row_path = receipt.parent/"native"/"tripod_control_272.json"
    row = sources.checked_json(row_path,evidence["last_row_sha256"])
    if (outer["exit_code"]!=0 or outer["debugger_inferior_exit_code"]!=0 or outer["forced_exit"]
            or outer["exceeded_budget"] or row["physics_step"]!=272 or row["guard_failure"] is not None
            or native.sha(proof)!=WAYPOINT_PROOF_SHA):
        raise ValueError("the exact original guarded approach and waypoint screen are required")
    goals, checked = {}, []
    for side, i in WAYPOINT_INDEX.items():
        route = json.loads(screen.ROUTES[side].read_bytes())
        goals[side] = route["waypoints"][i]
    for line in proof.read_text().splitlines():
        support = json.loads(line)
        side = support["side"]
        if support["waypoint"] != WAYPOINT_INDEX[side]:
            continue
        limits = support["min_sampled_margins_m"]
        if min(limits[:2])<=.00025 or min(limits[2:])<=0.:
            raise ValueError("Tripod waypoint screening leaves insufficient rod or support reserve")
        checked.append(side)
    if sorted(checked) != sorted(WAYPOINT_INDEX):
        raise ValueError("both original Tripod waypoint checks are required")
    return row, waypoint_controls(config,decode_owned(row),goals), (WAYPOINT_CONTRACT,receipt,row_path,proof)


def load_precontact(receipt, proof, config):
    from scripts import screen_dh116_tripod_measured_precontact as measured
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

    rows, _, files = measured.archive(receipt)
    request = json.loads(measured.REQUEST.read_bytes())
    audit = sources.checked_json(proof, PRECONTACT_PROOF_SHA)
    if (audit["all_passed"] is not True or audit["version"] != request["version"]
            or audit["outer_sha256"] != native.sha(receipt)
            or audit["request_sha256"] != native.sha(measured.REQUEST)
            or audit["last_row_sha256"] != native.sha(files[-2])
            or audit["before_sdk"] != measured.SDK or audit["route_sample_count"] != measured.SAMPLES
            or set(audit["samples"]) != set(measured.SIDES)
            or not all(measured.check_route(value) for value in audit["samples"].values())):
        raise ValueError("exact independently screened bilateral Tripod precontact path required")
    return rows[-1], waypoint_controls(config, decode_owned(rows[-1]), request["goals"], windows=12), (
        *files, measured.REQUEST, Path(measured.__file__), proof)


def load_distal_contact(receipt, proof, config):
    from scripts import screen_dh116_tripod_right_first_contact as first
    from scripts import verify_dh116_supported_distal_transfer as transfer
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116 import tabletop_power_contact_evidence as force

    rows, _, files = first.measured.archive(receipt, contract=first.CONTRACT, sdk=first.SDK)
    audit = sources.checked_json(proof, DISTAL_CONTACT_PROOF_SHA)
    request = first.read_request()
    if (audit["all_passed"] is not True or audit["version"] != request["version"]
            or audit["outer_sha256"] != native.sha(receipt)
            or audit["request_sha256"] != native.sha(first.REQUEST)
            or audit["last_row_sha256"] != native.sha(files[-2]) or audit["before_sdk"] != first.SDK
            or not first.route_passes(audit["samples"])):
        raise ValueError("exact right Tripod first-contact evidence is required")
    approach = waypoint_controls(config, decode_owned(rows[-1]), request["goals"], windows=first.DRIVE_WINDOWS)
    actions = np.vstack((approach, np.zeros((first.COAST_WINDOWS, 24))))
    dependencies = (*files, first.REQUEST, proof, Path(first.__file__), Path(first.measured.__file__),
                    Path(transfer.__file__), Path(force.__file__),
                    Path(force.__file__).with_name("tabletop_teacher_force_closure.py"))
    return rows[-1], actions, dependencies, dict(
        boundary_sdk=first.SDK, stations=request["target_stations"],
        max_pad_penetration_m=first.MAX_PAD_SDF_PENETRATION_M)


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
    archived = waypoint = precontact = distal_archive = None
    if (args.distal_reference is None) != (args.distal_screen is None):
        raise ValueError("provide both right Tripod distal contact inputs")
    if args.distal_reference is not None and args.precontact_reference is None:
        raise ValueError("first distal contact requires exact precontact replay")
    if (args.precontact_reference is None) != (args.precontact_screen is None):
        raise ValueError("provide both Tripod precontact inputs")
    if args.precontact_reference is not None and args.waypoint_reference is None:
        raise ValueError("Tripod precontact continuation requires original waypoint evidence")
    if (args.waypoint_reference is None) != (args.waypoint_geometry is None):
        raise ValueError("provide the paired Tripod waypoint evidence inputs")
    if args.waypoint_reference is not None and args.approach_reference is None:
        raise ValueError("Tripod waypoint continuation requires the first approach reference")
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
        if args.waypoint_reference is not None:
            waypoint, appended, bound = load_waypoint(args.waypoint_reference,args.waypoint_geometry,config)
            plan["frozen_files"].update({str(path.resolve(strict=True)):native.sha(path) for path in bound})
            actions = np.vstack((actions,appended))
        if args.precontact_reference is not None:
            precontact, appended, bound = load_precontact(args.precontact_reference,args.precontact_screen,config)
            plan["frozen_files"].update({str(path.resolve(strict=True)):native.sha(path) for path in bound})
            actions = np.vstack((actions,appended))
        if args.distal_reference is not None:
            distal_archive, appended, bound, parameters = load_distal_contact(
                args.distal_reference, args.distal_screen, config)
            plan["frozen_files"].update({str(path.resolve(strict=True)):native.sha(path) for path in bound})
            plan["right_tripod_distal_contact"] = parameters
            actions = np.vstack((actions,appended))
        sdk_end = SDK_END + 8*len(actions)
        plan["tripod_approach"] = actions.tolist()
        plan["budget"].update(mode=("tripod_guarded_waypoint_continuation" if waypoint
                                    else "tripod_supported_guarded_approach"), sdk_steps=sdk_end,
                              control_steps=len(actions), solver_steps=(sdk_end-2)*shared.SUBSTEPS,
                              soft_wall_seconds=300 if precontact else (240 if waypoint else plan["budget"]["soft_wall_seconds"]))
        if precontact is not None:
            plan["budget"]["mode"] = "tripod_measured_guarded_precontact"
        if distal_archive is not None:
            plan["budget"].update(mode="tripod_right_measured_distal_contact", soft_wall_seconds=420)
    return plan, config, archived, waypoint, precontact, distal_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("output","plan","geometry-audit","archived-settling-receipt","rays","left-bake","right-bake",
                   "proximal-capture","proximal-bake"):
        parser.add_argument("--"+option, required=True, type=Path)
    parser.add_argument("--approach-reference", type=Path)
    parser.add_argument("--waypoint-reference", type=Path)
    parser.add_argument("--waypoint-geometry", type=Path)
    parser.add_argument("--precontact-reference", type=Path)
    parser.add_argument("--precontact-screen", type=Path)
    parser.add_argument("--distal-reference", type=Path)
    parser.add_argument("--distal-screen", type=Path)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native.validate_execution_context(screen.ROOT)
    plan, config, archived, waypoint, precontact, distal_archive = freeze(args)
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
    distal = None if distal_archive is None else dict(plan["right_tripod_distal_contact"], archive=distal_archive)
    reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(screen.ROOT),
        args=args, supported_acquire=partial(supported.acquire_supported_reset, config=config, sdk_end=sdk_end,
        task="tripod"), sdk_end=sdk_end, require_no_hand_contact=True, task="tripod", tripod_archive=archived,
        action_after_settle=plan.get("tripod_approach"), live_support_scene=config["support_scene"],
        waypoint_archive=waypoint, precontact_archive=precontact, distal_contact=distal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
