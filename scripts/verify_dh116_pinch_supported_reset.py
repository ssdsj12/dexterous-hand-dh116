#!/usr/bin/env python3
"""Verify the independently positioned Pinch hands and original supported free rod."""

import argparse
from functools import partial
import json
from pathlib import Path

from scripts import screen_dh116_pinch_supported_reset as screen
from scripts import verify_dh116_power_supported_task as supported

shared = supported.shared
native, reset, sources = shared.native, shared.reset, shared.sources
SDK_END = 240
DISTAL_PROOF_SHA = "99522e21bcbf476db4eaa51cfb5d501669e89dbd7b3c7d0e07aee111da85ee97"
APPROACH_PROOF_SHA = "54f72ae776a56f688c4746971819ce320ffdb0e31adcc647e053f9d6e23a67e4"
REQUEST = screen.ROOT/"configs/dh116_tasks/flexible_pinch_end_supported_seed_v1.json"


def load_screen(path):
    seed = json.loads(REQUEST.read_bytes())
    audit = sources.checked_json(path,seed["geometry_audit_sha256"])
    if (seed["version"]!="dh116-pinch-end-supported-seed-v1" or seed["scope"]!=screen.SCOPE
            or seed["shared_support_config_sha256"]!=native.sha(screen.BASE_SCENE)
            or audit["scope"]!=screen.SCOPE or audit["config"]!=seed["config"]
            or audit["all_passed"] is not True or audit["init_and_archived_samples"]!=239):
        raise ValueError("exact independently screened two-hand Pinch seed required")
    return seed["config"]


def load_approach(receipt, proof, config):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from scripts import screen_dh116_pinch_supported_approach as approach
    from scripts.verify_dh116_tripod_supported_reset import waypoint_controls

    rows, supports, files = approach.archive(receipt)
    audit = sources.checked_json(proof, APPROACH_PROOF_SHA)
    request = json.loads(approach.REQUEST.read_bytes())
    if (audit["all_passed"] is not True or audit["version"] != request["version"]
            or audit["outer_sha256"] != native.sha(receipt)
            or audit["request_sha256"] != native.sha(approach.REQUEST)
            or audit["last_row_sha256"] != native.sha(files[-1])
            or audit["before_sdk"] != SDK_END or audit["route_sample_count"] != approach.measured.SAMPLES
            or set(audit["samples"]) != set(shared.SIDES)
            or supports != config["support_scene"]["supports"]
            or not all(approach.measured.check_route(value, pad_count=len(approach.PADS))
                       for value in audit["samples"].values())):
        raise ValueError("exact bilateral supported Pinch approach proof required")
    state = approach.settled_control_state(decode_owned(rows[-1]), config)
    actions = waypoint_controls(config, state, request["goals"], windows=approach.WINDOWS)
    return rows[-1], actions, (*files, approach.REQUEST, proof, Path(approach.__file__),
        Path(approach.measured.__file__), screen.ROOT / "scripts/verify_dh116_tripod_supported_reset.py")


def load_distal_contact(receipt, proof, config):
    import numpy as np
    from scripts import screen_dh116_pinch_right_first_contact as first
    from scripts import verify_dh116_supported_distal_transfer as transfer
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116 import tabletop_power_contact_evidence as force
    from scripts.verify_dh116_tripod_supported_reset import waypoint_controls

    rows, _, files = first.measured.archive(receipt, contract=first.CONTRACT, sdk=first.SDK, task="pinch")
    audit = sources.checked_json(proof, DISTAL_PROOF_SHA)
    request = first.read_request()
    if (audit["all_passed"] is not True or audit["version"] != request["version"]
            or audit["outer_sha256"] != native.sha(receipt)
            or audit["request_sha256"] != native.sha(first.REQUEST)
            or audit["last_row_sha256"] != native.sha(files[-2]) or audit["before_sdk"] != first.SDK
            or not first.route_passes(audit["samples"], pads=first.PADS, stations=first.STATIONS)):
        raise ValueError("exact right Pinch first-contact proof is required")
    drive = waypoint_controls(config, decode_owned(rows[-1]), request["goals"], windows=first.DRIVE_WINDOWS)
    actions = np.vstack((drive, np.zeros((first.COAST_WINDOWS, 24))))
    dependencies = (*files, first.REQUEST, proof, Path(first.__file__), Path(first.measured.__file__),
                    Path(transfer.__file__), Path(force.__file__), Path(force.__file__).with_name(
                    "tabletop_teacher_force_closure.py"), screen.ROOT / "scripts/screen_dh116_tripod_right_first_contact.py")
    return rows[-1], actions, dependencies, dict(
        boundary_sdk=first.SDK, stations=request["target_stations"], grasp="pinch",
        max_pad_penetration_m=first.MAX_PAD_SDF_PENETRATION_M)


def freeze(args):
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task

    config = load_screen(args.geometry_audit)
    if (native.sha(args.pinch_receipt)!=screen.PINCH_OUTER_SHA
            or native.sha(args.parked_receipt)!=screen.SUPPORTED_ROD_SHA):
        raise ValueError("the independent material reference receipts must match")
    task = load_flexible_grasp_task(screen.ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",screen.ROOT)
    table = dict(center_m=task["scene"]["table_center_m"],size_m=task["scene"]["table_size_m"])
    seed,_,_,_ = prepare_initial_state(config,table,expected_scope=screen.SCOPE)
    plan = reset.freeze_plan(args.output)
    plan.update(version="dh116-pinch-end-supported-seed-native-v1", preserved_scope=screen.SCOPE,
                pinch_supported_seed_sha256=seed["sha256"])
    plan["budget"].update(mode="pinch_supported_seed_no_hand_contact", sdk_steps=SDK_END,
        control_steps=0, solver_steps=(SDK_END-2)*shared.SUBSTEPS, new_power_mesh_registrations=6)
    _, proximal_files = shared.proximal_registration_inputs(args.proximal_capture,args.proximal_bake)
    files=[REQUEST,screen.BASE_SCENE,Path(__file__),Path(screen.__file__),Path(supported.__file__),
           Path(shared.__file__),args.rays,args.geometry_audit,args.pinch_receipt,args.parked_receipt,*proximal_files]
    rays=sources.checked_json(args.rays,sources.RAYS_SHA)["palms"]
    for side,folder in (("left",args.left_bake),("right",args.right_bake)):
        _,_,record,_=sources.fixture(side,rays[side],folder)
        archive=args.rays.parent/f"{side}_base_link.json"
        sources.checked_json(archive,record["gpu_row_sha256"])
        files.extend((archive,folder/"result.json",folder/f"{side}.base_link.npz"))
    boundary = actions = distal = None
    if (args.distal_reference is None) != (args.distal_screen is None):
        raise ValueError("provide Pinch right distal source and proof together")
    if args.distal_reference is not None and args.approach_reference is None:
        raise ValueError("Pinch distal motion requires replay of the screened approach")
    if (args.approach_reference is None) != (args.approach_screen is None):
        raise ValueError("both Pinch approach source and proof are required")
    if args.approach_reference is not None:
        boundary, actions, dependencies = load_approach(args.approach_reference, args.approach_screen, config)
        files.extend(dependencies)
        if args.distal_reference is not None:
            import numpy as np
            end_row, more, dependencies, parameters = load_distal_contact(
                args.distal_reference, args.distal_screen, config)
            actions = np.vstack((actions, more))
            files.extend(dependencies)
            plan["right_pinch_distal_contact"] = parameters
            distal = dict(parameters, archive=end_row)
        sdk_end = SDK_END + 8 * len(actions)
        plan["pinch_approach"] = actions.tolist()
        plan["budget"].update(mode="pinch_measured_guarded_approach", sdk_steps=sdk_end,
            control_steps=len(actions), solver_steps=(sdk_end-2)*shared.SUBSTEPS, soft_wall_seconds=240)
        if distal is not None:
            plan["budget"].update(mode="pinch_right_measured_distal_contact", soft_wall_seconds=360)
    for source in files:
        path=str(source.resolve(strict=True))
        digest=native.sha(source)
        if plan["frozen_files"].setdefault(path,digest)!=digest:
            raise ValueError("inconsistent Pinch supported source fingerprints")
    return plan,config,boundary,distal


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("output","plan","geometry-audit","pinch-receipt","parked-receipt","rays","left-bake","right-bake",
                 "proximal-capture","proximal-bake"):
        parser.add_argument("--"+flag,required=True,type=Path)
    parser.add_argument("--approach-reference",type=Path)
    parser.add_argument("--approach-screen",type=Path)
    parser.add_argument("--distal-reference",type=Path)
    parser.add_argument("--distal-screen",type=Path)
    parser.add_argument("--approved-frozen-sha256")
    args=parser.parse_args(argv)
    native.validate_execution_context(screen.ROOT)
    plan,config,boundary,distal=freeze(args)
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    assert_clean_import_boundary()
    digest=native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan,plan)
        print(json.dumps(dict(frozen_sha256=digest,budget=plan["budget"])))
        return 0
    if (args.approved_frozen_sha256!=digest or json.loads(args.plan.read_bytes())!=plan or args.output.exists()):
        raise ValueError("exact approved supported Pinch recipe and fresh output required")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    args.verify_near_open_settling=True
    args.measure_contact_evidence=False
    sdk_end = plan["budget"]["sdk_steps"]
    reset.execute(plan,acquire_operation=partial(shared.run_shared,profile=BackendProfile.load(screen.ROOT),
        args=args,supported_acquire=partial(supported.acquire_supported_reset,config=config,sdk_end=sdk_end,
        task="pinch"),sdk_end=sdk_end,require_no_hand_contact=True,task="pinch",
        live_support_scene=config["support_scene"], pinch_archive=boundary,
        action_after_settle=plan.get("pinch_approach"), distal_contact=distal))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
