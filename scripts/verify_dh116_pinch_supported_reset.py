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
    for source in files:
        path=str(source.resolve(strict=True))
        digest=native.sha(source)
        if plan["frozen_files"].setdefault(path,digest)!=digest:
            raise ValueError("inconsistent Pinch supported source fingerprints")
    return plan,config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("output","plan","geometry-audit","pinch-receipt","parked-receipt","rays","left-bake","right-bake",
                 "proximal-capture","proximal-bake"):
        parser.add_argument("--"+flag,required=True,type=Path)
    parser.add_argument("--approved-frozen-sha256")
    args=parser.parse_args(argv)
    native.validate_execution_context(screen.ROOT)
    plan,config=freeze(args)
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
    reset.execute(plan,acquire_operation=partial(shared.run_shared,profile=BackendProfile.load(screen.ROOT),
        args=args,supported_acquire=partial(supported.acquire_supported_reset,config=config,sdk_end=SDK_END,
        task="pinch"),sdk_end=SDK_END,require_no_hand_contact=True,task="pinch",
        live_support_scene=config["support_scene"]))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
