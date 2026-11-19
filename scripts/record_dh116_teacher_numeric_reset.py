#!/usr/bin/env python3
"""Source-bound Pinch numeric Reset capture before opening a PPO worker."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts import diagnose_dh116_teacher_initialization as native
from scripts import diagnose_dh116_teacher_stat_cache_release as cache

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dh116-fresh-teacher-numeric-reset-v1"
READBACK_FILE = "mesh_reset_state.json"
NUMERIC_FILE = "mesh_reset_numeric_state.json"
REQUIRED_EVIDENCE = (READBACK_FILE, NUMERIC_FILE, "reset_capture.json", "result.json",
                     "app_close_returned.json", "cleanup_stages.json", "reset_geometry.json",
                     "pre_action_common.json")


def freeze_plan(output, *, pair_proposal=None):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    result = cache.freeze_plan(ROOT, output, native.BINDINGS)
    profile = BackendProfile.load(ROOT)
    for item in profile.inventory:
        path = str(Path(item.path).resolve(strict=True))
        current = result["frozen_files"].setdefault(path,item.sha256)
        if current != item.sha256:
            raise ValueError("conflicting frozen backend inventory: "+path)
    result.update(version=SCHEMA,backend_profile_sha256=profile.digest,
                  preserved_scope="original Pinch scene, measured Reset only, no camera/control/PPO")
    if pair_proposal is not None:
        from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_bound_pair_proposal

        load_bound_pair_proposal(pair_proposal)
        result["pair_proposal"] = pair_proposal
        result["frozen_files"][pair_proposal["path"]] = pair_proposal["sha256"]
    result["budget"]["episode_resets"] = 1
    return result


def _verify_outer(run, receipt, expected_sha):
    if receipt.is_symlink() or native.sha(receipt) != expected_sha:
        raise ValueError("independent outer Reset receipt SHA differs")
    data = json.loads(receipt.read_bytes())
    if (data.get("output") != str(run.resolve()) or data.get("style") != "pinch"
            or type(data.get("exit_code")) is not int or data["exit_code"] != 0):
        raise ValueError("natural exit 0 from the same Pinch Reset run required")
    frozen = run / "frozen_plan.json"
    if frozen.is_symlink() or native.sha(frozen) != data.get("plan_sha256"):
        raise ValueError("outer Reset plan SHA differs")
    inventory = data.get("artifact_sha256", {})
    for name in REQUIRED_EVIDENCE:
        path = run/name
        if path.is_symlink() or native.sha(path) != inventory.get(name):
            raise ValueError("outer Reset artifact SHA differs: "+name)
    plan = json.loads(frozen.read_bytes())
    capture = json.loads((run / "reset_capture.json").read_bytes())
    if plan.get("pair_proposal") != data.get("pair_proposal") or plan.get("pair_proposal") != capture.get("pair_proposal"):
        raise ValueError("outer Reset proposal binding differs from plan or acquisition")
    return data


def seal_manifest(run, receipt, receipt_sha, destination):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import (
        RESET_MANIFEST_SCHEMA, candidate_config, load_reset_manifest)
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import TARGET_SCHEMA_VERSION, digest

    run, receipt = Path(run), Path(receipt)
    data = _verify_outer(run, receipt, receipt_sha)
    acquisition = json.loads((run/"reset_capture.json").read_bytes())
    lifecycle = json.loads((run/"result.json").read_bytes())
    plan = json.loads((run/"frozen_plan.json").read_bytes())
    if (plan.get("version") != SCHEMA or acquisition.get("sdk_step") != 2
            or acquisition.get("new_control_steps") != 0 or acquisition.get("clearance_passed") is not True
            or lifecycle.get("failure") is not None or lifecycle.get("close_mode") != "normal"):
        raise ValueError("qualified current-source Reset acquisition and lifecycle required")
    from go2_pvcnn.control.dh116.tabletop_mesh_observation_source import source_scope
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_readback import validate_reset_readback
    mesh = decode_owned(json.loads((run/READBACK_FILE).read_bytes()))
    readback = mesh["readback"]
    reset = readback.pop("completed_reset_receipt")
    with source_scope():
        validate_reset_readback(readback,reset_receipt=reset,mesh_plan=mesh["mesh_plan"])
    audit = decode_owned(json.loads((run/NUMERIC_FILE).read_bytes()))
    manifest = dict(schema=RESET_MANIFEST_SCHEMA, style="pinch", pose_preview_only=False,
        target_schema=TARGET_SCHEMA_VERSION, target_material_ids=list(range(41)),
        source_refs=dict(original_receipt=dict(path=str(receipt.resolve()),sha256=receipt_sha)),
        reset_seed=reset["seed"],
        rod_state={field:audit["samples"][1]["state"][field].tolist()
                   for field in ("positions","velocities","quaternions")},
        mesh_bindings_sha256=digest(encode_owned(reset["rod_reset"]["mesh_bindings"])))
    for key,name in (("reset_readback",READBACK_FILE),("numeric_reset",NUMERIC_FILE)):
        manifest["source_refs"][key]=dict(path=str((run/name).resolve()),sha256=data["artifact_sha256"][name])
    if "pair_proposal" in data:
        manifest["pair_proposal"] = data["pair_proposal"]
    # Validate the exact staged bytes using the unchanged production Reset reader.
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    import tempfile
    with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
        staged = Path(folder)/destination.name
        native.write_new(staged,manifest)
        config = candidate_config()
        config["resolved_reset_manifest"]=dict(path=str(staged),sha256=native.sha(staged))
        load_reset_manifest(config, project=ROOT)
        os.link(staged,destination)
    return dict(path=str(destination),sha256=native.sha(destination),resolved_reset_manifest_valid=True)


def _acquire(sim, owners, trace, output, profile, *, pair_proposal=None):
    from uuid import uuid4

    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners
    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as episode_api
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_mesh_observation_evidence import make_reset_type, _write
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_numeric_state import numeric_reset_state_scope
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope

    prepared = scene.prepare_scene(ROOT)
    run = "teacher-reset-"+uuid4().hex
    episode_uid = run+":0"
    binary = Path(profile.binary_path)
    def record(event):
        trace.event("physical_boundary",kind=event["kind"])
    with profile_reader_scope(profile), bindings((builder,"BACKEND",binary),
                                                  (builder,"BACKEND_SHA",profile.binary_sha256)):
        packet = (builder.load_pair_proposal() if pair_proposal is None
                  else builder.load_bound_pair_proposal(pair_proposal))
        seed = packet["reset_seed"]
        fresh = builder.build_fresh_owned(sim,owners,project=ROOT,prepared=prepared,ownership=trace,
                   record=record,run_uuid=run,episode_uid=episode_uid,physics_budget_end=10,
                   expected_reset_seed=seed, pair_proposal=pair_proposal)
        episode = fresh.episode
        factory = make_reset_type(ROOT,prepared,output)
        with bindings((episode_api,"DualHandResetTransaction",factory)), numeric_reset_state_scope(
                sim=sim,rod=episode._rod,record=lambda value:_write(output/NUMERIC_FILE,value)):
            result = episode.reset(seed,episode_uid=episode_uid,owner_resets=diagnostic_reset_owners())
        geometry = list(fresh.initial_geometry())
        if len(geometry) != 2 or native.clock(sim)["sdk_step"] != 2 or episode._rod.clock()["rod_tick"] != 0:
            raise ValueError("bounded same-clock numeric Reset required")
        pre = episode.pre_action_state()
        common = pre["common"]
        outcome = dict(run_uuid=run,episode_uid=episode_uid,**native.clock(sim),
            clearance_passed=result["clearance_passed"],physics_rows=0,new_control_steps=0,
            actual_rod_tick=episode._rod.clock()["rod_tick"],ppo_updates=0,grasp_qualified=False)
        if pair_proposal is not None:
            outcome["pair_proposal"] = pair_proposal
        _write(output/"reset_geometry.json",geometry)
        _write(output/"pre_action_common.json",common)
        native.write_new(output/"reset_capture.json",outcome)
        return outcome


def execute(plan):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock

    profile = BackendProfile.load(ROOT)
    if profile.digest != plan["backend_profile_sha256"]:
        raise ValueError("current backend profile differs from frozen numeric Reset")
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    output = Path(plan["output"])
    with native_instance_lock():
        output.mkdir()
        native.write_new(output/"frozen_plan.json",plan)
        sys.path.insert(0,str(Path(profile.binary_path).parents[1]))
        from ropesim import _ropesim
        native.validate_native_module(_ropesim,Path(profile.binary_path),plan["frozen_files"])
        from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
        assert_clean_import_boundary()
        from isaaclab.app import AppLauncher
        argv=sys.argv
        try:
            sys.argv=argv[:1]
            app=AppLauncher(headless=True,device="cuda:0",enable_cameras=False,fast_shutdown=False,
                            experience=plan["experience"]).app
        finally:
            sys.argv=argv
        entered=False
        try:
            import carb.settings
            import isaaclab.sim as sim_utils
            import omni.timeline
            import torch
            import weakref
            from isaacsim.core.simulation_manager import SimulationManager
            from scripts import dh116_pinch_acquisition_v2 as lifecycle_bindings
            from scripts.dh116_fault_safe_lifecycle_v3 import run_lifecycle
            from scripts.dh116_pinch_startup_v3 import StartupTrace,snapshot_for_startup
            from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
            from go2_pvcnn.control.dh116.tabletop_preview_launcher import commit_preview_stop
            from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import (
                clear_known_simulation_instance, simulation_source_contract)

            settings = carb.settings.get_settings()
            cleanup_contract = simulation_source_contract()
            cache_release = cache.make_before_collect(plan)
            sim_reference = None
            names=tuple(side+"_"+suffix for side in ("left","right") for suffix in
                ("hand","view","asset_sim_view","permanent_composer","instantaneous_composer"))
            def freeze_player():
                settings.set_bool("/app/player/playSimulations",False)
                return settings.get("/app/player/playSimulations")
            def make_sim():
                nonlocal sim_reference
                sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
                    dt=1/480,device="cuda:0",render_interval=1,gravity=(0.,0.,-9.81),
                    physx=sim_utils.PhysxCfg(solver_type=0)))
                sim_reference = weakref.ref(sim)
                return sim
            def acquire(sim, owners, trace, cleanup):
                result = _acquire(sim,owners,trace,output,profile,pair_proposal=plan.get("pair_proposal"))
                # Clear the known Kit stat cache before the existing owner GC.
                cache_release(sim,expected_sdk_step=2)
                return result
            def read_clock():
                observed = sim_reference() if sim_reference is not None else None
                return native.clock(observed) if observed is not None else None
            def stop():
                result = commit_preview_stop(omni.timeline.get_timeline_interface(),
                                             freeze=freeze_player,read_clock=read_clock)
                native.write_new(output/"stop_commit.json",result)
            native.validate_loaded_project_modules(ROOT,frozen_files=plan["frozen_files"])
            with bindings((lifecycle_bindings,"OWNED_NAMES",names+("runtime","rope","episode")),
                          (lifecycle_bindings,"LifetimeTrace",StartupTrace),
                          (lifecycle_bindings,"snapshot_weakrefs",snapshot_for_startup)):
                entered=True
                run_lifecycle(SimpleNamespace(update=app.update,close=lambda:app.close(wait_for_replicator=False)),
                    make_sim=make_sim,acquire=acquire,
                    synchronize=lambda:torch.cuda.synchronize("cuda:0"),
                    clear_sim=lambda sim:clear_known_simulation_instance(sim,contract=cleanup_contract),
                    stop=stop,
                    manager_is_none=lambda:SimulationManager.get_physics_sim_view() is None,
                    freeze=freeze_player,output=output,cleanup="explicit")
        finally:
            if not entered:
                app.close(wait_for_replicator=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--outer-receipt",type=Path)
    parser.add_argument("--outer-sha256")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--pair-proposal", type=Path)
    parser.add_argument("--pair-sha256")
    args=parser.parse_args(argv)
    if (args.pair_proposal is None) != (args.pair_sha256 is None):
        parser.error("--pair-proposal and --pair-sha256 must be supplied together")
    pair_proposal = None if args.pair_proposal is None else dict(path=str(args.pair_proposal), sha256=args.pair_sha256)
    if args.manifest:
        if pair_proposal is not None:
            parser.error("manifest takes its proposal only from the native receipt")
        if not args.outer_receipt or not args.outer_sha256:
            parser.error("manifest requires independent outer Reset receipt and SHA")
        print(json.dumps(seal_manifest(args.output,args.outer_receipt,args.outer_sha256,args.manifest)))
        return 0
    if args.plan is None:
        parser.error("an independent freeze plan is required")
    native.validate_execution_context(ROOT)
    expected=freeze_plan(args.output, pair_proposal=pair_proposal)
    if not args.approved_frozen_sha256:
        native.write_new(args.plan,expected)
        print(json.dumps(dict(frozen_sha256=native.digest(expected),budget=expected["budget"])))
        return 0
    plan=json.loads(args.plan.read_bytes())
    if plan!=expected or native.digest(plan)!=args.approved_frozen_sha256 or args.output.exists():
        raise ValueError("exact current-source fingerprint and fresh output required")
    execute(plan)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
