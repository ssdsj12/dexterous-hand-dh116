#!/usr/bin/env python3
"""Acquire fresh17 Tripod Reset evidence using the existing single-App lifecycle.

Two measured SDK initialization steps, one zero-step episode Reset, no control,
solver update, policy/model query or PPO. Outer natural exit is verified separately.
"""
import argparse
from functools import partial
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from scripts import record_dh116_teacher_numeric_reset as lifecycle

ROOT = lifecycle.ROOT
native = lifecycle.native
VERSION = "dh116-fresh-supported-tripod-reset-capture-v1"


def freeze_plan(output, *, task_config, task_config_sha256, extra_sources):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    from go2_pvcnn.control.dh116.tabletop_fresh_supported_builder import SOURCE_KEYS
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state, TRIPOD_SCOPE
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    from scripts import verify_dh116_power_shared_rod as shared

    task = Path(task_config)
    config = json.loads(checked_file(task,task_config_sha256))
    if not task.is_absolute() or task.resolve(strict=True) != task:
        raise ValueError("canonical task configuration path required")
    prepare_initial_state(config,prepare_scene(ROOT)["table"],expected_scope=TRIPOD_SCOPE)
    if not isinstance(extra_sources,dict) or set(extra_sources) != SOURCE_KEYS:
        raise ValueError("complete independent palm and proximal sources required")
    extra = {key:Path(value) for key,value in extra_sources.items()}
    rays = shared.sources.checked_json(extra["rays"],shared.sources.RAYS_SHA)["palms"]
    inputs = [task,Path(__file__),extra["rays"]]
    for side in ("left","right"):
        bake = extra[side+"_bake"]
        _,_,frozen,_ = shared.sources.fixture(side,rays[side],bake)
        archive = extra["rays"].parent/(side+"_base_link.json")
        shared.sources.checked_json(archive,frozen["gpu_row_sha256"])
        inputs.extend((archive,bake/"result.json",bake/(side+".base_link.npz")))
    _,proximal_files = shared.proximal_registration_inputs(extra["proximal_capture"],extra["proximal_bake"])
    inputs.extend(proximal_files)
    plan = lifecycle.freeze_plan(output)
    # The caller uses the existing lifecycle, not its historical Pinch acquisition.
    plan.update(version=VERSION,style="tripod",mesh_count=17,pair_capacity=41*17*12,
        task_config=dict(path=str(task),sha256=task_config_sha256),
        extra_sources={key:str(value) for key,value in extra.items()},
        preserved_scope="fresh supported Tripod17, one measured Reset; no settlement/control/camera/PPO")
    plan["budget"].update(mode="fresh_tripod_reset17",sdk_steps=2,episode_resets=1,
                          new_power_mesh_registrations=6)
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular immutable input required: "+str(path))
        key = str(path.resolve(strict=True))
        digest = native.sha(path)
        if plan["frozen_files"].setdefault(key,digest) != digest:
            raise ValueError("conflicting frozen Tripod Reset source: "+key)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    from go2_pvcnn.control.dh116.tabletop_fresh_supported_builder import build_fresh_supported_owned
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_numeric_state import numeric_reset_state_scope
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, profile_reader_scope
    from go2_pvcnn.control.dh116.tabletop_mesh_observation_evidence import _write
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners

    fresh = episode = None
    try:
        profile = BackendProfile.load(ROOT)
        if profile.digest != plan["backend_profile_sha256"] or plan["budget"]["sdk_steps"] != 2:
            raise ValueError("exact frozen current profile and two-step Reset scope required")
        run = "fresh-tripod17-"+uuid4().hex
        episode_uid = run+":0"
        def record(row):
            trace.event("tripod_fresh_boundary",kind=row["kind"])
            if row["kind"] == "reset":
                _write(output/"episode_reset.json",row)
        fresh = build_fresh_supported_owned(sim,owners,project=ROOT,prepared=None,ownership=trace,
            record=record,backend_profile=profile,task_config=Path(plan["task_config"]["path"]),
            task_config_sha256=plan["task_config"]["sha256"],extra_sources=plan["extra_sources"],
            run_uuid=run,episode_uid=episode_uid,physics_budget_end=10)
        episode = fresh.episode
        def readback(row):
            _write(output/"mesh_reset_state.json",dict(readback=row,mesh_plan=fresh.mesh_plan))
        with profile_reader_scope(profile), numeric_reset_state_scope(sim=sim,rod=episode._rod,
                record=lambda row:_write(output/"mesh_reset_numeric_state.json",row)):
            result = episode.reset(fresh.reset_seed,episode_uid=episode_uid,
                owner_resets=diagnostic_reset_owners(),supported_mesh_plan=fresh.mesh_plan,
                record_reset_readback=readback)
        geometry = list(fresh.initial_geometry())
        pre = episode.pre_action_state()
        identity = pre["common"]["identity"]
        if (len(geometry) != 2 or native.clock(sim)["sdk_step"] != 2
                or identity["physics_step"] != 2 or identity["rod_tick"] != 0
                or pre["common"]["valid"].sum().item() != 66):
            raise ValueError("same-clock two-step Reset and original66 common fields required")
        _write(output/"reset_geometry.json",geometry)
        _write(output/"pre_action_common.json",pre["common"])
        outcome = dict(version=VERSION,style="tripod",run_uuid=run,episode_uid=episode_uid,
            **native.clock(sim),actual_rod_tick=identity["rod_tick"],mesh_count=17,pair_capacity=8364,
            reset_seed_sha256=fresh.reset_seed_sha256,mesh_plan_sha256=fresh.mesh_plan["sha256"],
            task_config=plan["task_config"],clearance_passed=result["clearance_passed"],
            new_control_steps=0,physics_rows=0,solver_steps=0,ppo_updates=0,
            native_exit_verified=False,grasp_qualified=False,teacher_admission=False)
        native.write_new(output/"reset_capture.json",outcome)
        return outcome
    finally:
        sim = owners = trace = fresh = episode = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output","plan","task-config","rays","left-bake","right-bake","proximal-capture","proximal-bake"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--task-sha256",required=True)
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--verify-existing-plan",action="store_true",help="CPU-only source verification child; never launches an App")
    args = parser.parse_args(argv)
    native.validate_execution_context(ROOT)
    extra = {key:getattr(args,key) for key in
             ("rays","left_bake","right_bake","proximal_capture","proximal_bake")}
    if args.verify_existing_plan or args.approved_frozen_sha256 is None:
        if args.verify_existing_plan and args.approved_frozen_sha256 is not None:
            parser.error("CPU verifier cannot execute an approved native plan")
        expected = freeze_plan(args.output,task_config=args.task_config,
                              task_config_sha256=args.task_sha256,extra_sources=extra)
        if args.verify_existing_plan:
            if json.loads(args.plan.read_bytes()) != expected or args.output.exists():
                raise ValueError("current independently rebuilt source plan differs")
        else:
            native.write_new(args.plan,expected)
        print(json.dumps(dict(frozen_sha256=native.digest(expected),budget=expected["budget"])))
        return 0
    # CPU source inspection imports USD. Run it in a separate interpreter before
    # entering the original R06 launcher, never unload SDK modules or skip its gate.
    expected = json.loads(args.plan.read_bytes())
    if native.digest(expected) != args.approved_frozen_sha256 or args.output.exists():
        raise ValueError("exact fresh published fingerprint and exclusive output required")
    verification = [sys.executable,"-B","-m","scripts.record_dh116_tripod_supported_reset",
        "--verify-existing-plan","--output",str(args.output),"--plan",str(args.plan),
        "--task-config",str(args.task_config),"--task-sha256",args.task_sha256]
    for key,value in extra.items():
        verification.extend(("--"+key.replace("_","-"),str(value)))
    subprocess.run(verification,check=True)
    # Recheck every source byte after the child exits; no native import is needed.
    for path,digest in expected["frozen_files"].items():
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
        checked_file(path,digest)
    if json.loads(args.plan.read_bytes()) != expected or args.output.exists():
        raise ValueError("frozen plan or exclusive output changed after verification")
    lifecycle.execute(expected,acquire_operation=partial(acquire,plan=expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
