#!/usr/bin/env python3
"""Run the source-locked two-articulation tensor Teacher in a supervised native child."""
import argparse
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import uuid

ROOT = Path(__file__).resolve().parents[1]


def load_candidate(profile):
    name = "ropesim._ropesim"
    binary = Path(profile.binary_path).resolve(strict=True)
    loaded = sys.modules.get(name)
    if loaded is not None:
        if Path(loaded.__file__).resolve() != binary:
            raise RuntimeError("another native ropesim binary already loaded")
        return loaded
    spec = importlib.util.spec_from_file_location(name,binary)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module


def train(sim, owners, trace, project, prepared, archive, lanes, output, gpu, end, *,args,reset,profile,height):
    import torch
    from scripts.audit_dh116_deformx_shared_pairs import make_solver
    from scripts.diagnose_dh116_teacher_initialization import write_new
    from go2_pvcnn.control.dh116.tabletop_regex_task_setup import prepare_regex_task
    from go2_pvcnn.control.dh116.tabletop_tensor_collector import NativeTensorCollector
    from go2_pvcnn.control.dh116.tabletop_cuda_geometry import geometry_measurement_identity
    from go2_pvcnn.control.dh116.tabletop_tensor_rod_owner import TensorRodOwner
    batch,seed,_,bindings,initial = prepare_regex_task(sim,owners,trace,project,prepared,archive,lanes,seed_height=height)
    names = tuple(row["name"] for row in archive.mesh_plan["entries"])
    backend = load_candidate(profile)
    solver = make_solver(backend,archive.mesh_plan,reset,initial["env_origins"].float().cpu().numpy(),
        style=archive.style,capacity=41*len(names)*12,empty_lane=False)
    owners.keep("shared_rod_solver",solver)
    run_uuid=uuid.uuid4().hex
    runtime = TensorRodOwner(solver,run_uuid=run_uuid,physics_step=sim.current_time_step_index,
        episodes=tuple(f"{run_uuid}:{i}:1" for i in range(len(lanes))),mesh_names=names,device="cuda:0")
    torch.manual_seed(args.policy_seed)
    torch.cuda.manual_seed_all(args.policy_seed)
    collector = NativeTensorCollector(project,sim,batch,prepared,archive,seed,initial,bindings,runtime,profile)
    probe = bool(args.probe_control_windows)
    if probe and len(lanes)==2:
        collector.arm_probe(output)
    total = 2 if probe else args.updates*32
    iterations = []
    window_seconds = []
    window_phases = []
    rollout_start=perf_counter()
    try:
        for index in range(total):
            started_window=perf_counter()
            reward = collector.window(end,probe_reset=probe and len(lanes)==2 and index==0)
            window_seconds.append(perf_counter()-started_window)
            window_phases.append(dict(collector.timings))
            if collector.trainer.rollout.full:
                if probe:
                    raise RuntimeError("probe must not update a policy")
                torch.cuda.synchronize()
                start=perf_counter()
                metrics=collector.trainer.update()
                torch.cuda.synchronize()
                finish=perf_counter()
                if metrics["minibatch_updates"]<=0:
                    raise RuntimeError("PPO rejected every minibatch")
                update=collector.trainer.counters["updates"]
                record=dict(update=update,sdk_step=runtime.physics_step,metrics=metrics,
                    rollout_seconds=start-rollout_start,optimizer_seconds=finish-start,
                    window_seconds=window_seconds[-32:],window_phases=window_phases[-32:])
                path=output/f"tensor_{archive.style}_update_{update:04d}.pt"
                record["checkpoint_sha256"]=save_snapshot(path,collector.trainer,args,profile,seed)
                record["iteration_seconds"]=perf_counter()-rollout_start
                write_new(output/f"tensor_update_{update:04d}.json",record)
                iterations.append(record)
                rollout_start=perf_counter()
            print("DH116_TENSOR_WINDOW",index+1,"sdk",runtime.physics_step,"reward",reward,flush=True)
    except BaseException as exc:
        collector.ctrl.invalidate()
        write_new(output/"tensor_failure.json",dict(error=str(exc),type=type(exc).__name__,
            completed_windows=collector.controls,sdk_step=sim.current_time_step_index,rod_sdk=runtime.physics_step,
            updates=collector.trainer.counters["updates"],teacher_admission=False))
        raise
    updates=collector.trainer.counters["updates"]
    if (runtime.physics_step!=end or collector.controls!=total or updates!=(0 if probe else args.updates)
            or (probe and runtime.resets != ([1,2] if len(lanes)==2 else [1]*len(lanes)))):
        raise ValueError("incomplete actual bounded run")
    return dict(throughput_probe_only=probe,updates=updates,controls=total,environments=len(lanes),
        sdk_step=end,iterations=iterations,window_seconds=window_seconds,window_phases=window_phases,sdk_initialization_steps=2,physics_hz=480,control_hz=60,
        rod_substeps=12,articulations=2,action_size=24,run_uuid=run_uuid,
        geometry_backend=getattr(collector.geometry,"backend",None),
        geometry_measurement=geometry_measurement_identity(collector.geometry),
        actual_measurement_backend=geometry_measurement_identity(collector.geometry)["actual_measurement_backend"],
        measurement_scope="bounded_diagnostic", measurement_native_evidence=args.measurement_native_evidence,
        native_readback_mode=getattr(batch,"readback_mode","raw"),
        source_cache_metadata=batch.readback_metadata() if hasattr(batch,"readback_metadata") else None,readback_audit=batch.readback_audit() if hasattr(batch,"readback_audit") else None,
        reset_ids=runtime.resets,reset_checks=collector.reset_proofs,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),teacher_admission=False,grasp_qualified=False)


def save_snapshot(path,trainer,args,profile,seed):
    """Use the separately verified v2 optimizer/RNG restart format."""
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
    from go2_pvcnn.control.dh116.tabletop_tensor_checkpoint import save_checkpoint
    recipe=digest(dict(schema="dh116-tensor-native-ppo-recipe-v2",
        model=trainer.model.model_identity,rollout_steps=trainer.rollout_steps,
        num_envs=trainer.num_envs,receipt_sha256=args.receipt_sha256,
        profile_sha256=profile.digest,seed_sha256=seed["sha256"],policy_seed=args.policy_seed,
        physics_hz=480,control_hz=60,solver_substeps=12,action_size=24,
        geometry_backend=os.environ.get("DH116_TENSOR_GEOMETRY_BACKEND","eager"),
        readback_backend=os.environ.get("DH116_NATIVE_READBACK_BACKEND","raw"),
        geometry_measurement=__import__("go2_pvcnn.control.dh116.tabletop_cuda_geometry",
            fromlist=["CudaGeometryProfile"]).CudaGeometryProfile.load().identity()
            if os.environ.get("DH116_TENSOR_GEOMETRY_BACKEND")=="cuda_cabi" else None))
    result=save_checkpoint(path,trainer,recipe_sha256=recipe,
        receipt_sha256=args.receipt_sha256,backend_profile=profile,
        seed_sha256=seed["sha256"],policy_seed=args.policy_seed)
    return result["sha256"]


def worker(args, end):
    from scripts.audit_dh116_isaac_batch_scene import _run,load_grasp_archive,_gpu_memory
    from scripts.audit_dh116_deformx_shared_pairs import load_inputs
    from scripts.run_dh116_work_packages import NATIVE_LOCK,native_file_lock
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    from go2_pvcnn.control.dh116.tabletop_regex_task_setup import OWNED_NAMES
    from go2_pvcnn.control.dh116.tabletop_tensor_backend_profile import TensorBackendProfile
    if os.getenv("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("explicit matching physical GPU required")
    profile=TensorBackendProfile.load()
    archive=load_grasp_archive(args.grasp,args.receipt,args.receipt_sha256)
    _,reset=load_inputs(args.grasp,args.receipt,args.receipt_sha256,args.smoke,args.smoke_sha256,args.smoke_ledger_sha256)
    height=None
    if args.height_candidate:
        candidate=json.loads(checked_file(args.height_candidate,args.height_candidate_sha256))
        if candidate["schema"]!="dh116-shared-training-height-candidate-v1" or candidate["grasp"]!=args.grasp:
            raise ValueError("grasp-specific original height candidate required")
        checked_file(candidate["failure_evidence"],candidate["evidence_sha256"])
        height={k:candidate[k] for k in ("delta_z_m","evidence_sha256")}
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu)>512:
            raise RuntimeError("physical GPU already occupied")
        _run(ROOT,archive,args.gpu,args.num_envs,args.output,coupled=True,max_sdk_step=end,
             author_and_exercise=partial(train,args=args,reset=reset,profile=profile,height=height),owned_names=OWNED_NAMES,runtime_source_audit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp",choices=("pinch","tripod","power"),required=True)
    parser.add_argument("--num-envs",type=int,choices=(2,128),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--receipt-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--smoke",type=Path)
    parser.add_argument("--smoke-sha256")
    parser.add_argument("--smoke-ledger-sha256")
    parser.add_argument("--height-candidate",type=Path)
    parser.add_argument("--height-candidate-sha256")
    parser.add_argument("--policy-seed",type=int,default=20261123)
    parser.add_argument("--measurement-native-receipt",type=Path,action="append",default=[],
        help="pinned outer.json; repeat for N2 contact and N128 throughput diagnostics")
    parser.add_argument("--measurement-native-receipt-sha256",action="append",default=[])
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--updates",type=int)
    group.add_argument("--probe-control-windows",type=int,choices=(2,))
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.updates and os.environ.get("DH116_NATIVE_READBACK_AUDIT")=="1":
        parser.error("cache/raw oracle audit is N2 diagnostic only and cannot collect PPO")
    from go2_pvcnn.control.dh116.tabletop_runtime_execution import validate_runtime_execution
    try:
        validate_runtime_execution()
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.measurement_native_receipt)!=len(args.measurement_native_receipt_sha256):
        parser.error("measurement native receipt paths and SHA256 values must be paired")
    args.measurement_native_evidence=None
    measurement_mode=os.environ.get("DH116_TENSOR_GEOMETRY_BACKEND","eager")
    if args.updates and measurement_mode=="cuda_cabi" and not args.measurement_native_receipt:
        parser.error("CUDA geometry requires verified native lifecycle evidence before PPO; use bounded diagnostics only")
    if args.measurement_native_receipt:
        if measurement_mode!="cuda_cabi":
            parser.error("measurement native receipts require cuda_cabi mode")
        from go2_pvcnn.control.dh116.tabletop_cuda_geometry import verify_measurement_native_receipts
        try:
            args.measurement_native_evidence=verify_measurement_native_receipts(
                list(zip(args.measurement_native_receipt,args.measurement_native_receipt_sha256)),style=args.grasp)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            parser.error("invalid measurement native evidence: "+str(exc))
    if args.grasp=="tripod":
        parser.error("Tripod native support/route guard adapter is unavailable")
    if bool(args.height_candidate)!=bool(args.height_candidate_sha256):
        parser.error("height candidate path and SHA must be paired")
    if args.updates is not None and args.updates<2:
        parser.error("training requires at least two PPO updates")
    total=args.probe_control_windows or 32*args.updates
    end=2+8*total
    if args.worker:
        worker(args,end)
        return
    from scripts.diagnose_dh116_teacher_initialization import write_new
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    native=output/"native"
    native.mkdir()
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--output",str(native),"--worker"]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER="PCI_BUS_ID")
    with (output/"native.log").open("wb") as log:
        child=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
    write_new(output/"process_exit.json",dict(returncode=child.returncode))
    if child.returncode:
        raise RuntimeError(f"native worker exited with {child.returncode}; see {output/'native.log'}")
    lifecycle=json.loads((native/"result.json").read_text())
    close=json.loads((native/"app_close_returned.json").read_text())
    actual=lifecycle["acquisition"]
    if (close.get("app_close_returned") is not True or close.get("close_mode")!="normal"
            or lifecycle.get("failure") is not None or lifecycle.get("close_mode")!="normal"
            or lifecycle.get("lifetime_released_before_close") is not True or lifecycle.get("physics_steps")!=end
            or actual.get("sdk_step")!=end or actual.get("controls")!=total or actual.get("updates")!=(args.updates or 0)):
        raise ValueError("actual complete acquisition and verified natural close required")
    write_new(output/("tensor_ppo_probe.json" if args.probe_control_windows else "tensor_ppo_result.json"),
        dict(actual,process_exit_verified=True,close_mode="normal",binary_source="flat_cuda_v1"))


if __name__=="__main__":
    main()

