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
        rod_substeps=12,articulations=2,action_size=24,run_uuid=run_uuid,reset_ids=runtime.resets,reset_checks=collector.reset_proofs,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),teacher_admission=False,grasp_qualified=False)


def save_snapshot(path,trainer,args,profile,seed):
    from copy import deepcopy
    from dataclasses import asdict
    import torch
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_checkpoint import _rng_state,_decode_rng
    from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import _clean,_cpu_tree,_finite_tree,_validate_adam,_validate_rng
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_checkpoint import verify_roundtrip
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    from tempfile import NamedTemporaryFile
    _clean(trainer,"clear")
    rng=_rng_state()
    rng["cuda"]=torch.cuda.get_rng_state_all() if trainer.device.type=="cuda" else []
    packet=dict(schema="dh116-tensor-native-checkpoint-v1",model=trainer.model.model_identity,
        receipt_sha256=args.receipt_sha256,backend_profile_sha256=profile.digest,seed_sha256=seed["sha256"],
        policy_seed=args.policy_seed,ppo_config=asdict(trainer.config),policy=_cpu_tree(trainer.model.state_dict()),
        optimizer=_cpu_tree(trainer.optimizer.state_dict()),rng=rng,policy_version=trainer.policy_version,
        counters=deepcopy(trainer.counters),physical_state_saved=False,restart_requires_verified_reset=True)
    packet["driver_source_sha256"]={p.name:file_sha256(p) for p in (
        Path(__file__),ROOT/"src/go2_pvcnn/control/dh116/tabletop_tensor_collector.py")}
    _finite_tree(packet)
    with NamedTemporaryFile(dir=path.parent) as stream:
        torch.save(packet,stream)
        stream.flush()
        os.fsync(stream.fileno())
        saved=torch.load(stream.name,map_location="cpu",weights_only=True)
        _validate_adam(saved["optimizer"],trainer.model,trainer.config,saved["counters"]["updates"])
        saved_rng=saved["rng"]
        if (not torch.equal(saved_rng["torch"],rng["torch"]) or saved_rng["python"]!=rng["python"]
                or not torch.equal(saved_rng["numpy"][1],rng["numpy"][1])
                or saved_rng["numpy"][0]!=rng["numpy"][0] or saved_rng["numpy"][2:]!=rng["numpy"][2:]
                or len(saved_rng["cuda"])!=len(rng["cuda"])
                or any(not torch.equal(a,b) for a,b in zip(saved_rng["cuda"],rng["cuda"],strict=True))):
            raise ValueError("serialized checkpoint changed RNG state")
        decoded=_decode_rng(dict(saved["rng"],cuda=[]))
        _validate_rng(dict(decoded,cuda=saved["rng"]["cuda"]))
        restored=type(trainer)(deepcopy(trainer.model),trainer.config,rollout_steps=32,num_envs=trainer.num_envs,device=trainer.device)
        restored.model.load_state_dict(saved["policy"])
        restored.optimizer.load_state_dict(saved["optimizer"])
        restored.counters,restored.policy_version=saved["counters"],saved["policy_version"]
        verify_roundtrip(trainer,restored)
        os.link(stream.name,path)
    return file_sha256(path)


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
             author_and_exercise=partial(train,args=args,reset=reset,profile=profile,height=height),owned_names=OWNED_NAMES)


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
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--updates",type=int)
    group.add_argument("--probe-control-windows",type=int,choices=(2,))
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
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

