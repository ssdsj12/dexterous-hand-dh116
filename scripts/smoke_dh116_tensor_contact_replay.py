#!/usr/bin/env python3
"""N2 archived-command contact diagnostic, never an on-policy PPO rollout.

Reuse NativeTensorCollector.sdk/reset without overriding sampled actions or
entering any model inference, collection, rollout writer or optimizer API.
The original 26 issued windows and final zero continuation are unchanged.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTACT_RECEIPT_SHA256 = "79498873c2211b8722338bc1be20292c74a496dd6918ea593bdea8530ef0d63d"
ACTION_SOURCE = "archived_command_replay"
WINDOWS = 27
SDK_END = 218


def bind_replay_source(archive, path, expected_sha):
    """Authenticate the old loader and retain its exact pair and issued vectors."""
    from scripts.smoke_dh116_teacher_batch_dynamic import archived_contact_prefix
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    if expected_sha != CONTACT_RECEIPT_SHA256:
        raise ValueError("locked original contact receipt SHA required")
    pair, commands = archived_contact_prefix(path, expected_sha)
    receipt = json.loads(checked_file(path, expected_sha))
    archive.pair = pair
    return commands, dict(action_source=ACTION_SOURCE, contact_receipt=str(path.resolve()),
        contact_receipt_sha256=expected_sha, source_artifact_sha256=receipt["artifact_sha256"],
        replayed_prefix_windows=26, continuation_windows=1, reset_lane=0, reset_after_window=26,
        ppo_rollout_admission=False, teacher_admission=False)


def contact_stats(packet, mesh_names, bindings, *, n):
    """Cold diagnostic counts from the actual validated flat-owner packet."""
    import torch
    empty = dict(pair_samples=[0]*n, hand_pair_samples=[0]*n,
        table_pair_samples=[0]*n, nonzero_hand_force=[False]*n,
        maximum_hand_force_n=[0.]*n)
    if packet is None:
        return empty
    slots = [i for i, name in enumerate(mesh_names) if name in bindings]
    if not slots:
        raise ValueError("original named hand mesh bindings required")
    w = packet["mesh_wrenches"]
    env, indices = packet["env_ids"], packet["indices"]
    if (w.shape != (n, len(mesh_names), 6) or env.ndim != 1
            or indices.shape != (len(env),3) or not torch.isfinite(w).all()
            or bool(((env<0)|(env>=n)).any())
            or bool(((indices[:,2]<0)|(indices[:,2]>=len(mesh_names))).any())):
        raise ValueError("complete actual N2 flat-pair/wrench shape required")
    hands = torch.isin(indices[:,2], indices.new_tensor(slots))
    tables = torch.isin(indices[:,2], indices.new_tensor(
        [i for i,name in enumerate(mesh_names) if name == "table"]))
    count = torch.bincount(env, minlength=n)
    hand_count = torch.bincount(env[hands], minlength=n)
    table_count = torch.bincount(env[tables], minlength=n)
    force = w[:,slots,:3].norm(dim=-1).amax(-1)
    nonzero = force > 0
    if bool((nonzero & (hand_count==0)).any()):
        raise ValueError("nonzero hand reaction lacks a real hand pair")
    return dict(pair_samples=count.tolist(),hand_pair_samples=hand_count.tolist(),
        table_pair_samples=table_count.tolist(),nonzero_hand_force=nonzero.tolist(),
        maximum_hand_force_n=force.tolist())


def require_contact(coverage):
    # Table support alone cannot qualify hand–rod coupling.
    if not all(sum(coverage[key])>0 for key in (
            "hand_pair_samples","nonzero_captured_hand_steps","nonzero_applied_hand_steps")):
        raise ValueError("actual hand pair and nonzero captured/next-SDK hand reaction required")
    if (coverage["hand_pair_samples"][0]<=0
            or coverage["nonzero_applied_hand_steps"][0]<=0
            or coverage["nonzero_applied_hand_steps"][1]!=0):
        raise ValueError("original active lane hand reaction and holding peer separation required")


def run_replay(collector, commands, end, *, source_sha):
    """Run only original SDK transactions and reward windows, with lane-0 Reset."""
    import torch
    from go2_pvcnn.control.dh116.tabletop_tensor_collector import require_substep_guard
    c = collector
    if c.n != 2:
        raise ValueError("N2 actual diagnostic required")
    if len(commands)!=WINDOWS or end!=SDK_END or c.runtime.physics_step!=2 or c.controls:
        raise ValueError("27 original windows and exactly 218 bounded SDK steps required")
    for command in commands:
        a = np.asarray(command)
        if a.shape!=(24,) or a.dtype!=np.float64 or not np.isfinite(a).all() or (abs(a)>1).any():
            raise ValueError("archived finite normalized float64 24D commands required")
    if np.any(commands[-1]):
        raise ValueError("original post-Reset zero continuation required")
    c.first_sdk = c.runtime.physics_step
    coverage = {key:[0,0] for key in ("pair_samples","hand_pair_samples","table_pair_samples",
        "nonzero_captured_hand_steps","nonzero_applied_hand_steps")}
    coverage["maximum_hand_force_n"]=[0.,0.]
    rows, durations, next_reset = [], [], None
    # Failed native attempts retain measured table/hand counts as well.
    c.diagnostic_coverage, c.diagnostic_steps = coverage, rows
    for index, command in enumerate(commands):
        started=perf_counter()
        c.timings={}
        actions=torch.zeros(2,24,dtype=torch.float64,device=c.device)
        actions[0]=torch.as_tensor(command,device=c.device)
        token=f"{ACTION_SOURCE}:{source_sha}:{index}"
        tokens=(token,)*2
        c.begin_action_window(actions,token=token,policy_uids=(ACTION_SOURCE,)*2,end=end)
        for substep in range(8):
            pre=c.runtime.physics_step
            applied=contact_stats(c.runtime._reaction,c.runtime.mesh_names,c.bindings,n=2)
            resets=list(c.runtime.resets)
            ticks=list(c.runtime.ticks)
            ready=c.sdk(actions,end)
            if substep<7:
                require_substep_guard(ready,"terminal lane before end of archived diagnostic window")
            require_substep_guard(c.reward._terminal,"terminal lane in archived diagnostic window")
            captured=contact_stats(c.runtime._reaction,c.runtime.mesh_names,c.bindings,n=2)
            row=dict(sdk_step=c.runtime.physics_step,source_sdk_step=pre,
                applied_sdk_step=c.runtime.physics_step,reset_ids=resets,rod_ticks_pre=ticks,
                applied=applied,captured=captured,window=index+1,substep=substep)
            rows.append(row)
            if index==26 and substep==0:
                if resets!=[2,1] or ticks!=[0,2496] or applied["nonzero_hand_force"][0]:
                    raise ValueError("post-Reset next SDK retained selected reaction or changed peer epoch")
                next_reset=row
            for i in (0,1):
                for key in ("pair_samples","hand_pair_samples","table_pair_samples"):
                    coverage[key][i]+=captured[key][i]
                coverage["nonzero_captured_hand_steps"][i]+=int(captured["nonzero_hand_force"][i])
                coverage["nonzero_applied_hand_steps"][i]+=int(applied["nonzero_hand_force"][i])
                coverage["maximum_hand_force_n"][i]=max(coverage["maximum_hand_force_n"][i],
                    captured["maximum_hand_force_n"][i],applied["maximum_hand_force_n"][i])
        c.record_window_request()
        truncated=torch.zeros(2,dtype=torch.bool,device=c.device)
        if index==25:
            truncated[0]=True
        packet=c.complete_action_window(truncated=truncated)
        c.snapshot("window_"+str(index+1),dict(schemas=tuple(c.reward.schemas),packet=packet,
            actions=actions,action_source=ACTION_SOURCE,ppo_rollout_admission=False))
        if index==25:
            c.reset((0,))
        durations.append(perf_counter()-started)
        print("DH116_ARCHIVED_COMMAND_WINDOW",index+1,"sdk",c.runtime.physics_step,flush=True)
    if (c.runtime.physics_step!=end or c.runtime.resets!=[2,1]
            or c.runtime.ticks!=(96,2592) or len(c.reset_proofs)!=1
            or (getattr(c,"trainer",None) is not None and
                (c.trainer.counters["updates"] or len(c.trainer.rollout)))):
        raise ValueError("incomplete isolated Reset diagnostic or unexpected PPO admission")
    require_contact(coverage)
    return dict(schema="dh116-tensor-archived-contact-diagnostic-v1",action_source=ACTION_SOURCE,
        updates=0,ppo_updates=0,ppo_rollout_rows=0,ppo_rollout_admission=False,teacher_admission=False,
        native_qualification=False,grasp_qualified=False,sdk_step=end,controls=WINDOWS,environments=2,
        sdk_initialization_steps=2,physics_hz=480,control_hz=60,rod_substeps=12,
        articulations=2,action_size=24,reset_ids=list(c.runtime.resets),reset_checks=c.reset_proofs,
        contact_coverage=coverage,post_reset_next_sdk=next_reset,window_seconds=durations,steps=rows)


def exercise(sim, owners, trace, project, prepared, archive, lanes, output, gpu, end,
             *, commands, reset, profile, provenance):
    import torch
    from scripts.audit_dh116_deformx_shared_pairs import make_solver
    from scripts.diagnose_dh116_teacher_initialization import write_new
    from scripts.train_dh116_tensor_native_teacher import load_candidate
    from go2_pvcnn.control.dh116.tabletop_regex_task_setup import prepare_regex_task
    from go2_pvcnn.control.dh116.tabletop_tensor_physics_runtime import NativeTensorPhysicsRuntime
    from go2_pvcnn.control.dh116.tabletop_cuda_geometry import geometry_measurement_identity
    from go2_pvcnn.control.dh116.tabletop_tensor_rod_owner import TensorRodOwner
    batch,seed,_,bindings,initial=prepare_regex_task(sim,owners,trace,project,prepared,archive,lanes)
    names=tuple(row["name"] for row in archive.mesh_plan["entries"])
    backend=load_candidate(profile)
    solver=make_solver(backend,archive.mesh_plan,reset,initial["env_origins"].float().cpu().numpy(),
        style=archive.style,capacity=41*len(names)*12,empty_lane=False)
    owners.keep("shared_rod_solver",solver)
    run_uuid=uuid.uuid4().hex
    runtime=TensorRodOwner(solver,run_uuid=run_uuid,physics_step=sim.current_time_step_index,
        episodes=tuple(f"{run_uuid}:{i}:1" for i in range(2)),mesh_names=names,device="cuda:0")
    c=NativeTensorPhysicsRuntime(project,sim,batch,prepared,archive,seed,initial,bindings,runtime,profile)
    c.arm_probe(output)
    write_new(output/"tensor_contact_replay_plan.json",dict(provenance,
        backend_profile_sha256=profile.digest,seed_sha256=seed["sha256"],max_sdk_steps=end,
        commands_float64_sha256=hashlib.sha256(np.stack(commands).tobytes()).hexdigest(),
        model_sampling=False,optimizer_updates=0,ppo_rollout_admission=False))
    try:
        actual=run_replay(c,commands,end,source_sha=provenance["contact_receipt_sha256"])
    except BaseException as exc:
        c.ctrl.invalidate()
        write_new(output/"tensor_contact_replay_failure.json",dict(error=str(exc),type=type(exc).__name__,
            completed_windows=c.controls,sdk_step=sim.current_time_step_index,rod_sdk=runtime.physics_step,
            updates=0,ppo_rollout_rows=0,
            action_source=ACTION_SOURCE,ppo_rollout_admission=False,teacher_admission=False,
            contact_coverage=getattr(c,"diagnostic_coverage",None),
            steps=getattr(c,"diagnostic_steps",None)))
        raise
    actual.update(provenance,backend_profile_sha256=profile.digest,run_uuid=run_uuid,
        geometry_measurement=geometry_measurement_identity(c.geometry),
        actual_measurement_backend=geometry_measurement_identity(c.geometry)["actual_measurement_backend"],
        measurement_scope="bounded_diagnostic",
        native_readback_mode=getattr(batch,"readback_mode","raw"),
        source_cache_metadata=batch.readback_metadata() if hasattr(batch,"readback_metadata") else None,readback_audit=batch.readback_audit() if hasattr(batch,"readback_audit") else None,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    write_new(output/"tensor_contact_replay_dynamic.json",actual)
    return actual


def worker(args):
    from scripts.audit_dh116_isaac_batch_scene import _run,load_grasp_archive,_gpu_memory
    from scripts.audit_dh116_deformx_shared_pairs import load_inputs
    from scripts.run_dh116_work_packages import NATIVE_LOCK,native_file_lock
    from go2_pvcnn.control.dh116.tabletop_regex_task_setup import OWNED_NAMES
    from go2_pvcnn.control.dh116.tabletop_tensor_backend_profile import TensorBackendProfile
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    if os.getenv("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise ValueError("explicit matching physical GPU required")
    profile=TensorBackendProfile.load()
    archive=load_grasp_archive("pinch",args.receipt,args.receipt_sha256)
    _,reset=load_inputs("pinch",args.receipt,args.receipt_sha256,args.smoke,
        args.smoke_sha256,args.smoke_ledger_sha256)
    commands,provenance=bind_replay_source(archive,args.contact_receipt,args.contact_receipt_sha256)
    baseline=json.loads(checked_file(args.baseline,args.baseline_sha256))
    if (baseline.get("sdk_step")!=218 or len(baseline.get("window_seconds",[]))!=27
            or baseline.get("nonzero_routed_hand_steps")!={"0":14,"1":0}
            or baseline.get("ppo_updates")!=0):
        raise ValueError("original 218-SDK 27-window real-contact baseline required")
    provenance.update(reset_receipt=str(args.receipt.resolve()),reset_receipt_sha256=args.receipt_sha256,
        smoke_receipt=str(args.smoke.resolve()),smoke_receipt_sha256=args.smoke_sha256,
        smoke_ledger_sha256=args.smoke_ledger_sha256,baseline_result=str(args.baseline.resolve()),
        baseline_result_sha256=args.baseline_sha256,baseline_nonzero_routed_hand_steps={"0":14,"1":0})
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu)>512:
            raise RuntimeError("physical GPU already occupied")
        _run(ROOT,archive,args.gpu,2,args.output,coupled=True,max_sdk_step=SDK_END,
            author_and_exercise=partial(exercise,commands=commands,reset=reset,profile=profile,
                provenance=provenance),owned_names=OWNED_NAMES,runtime_source_audit=True)


def verify_reaction_link(row, pending, stats):
    """CPU check of same-owner immediate-next-SDK consumption, including Reset."""
    from go2_pvcnn.control.dh116.tabletop_tensor_collector_replay import compare
    pre, reaction = row["pre"], row["reaction"]
    expected = dict(source_solver_id=pre["parent_solver_id"],parent_num_envs=pre["parent_num_envs"],
        run_uuid=pre["run_uuid"],source_physics_step=pre["sdk_step"],
        applied_physics_step=pre["sdk_step"]+1,rod_ticks=pre["rod_ticks"],
        episode_uids=pre["episode_uids"],reset_ids=pre["reset_ids"],solver_time_s=pre["solver_time_s"])
    if any(reaction.get(k)!=v for k,v in expected.items()):
        raise ValueError("wrong original same-lane immediate-next-SDK reaction ownership")
    if pending is None:
        if (reaction.get("bootstrap_empty") is not True or any(pre["rod_ticks"])
                or tuple(pre["reset_ids"])!=(1,1) or reaction["mesh_wrenches"].count_nonzero()):
            raise ValueError("only original unadvanced empty bootstrap reaction allowed")
        return
    if reaction.get("bootstrap_empty"):
        raise ValueError("bootstrap reaction reused after an actual solver step")
    for field in ("counts","env_ids","indices","vectors","normal_force","node_forces","mesh_wrenches"):
        compare(stats,"pending_reaction."+field,reaction[field],pending[field])


def reset_pending_reaction(pending, ids):
    """Independent CPU construction of the exact selected Reset reaction filter."""
    import torch
    result={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in pending.items()}
    keep=~torch.isin(result["env_ids"],result["env_ids"].new_tensor(ids))
    for field in ("env_ids","indices","vectors","normal_force"):
        result[field]=result[field][keep].clone()
    for field in ("counts","node_forces","mesh_wrenches"):
        result[field][list(ids)]=0
    return result


def replay_snapshots(directory):
    """Replay all 216 measured frames through the frozen scalar original readers."""
    import torch
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116 import tabletop_tensor_collector_replay as old
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    plan=json.loads((directory/"tensor_contact_replay_plan.json").read_text())
    commands,_=bind_replay_source(SimpleNamespace(pair=None),Path(plan["contact_receipt"]),
        plan["contact_receipt_sha256"])
    hashes={}
    def load(name):
        path=directory/("tensor_n2_"+name+".pt")
        hashes[path.name]=file_sha256(path)
        return torch.load(path,map_location="cpu",weights_only=True)
    meta=load("startup")
    cfg=old.load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json",ppo_gamma=.995)
    oracle=old.PinchOracle(meta)
    engines=[old.TrainingReward(cfg),old.TrainingReward(cfg)]
    original=[old.TrainingReward(cfg),old.TrainingReward(cfg)]
    last=[None,None]
    last_original=[None,None]
    stats={}
    coverage={key:[0,0] for key in ("pair_samples","hand_pair_samples","table_pair_samples",
        "nonzero_captured_hand_steps","nonzero_applied_hand_steps")}
    def sample(state,frame,identity,lane,distance):
        return old.endpoint("pinch",old.clock(identity,lane),distance_m=float(distance),health=0,
            guard=None,hands=old.hands(meta,state,lane),finger_goal=cfg.finger_goal,scope="owner_source_unqualified")
    def reset(ids,state,nodes,frame,identity):
        for lane in ids:
            distance=oracle.distance(state,nodes,lane,post=False)
            actual=float(frame["fields"]["distance"][lane])
            old.compare(stats,"distance_m",actual,distance,atol=2e-6)
            for group,history,d in ((engines,last,actual),(original,last_original,distance)):
                h=sample(state,frame,identity,lane,d)
                group[lane].reset_from_verified_state(h,previous_request=old.SubmittedRequest(
                    "reset:"+h.stamp.episode_uid,"controller_reset_zero",h.stamp,(0.,)*24,
                    "normalized_submitted_command24_preprojection"))
                history[lane]=h
    src=meta["source"]
    if src["sdk_step"]!=2 or tuple(src["reset_ids"])!=(1,1):
        raise ValueError("original two-SDK initial Reset required")
    reset((0,1),meta["state"],meta["nodes"],meta["reset"],src)
    pending=None
    for window,command in enumerate(commands):
        record=load("window_"+str(window+1))
        settlement=record["packet"]
        action=record["actions"]
        expected=torch.zeros(2,24,dtype=torch.float64)
        expected[0]=torch.as_tensor(command)
        old.compare(stats,"archived_command",action,expected)
        if (record.get("action_source")!=ACTION_SOURCE or record.get("ppo_rollout_admission") is not False
                or tuple(settlement["policy_uids"])!=(ACTION_SOURCE,)*2):
            raise ValueError("explicit archived-command diagnostic reward identity required")
        for lane in (0,1):
            for group,history in ((engines,last),(original,last_original)):
                group[lane].begin(old.SubmittedRequest(settlement["tokens"][lane],ACTION_SOURCE,
                    history[lane].stamp,tuple(action[lane].tolist()),
                    "normalized_submitted_command24_preprojection"),remaining_substeps=8)
        for substep in range(8):
            sdk=3+8*window+substep
            row=load("sdk_"+str(sdk))
            if src!=row["pre"] or row["post"]["sdk_step"]!=sdk:
                raise ValueError("nonconsecutive actual source/Reset identity")
            verify_reaction_link(row,pending,stats)
            old.compare(stats,"issued_archived_action",row["action"],action)
            captured=contact_stats(row["capture"],meta["mesh_names"],meta["bindings"],n=2)
            applied=(contact_stats(pending,meta["mesh_names"],meta["bindings"],n=2)
                if pending is not None else contact_stats(None,meta["mesh_names"],meta["bindings"],n=2))
            for lane in (0,1):
                for key in ("pair_samples","hand_pair_samples","table_pair_samples"):
                    coverage[key][lane]+=captured[key][lane]
                coverage["nonzero_captured_hand_steps"][lane]+=int(captured["nonzero_hand_force"][lane])
                coverage["nonzero_applied_hand_steps"][lane]+=int(applied["nonzero_hand_force"][lane])
            residual=row["capture"]["node_forces"].sum(1)+row["capture"]["mesh_wrenches"][...,:3].sum(1)
            if (residual.norm(dim=-1)>1e-3).any():
                raise ValueError("original rod/mesh force balance guard")
            old.original_targets(meta,row,stats)
            old.original_composer(meta,row,stats)
            state=row["state_post"]
            clearance,_=old.collision_clearance(state["link_pose_wxyz"].numpy(),meta["shapes"])
            old.compare(stats,"raw_source_clearance_m",row["post_clearance"],clearance,atol=1e-12)
            if np.any(clearance<old.CLEARANCE_M):
                raise ValueError("original scalar raw table clearance guard")
            for lane in (0,1):
                routes=old.original_routes(meta,state,row["capture"]["mesh_wrenches"],lane)
                guard=old.FlexiblePhysicalExchange._guard_failure(None,old.hands(meta,state,lane),routes)
                if guard:
                    raise ValueError("original scalar joint/force guard: "+guard)
                tracking=(state["base_link_pose_w"][lane,:,:3].double()
                    -row["projection"]["projected_root_poses"][lane,:,:3]).norm(dim=-1)
                speed=state["base_link_com_velocity_w"][lane,:,:3].double().norm(dim=-1)
                if (tracking>old.WRIST["tracking_guard_m"]).any() or (speed>old.WRIST["linear_speed_guard_m_s"]).any():
                    raise ValueError("original scalar wrist transport guard")
                for side,names in enumerate(meta["joint_names"]):
                    if any(abs(float(state["q"][lane,side,names.index(k)]))>old.math.pi/2 for k in old.WRIST_ANGULAR):
                        raise ValueError("original scalar wrist angular limit")
                distance=oracle.distance(state,row["nodes_post"],lane,post=True,geometry=row["geometry"])
                actual=float(row["post_frame"]["fields"]["distance"][lane])
                old.compare(stats,"distance_m",actual,distance,atol=2e-6)
                for group,history,d in ((engines,last,actual),(original,last_original,distance)):
                    h=sample(state,row["post_frame"],row["post"],lane,d)
                    group[lane].observe(h)
                    history[lane]=h
            src=row["post"]
            pending=row["capture"]
        for lane in (0,1):
            def settle(group):
                return group[lane].settle_once(settlement["tokens"][lane],
                    truncated=bool(settlement["truncated"][lane]),buffer_full=bool(settlement["buffer_full"][lane]))
            same,scalar=settle(engines),settle(original)
            old.compare(stats,"reward_same_measured_distance",settlement["reward"][lane],same.reward,atol=2e-12)
            difference=abs(same.phi_pre-scalar.phi_pre)+cfg.gamma*abs(same.endpoint_phi-scalar.endpoint_phi)
            old.compare(stats,"reward_original_sdf_distance",settlement["reward"][lane],scalar.reward,atol=difference+2e-12)
            if bool(settlement["done"][lane])!=scalar.done or bool(settlement["terminated"][lane])!=scalar.terminated:
                raise ValueError("original terminal semantics differ")
        if window==25:
            receipt=load("reset_210")
            after=receipt["source"]
            if (tuple(receipt["ids"])!=(0,) or tuple(after["reset_ids"])!=(2,1)
                    or tuple(after["rod_ticks"])!=(0,2496) or after["sdk_step"]!=210
                    or after["episode_uids"][1]!=src["episode_uids"][1]):
                raise ValueError("missing isolated lane-0 Reset before continuation")
            pending=reset_pending_reaction(pending,(0,))
            src=after
            reset((0,),receipt["state"],receipt["nodes"],receipt["references"],src)
    require_contact(coverage)
    return dict(scope="N2 archived command measured-frame replay through original scalar readers",
        sdk_steps=216,lanes=2,windows=27,updates=0,action_source=ACTION_SOURCE,
        original_guard_passed=True,maximum_absolute_errors=stats,actual_contact_coverage=coverage,
        input_sha256=hashes,ppo_rollout_admission=False,native_qualification=False,teacher_admission=False)


def main():
    if len(sys.argv)>1 and sys.argv[1]=="replay":
        parser=argparse.ArgumentParser(description="CPU replay of all original contact diagnostic snapshots")
        parser.add_argument("native",type=Path)
        parser.add_argument("--output",type=Path,required=True)
        args=parser.parse_args(sys.argv[2:])
        from scripts.diagnose_dh116_teacher_initialization import write_new
        write_new(args.output,replay_snapshots(args.native))
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu",type=int,required=True)
    for name in ("receipt","smoke","contact-receipt","baseline"):
        parser.add_argument("--"+name,type=Path,required=True)
        parser.add_argument("--"+name+"-sha256",required=name!="contact-receipt",
            default=CONTACT_RECEIPT_SHA256 if name=="contact-receipt" else None)
    parser.add_argument("--smoke-ledger-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args)
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
            or lifecycle.get("lifetime_released_before_close") is not True
            or lifecycle.get("physics_steps")!=SDK_END or actual.get("sdk_step")!=SDK_END
            or actual.get("controls")!=WINDOWS or actual.get("updates")!=0
            or actual.get("ppo_rollout_rows")!=0):
        raise ValueError("complete actual diagnostic, zero PPO admission and natural close required")
    require_contact(actual["contact_coverage"])
    write_new(output/"tensor_contact_replay_result.json",dict(actual,
        process_exit_verified=True,close_mode="normal",binary_source="flat_cuda_v1"))


if __name__=="__main__":
    main()
