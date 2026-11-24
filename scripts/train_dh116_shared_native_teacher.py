#!/usr/bin/env python3
"""Bounded real shared-DeformX Teacher PPO using original rewards and PPO writer."""
from collections import deque
from copy import copy, deepcopy
from dataclasses import asdict
from functools import partial
import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter


class NativeSharedPPO:
    def __init__(self,runtime,sim,scenes,seed,bindings,hand_bindings,archive,output,*,updates=2,policy_seed=20261123,probe_windows=0,profile_first_sdk=False):
        import torch
        torch.manual_seed(policy_seed)
        torch.cuda.manual_seed_all(policy_seed)
        from scripts.audit_dh116_shared_teacher_native_inputs import NativeInputAudit
        from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
        from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
        from go2_pvcnn.control.dh116.tabletop_shared_parameter_radius import (
            SharedParameterRadius,SharedPinchGeometry,SharedTripodGeometry)
        from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
        self.audit=NativeInputAudit(runtime,sim,scenes,seed,bindings,hand_bindings,archive,output,
                                    action_owner="policy")
        self.runtime,self.sim,self.scenes,self.bindings,self.hands=runtime,sim,scenes,bindings,hand_bindings
        if probe_windows not in (0,2):
            raise ValueError("explicit two-window throughput probe or actual PPO required")
        self.probe_windows=probe_windows
        self.profile_first_sdk=bool(profile_first_sdk)
        self._cpu_profiles={};self.active_sdk=None
        self.archive,self.output,self.updates=archive,output,(0 if probe_windows else updates)
        self.window_metrics=[]
        self.sampled_peak_gpu_used_mib=0
        self.source,self.refs,self.model=self.audit.source,self.audit.refs,self.audit.model
        self.config=self.audit.reward_config
        pp=self.config.ppo
        self.trainer=DeformablePPOTrainer(self.model,PPOConfig(lr=pp["learning_rate"],
            gamma=pp["gamma"],lam=pp["gae_lambda"],clip=pp["clip"],epochs=pp["epochs"],
            minibatches=pp["minibatches"],value_coef=pp["value"],entropy_coef=pp["entropy"],
            max_grad_norm=pp["max_grad_norm"],target_kl=pp["target_kl"],
            kl_backtrack_steps=pp["kl_backtrack_steps"]),rollout_steps=32,num_envs=len(scenes),device="cuda:0")
        self.refs.episode_cap_s=32/60
        self.radius=SharedParameterRadius(self.source)
        self.guard=[]
        if archive.style=="pinch":
            from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history
            with _canonical_pinch_history():
                prototype=SharedPinchGeometry(ROOT,scope=self.source.scope,radius_source=self.radius)
            self.guard=[prototype]*len(scenes)
        elif archive.style=="tripod":
            task=archive.mesh_plan["task_config"]
            prototype=SharedTripodGeometry(ROOT,scope=self.source.scope,radius_source=self.radius,
                task_config=Path(task["path"]),task_config_sha256=task["sha256"],mesh_bindings=bindings[0])
            from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
            from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
            for scene,names in zip(scenes,bindings,strict=True):
                guard=copy(prototype)
                guard._mesh_bindings=deepcopy(names)
                guard._support=deepcopy(prototype._support)
                for box in guard._support["supports"]:
                    import numpy as np
                    box["center_m"]=(np.asarray(box["center_m"])+scene["origin"]).tolist()
                guard.sources=dict(prototype.sources,tripod=dict(prototype.sources["tripod"],
                    mesh_bindings_sha256=digest(encode_owned(names))))
                self.guard.append(guard)
        self.rewards={}
        self.last={}
        self.sample=None
        self.rows={i:[] for i in range(len(scenes))}
        self.ring=deque(maxlen=2)
        self.metrics=[]
        self.control=0
        self.reset_controls=[0]*len(scenes)
        self.started=perf_counter()
        self.rollout_started=self.started
        self.timings={}
        self.log=NativeJournal(output/"ppo_updates.jsonl")
        self.policy_uid=_digest(self.model.state_dict())
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
        from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
        self.recipe_sha256=digest(dict(model=self.model.model_identity,reward=self.config.config_sha256,
            reset_archive=archive.reset_receipt_sha256,actual_hand_seed=seed["sha256"],
            ppo=asdict(self.trainer.config),seed=policy_seed,
            rollout_steps=32,source_sha256={str(p):file_sha256(p) for p in (
                Path(__file__),Path(__file__).with_name("smoke_dh116_shared_rod_native.py"),
                ROOT/"src/go2_pvcnn/control/dh116/tabletop_shared_teacher_reward.py",
                ROOT/"src/go2_pvcnn/control/dh116/tabletop_shared_parameter_radius.py",
                ROOT/"src/go2_pvcnn/control/dh116/tabletop_shared_teacher_checkpoint.py")}))
        from scripts.diagnose_dh116_teacher_initialization import write_new
        write_new(output/"ppo_plan.json",dict(grasp=archive.style,environments=len(scenes),
            planned_updates=self.updates,probe_control_windows=probe_windows,rollout_steps=32,policy_seed=policy_seed,
            recipe_sha256=self.recipe_sha256,ppo=asdict(self.trainer.config),source_family="shared_batched_solver_v1",
            source_profile_sha256=self.source.profile.digest,teacher_admission=False))
        self.radius.capture()
        for i in range(len(scenes)):
            self.reset_reward(i)

    def clock(self,lane):
        from go2_pvcnn.control.dh116.tabletop_shared_teacher_live_source import lane_clock
        return lane_clock(self.source,lane,policy_step=self.control,
            substep=(self.runtime.physics_step-self.audit.start-1)%8 if self.sample is not None else 0)

    def reset_reward(self,lane):
        from go2_pvcnn.control.dh116.tabletop_training_reward import TrainingReward,SubmittedRequest
        from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _capture
        from go2_pvcnn.control.dh116.tabletop_shared_teacher_reward import endpoint
        clock=self.clock(lane)
        hands=_capture(self.hands[lane],(lane,),lambda:[clock])["hands"]
        pre=endpoint(self.archive.style,clock,distance_m=self.refs.rows[lane]["distance"],
            health=int(self.runtime.solver.last_error),guard=None,hands=hands,
            finger_goal=self.config.finger_goal,scope=self.source.scope)
        reward=TrainingReward(self.config)
        request=SubmittedRequest("reset:"+clock["episode_uid"],"controller_reset_zero",pre.stamp,
            (0.,)*24,"normalized_submitted_command24_preprojection")
        reward.reset_from_verified_state(pre,previous_request=request)
        self.rewards[lane]=reward;self.last[lane]=pre
        self.reset_controls[lane]=self.control

    def profile_call(self,label,operation,*args,**kwargs):
        if not self.profile_first_sdk or self.active_sdk!=self.audit.start+1:
            return operation(*args,**kwargs)
        import cProfile
        profile=self._cpu_profiles.setdefault(label,cProfile.Profile())
        return profile.runcall(operation,*args,**kwargs)

    def action(self,sdk,lane):
        self.active_sdk=sdk
        import torch
        from go2_pvcnn.control.dh116.tabletop_training_reward import SubmittedRequest
        if self.sample is None:
            started=perf_counter()
            self.window_started=started
            packet,obs=self.audit.capture_inputs()
            self.sample=self.trainer.collect_step(obs,obs)
            self.pre_source=packet.source
            self.timings["source_and_sample_s"]=self.timings.get("source_and_sample_s",0.)+perf_counter()-started
            for i,reward in self.rewards.items():
                reward.begin(SubmittedRequest(self.sample["collection_token"],self.policy_uid,
                    self.last[i].stamp,tuple(self.sample["actions"][i].cpu().tolist()),
                    "normalized_submitted_command24_preprojection"),remaining_substeps=8)
        if sdk-self.pre_source.sdk_step not in range(1,9):
            raise RuntimeError("policy sample does not own current SDK window")
        return self.sample["actions"][lane].detach().cpu().double()

    def after_step(self,action_rows,transactions,capture,measured,issued_routes):
        import numpy as np
        import torch
        from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
        from go2_pvcnn.control.dh116.tabletop_shared_teacher_reward import endpoint
        from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import required_sdf_valid
        from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import route_world_wrenches as pinch_routes
        from go2_pvcnn.control.dh116.tabletop_power_shared_rod import route_world_wrenches as supported_routes
        started=perf_counter()
        p=torch.as_tensor(self.runtime.solver.positions()).cpu().numpy().copy()
        v=torch.as_tensor(self.runtime.solver.velocities()).cpu().numpy().copy()
        force=capture["node_forces"].cpu().numpy()
        wrench=capture["mesh_wrenches"].cpu().numpy()
        self.radius.capture()
        current=[]
        for i,(readback,transaction,(poses,centers)) in enumerate(zip(action_rows,transactions,measured,strict=True)):
            clock=self.clock(i)
            named={name:wrench[i,j].copy() for j,name in enumerate(self.runtime.mesh_names)}
            route=pinch_routes if self.archive.style=="pinch" else supported_routes
            routes,table=route({key:named[key] for key in (*self.bindings[i],"table")},
                              self.bindings[i],poses,centers)
            pair_count=int(capture["counts"][i])
            rod=dict(clock,node_positions=p[i:i+1],velocities=v[i:i+1],
                distributed_contact=force[i:i+1],hand_wrenches_com_world=routes,
                table_wrench_world=table,mesh_names=self.runtime.mesh_names,
                mesh_bindings=self.bindings[i],named_wrenches=named,
                external_support_wrenches_world={key:named[key] for key in named if key in ("table","support_0","support_1")},
                pair_capture_status="validated_solver_samples",rod_mesh_contact_pairs=dict(
                    physics_step=clock["physics_step"],rod_tick=clock["rod_tick"],
                    solver_buffer=capture["buffer_metadata"][i],
                    pairs=capture["indices"][capture["env_ids"]==i].cpu().numpy()))
            row=dict(physics_step=clock["physics_step"],policy_step=self.control,substep=clock["substep"],
                rod=rod,action_readback=readback,wrist_projection=transaction[1],guard_failure=None,
                material_director=dict(source_scope=self.source.scope),
                issued_composer_wrench=issued_routes[i],
                reaction_source_physics_step=None if clock["rod_tick"]==12 else clock["physics_step"]-1)
            current.append(row)
            self.rows[i].append(row)
            if self.archive.style in ("pinch","tripod"):
                geometry=self.profile_call("geometry",self.guard[i].capture,
                    row,sim=self.sim,bindings=self.hands[i],read_clock=lambda i=i:self.clock(i))
                row["closure_geometry"]=geometry
                if not geometry["source_complete"] or geometry["failure_reasons"]:
                    row["guard_failure"]=geometry.get("terminal_reason") or "source_geometry_unavailable"
                    self.ring.append(current)
                    self.fail("original_geometry_guard",geometry)
                    raise SourceFailure("hard guard/source failure: "+repr(geometry["failure_reasons"]))
                distance=math.fsum(max(0.,geometry["sides"][side]["required_pad_sdf"][name]["minimum_gap_m"])
                    for side in ("left","right") for name in self.guard[i].REQUIRED_PADS)/(2*len(self.guard[i].REQUIRED_PADS))
            else:
                from go2_pvcnn.control.dh116.tabletop_native_teacher_power_distance import power_pose_error
                from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT,checked_file
                goal=json.loads(checked_file(ROOT/"configs/dh116_tasks/flexible_power_near_open_settle_v1.json",
                               self.archive.task["near_open_source_sha256"]))
                targets=np.array([goal["candidate"][side]["target_pose_wxyz"][:3] for side in ("left","right")])+self.scenes[i]["origin"]
                palms=np.stack([poses[side+".base_link"][0] for side in ("left","right")])
                ds,_=power_pose_error(self.runtime._initial[i].cpu().numpy(),targets,p[i],palms)
                distance=math.fsum(ds)/2
            sample=endpoint(self.archive.style,clock,distance_m=float(distance),
                health=int(self.runtime.solver.last_error),guard=None,hands=readback["post"]["hands"],
                finger_goal=self.config.finger_goal,scope=self.source.scope)
            self.rewards[i].observe(sample);self.last[i]=sample
        self.ring.append(current)
        self.timings["guard_and_r0_s"]=self.timings.get("guard_and_r0_s",0.)+perf_counter()-started
        if next(iter(self.rows.values()))[-1]["substep"]!=7:
            return ()
        return self.complete_window()

    def complete_window(self):
        import torch
        from go2_pvcnn.control.dh116.tabletop_teacher_batch_rollout import record_group_window,GAE_FIELDS
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
        sample=self.sample
        self.refs.set_previous_request(sample["actions"],sample["collection_token"])
        packet,obs=self.audit.capture_inputs()
        with torch.inference_mode():
            values=self.model.value(obs).cpu().tolist()
        done={};settlements={};bootstrap={}
        for i,reward in self.rewards.items():
            timeout=(self.control+1-self.reset_controls[i])>=32 and not reward._terminal
            if timeout:done[i]="time_limit"
            elif reward._terminal:done[i]="reward:"+"|".join(reward._reasons)
            value=reward.settle_once(sample["collection_token"],truncated=timeout,
                buffer_full=not timeout and not reward._terminal and self.trainer.rollout.full is False
                            and len(self.trainer.rollout)+1==self.trainer.rollout_steps)
            settlements[i]=value
            if not value.terminated:
                clock=self.rows[i][-1]["rod"]
                bootstrap[i]=(values[i],dict({k:clock[k] for k in GAE_FIELDS},phase="post_action_pre_reset"))
        report=dict(env_ids=list(range(len(self.scenes))),terminal_reasons=done,aborted=False,
                    physical_samples_per_env=8,control_step=self.control)
        record_group_window(self.trainer,sample,report,self.rows,settlements,bootstrap,
                            policy_uid=self.policy_uid,reward_config=self.config)
        self.source.validate_fresh(packet)
        duration=perf_counter()-self.window_started
        self.timings["control_windows_s"]=self.timings.get("control_windows_s",0.)+duration
        self.control+=1;self.sample=None
        self.rows={i:[] for i in range(len(self.scenes))}
        from scripts.audit_dh116_isaac_batch_scene import _gpu_memory
        import resource
        self.sampled_peak_gpu_used_mib=max(self.sampled_peak_gpu_used_mib,
                                          _gpu_memory(int(os.environ["CUDA_VISIBLE_DEVICES"])))
        window=dict(grasp=self.archive.style,environments=len(self.scenes),
            sdk_step=self.runtime.physics_step,control=self.control,window_seconds=duration,
            reward_mean=sum(x.reward for x in settlements.values())/len(settlements),
            done_envs=list(done),sampled_peak_gpu_used_mib=self.sampled_peak_gpu_used_mib,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            phase_seconds=dict(self.timings))
        self.window_metrics.append(window)
        print("DH116_SHARED_PPO_WINDOW",json.dumps(window),flush=True)
        if self.trainer.rollout.full:
            started=perf_counter()
            metrics=self.trainer.update()
            record=dict(grasp=self.archive.style,environments=len(self.scenes),
                updates=self.trainer.counters["updates"],sdk_step=self.runtime.physics_step,
                rollout_seconds=started-self.rollout_started,optimizer_seconds=perf_counter()-started,
                phase_seconds=dict(self.timings),metrics=metrics,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                source_family="shared_batched_solver_v1",teacher_admission=False,grasp_qualified=False)
            if metrics["minibatch_updates"]<=0:
                self.log.emit("rejected_ppo_update",record)
                raise RuntimeError("no accepted original PPO minibatch; cannot claim trained update")
            from go2_pvcnn.control.dh116.tabletop_shared_teacher_checkpoint import save_checkpoint,load_checkpoint
            from copy import deepcopy
            started_checkpoint=perf_counter()
            saved=save_checkpoint(self.output/f"teacher_{self.archive.style}_shared_update_{record['updates']:04d}.pt",
                self.trainer,recipe_sha256=self.recipe_sha256)
            restored=type(self.trainer)(deepcopy(self.model),self.trainer.config,rollout_steps=32,
                                       num_envs=len(self.scenes),device=self.trainer.device)
            loaded=load_checkpoint(saved["path"],restored,recipe_sha256=self.recipe_sha256,
                                   expected_sha256=saved["sha256"])
            from go2_pvcnn.control.dh116.tabletop_shared_teacher_checkpoint import verify_roundtrip
            verify_roundtrip(self.trainer,restored)
            del restored
            record["checkpoint_sha256"]=saved["sha256"]
            record["checkpoint_roundtrip_verified"]=True
            record["checkpoint_seconds"]=perf_counter()-started_checkpoint
            record["iteration_seconds"]=record["rollout_seconds"]+record["optimizer_seconds"]+record["checkpoint_seconds"]
            self.metrics.append(record);self.log.emit("ppo_update",record)
            print("DH116_SHARED_PPO_UPDATE",json.dumps(record),flush=True)
            self.policy_uid=_digest(self.model.state_dict())
            self.timings={};self.rollout_started=perf_counter()
        return tuple(done)

    def after_reset(self,lane):
        self.audit.reset_reference(lane)
        self.radius.capture()
        self.reset_reward(lane)

    def fail(self,reason,evidence=None):
        from scripts.diagnose_dh116_teacher_initialization import write_new
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
        path=self.output/"ppo_failure_blackbox.json"
        if not path.exists():
            write_new(path,encode_owned(dict(reason=reason,evidence=evidence,
                last_two_sdk_steps=list(self.ring),sdk_step=self.runtime.physics_step,
                committed_controls=self.control,updates=self.trainer.counters["updates"],
                measured_sdk_step=self.sim.current_time_step_index,
                measured_sdk_time_s=float(self.sim.current_time),
                source_family="shared_batched_solver_v1",teacher_admission=False)))

    def finish(self):
        from scripts.diagnose_dh116_teacher_initialization import write_new
        if (self.sample is not None or self.trainer.counters["updates"]!=self.updates
                or (self.probe_windows and self.control!=self.probe_windows)):
            raise RuntimeError("actual bounded PPO updates incomplete")
        result=dict(grasp=self.archive.style,environments=len(self.scenes),
            updates=self.trainer.counters["updates"],controls=self.control,
            throughput_probe_only=bool(self.probe_windows),windows=self.window_metrics,
            sampled_peak_gpu_used_mib=self.sampled_peak_gpu_used_mib,
            sdk_step=self.runtime.physics_step,metrics=self.metrics,teacher_admission=False,
            grasp_qualified=False,source_family="shared_batched_solver_v1")
        write_new(self.output/("shared_ppo_probe.json" if self.probe_windows else "shared_ppo_result.json"),result)
        return result

    def close(self):
        for label,profile in self._cpu_profiles.items():
            profile.dump_stats(str(self.output/("cpu_profile_"+label+".stats")))
        self._cpu_profiles.clear()
        self.log.close()
        self.audit.close()
        self.runtime=self.sim=self.scenes=self.bindings=self.hands=self.source=self.refs=self.model=self.trainer=self.guard=self.audit=self.radius=None
        self.rows={};self.ring.clear()


def main():
    from scripts.audit_dh116_isaac_batch_scene import _run,load_grasp_archive,_gpu_memory
    from scripts.audit_dh116_deformx_shared_pairs import load_inputs
    from scripts.smoke_dh116_shared_rod_native import coupled
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    from scripts.run_dh116_work_packages import NATIVE_LOCK,native_file_lock
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp",choices=("pinch","tripod","power"),required=True)
    for name in ("receipt","backend","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("receipt-sha256","backend-sha256"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--smoke",type=Path);parser.add_argument("--smoke-sha256")
    parser.add_argument("--smoke-ledger-sha256")
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--num-envs",type=int,choices=(2,128),required=True)
    parser.add_argument("--updates",type=int,choices=(1,2),default=2)
    parser.add_argument("--height-candidate",type=Path)
    parser.add_argument("--height-candidate-sha256")
    parser.add_argument("--profile-first-sdk",action="store_true")
    parser.add_argument("--probe-control-windows",type=int,choices=(2,),default=0)
    args=parser.parse_args()
    height=None
    if args.height_candidate is not None:
        candidate=json.loads(checked_file(args.height_candidate,args.height_candidate_sha256))
        if candidate["grasp"]!=args.grasp or candidate["schema"]!="dh116-shared-training-height-candidate-v1":
            raise ValueError("grasp-specific height candidate required")
        checked_file(Path(candidate["failure_evidence"]),candidate["evidence_sha256"])
        height={key:candidate[key] for key in ("delta_z_m","evidence_sha256")}
    elif args.height_candidate_sha256 is not None:
        raise ValueError("height candidate path required")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise ValueError("explicit physical GPU and exclusive local cuda:0 mapping required")
    checked_file(args.backend,args.backend_sha256)
    sys.path.insert(0,str(args.backend.parents[1]))
    from ropesim import _ropesim
    if Path(_ropesim.__file__).resolve()!=args.backend.resolve():
        raise ValueError("actual source-locked shared native binary required")
    archive=load_grasp_archive(args.grasp,args.receipt,args.receipt_sha256)
    _,reset=load_inputs(args.grasp,args.receipt,args.receipt_sha256,
        args.smoke,args.smoke_sha256,args.smoke_ledger_sha256)
    end=10+8*(args.probe_control_windows or 32*args.updates)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu)>512:
            raise RuntimeError("physical GPU already occupied; no unsafe sharing")
        args.output.mkdir(exist_ok=False)
        _run(Path(__file__).resolve().parents[1],archive,args.gpu,args.num_envs,args.output,
             coupled=True,max_sdk_step=end,extra_exercise=partial(coupled,reset=reset,backend=_ropesim,
                 training_height=height,policy_factory=partial(NativeSharedPPO,updates=args.updates,
                                                                probe_windows=args.probe_control_windows,
                                                                profile_first_sdk=args.profile_first_sdk)))


if __name__=="__main__":
    main()
