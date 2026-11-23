"""Actual shared Teacher inputs during authenticated archived action replay."""
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedBackendProfile, SharedTeacherSource
from go2_pvcnn.control.dh116.tabletop_shared_teacher_live_source import SharedSmokeReferences, read_live_common
from go2_pvcnn.control.dh116.tabletop_shared_native_teacher import SharedNativeTeacher
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import required_sdf_valid


class NativeInputAudit:
    """No policy-issued actions, optimizer calls, or training admission."""

    def __init__(self, runtime, sim, scenes, seed, bindings, hand_bindings, archive, output):
        if archive.style != "pinch":
            raise ValueError("this actual input audit uses the qualified Pinch contact prefix")
        self.runtime,self.sim,self.scenes,self.seed=runtime,sim,scenes,seed
        self.bindings,self.hands,self.output=bindings,hand_bindings,output
        self.start=runtime.physics_step
        from scripts.diagnose_dh116_teacher_initialization import write_new
        write_new(output/"shared_source_clock_input.json",dict(
            sdk_step=sim.current_time_step_index,measured_sdk_time_s=float(sim.current_time),
            configured_sdk_dt_s=float(sim.get_physics_dt()),solver_time_s=float(runtime.solver.time),
            owned_solver_time_s=runtime._time,owner_sdk_step=runtime.physics_step,
            solver_last_error=int(runtime.solver.last_error)))
        profile=SharedBackendProfile.load()
        origins=np.array([scene["origin"]+np.asarray(seed["origin"]) for scene in scenes])
        self.source=SharedTeacherSource(runtime,sim,backend_profile=profile,env_origins=origins,
            rod_uids=tuple("rod:"+str(i) for i in range(len(scenes))),
            scope="owner_source_unqualified",device="cuda:0")
        self.refs=SharedSmokeReferences(self.source,grasp="pinch",episode_cap_s=8.,wall_cap_s=180.)
        from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history
        from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import MeasuredClosureGeometry
        with _canonical_pinch_history():
            self.geometry=MeasuredClosureGeometry(ROOT)
        cfg=json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_pinch_v1.json").read_text())
        reward=load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json",ppo_gamma=.995)
        self.model=SharedNativeTeacher(cfg,backend_profile=profile,num_envs=len(scenes),reward_config=reward).cuda()
        self.rows=[];self.pending=None;self.count=0
        for i in range(len(scenes)):
            self.reset_reference(i)
        self.capture("initial_reset")

    def reset_reference(self,lane):
        from scripts.audit_dh116_batch_rod_exchange import _live_contact_geometry
        poses,_=_live_contact_geometry(self.scenes[lane],self.bindings[lane])
        nodes=torch.as_tensor(self.runtime.solver.positions())[lane].cpu().numpy()
        radius=float(self.runtime.solver.parameter_snapshot()["contact_radius_m"])
        distances=[]
        for side in ("left","right"):
            pads=[]
            for name in ("finger14_link","finger23_link"):
                xyz,quat=poses[side+"."+name]
                transform=self.geometry._pose_matrix(np.r_[xyz,quat[3],quat[:3]])
                sample=self.geometry._sdf(self.geometry._grids[side][name],transform,nodes,radius=radius)
                if not required_sdf_valid(sample):
                    raise ValueError("actual reset required pad SDF source unavailable")
                pads.append(max(0.,sample["minimum_gap_m"]))
            distances.append(sum(pads)/2)
        self.refs.reset_lane(lane,distance_mean_m=sum(distances)/2)

    def capture(self,phase):
        started=perf_counter()
        policy=(self.runtime.physics_step-self.start)//8
        common=read_live_common(self.source,self.hands,self.seed,policy_step=policy)
        contexts=self.refs.contexts()
        packet=self.source.capture(common,contexts)
        observations=self.model.pack_shared(packet,self.source)
        with torch.inference_mode():
            mean=self.model.deterministic(observations)
        if not torch.isfinite(mean).all():
            raise ValueError("actual native Teacher diagnostic output nonfinite")
        row=dict(phase=phase,sdk_step=self.runtime.physics_step,envs=len(common),
            common_valid=int(observations["common_valid"].sum()),
            parent_parameter_envs=packet.parent_parameters["num_envs"],
            measured_sdk_time_s=packet.source.sdk_time_s,sdk_dt_s=packet.source.sdk_dt_s,
            nominal_policy_time_s=packet.source.policy_time_s,
            resets=list(packet.source.reset_ids),rod_ticks=list(packet.source.rod_ticks),
            max_abs_diagnostic_mean=float(mean.abs().max()),capture_seconds=perf_counter()-started,
            ppo_updates=0,policy_action_owner="archived_command_replay",
            teacher_admission=False)
        self.rows.append(row)
        print("DH116_SHARED_NATIVE_INPUT",json.dumps(row),flush=True)

    def after_step(self,action_rows):
        actions=np.concatenate([np.asarray(row["request"]["action24"]) for row in action_rows])
        if self.pending is None:
            self.pending=actions.copy()
        elif not np.array_equal(actions,self.pending):
            raise ValueError("archived request changed inside a control window")
        self.count+=1
        if self.count==8:
            self.refs.set_previous_request(torch.as_tensor(actions),
                "actual-archive-window:"+str(self.runtime.physics_step))
            self.pending=None;self.count=0
            self.capture("post_window_pre_reset")

    def after_reset(self,lane):
        if self.count:
            raise ValueError("input references Reset inside a control window")
        self.reset_reference(lane)
        self.capture("post_local_reset")

    def finish(self):
        if self.count:
            raise ValueError("incomplete control window in source audit")
        result=dict(captures=self.rows,sdk_steps=self.runtime.physics_step,
            native_source_captures=len(self.rows),native_ppo_updates=0,teacher_admission=False)
        from scripts.diagnose_dh116_teacher_initialization import write_new
        write_new(self.output/"shared_teacher_inputs.json",result)
        return result

    def close(self):
        self.source=self.runtime=self.sim=self.scenes=self.hands=self.bindings=self.model=self.geometry=self.refs=None
