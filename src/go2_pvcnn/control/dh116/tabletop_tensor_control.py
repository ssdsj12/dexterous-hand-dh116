"""Two-side GPU command and reaction arithmetic using the original kernels."""
import math
import torch
from .tabletop_deformable_solver_clock import solver_clock_matches
from .linear_coupling import ACTIVE,MIMIC,active_limits
from .tabletop_surface_contract import ACTION_CONTRACT,validate_actions
from .tabletop_command_kernels import finger_pre_guard,surface_final_guard,wrist_requested_targets
from .tabletop_tensor_wrist_projection import project_tensor_policy_wrist_targets
from .tabletop_palm_mount import local_translation,JOINT_NAMES,ANGULAR_JOINT_NAMES,CONTRACT as MOUNT
DT=1/480

class TensorControl:
    def __init__(self,batch,root_seed_w,bindings,mesh_names):
        self.batch=batch;self.roots=root_seed_w.detach().double().clone()
        self.device=self.roots.device;self.n=len(self.roots)
        self.palms=torch.zeros(self.n,2,3,device=self.device,dtype=torch.float64)
        self.angles=torch.zeros_like(self.palms)
        self.active=[];self.write_ids=[];self.mimic=[]
        self.slots=[];self.contact_bodies=[]
        for side in ("left","right"):
            names=batch.sides[side].joint_names
            self.active.append([names.index(k) for k in ACTIVE])
            self.write_ids.append(self.active[-1]+[names.index(k) for k in JOINT_NAMES+ANGULAR_JOINT_NAMES])
            self.mimic.append([(names.index(f),names.index(m),ratio,offset) for f,m,ratio,offset in MIMIC])
            keys=[k for k,v in bindings.items() if v["side"]==side]
            self.slots.append([mesh_names.index(k) for k in keys])
            self.contact_bodies.append([bindings[k]["body_index"] for k in keys])
        self.low,self.high=(self.roots.new_tensor(v).repeat(2) for v in active_limits())
        self.rates=dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],
            wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
            rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
        self.pending=None;self.epoch=0
        self.mesh_count=len(mesh_names)
        self._failed=False

    def invalidate(self,env_ids=None):
        if env_ids is None:
            self.palms.zero_();self.angles.zero_()
        else:
            self.palms[env_ids]=0;self.angles[env_ids]=0
        self.pending=None;self.epoch+=1

    def joint_failures(self,state):
        q=state["q"].double()
        active=torch.cat([q[:,s,self.active[s]] for s in (0,1)],1)
        mimic=torch.stack([q[:,s,f]-ratio*q[:,s,m]-offset for s in (0,1)
                           for f,m,ratio,offset in self.mimic[s]],1)
        return ((active<self.low-.02)|(active>self.high+.02)).any(1)|(mimic.abs()>.02).any(1)|~torch.isfinite(q).all((1,2))|~torch.isfinite(state["qd"]).all((1,2))

    def prepare(self,actions,state,clearance_m,base_poses_wxyz,*,stamp):
        if self._failed:raise ValueError("failed control owner requires a new native scope")
        if self.pending is not None:raise ValueError("previous SDK transaction unfinished")
        if (not isinstance(actions,torch.Tensor) or actions.shape!=(self.n,24)
                or actions.device!=self.device or not actions.is_floating_point()):
            raise ValueError("same-device full normalized floating batch24 required")
        finite_inputs=[actions]
        expected_joints=len(self.batch.sides["left"].joint_names)
        for key in ("q","qd","backend_targets"):
            value=state[key]
            if (not isinstance(value,torch.Tensor) or value.shape!=(self.n,2,expected_joints)
                    or value.device!=self.device or not value.is_floating_point()):
                self._failed=True
                raise ValueError("finite complete original q/qd/backend targets before command projection required")
            finite_inputs.append(value)
        for value,shape in ((state["root_pose_wxyz"],(self.n,2,7)),
                (base_poses_wxyz,(self.n,2,7)),(clearance_m,(self.n,2))):
            if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=self.device
                    or not value.is_floating_point()):
                self._failed=True
                raise ValueError("finite complete actual pre-action roots/base/clearance required")
            finite_inputs.append(value)
        # Shapes/devices/dtypes precede one current-frame numeric decision.
        finite=torch.isfinite(torch.cat([v.reshape(-1) for v in finite_inputs])).all()
        if not bool(finite & ~(actions.abs()>1).any()):
            validate_actions(actions)  # Preserve action failure semantics on the cold error path.
            self._failed=True
            raise ValueError("finite complete actual pre-action state/base/clearance required")
        q=state["q"].double()
        command=torch.cat([state["backend_targets"][:,s,self.active[s]].double() for s in (0,1)],1)
        actual=torch.cat([q[:,s,self.active[s]] for s in (0,1)],1)
        mimic=torch.stack([q[:,s,f]-ratio*q[:,s,m]-offset for s in (0,1)
                           for f,m,ratio,offset in self.mimic[s]],1)
        action=actions.double()
        target=finger_pre_guard(command,action,actual,mimic,self.low,self.high,DT,self.rates,policy_owned=True)
        _,target=surface_final_guard(command,target,self.low,self.high,DT)
        palm,angle=wrist_requested_targets(self.palms,self.angles,action,self.rates,DT)
        projection=project_tensor_policy_wrist_targets(root_seed_w=self.roots,measured_base_w=base_poses_wxyz.double(),
            common_lift_offset_m=self.roots.new_zeros(self.n),measured_clearance_m=clearance_m.double(),
            palm_offsets_m=self.palms,requested_palm_offsets_m=palm,
            angles_rad=self.angles,requested_angles_rad=angle,
            allowed=torch.ones(self.n,device=self.device,dtype=torch.bool),
            live=torch.ones(self.n,device=self.device,dtype=torch.bool))
        anchors=state["root_pose_wxyz"].double()
        local=torch.stack([local_translation(projection["projected_root_poses"][:,s],anchors[:,s]) for s in (0,1)],1)
        if bool((local.abs()>MOUNT["limit_m"]).any()):raise ValueError("original mount translation hard limit")
        expected=[]
        for s,side in enumerate(("left","right")):
            values=torch.cat((target[:,6*s:6*s+6],local[:,s],projection["projected_angles"][:,s]),1).float()
            expected.append(values)
        self.pending=dict(stamp=stamp,epoch=self.epoch,expected=expected,projection=projection,phase="prepared")
        return self.pending

    def submit(self,transaction,*,stamp,wrenches,state,reaction_source):
        if (transaction is not self.pending or stamp!=transaction["stamp"]
                or self.epoch!=transaction["epoch"] or transaction["phase"]!="prepared"):
            raise ValueError("one owned unsubmitted pre-action GPU transaction required")
        try:
            expected_source=dict(source_solver_id=stamp.parent_solver_id,parent_num_envs=stamp.parent_num_envs,
                run_uuid=stamp.run_uuid,source_physics_step=stamp.sdk_step,
                applied_physics_step=stamp.sdk_step+1,rod_ticks=stamp.rod_ticks,
                episode_uids=stamp.episode_uids,reset_ids=stamp.reset_ids,
                solver_time_s=stamp.solver_time_s)
            if (not isinstance(reaction_source,dict)
                    or any(reaction_source.get(k)!=v for k,v in expected_source.items())
                    or reaction_source.get("mesh_wrenches") is not wrenches):
                raise ValueError("owned immediate-next-SDK reaction and current source identity required")
            if (not isinstance(wrenches,torch.Tensor)
                    or wrenches.shape!=(self.n,self.mesh_count,6)
                    or wrenches.device!=self.device
                    or not wrenches.is_floating_point()):
                raise ValueError("finite complete same-device actual mesh wrenches required")
            for key in ("link_pose_xyzw","body_com_pose_w"):
                value=state[key]
                if (not isinstance(value,torch.Tensor) or value.ndim!=4
                        or value.shape[:2]!=(self.n,2) or value.shape[-1]!=7
                        or value.device!=self.device or not value.is_floating_point()
                        or any(max(ids)>=value.shape[2] for ids in self.contact_bodies)):
                    raise ValueError("finite complete actual link and COM poses required")
            for s in (0,1):
                value=transaction["expected"][s]
                if (not isinstance(value,torch.Tensor) or value.shape!=(self.n,len(self.write_ids[s]))
                        or value.device!=self.device or not value.is_floating_point()):
                    raise ValueError("finite complete projected command before native write required")
            finite_inputs=[wrenches,state["link_pose_xyzw"],state["body_com_pose_w"],*transaction["expected"]]
            finite=torch.isfinite(torch.cat([v.reshape(-1) for v in finite_inputs])).all()
            forces=[];torques=[];over_force=[]
            for s in (0,1):
                bodies=self.contact_bodies[s];w=wrenches[:,self.slots[s]].double()
                origin=state["link_pose_xyzw"][:,s,bodies,:3].double()
                com=state["body_com_pose_w"][:,s,bodies,:3].double()
                # The original guard is evaluated before the final SDK FP32 cast.
                over_force.append((torch.linalg.vector_norm(w[...,:3],dim=-1)>30).any())
                forces.append(w[...,:3].float())
                torques.append((w[...,3:]+torch.linalg.cross(origin-com,w[...,:3],dim=-1)).float())
            representable=torch.isfinite(torch.cat([v.reshape(-1) for v in forces+torques])).all()
            over_force=torch.stack(over_force).any()
            if not bool(finite & ~over_force & representable):
                raise ValueError("finite complete command/link/COM/wrench, original 30 N reaction guard and representable SDK force/COM torque required")
            # Complete validation precedes every native mutation.
            for s,side in enumerate(("left","right")):
                owner=self.batch.sides[side]
                owner.hand.permanent_wrench_composer.reset()
                owner.set_wrenches(forces=forces[s],torques=torques[s],body_ids=self.contact_bodies[s],is_global=True)
                owner.set_joint_position_target(transaction["expected"][s],joint_ids=self.write_ids[s])
            self.batch.write_data_to_sim()
            transaction["phase"]="submitted"
        except BaseException:
            transaction["phase"]="failed";self.pending=None;self._failed=True
            raise

    def finish(self,transaction,state,*,stamp):
        pre=transaction["stamp"]
        if (transaction is not self.pending or self.epoch!=transaction["epoch"]
                or transaction["phase"]!="submitted"
                or stamp.sdk_step!=pre.sdk_step+1
                or stamp.rod_ticks!=tuple(t+12 for t in pre.rod_ticks)
                or not solver_clock_matches(pre.solver_time_s,stamp.solver_time_s,steps=1)
                or any(getattr(stamp,k)!=getattr(pre,k) for k in
                    ("parent_solver_id","parent_num_envs","env_rows","run_uuid","episode_uids",
                     "rod_uids","reset_ids","profile_sha256","scope"))):
            self._failed=True;self.pending=None
            raise ValueError("one actual SDK/rod step with unchanged source identity required by owned batch")
        bad=self.joint_failures(state)
        for s in (0,1):
            actual=state["backend_targets"][:,s,self.write_ids[s]]
            bad|=~torch.isclose(actual,transaction["expected"][s],rtol=0,atol=1e-7).all(1)
        if bool(bad.any()):
            self.pending=None;self._failed=True;raise ValueError("actual SDK target/finite/joint/MIMIC guard failed")
        self.palms=transaction["projection"]["palm_offsets_m"]
        self.angles=transaction["projection"]["projected_angles"]
        transaction["phase"]="finished"
        self.pending=None
        return bad
