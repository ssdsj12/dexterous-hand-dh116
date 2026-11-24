"""Native two-articulation full-action collector with synchronous substep guards."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import json
import math
from time import perf_counter
import uuid

import numpy as np
import torch

from .tabletop_batch_geometry import collision_clearance
from .tabletop_deformable_wrist_projection import CLEARANCE_M, SOURCE
from .tabletop_device_ppo import DeviceDeformablePPOTrainer
from .tabletop_joint_target_readback import WRIST_ANGULAR
from .tabletop_native_teacher_profile import checked_file
from .tabletop_palm_mount import CONTRACT as WRIST
from .tabletop_privileged_ppo import PPOConfig
from .tabletop_regex_task_setup import capture_regex_state, restore_regex_seed
from .tabletop_tensor_control import TensorControl
from .tabletop_tensor_geometry import TensorGeometry, DECISION_MARGIN_M
from .tabletop_tensor_mesh_bridge import TensorMeshBridge
from .tabletop_tensor_native_inputs import pack_shared_tensor_inputs
from .tabletop_tensor_native_teacher import TensorNativeTeacher
from .tabletop_tensor_source_owner import TensorSourceOwner
from .tabletop_tensor_training_reward import (TensorLaneSchema, TensorRewardFrame,
    TensorTrainingReward, measured_finger_errors)
from .tabletop_training_reward import ERROR_FIELDS, load_config

SIDES = ("left", "right")


def probe_copy(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):
        return torch.from_numpy(value.copy())
    if is_dataclass(value):
        return {f.name:probe_copy(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):
        return {k:probe_copy(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(probe_copy(v) for v in value)
    return value



class RawTableClearance:
    """Full original vertex sets and raw SDK quaternion before each rod step."""
    def __init__(self, prepared, body_names, device):
        self.shapes = [(i, body_names[i].index(name), row["vertices"])
            for i, side in enumerate(SIDES)
            for name, row in prepared["hands"][side]["source_colliders"].items()]
        if len(self.shapes) != 34:
            raise ValueError("all 34 source colliders required")
        self.sides = []
        for i in range(2):
            bodies, vertices = zip(*[(j, v) for side, j, v in self.shapes if side == i])
            self.sides.append((torch.as_tensor(np.repeat(bodies, [len(v) for v in vertices]), device=device),
                torch.as_tensor(np.concatenate(vertices), dtype=torch.float64, device=device)))

    def __call__(self, poses):
        clear = []
        for side, (links, vertices) in enumerate(self.sides):
            p = poses[:, side].double()
            w, x, y, z = p[..., 3:].unbind(-1)
            row = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
            heights = (row[:, links]*vertices).sum(-1) + p[:, links, 2]
            clear.append(heights.amin(-1)-.30)
        clear = torch.stack(clear, -1)
        near = ((clear-CLEARANCE_M).abs() <= DECISION_MARGIN_M).any(-1)
        if bool(near.any()):
            exact, _ = collision_clearance(poses[near].detach().cpu().numpy(), self.shapes)
            clear[near] = clear.new_tensor(exact)
        if bool((~torch.isfinite(clear) | (clear < CLEARANCE_M)).any()):
            raise ValueError("original pre-rod raw source table clearance guard failed")
        return clear


def mesh_reset(solver, bridge, state, ids, *, before=None):
    """Cold native Reset ABI, with all three actual phase and velocity readbacks."""
    positions, quaternions = bridge.targets(state["link_pose_xyzw"])
    idx = list(ids)
    p = positions[idx].cpu().numpy()
    q = quaternions[idx].cpu().numpy()
    clock = float(solver.time)
    solver.reset_env_mesh_poses(np.asarray(idx, dtype=np.int32), p, q)
    after = solver.mesh_state_snapshot()
    for phase in ("current", "target", "solver"):
        np.testing.assert_array_equal(after[phase+"_positions_world_m"][idx], p)
        np.testing.assert_allclose(after[phase+"_quaternions_xyzw"][idx], q, atol=2e-6, rtol=0)
    for field in ("solver_linear_velocity_world_m_s", "solver_angular_velocity_world_rad_s",
                  "motion_delta_world_m", "motion_rotation_world_rad", "pending_teleport"):
        if np.any(after[field][idx]):
            raise ValueError("selected mesh Reset retained motion")
    if before is not None:
        peers = sorted(set(range(solver.num_envs))-set(ids))
        for key, value in before.items():
            if isinstance(value, np.ndarray) or key == "pair_counts":
                a, b = np.asarray(value)[peers], np.asarray(after[key])[peers]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError("peer native mesh Reset readback changed: "+key)
    if float(solver.time) != clock or solver.last_error:
        raise ValueError("mesh reset advanced native time or failed")


def require_substep_guard(failures, label):
    if bool(failures.any()):
        raise ValueError(label + " at lanes " + str(failures.nonzero().flatten().tolist()))


class NativeTensorCollector:
    def __init__(self, project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile):
        self.sim, self.batch, self.runtime = sim, batch, runtime
        self.initial, self.state = initial, initial
        self.device = runtime.device
        self.style, self.n = archive.style, runtime.solver.num_envs
        self.ids = tuple(range(self.n))
        self.origins = initial["env_origins"] + initial["env_origins"].new_tensor(seed["origin"])
        self.source = TensorSourceOwner(runtime, sim, profile, self.origins, self.device,
            self.style, 32/60, 180., rod_uids=tuple(f"rod:{i}" for i in self.ids))
        self.joints = tuple(tuple(batch.sides[s].joint_names) for s in SIDES)
        names = tuple(tuple(batch.sides[s].body_names) for s in SIDES)
        for side, body_names in zip(SIDES,names,strict=True):
            owner=batch.sides[side]
            paths=owner.hand.root_physx_view.link_paths
            for prefix, idx in zip(batch.prefixes,owner.row_indices.tolist(),strict=True):
                if tuple(paths[idx])!=tuple(prefix+"/"+side+"Hand/"+name for name in body_names):
                    raise ValueError("actual link identity differs from original source geometry")
        self.reset_proofs=[]
        self.probe_dir=None
        self.bindings=bindings
        self._views=tuple(batch.sides[s].hand.root_physx_view for s in SIDES)
        roots = initial["root_seed_w"].new_tensor(seed["reset_root_pose_w"])[None].repeat(self.n,1,1)
        roots[..., :3] += initial["env_origins"][:, None]
        self.ctrl = TensorControl(batch, roots, bindings, runtime.mesh_names)
        self.effort = float(prepared["candidate_physics"]["hand"]["active_effort_limit_nm"])
        self.raw_clearance = RawTableClearance(prepared, names, self.device)
        self.bridge = TensorMeshBridge(runtime.mesh_names, bindings, initial["mesh_positions_world_m"],
            initial["mesh_quaternions_xyzw"])
        mesh_reset(runtime.solver, self.bridge, initial, self.ids)
        self.radius = float(runtime.solver.parameter_snapshot()["contact_radius_m"])
        self.geometry = None
        if self.style == "pinch":
            from .tabletop_native_teacher_sources import _canonical_pinch_history
            from .tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry
            with _canonical_pinch_history():
                original = PolicyOwnedPinchGeometry(project, scope=self.source.scope)
            if not math.isclose(self.radius, original._expected_radius(), abs_tol=1e-9, rel_tol=2e-6):
                raise ValueError("locked Pinch material radius differs")
            for side in SIDES:
                if len(self.ctrl.slots[SIDES.index(side)]) != original.ROUTE_COUNT:
                    raise ValueError("original Pinch reaction route layout required")
                for name, locked in original.sources[side]["sdf_binding"].items():
                    row = bindings[side+"."+name]
                    if row["sdf_sha256"] != locked["sha256"] or row["capture_sha256"] != locked["capture_sha256"]:
                        raise ValueError("original locked SDF or capture differs")
            self.geometry = TensorGeometry.from_policy_geometry(original, names, device=self.device)
        elif self.style == "power":
            goal = json.loads(checked_file(project/"configs/dh116_tasks/flexible_power_near_open_settle_v1.json",
                archive.task["near_open_source_sha256"]))
            targets = [goal["candidate"][s]["target_pose_wxyz"][:3] for s in SIDES]
            self.targets = roots.new_tensor(targets)[None]+initial["env_origins"][:,None]
            self.stations = (self.targets[:,:,None]-runtime._initial[:,None].double()).norm(dim=-1).argmin(-1)
            self.row = torch.arange(self.n,device=self.device)[:,None]
        else:
            raise ValueError("Tripod requires its complete native support and route guard")
        self.config = load_config(project/f"configs/dh116_training/{self.style}_finger_goal_reward_v1.json",ppo_gamma=.995)
        cfg = json.loads((project/f"configs/dh116_training/privileged_deformable_teacher_{self.style}_v1.json").read_text())
        self.model = TensorNativeTeacher(cfg, backend_profile=profile, num_envs=self.n,reward_config=self.config).to(self.device)
        pp = self.config.ppo
        self.trainer = DeviceDeformablePPOTrainer(self.model,PPOConfig(lr=pp["learning_rate"],gamma=pp["gamma"],
            lam=pp["gae_lambda"],clip=pp["clip"],epochs=pp["epochs"],minibatches=pp["minibatches"],
            value_coef=pp["value"],entropy_coef=pp["entropy"],max_grad_norm=pp["max_grad_norm"],
            target_kl=pp["target_kl"],kl_backtrack_steps=pp["kl_backtrack_steps"]),
            rollout_steps=32,num_envs=self.n,device=self.device)
        self.reward = TensorTrainingReward(self.config, num_envs=self.n, device=self.device)
        self.ages = torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.controls = 0
        self._reset_refs(self.ids)

    def source_guards(self):
        state=self.state
        bad=self.ctrl.joint_failures(state)
        for name in ("backend_targets","root_pose_wxyz","body_com_velocity_w","link_pose_xyzw","body_com_pose_w"):
            bad |= ~torch.isfinite(state[name]).flatten(1).all(1)
        for i,s in enumerate(SIDES):
            owner=self.batch.sides[s]
            hand=owner.hand
            if hand.root_physx_view is not self._views[i] or tuple(hand.joint_names)!=self.joints[i]:
                raise ValueError("actual mounted hand view/joint source changed")
            root=hand.data.root_pose_w.index_select(0,owner.row_indices).double()
            target=hand.data.joint_pos_target.index_select(0,owner.row_indices)
            if root.shape!=(self.n,7) or target.shape!=state["q"][:,i].shape:
                raise ValueError("incomplete SDK root/target cache")
            bad |= ~torch.isfinite(target).all(1) | ~torch.isfinite(root).all(1)
            bad |= (root[:,3:].norm(dim=-1)-1).abs()>1e-5
            bad |= (root-state["root_pose_wxyz"][:,i].double()).abs().amax(-1)>1e-6
        require_substep_guard(bad,"original mounted17 source/joint/cache guards")

    def arm_probe(self,output):
        if self.n!=2:
            raise ValueError("raw parity snapshots are N2 only")
        self.probe_dir=output
        self.snapshot("startup",dict(state=self.state,roots=self.ctrl.roots,source=self.source._boundary(),
            radius=self.radius,nodes=self.nodes,velocity=self.velocity,reset=self.last,
            sources=self.reward.schemas,origins=self.origins,write_ids=self.ctrl.write_ids,
            active_ids=self.ctrl.active,contact_ids=self.ctrl.contact_bodies,mesh_slots=self.ctrl.slots,
            mesh_names=self.runtime.mesh_names,bindings=self.bindings,shapes=self.raw_clearance.shapes,
            joint_names=self.joints,body_names=tuple(self.batch.sides[s].body_names for s in SIDES)))

    def snapshot(self,name,row):
        if self.probe_dir is not None:
            with (self.probe_dir/("tensor_n2_"+name+".pt")).open("xb") as stream:
                torch.save(probe_copy(row),stream)

    def rod_state(self):
        solver = self.runtime.solver
        p, v = (torch.as_tensor(x) for x in (solver.positions(),solver.velocities()))
        if any(x.device != self.device or x.shape != (self.n,41,3) for x in (p,v)):
            raise ValueError("actual same-device full rod position and velocity required")
        require_substep_guard(~(torch.isfinite(p)&torch.isfinite(v)).all((1,2)), "nonfinite rod")
        return p,v

    def distance(self, stamp, *, reset=False):
        p,v = self.rod_state()
        self.nodes,self.velocity=p,v
        if self.geometry is None:
            target = self.targets + p[self.row,self.stations].double() - self.runtime._initial[self.row,self.stations].double()
            return (self.state["base_link_pose_w"][...,:3].double()-target).norm(dim=-1).mean(-1)
        geometry = self.geometry.capture(self.state["link_pose_xyzw"],p,frame=stamp,radius=self.radius)
        accepted = geometry.source_complete if reset else geometry.geometry_passed
        require_substep_guard(~accepted, "original source geometry or SDF coverage")
        return geometry.positive_pad_mean_m.mean(-1)

    def frame(self, stamp, distance):
        q = torch.stack([self.state["q"][:,s,self.ctrl.active[s]] for s in (0,1)],dim=1)
        errors = measured_finger_errors(q,self.config.finger_goal)
        healthy = torch.full_like(distance,self.runtime.solver.last_error==0,dtype=torch.bool)
        guard = self.ctrl.joint_failures(self.state)
        fields = dict(distance=distance,numerical_health=healthy,constraint_failure=guard,
            original_guard_joint_force_clearance_geometry=guard)
        fields.update(zip(ERROR_FIELDS,errors.unbind(-1),strict=True))
        return TensorRewardFrame(torch.full_like(distance,stamp.sdk_step,dtype=torch.long),
            torch.full_like(distance,stamp.solver_time_s,dtype=torch.float64),fields)

    def _reset_refs(self, ids):
        stamp = self.source._boundary()
        if self.radius != float(self.runtime.solver.parameter_snapshot()["contact_radius_m"]):
            raise ValueError("Reset changed the verified contact radius")
        self.source_guards()
        self.clear = self.raw_clearance(self.state["link_pose_wxyz"])
        d = self.distance(stamp,reset=True)
        self.last = self.frame(stamp,d)
        self.source.reset(ids,d[list(ids)])
        selected=list(ids)
        for field,seed_field in (("q","seed_q"),("qd","seed_qd"),("root_pose_wxyz","seed_root_pose_w")):
            if not torch.allclose(self.state[field][selected],self.initial[seed_field][selected],atol=1e-6,rtol=0):
                raise ValueError("selected hand Reset readback differs from seed: "+field)
        if self.state["root_velocity"][selected].count_nonzero() or self.velocity[selected].count_nonzero():
            raise ValueError("selected hand or rod Reset retained velocity")
        ds = ("NativePowerPoseApproach.fixed_material_station" if self.style=="power" else
              "PolicyOwned"+self.style.title()+"Approach.fixed_material41")
        guardsource = "original_joint_force_clearance_and_policy_source_geometry_guards"
        specs = (("distance","m",ds),("numerical_health","dimensionless","RopeBatch.last_error.same_owner"),
            ("constraint_failure","dimensionless",guardsource),("original_guard_joint_force_clearance_geometry","dimensionless",guardsource),
            *((name,"rad","sdk_named_active_q") for name in ERROR_FIELDS))
        schemas = tuple(TensorLaneSchema(stamp.episode_uids[i],i,str(stamp.reset_ids[i]),stamp.rod_uids[i],
            "unqualified_source","policy_owned_"+self.style+"_endpoint_v1",tuple(map(str,range(41))),specs) for i in ids)
        f = self.last
        selected = list(ids)
        reset = replace(f,physics_tick=f.physics_tick[selected],solver_time_s=f.solver_time_s[selected],
            fields={k:v[selected] for k,v in f.fields.items()})
        self.reward.reset_selected(ids,schemas,reset,previous_request=self.source.previous_request[selected],
            tokens=tuple(self.source.previous_request_tokens[i] for i in ids),policy_uids=("controller_reset_zero",)*len(ids))
        self.ages[selected]=0

    def observation(self):
        stamp = self.source._boundary()
        cv,cm,ci = self.source.contexts()
        packet = pack_shared_tensor_inputs(self.state["q"],self.state["qd"],joint_names=self.joints,
            body_link_pose_w=self.state["base_link_pose_w"],body_com_velocity_w=self.state["base_link_com_velocity_w"],
            root_seed_w=self.initial["root_seed_w"],env_origins=self.origins,rod_positions_w=self.nodes,
            rod_velocities_w=self.velocity,rest_lengths_m=self.source.rest_lengths_m,task_context=cv,task_context_valid=cm,
            source_identity=stamp,context_identities=ci,backend_profile_sha256=stamp.profile_sha256,
            grasp=self.style,scope=self.source.scope)
        return packet.for_teacher(self.model,current_identity=self.source._boundary())

    def sdk(self, actions, end):
        from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics
        phase_start=perf_counter()
        def mark(name):
            nonlocal phase_start
            now=perf_counter()
            self.timings[name]=self.timings.get(name,0.)+now-phase_start
            phase_start=now
        pre = self.source._boundary()
        trans = self.ctrl.prepare(actions,self.state,self.clear,self.state["base_link_pose_w"],stamp=pre)
        if self.controls == 0 and pre.sdk_step == self.first_sdk:
            reaction = self.runtime.initial_reaction(physics_step=pre.sdk_step+1)
        else:
            reaction = self.runtime.consume(physics_step=pre.sdk_step+1)
        evidence=None
        if self.probe_dir is not None:
            evidence=probe_copy(dict(pre=pre,state_pre=self.state,rod_pre=self.nodes,velocity_pre=self.velocity,
                pre_frame=self.last,pre_clearance=self.clear,palms=self.ctrl.palms,angles=self.ctrl.angles,
                action=actions,projection=trans["projection"],command=trans["expected"],reaction=reaction))
        self.ctrl.submit(trans,stamp=pre,wrenches=reaction["mesh_wrenches"],state=self.state,reaction_source=reaction)
        if evidence is not None:
            evidence["composer"]={}
            for s in SIDES:
                owner=self.batch.sides[s]
                comp=owner.hand.permanent_wrench_composer
                evidence["composer"][s]=probe_copy({k:getattr(comp,"composed_"+k+"_as_torch").index_select(0,owner.row_indices)
                    for k in ("force","torque")})
        mark("project_reaction_submit")
        sdk = step_thumb_physics(self.sim,pre.sdk_step,budget_end=end)
        mark("physx_step")
        self.batch.update(1/480)
        self.state = capture_regex_state(self.batch,body_com_pose_b_xyzw=self.initial["body_com_pose_b_xyzw"])
        mark("sdk_readback")
        self.clear = self.raw_clearance(self.state["link_pose_wxyz"])
        self.source_guards()
        self.bridge.submit(self.runtime.solver,self.state["link_pose_xyzw"])
        mark("raw_source_guard_mesh_submit")
        capture = self.runtime.advance(physics_step=sdk)
        mark("rod_flat_export_reduce")
        post = self.source._boundary()
        self.ctrl.finish(trans,self.state,stamp=post)
        if self.geometry is not None:
            projection = trans["projection"]
            if projection["projection_source"] != SOURCE:
                raise ValueError("original wrist projection source differs")
            tracking = (self.state["base_link_pose_w"][...,:3].double()-projection["projected_root_poses"][...,:3]).norm(dim=-1)
            speed = self.state["base_link_com_velocity_w"][...,:3].double().norm(dim=-1)
            angular = torch.stack([self.state["q"][:,s,[self.joints[s].index(k) for k in WRIST_ANGULAR]] for s in (0,1)],1)
            wrench = capture["mesh_wrenches"][:,self.ctrl.slots[0]+self.ctrl.slots[1]]
            bad = ((tracking>WRIST["tracking_guard_m"])|(speed>WRIST["linear_speed_guard_m_s"])).any(1)
            bad |= (angular.double().abs()>math.pi/2).any((1,2)) | (wrench[...,:3].norm(dim=-1)>30).any(1)
            bad |= ~torch.isfinite(self.state["body_com_velocity_w"]).all((1,2,3)) | ~torch.isfinite(self.state["backend_targets"]).all((1,2))
            require_substep_guard(bad,"original force or wrist guard")
        self.last = self.frame(post,self.distance(post))
        ready=self.reward.observe(self.last)
        mark("geometry_r0_and_post_guards")
        if evidence is not None:
            evidence.update(post=post,state_post=self.state,nodes_post=self.nodes,velocity_post=self.velocity,
                post_clearance=self.clear,capture=capture,post_frame=self.last,
                geometry=None if self.geometry is None else self.geometry.result)
            self.snapshot("sdk_"+str(sdk),evidence)
            mark("n2_cold_parity_evidence")
        return ready

    def window(self,end,*,probe_reset=False):
        self.timings={}
        phase_start=perf_counter()
        if self.controls==0:
            self.first_sdk = self.runtime.physics_step
        obs = self.observation()
        sample = self.trainer.collect_step(obs,obs)
        actions, token = sample["actions"],sample["collection_token"]
        if probe_reset and torch.equal(actions[0],actions[1]):
            raise ValueError("probe requires independent sampled 24D lane actions")
        self.timings["observation_policy_sample"]=perf_counter()-phase_start
        phase_start=perf_counter()
        tokens = (token,)*self.n
        policy = f"{self.trainer.session_id}:{self.trainer.policy_version}"
        self.reward.begin(tuple(self.reward.schemas),request=actions,tokens=tokens,policy_uids=(policy,)*self.n,
            physics_tick=self.last.physics_tick,solver_time_s=self.last.solver_time_s,remaining_substeps=end-self.runtime.physics_step)
        self.timings["reward_window_begin"]=perf_counter()-phase_start
        for substep in range(8):
            ready = self.sdk(actions,end)
            if substep<7:
                require_substep_guard(ready, "terminal lane before end of shared SDK window")
        phase_start=perf_counter()
        self.source.set_previous_request(actions,token)
        with torch.no_grad():
            value=self.model.value(self.observation())
        self.ages+=1
        timeout = (self.ages>=32) & ~self.reward._terminal
        if probe_reset:
            timeout[1] = ~self.reward._terminal[1]
        full = len(self.trainer.rollout)+1 == self.trainer.rollout_steps
        packet=self.reward.settle_once(tokens,truncated=timeout,buffer_full=~timeout & ~self.reward._terminal & full)
        self.trainer.record_step(packet.reward.float(),packet.done,value,truncated=packet.truncated,
            bootstrap_mask=~packet.terminated,executed_actions=actions,collection_token=token)
        self.snapshot("window_"+str(self.controls+1),dict(schemas=tuple(self.reward.schemas),
            packet=packet,post_value=value,actions=actions))
        self.controls+=1
        self.timings["value_and_reward_settlement"]=perf_counter()-phase_start
        phase_start=perf_counter()
        if bool(packet.done.any()):
            ids=tuple(packet.done.nonzero().flatten().tolist())
            self.reset(ids)
        self.timings["window_reset"]=perf_counter()-phase_start
        return float(packet.reward.mean())

    def peer_state(self, peers):
        """Cold probe only. Keep exact byte patterns including signed zero and NaNs."""
        tensors = {**self.state,"rod_nodes":self.nodes,"rod_v":self.velocity,"palm":self.ctrl.palms,
            "angles":self.ctrl.angles,"age":self.ages,"rest":self.source.rest_lengths_m,
            "request":self.source.previous_request,"reset_z":self.source.reset_z,"reset_distance":self.source.reset_distance_m}
        for name,value in vars(self.reward).items():
            if isinstance(value,torch.Tensor) and value.ndim and len(value)==self.n:
                tensors["reward."+name]=value
        for name in ("counts","node_forces","mesh_wrenches"):
            tensors["reaction."+name]=self.runtime._reaction[name]
        for s in SIDES:
            owner=self.batch.sides[s]
            for k in ("permanent","instantaneous"):
                composer=getattr(owner.hand,k+"_wrench_composer")
                for field in ("force","torque"):
                    value=getattr(composer,"composed_"+field+"_as_torch")
                    tensors[".".join((s,k,field))]=value.index_select(0,owner.row_indices)
        result={k:v[list(peers)].contiguous().view(torch.uint8).clone() for k,v in tensors.items()}
        packet=self.runtime._reaction
        mask=torch.isin(packet["env_ids"],torch.tensor(peers,device=self.device))
        for k in ("env_ids","indices","vectors","normal_force"):
            result["flat_pairs."+k]=packet[k][mask].contiguous().view(torch.uint8).clone()
        for name in ("_terms","_last_values"):
            for key,value in getattr(self.reward,name).items():
                if isinstance(value,torch.Tensor):
                    result["reward."+name+"."+key]=value[list(peers)].contiguous().view(torch.uint8).clone()
        return result

    def reset(self,ids):
        peers = tuple(sorted(set(self.ids)-set(ids)))
        def peer_epochs():
            return tuple((self.runtime.episodes[i],self.runtime.resets[i],self.runtime.ticks[i],
                self.reward.schemas[i],self.source.previous_request_tokens[i]) for i in peers)
        epochs=peer_epochs()
        before=self.peer_state(peers) if peers else {}
        mesh=self.runtime.solver.mesh_state_snapshot() if peers else None
        sdk=self.runtime.physics_step
        self.state=restore_regex_seed(self.sim,self.batch,self.initial,env_ids=list(ids),active_effort_limit_nm=self.effort)
        self.runtime.reset_lanes(ids,episode_uids=tuple(uuid.uuid4().hex for _ in ids))
        mesh_reset(self.runtime.solver,self.bridge,self.state,ids,before=mesh)
        self.ctrl.invalidate(list(ids))
        self._reset_refs(ids)
        if peers:
            after=self.peer_state(peers)
            if epochs!=peer_epochs() or set(before)!=set(after) or any(not torch.equal(before[k],after[k]) for k in before):
                raise ValueError("selected Reset changed peer hand/rod/reaction/controller/source/reward state")
        self.snapshot("reset_"+str(sdk),dict(ids=ids,state=self.state,nodes=self.nodes,velocities=self.velocity,
            source=self.source._boundary(),references=self.last,schemas=tuple(self.reward.schemas)))
        if self.sim.current_time_step_index != sdk:
            raise ValueError("Reset unexpectedly advanced PhysX")
        self.reset_proofs.append(dict(sdk_step=sdk,selected=list(ids),peers=list(peers),
            peer_fields=list(before),native_mesh_verified=True,reset_ids=list(self.runtime.resets)))

