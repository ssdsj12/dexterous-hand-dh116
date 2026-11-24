"""Shared read-once flat-CUDA solver ownership; no per-lane step dictionaries."""
from functools import wraps
import math
import numpy as np
import torch
from .tabletop_deformable_solver_clock import solver_clock_matches
from .tabletop_tensor_pair_capture import flat_cuda_pair_capture

def poison_failure(operation):
    @wraps(operation)
    def run(self,*args,**kwargs):
        try:return operation(self,*args,**kwargs)
        except BaseException:
            self._failed=True;self._reaction=None
            raise
    return run

class TensorRodOwner:
    def __init__(self,solver,*,run_uuid,physics_step,episodes,mesh_names,solver_substeps=12,device="cuda:0"):
        n=solver.num_envs
        if (not 2<=n<=512 or solver.last_error!=0 or len(episodes)!=n
                or len(set(episodes))!=n or len(set(mesh_names))!=len(mesh_names)
                or solver_substeps!=12 or type(physics_step)is not int or physics_step<0):
            raise ValueError("complete healthy source-owned shared solver required")
        self.solver=solver;self.run_uuid=run_uuid;self.physics_step=physics_step
        self.mesh_names=tuple(mesh_names);self.episodes=tuple(episodes);self.substeps=solver_substeps
        self.resets=[1]*n;self._reset_step=[physics_step]*n
        self._time=float(solver.time);self.device=torch.device(device)
        self._initial=torch.as_tensor(solver.positions(),device=device).detach().clone()
        if self._initial.shape!=(n,41,3) or not bool(torch.isfinite(self._initial).all()):
            raise ValueError("actual complete free-rod initial nodes required")
        self._failed=False;self._consumed=True;self._reaction=None
        self._initial_step=physics_step;self._initial_used=False

    @property
    def ticks(self):
        return tuple((self.physics_step-step)*self.substeps for step in self._reset_step)

    def _healthy(self):
        if self._failed or self.solver.last_error!=0 or float(self.solver.time)!=self._time:
            self._failed=True
            raise ValueError("actual shared solver boundary changed or failed")

    def advance(self,*,physics_step):
        try:
            self._healthy()
            if physics_step!=self.physics_step+1 or not self._consumed:
                raise ValueError("one SDK step and consumed old reaction required")
            old=self._time;self.solver.step_free(1/480,self.substeps);new=float(self.solver.time)
            if self.solver.last_error!=0 or not solver_clock_matches(old,new,steps=1):
                raise ValueError("actual shared CUDA solver clock failed")
            packet=flat_cuda_pair_capture(self.solver.contact_pair_samples_flat_cuda_v1(),
                n=self.solver.num_envs,vertices=41,meshes=len(self.mesh_names),substeps=self.substeps,
                device=self.device,expected_owner_time_s=new)
            residual=packet["node_forces"].sum(1)+packet["mesh_wrenches"][...,:3].sum(1)
            if bool((residual.norm(dim=-1)>1e-3).any()):
                raise ValueError("original rod/mesh force balance guard failed")
            self.physics_step=physics_step;self._time=new;self._consumed=False;self._reaction=packet
            return {k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in packet.items()}
        except BaseException:
            self._failed=True;self._reaction=None
            raise

    def consume(self,*,physics_step):
        self._healthy()
        if self._consumed or self._reaction is None or physics_step!=self.physics_step+1:
            self._failed=True;raise ValueError("immediate next-SDK exclusive reaction ownership required")
        self._consumed=True
        result=self._reaction;self._reaction=None
        return self._source_packet(result,physics_step)

    def _source_packet(self,packet,applied_step):
        return dict(packet,source_solver_id=id(self.solver),
            parent_num_envs=self.solver.num_envs,run_uuid=self.run_uuid,
            source_physics_step=self.physics_step,applied_physics_step=applied_step,
            rod_ticks=self.ticks,episode_uids=self.episodes,reset_ids=tuple(self.resets),
            solver_time_s=self._time)

    @poison_failure
    def initial_reaction(self,*,physics_step):
        """One explicitly empty pending reaction before any solver advance."""
        self._healthy()
        if (self._initial_used or self.physics_step!=self._initial_step
                or physics_step!=self.physics_step+1 or not self._consumed
                or any(self.ticks) or any(r!=1 for r in self.resets)):
            raise ValueError("only the original unadvanced empty pending reaction is available")
        self._initial_used=True
        return self._source_packet(dict(bootstrap_empty=True,
            mesh_wrenches=torch.zeros(self.solver.num_envs,len(self.mesh_names),6,
                device=self.device,dtype=torch.float64)),physics_step)

    @poison_failure
    def reset_lanes(self,env_ids,*,episode_uids):
        """Only the task owner may call at a completed window with pending reaction."""
        self._healthy()
        ids=tuple(env_ids)
        if (not ids or len(set(ids))!=len(ids) or len(ids)!=len(episode_uids)
                or any(type(i)is not int or not 0<=i<self.solver.num_envs for i in ids)
                or any(not isinstance(u,str) or not u or u in self.episodes for u in episode_uids)
                or len(set(episode_uids))!=len(ids) or self._consumed or self._reaction is None):
            raise ValueError("distinct selected Reset IDs/epochs with pending reaction required")
        before=torch.as_tensor(self.solver.positions(),device=self.device).clone()
        self.solver.reset_envs(np.asarray(ids,dtype=np.int32))
        after=torch.as_tensor(self.solver.positions(),device=self.device)
        selected=torch.tensor(ids,device=self.device)
        keep=torch.ones(self.solver.num_envs,dtype=torch.bool,device=self.device);keep[selected]=False
        if (self.solver.last_error!=0 or float(self.solver.time)!=self._time
                or not torch.equal(before[keep],after[keep]) or not torch.equal(after[selected],self._initial[selected])):
            self._failed=True;raise ValueError("zero-time selected rod Reset mutated peer or time")
        packet=self._reaction;retained=~torch.isin(packet["env_ids"],selected)
        for k in ("env_ids","indices","vectors","normal_force"):packet[k]=packet[k][retained].clone()
        for k in ("counts","node_forces","mesh_wrenches"):packet[k][selected]=0
        packet["pair_lane_meta"][selected,3]=0
        epochs=list(self.episodes)
        for i,u in zip(ids,episode_uids):
            epochs[i]=u;self.resets[i]+=1;self._reset_step[i]=self.physics_step
        self.episodes=tuple(epochs)
