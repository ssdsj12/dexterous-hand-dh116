import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_rod_owner import TensorRodOwner

class Solver:
    num_envs=2
    def __init__(self):
        self.time=0.;self.last_error=0;self.initial=torch.arange(246).reshape(2,41,3).float()/100
        self.p=self.initial.clone();self.ready=False;self.used=False
    def positions(self):return self.p
    def step_free(self,dt,substeps):
        self.time=float(np.float32(self.time)+np.float32(dt));self.p+=.001;self.ready=True;self.used=False
    def reset_envs(self,ids):
        self.p[ids]=self.initial[ids];self.ready=False
    def contact_pair_samples_flat_cuda_v1(self):
        if not self.ready or self.used:raise ValueError("not fresh")
        self.used=True
        return dict(schema="dh116.sdf_pair_flat_cuda.v1",num_envs=2,mesh_count=1,substeps=12,
            owner_time_s_before=self.time,owner_time_s_after=self.time,
            indices_gpu=torch.empty(0,4,dtype=torch.int32),vectors_gpu=torch.empty(0,12),
            lane_meta_gpu=torch.tensor([[0,0,1,0],[0,0,1,0]],dtype=torch.int32),
            node_forces_gpu=torch.zeros(2,41,3),mesh_wrenches_gpu=torch.zeros(2,1,6))

def make():
    solver=Solver()
    return solver,TensorRodOwner(solver,run_uuid="fixture",physics_step=2,
        episodes=("a","b"),mesh_names=("table",),device="cpu")

def test_read_once_complete_window_and_zero_time_selected_reset():
    solver,owner=make()
    owner.advance(physics_step=3);owner.consume(physics_step=4)
    owner.advance(physics_step=4);peer=solver.p[1].clone();time=solver.time
    owner.reset_lanes((0,),episode_uids=("new",))
    assert torch.equal(peer,solver.p[1]) and solver.time==time
    assert owner.ticks==(0,24) and owner.resets==[2,1]
    owner.consume(physics_step=5);owner.advance(physics_step=5);owner.consume(physics_step=6)
    assert owner.ticks==(12,36)

@pytest.mark.parametrize("fault",["skip","double_read","clock","partial_reset"])
def test_failed_owned_exchange_is_poisoned(fault):
    solver,owner=make()
    if fault=="skip":
        with pytest.raises(ValueError):owner.advance(physics_step=4)
    else:
        owner.advance(physics_step=3)
        if fault=="double_read":
            owner.consume(physics_step=4)
            with pytest.raises(ValueError):owner.consume(physics_step=4)
        elif fault=="clock":
            solver.time+=.1
            with pytest.raises(ValueError):owner.consume(physics_step=4)
        else:
            def fail(ids):
                solver.p[0]=0
                raise RuntimeError("partial Reset fixture")
            solver.reset_envs=fail
            with pytest.raises(RuntimeError):owner.reset_lanes((0,),episode_uids=("new",))
    assert owner._failed
    with pytest.raises(ValueError):owner.consume(physics_step=4)

def test_consumed_reaction_carries_current_solver_clock_and_reset_identity():
    solver,owner=make()
    owner.advance(physics_step=3)
    owner.reset_lanes((0,),episode_uids=("new",))
    packet=owner.consume(physics_step=4)
    assert packet["source_solver_id"]==id(solver)
    assert packet["run_uuid"]=="fixture"
    assert packet["source_physics_step"]==3 and packet["applied_physics_step"]==4
    assert packet["rod_ticks"]==(0,12)
    assert packet["episode_uids"]==("new","b") and packet["reset_ids"]==(2,1)
    assert packet["solver_time_s"]==solver.time

def test_initial_reaction_is_explicitly_empty_and_not_reusable():
    solver,owner=make()
    packet=owner.initial_reaction(physics_step=3)
    assert packet["bootstrap_empty"] and not packet["mesh_wrenches"].any()
    assert packet["source_physics_step"]==2 and packet["applied_physics_step"]==3
    with pytest.raises(ValueError):owner.initial_reaction(physics_step=3)
    assert owner._failed
