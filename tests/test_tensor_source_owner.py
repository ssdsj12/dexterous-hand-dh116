"""CPU oracle for source-owned tensor references; never native admission proof."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedSourceIdentity
from go2_pvcnn.control.dh116.tabletop_shared_teacher_live_source import SharedSmokeReferences
from go2_pvcnn.control.dh116.tabletop_tensor_rod_owner import TensorRodOwner

ROOT=Path(__file__).resolve().parents[1]
MODULE="go2_pvcnn.control.dh116.tabletop_tensor_source_owner"


def api():
    assert importlib.util.find_spec(MODULE) is not None,"tensor source/context owner is missing"
    from go2_pvcnn.control.dh116 import tabletop_tensor_source_owner as result
    return result


class Profile:
    def __init__(self):
        self.digest=digest(json.loads((ROOT/"docs/contracts/dh116_flat_backend_candidate_v1.json").read_text()))
        self.calls=0
    def verify(self):
        self.calls+=1
        return self


class Solver:
    def __init__(self,n):
        self.num_envs=n;self.time=4.;self.last_error=0;self.calls=0;self.ready=False;self.used=False
        cfg=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())["rod"]
        rest=np.full((n,40),.02,dtype=np.float64)
        mass=np.concatenate((rest[:,:1]/2,(rest[:,:-1]+rest[:,1:])/2,rest[:,-1:]/2),1)*cfg["linear_density_kg_m"]
        self.raw=dict(schema="dh116.native_parameter_snapshot.v1",metadata_source="native_host_metadata",
            segment_source="cuda_device_copy",hardware_measured=False,num_envs=n,nodes_per_env=41,time_s=self.time,
            contact_radius_m=cfg["radius_m"],gravity_world_m_s2=cfg["gravity_world_m_s2"],
            contact_mu_static=cfg["mu_s"],contact_mu_kinetic=cfg["mu_k"],contact_v_slip_m_s=cfg["v_slip_m_s"],
            contact_kappa=cfg["kappa"],rest_lengths_m=rest,stored_k_stretch_n_m=rest*cfg["axial_stiffness_n"],
            rest_quaternion_norm_n_m=np.full_like(rest,4*cfg["bend_stiffness_n_m2"])/rest,
            stretch_ea_n=np.full_like(rest,cfg["axial_stiffness_n"]),bend_ei_n_m2=np.full_like(rest,cfg["bend_stiffness_n_m2"]),
            inverse_masses_kg_inv=1/mass)
        self.p=torch.zeros(n,41,3)
        self.p[:,:,1]=torch.arange(41)[None,:]*.02
        self.p[:,:,2]=torch.arange(41)[None,:]*.0003+.06+torch.arange(n)[:,None]*.001
        self.initial=self.p.clone()
    def positions(self):return self.p
    def parameter_snapshot(self):
        self.calls+=1
        result=deepcopy(self.raw);result["time_s"]=self.time
        return result
    def step_free(self,dt,substeps):
        self.time=float(np.float32(self.time)+np.float32(dt));self.p+=.0001;self.ready=True;self.used=False
    def reset_envs(self,ids):self.p[ids]=self.initial[ids]
    def contact_pair_samples_flat_cuda_v1(self):
        if not self.ready or self.used:raise ValueError("not fresh")
        self.used=True;n=self.num_envs
        return dict(schema="dh116.sdf_pair_flat_cuda.v1",num_envs=n,mesh_count=1,substeps=12,
            owner_time_s_before=self.time,owner_time_s_after=self.time,
            indices_gpu=torch.empty(0,4,dtype=torch.int32),vectors_gpu=torch.empty(0,12),
            lane_meta_gpu=torch.tensor([[0,0,1,0]]*n,dtype=torch.int32),
            node_forces_gpu=torch.zeros(n,41,3),mesh_wrenches_gpu=torch.zeros(n,1,6))


class Sim:
    def __init__(self):self.set(2)
    def get_physics_dt(self):return 1/480
    def set(self,step):
        self.current_time_step_index=step;self.current_time=step*float(np.float32(1/480))


def make(n=2,grasp="pinch",**kwargs):
    m=api();solver=Solver(n);sim=Sim();profile=Profile()
    runtime=TensorRodOwner(solver,run_uuid="fixture",physics_step=2,episodes=tuple(f"ep:{i}" for i in range(n)),
        mesh_names=("table",),device="cpu")
    origins=torch.tensor([[3.*i,-6.*i,0.] for i in range(n)],dtype=torch.float64)
    owner=m.TensorSourceOwner(runtime,sim,profile,origins,"cpu",grasp,episode_cap_s=2.,wall_cap_s=100.,**kwargs)
    return owner,runtime,solver,sim,profile


def advance(runtime,sim):
    if not runtime._consumed:runtime.consume(physics_step=runtime.physics_step+1)
    runtime.advance(physics_step=runtime.physics_step+1);sim.set(runtime.physics_step)


@pytest.mark.parametrize("n,grasp",[(2,"pinch"),(2,"tripod"),(128,"power")])
def test_complete_context_batch_exactly_matches_frozen_smoke_oracle(monkeypatch,n,grasp):
    m=api()
    import go2_pvcnn.control.dh116.tabletop_shared_teacher_live_source as old
    clock=[20.];monkeypatch.setattr(m,"perf_counter",lambda:clock[0]);monkeypatch.setattr(old,"perf_counter",lambda:clock[0])
    owner,runtime,solver,sim,profile=make(n,grasp)
    distances=torch.linspace(0.,.03,n,dtype=torch.float64)
    owner.reset(tuple(range(n)),distances)
    oracle=SharedSmokeReferences(owner,grasp=grasp,episode_cap_s=2.,wall_cap_s=100.)
    for i in range(n):oracle.reset_lane(i,distance_mean_m=float(distances[i]))
    actions=torch.linspace(-1.,1.,n*24,dtype=torch.float64).reshape(n,24)
    owner.set_previous_request(actions,"actual-request");oracle.set_previous_request(actions,"actual-request")
    for _ in range(5):advance(runtime,sim)
    clock[0]=30.;calls=solver.calls;verifies=profile.calls
    values,valid,identities=owner.contexts()
    expected=oracle.contexts()
    torch.testing.assert_close(values,torch.cat([p.values for p in expected]),rtol=0,atol=0,equal_nan=True)
    assert torch.equal(valid,torch.cat([p.valid for p in expected]))
    assert identities==tuple(p.metadata.identities[0] for p in expected)
    assert type(owner._boundary()) is SharedSourceIdentity
    assert solver.calls==calls and profile.calls==verifies
    assert values.dtype==torch.float32 and values.shape==(n,50)
    assert torch.isnan(values[~valid]).all() and valid.sum()==n*42


def test_selected_reset_replaces_only_selected_gpu_reference_row(monkeypatch):
    owner,runtime,solver,sim,profile=make()
    owner.reset((0,1),torch.tensor([.004,.02],dtype=torch.float64))
    owner.set_previous_request(torch.full((2,24),.4),"before-reset")
    advance(runtime,sim)
    before={name:getattr(owner,name).clone() for name in ("reset_z","reset_distance_m","rest_lengths_m","previous_request","reference_sdk")}
    runtime.reset_lanes((1,),episode_uids=("new:1",))
    owner.reset((1,),torch.tensor([.009],dtype=torch.float64))
    for name,saved in before.items():
        assert torch.equal(getattr(owner,name)[0],saved[0])
    assert owner.reset_distance_m.tolist()==[.004,.009]
    assert owner.previous_request[1].eq(0).all()
    values,valid,identities=owner.contexts()
    assert values[1,43]==0. and values[0,43]>0.
    assert identities[1].episode_uid=="new:1" and identities[0].episode_uid=="ep:0"
    with pytest.raises(ValueError,match="new|Reset|reset"):
        owner.reset((1,),torch.tensor([.01]))


def test_contexts_require_every_fresh_reset_epoch_and_do_not_refresh_snapshot():
    owner,runtime,solver,sim,profile=make()
    with pytest.raises(ValueError,match="Reset|reset|reference"):
        owner.contexts()
    owner.reset((0,1),torch.tensor([.01,.02]))
    calls=solver.calls;solver.parameter_snapshot=lambda:(_ for _ in ()).throw(AssertionError("substep snapshot forbidden"))
    owner.contexts();advance(runtime,sim);owner.contexts()
    runtime.reset_lanes((0,),episode_uids=("changed",))
    with pytest.raises(ValueError,match="epoch|Reset|reset"):
        owner.contexts()
    assert calls==2


@pytest.mark.parametrize("damage",["radius","mass","ea","ei","gravity","fixed"])
def test_original_actual_material_validation_is_kept(damage):
    m=api();solver=Solver(2);profile=Profile();sim=Sim()
    if damage=="radius":solver.raw["contact_radius_m"]*=2
    elif damage=="mass":solver.raw["inverse_masses_kg_inv"][0,1]*=2
    elif damage=="ea":solver.raw["stretch_ea_n"][0,1]*=2
    elif damage=="ei":solver.raw["bend_ei_n_m2"][0,1]*=2
    elif damage=="gravity":solver.raw["gravity_world_m_s2"]=[0.,0.,0.]
    else:solver.raw["inverse_masses_kg_inv"][0,1]=0
    runtime=TensorRodOwner(solver,run_uuid="fixture",physics_step=2,episodes=("a","b"),mesh_names=("table",),device="cpu")
    with pytest.raises(ValueError):m.TensorSourceOwner(runtime,sim,profile,torch.zeros(2,3),"cpu","pinch",2.,100.)


@pytest.mark.parametrize("damage",["sdk_step","sdk_time","solver_time","error","dt"])
def test_original_actual_sdk_solver_precision_boundary_is_kept(damage):
    owner,runtime,solver,sim,profile=make();owner.reset((0,1),torch.tensor([.01,.02]))
    if damage=="sdk_step":sim.current_time_step_index+=1
    elif damage=="sdk_time":sim.current_time+=1e-7
    elif damage=="solver_time":solver.time+=.001
    elif damage=="error":solver.last_error=1
    else:sim.get_physics_dt=lambda:1/60
    with pytest.raises(ValueError):owner.contexts()


def test_no_fake_native_scope_or_profile_digest_is_accepted():
    with pytest.raises(ValueError,match="native|loaded"):
        make(scope="owner_source_unqualified")
    m=api();owner,runtime,solver,sim,profile=make();profile.digest="0"*64
    with pytest.raises(ValueError,match="profile|candidate"):
        m.TensorSourceOwner(runtime,sim,profile,torch.zeros(2,3),"cpu","pinch",2.,100.)
    assert owner.native_qualified is False and owner.teacher_admission is False


def test_previous_request_and_returned_context_ownership_and_guards():
    owner,runtime,solver,sim,profile=make();owner.reset((0,1),torch.tensor([.01,.02]))
    values,valid,identities=owner.contexts();values.zero_();valid.zero_()
    again=owner.contexts();assert again[0][0,0]==1. and again[1].any()
    actions=torch.full((2,24),.3);owner.set_previous_request(actions,"recorded");actions.zero_()
    assert owner.previous_request.eq(torch.tensor(.3)).all()
    with pytest.raises(ValueError):owner.set_previous_request(torch.full((2,24),1.01),"bad")
    with pytest.raises(ValueError):owner.set_previous_request(torch.full((2,24),float("nan")),"bad")
    with pytest.raises(ValueError):owner.reset((0,0),torch.tensor([.01,.02]))
    origin=owner.origins;origin.zero_()
    assert owner.origins[1].tolist()==[3.,-6.,0.]


def test_selected_reset_refuses_changed_peer_rest_in_actual_snapshot():
    owner,runtime,solver,sim,profile=make();owner.reset((0,1),torch.tensor([.01,.02]))
    advance(runtime,sim);runtime.reset_lanes((1,),episode_uids=("new:1",))
    raw=solver.raw;raw["rest_lengths_m"][0]*=2;raw["stored_k_stretch_n_m"][0]*=2
    raw["rest_quaternion_norm_n_m"][0]/=2;raw["inverse_masses_kg_inv"][0]/=2
    with pytest.raises(ValueError,match="peer|material|rest"):
        owner.reset((1,),torch.tensor([.02]))


@pytest.mark.parametrize("field,value",[("teacher_admission",True),("native_mechanical_receipt",{}),("physics_kernel_changes",True)])
def test_candidate_manifest_cannot_be_promoted_to_qualified(monkeypatch,tmp_path,field,value):
    m=api();manifest=json.loads(m.CANDIDATE_MANIFEST.read_text());manifest[field]=value
    path=tmp_path/"candidate.json";path.write_text(json.dumps(manifest));monkeypatch.setattr(m,"CANDIDATE_MANIFEST",path)
    profile=Profile();profile.digest=digest(manifest);solver=Solver(2);sim=Sim()
    runtime=TensorRodOwner(solver,run_uuid="fixture",physics_step=2,episodes=("a","b"),mesh_names=("table",),device="cpu")
    with pytest.raises(ValueError,match="unqualified|candidate"):
        m.TensorSourceOwner(runtime,sim,profile,torch.zeros(2,3),"cpu","pinch",2.,100.)


def test_new_source_contexts_feed_actual_tensor_packet_and_freshness_gate():
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
    owner,runtime,solver,sim,profile=make();owner.reset((0,1),torch.tensor([.01,.02]))
    values,valid,identities=owner.contexts();pose=torch.zeros(2,2,7);pose[:,:,3]=1.
    names=JOINTS+WRIST_LINEAR+WRIST_ANGULAR
    packet=pack_shared_tensor_inputs(torch.zeros(2,2,17),torch.zeros(2,2,17),joint_names=(names,names),
        body_link_pose_w=pose,body_com_velocity_w=torch.zeros(2,2,6),root_seed_w=pose.clone(),
        env_origins=owner.origins,rod_positions_w=solver.positions(),rod_velocities_w=torch.zeros_like(solver.positions()),
        rest_lengths_m=owner.rest_lengths_m,task_context=values,task_context_valid=valid,
        source_identity=owner._boundary(),context_identities=identities,backend_profile_sha256=profile.digest,
        grasp="pinch",scope=owner.scope)
    assert packet.native_qualified is False and owner.validate_fresh(packet) is packet
    assert packet.inputs["task_context_valid"].equal(valid)
    assert torch.isnan(packet.inputs["common371"][~packet.inputs["common_valid"]]).sum()==610
    advance(runtime,sim)
    with pytest.raises(ValueError,match="epoch|clock|source"):
        owner.validate_fresh(packet)


def test_main_candidate_profile_binds_without_impersonating_old_profile_class():
    from go2_pvcnn.control.dh116.tabletop_tensor_backend_profile import TensorBackendProfile
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedTeacherSource,SharedBackendProfile
    m=api();profile=TensorBackendProfile.load();solver=Solver(2);sim=Sim()
    assert type(profile) is not SharedBackendProfile
    runtime=TensorRodOwner(solver,run_uuid="fixture",physics_step=2,episodes=("a","b"),mesh_names=("table",),device="cpu")
    owner=m.TensorSourceOwner(runtime,sim,profile,torch.zeros(2,3),"cpu","pinch",2.,100.)
    owner.reset((0,1),torch.tensor([.01,.02]));values,valid,identities=owner.contexts()
    assert values.shape==(2,50) and valid.sum()==84 and owner._boundary().profile_sha256==profile.digest
    with pytest.raises(ValueError,match="actual shared owner|profile"):
        SharedTeacherSource(runtime,sim,backend_profile=profile,env_origins=np.zeros((2,3)),scope="cpu_fixture",device="cpu")
