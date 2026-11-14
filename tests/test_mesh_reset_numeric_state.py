"""Real Python reset with CAI protocol CPU doubles; no device allocation/native import."""
from copy import deepcopy
import importlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_mesh_reset_order import native_semantics,plan as old_plan
from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as old
from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as profile

ROOT=Path(__file__).resolve().parents[1]
def api():
    assert (ROOT/"src/go2_pvcnn/control/dh116/tabletop_mesh_reset_numeric_state.py").exists()
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_mesh_reset_numeric_state")

class CudaView:
    def __init__(self,array):self.array=array
    @property
    def __cuda_array_interface__(self):
        return dict(shape=self.array.shape,strides=None,typestr=self.array.dtype.str,data=(4096,False),version=2)

def test_original_state_hashes_fresh_cuda_wrapper_addresses_not_numeric_values():
    p=np.zeros((1,41,3),np.float32);q=np.zeros((1,41,4),np.float32)
    rope=SimpleNamespace(parameter_snapshot=lambda:dict(time_s=0.),last_error=0,
        positions=lambda:CudaView(p),velocities=lambda:CudaView(p),quaternions=lambda:CudaView(q))
    a,b=old._state(rope),old._state(rope)
    assert old._digest(a["parameters"])==old._digest(b["parameters"])
    for name in ("positions","velocities","quaternions"):
        assert a[name].shape==() and a[name].dtype==object
        assert old._digest(a[name])!=old._digest(b[name])
    assert old._digest(a)!=old._digest(b)

@pytest.fixture
def case(monkeypatch):
    c=native_semantics(monkeypatch)
    nodes=np.zeros((41,3),np.float32);nodes[:,0]=np.linspace(0,.5,41);nodes[:,2]=.2
    c.rod._base.nodes=nodes.copy();c.rod._base.rest=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    c.rope.nodes=nodes[None].copy();c.rope.force=np.zeros_like(c.rope.nodes)
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import lumped_masses,verify_mass_readback
    c.rod._base.mass_readback=verify_mass_readback(c.rope.inv_masses(),lumped_masses(nodes,c.rope.density))
    c.plan=old_plan(c,monkeypatch)
    c.sim.device="cuda:0";c.rope.num_envs=1;c.rope.verts_per_rope=41;c.rope.fixed_root_orientation=False
    cfg=c.rod._base.cfg["rod"];rest=np.linalg.norm(np.diff(c.rod._base.nodes,axis=0),axis=1)[None].astype(np.float32)
    c.raw=dict(schema="dh116.native_parameter_snapshot.v1",num_envs=1,nodes_per_env=41,time_s=0.,
        metadata_source="native_host_metadata",segment_source="cuda_device_copy",hardware_measured=False,
        contact_radius_m=cfg["radius_m"],gravity_world_m_s2=np.array(cfg["gravity_world_m_s2"],np.float32),
        contact_mu_static=cfg["mu_s"],contact_mu_kinetic=cfg["mu_k"],contact_v_slip_m_s=cfg["v_slip_m_s"],contact_kappa=cfg["kappa"],
        rest_lengths_m=rest,stored_k_stretch_n_m=rest*cfg["axial_stiffness_n"],
        rest_quaternion_norm_n_m=4*cfg["bend_stiffness_n_m2"]/rest,
        stretch_ea_n=np.full((1,40),cfg["axial_stiffness_n"],np.float32),
        bend_ei_n_m2=np.full((1,40),cfg["bend_stiffness_n_m2"],np.float32),
        inverse_masses_kg_inv=np.array(c.rope.inv_masses(),np.float32).reshape(1,41))
    c.rope.parameter_snapshot=lambda:deepcopy(dict(c.raw,time_s=c.rope.time))
    c.p=c.rope.positions();c.v=c.rope.velocities();c.q=c.rope.quaternions()
    c.rope.positions=lambda:CudaView(c.p);c.rope.velocities=lambda:CudaView(c.v);c.rope.quaternions=lambda:CudaView(c.q)
    c.stream={"value":0};c.copies=[]
    monkeypatch.setattr(torch.cuda,"current_stream",lambda d:c.stream["value"])
    monkeypatch.setattr(torch.cuda,"default_stream",lambda d:0)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda d:c.copies.append("sync"))
    original=torch.as_tensor
    class Tensor:
        def __init__(self,a):self.a=a;self.device=torch.device("cuda:0");self.dtype=torch.from_numpy(a).dtype
        @classmethod
        def __torch_function__(cls,func,types,args=(),kwargs=None):
            return func(*(torch.from_numpy(v.a) if isinstance(v,cls) else v for v in args),**(kwargs or {}))
        def detach(self):return self
        def clone(self):c.copies.append("clone");return Tensor(self.a.copy())
        def cpu(self):return torch.from_numpy(self.a.copy())
    monkeypatch.setattr(torch,"as_tensor",lambda v,*a,**k:Tensor(v.array) if isinstance(v,CudaView) else original(v,*a,**k))
    backend=SimpleNamespace(__file__=str(profile.BACKEND),RopeBatch=type(c.rope))
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",backend)
    monkeypatch.setitem(sys.modules,"ropesim",SimpleNamespace(_ropesim=backend))
    return c

def execute(c,record):
    m=api()
    with profile.source_scope(),m.numeric_reset_state_scope(sim=c.sim,rod=c.rod,record=record):
        return old.RebuiltMeshReset(c.seed,episode_uid="numeric-reset",mesh_plan=c.plan).run(
            c.sim,c.rod,c.bindings,owner_resets=c.owners)

def test_true_numeric_copy_in_original_transaction_and_disk_validation(case):
    c=case;m=api();saved=[];before=old._state
    receipt=execute(c,lambda x:saved.append(x))
    assert old._state is before
    audit=saved[0];assert audit["operation_completed"] and audit["byte_equal"]
    assert len(audit["samples"])==2 and c.copies.count("sync")==2 and c.copies.count("clone")==8 # also original reset p/v
    for sample in audit["samples"]:
        for name,size in (("positions",3),("velocities",3),("quaternions",4)):
            assert sample["state"][name].shape==(1,41,size) and sample["state"][name].dtype==np.float32
    assert c.rope.steps==0 and c.sim.current_time_step_index==2
    receipt["mesh_state_readback"]=dict(before={"source_instance_id":id(c.rope)},after={"source_instance_id":id(c.rope)})
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    disk=json.loads(json.dumps(encode_owned(audit),allow_nan=False))
    assert m.validate_numeric_reset_audit(disk,reset_receipt=receipt)["numeric_state_unchanged"]
    c.p[:]=999
    assert audit["samples"][0]["state"]["positions"].max()<1
    assert audit["native_qualification"] is audit["teacher_admission"] is False

@pytest.mark.parametrize("fault",["positions","velocities","quaternions","radius","signed_zero"])
def test_original_invariance_guard_still_rejects_real_changes_and_saves_diff(case,fault):
    c=case;m=api();saved=[];original=c.rope.add_mesh;oldgetter=old._state
    def changed(*a,**k):
        slot=original(*a,**k)
        if slot==10:
            if fault=="radius":c.raw["contact_radius_m"]*=2
            elif fault=="signed_zero":c.v[0,0,0]=-0.
            else:{"positions":c.p,"velocities":c.v,"quaternions":c.q}[fault][0,0,0]+=.01
        return slot
    c.rope.add_mesh=changed
    with pytest.raises(ValueError,match="mesh-only reset changed"):execute(c,lambda x:saved.append(x))
    assert old._state is oldgetter and c.rod._needs_reset
    a=saved[0];assert not a["byte_equal"] and not a["operation_completed"]
    key="parameters.contact_radius_m" if fault=="radius" else "velocities" if fault=="signed_zero" else fault
    assert not a["field_differences"][key]["byte_equal"]
    if fault=="signed_zero":
        assert a["field_differences"][key]["numeric_equal"] is True
        assert a["field_differences"][key]["signed_zero_only"] is True

@pytest.mark.parametrize("fault",["shape","dtype","nan","object","stream","clock","solver_error"])
def test_bad_capture_rejected_without_faking_numeric_state(case,fault):
    c=case;m=api();saved=[]
    if fault=="shape":c.v=c.v[0]
    elif fault=="dtype":c.v=c.v.astype(np.float64)
    elif fault=="nan":c.v[0,0,0]=np.nan
    elif fault=="object":c.rope.velocities=lambda:object()
    elif fault=="stream":c.stream["value"]=1
    elif fault=="solver_error":c.rope.last_error=1
    elif fault=="clock":
        original=c.rope.velocities
        def drift():c.sim.current_time+=.001;return original()
        c.rope.velocities=drift
    with pytest.raises((ValueError,TypeError,RuntimeError)):execute(c,lambda x:saved.append(x))
    assert not saved[0]["operation_completed"]
    assert saved[0]["capture_errors"] or (not saved[0]["samples"] and saved[0]["operation_error"])
    assert old._state is m.ORIGINAL_STATE
    assert c.rope.steps==0

def test_scope_rejects_nesting_and_restores_after_external_exception(case):
    c=case;m=api();saved=[];getter=old._state
    with profile.source_scope():
        with pytest.raises(RuntimeError,match="caller failed"):
            with m.numeric_reset_state_scope(sim=c.sim,rod=c.rod,record=saved.append):
                with pytest.raises(ValueError,match="state reader binding"):
                    with m.numeric_reset_state_scope(sim=c.sim,rod=c.rod):pass
                raise RuntimeError("caller failed")
    assert old._state is getter and saved[0]["operation_completed"] is False

@pytest.mark.parametrize("damage",["state","clock","owner","diff","flag"])
def test_disk_tamper_cannot_become_unchanged(case,damage):
    c=case;m=api();saved=[];receipt=execute(c,saved.append);audit=deepcopy(saved[0])
    receipt["mesh_state_readback"]=dict(before={"source_instance_id":id(c.rope)},after={"source_instance_id":id(c.rope)})
    if damage=="state":audit["samples"][1]["state"]["velocities"][0,0,0]=1
    elif damage=="clock":audit["samples"][1]["clock_after"]["sdk_timestamp_s"]+=1e-4
    elif damage=="owner":audit["samples"][1]["source_instance_id"]+=1
    elif damage=="diff":audit["field_differences"]["positions"]["max_abs_difference"]=2.
    elif damage=="flag":audit["native_qualification"]=True
    with pytest.raises((ValueError,KeyError)):m.validate_numeric_reset_audit(audit,reset_receipt=receipt)


def test_record_failure_restores_binding_without_extra_owner_mutation(case):
    c=case;m=api();getter=old._state;captured=[]
    def failing(value):
        captured.append(value)
        raise OSError("CPU durable writer failure")
    with pytest.raises(OSError,match="durable writer"):execute(c,failing)
    assert old._state is getter
    assert captured[0]["operation_completed"] is True
    assert c.rope.steps==0 and c.sim.current_time_step_index==2
    assert c.rod._needs_reset is False # completed reset; failed recorder does not mutate physics

def test_failed_reader_frame_drops_borrowed_owners(case):
    c=case;m=api();c.stream["value"]=1
    with pytest.raises(ValueError) as caught:execute(c,lambda value:None)
    frame=caught.value.__traceback__
    while frame:
        if frame.tb_frame.f_code.co_name in ("numeric_reset_state_scope","_capture","_copy_velocity"):
            for name in ("sim","rod","rope","record","getter","view","tensor"):
                if name in frame.tb_frame.f_locals:assert frame.tb_frame.f_locals[name] is None
        frame=frame.tb_next

def test_larger_parameter_arrays_and_nonfloat_fields_rejected_before_hashing():
    m=api()
    with pytest.raises(ValueError):m.state_differences({"bad":np.array([object()],object)},{"bad":np.array([object()],object)})
    with pytest.raises(ValueError):m.state_differences({"bad":np.ones(1025,np.float32)},{"bad":np.ones(1025,np.float32)})
