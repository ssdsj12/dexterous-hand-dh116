"""CPU protocol doubles exercise the real opt-in reader; no CUDA backend import."""
from copy import deepcopy
import gc
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
import weakref
import numpy as np
import pytest
import torch

MODULE="go2_pvcnn.control.dh116.tabletop_material_director_readback"


def mod():
    return importlib.import_module(MODULE)


class CudaView:
    def __init__(self, array): self.array=array


class RopeBatch:
    def __init__(self,nodes,events):
        self.p=nodes[None].copy()
        self.q=np.zeros((1,41,4),np.float32);self.q[...,3]=1
        self.q[:,-1]=[2,3,4,5]  # finite unused padding need not be a unit frame
        self.events=events;self.time=0.;self.last_error=0
        self.num_envs=1;self.verts_per_rope=41
    def positions(self):
        self.events.append("positions");return CudaView(self.p)
    def quaternions(self):
        self.events.append("quaternions");return CudaView(self.q)


@pytest.fixture
def case(monkeypatch):
    m=mod();events=[]
    nodes=np.zeros((41,3),np.float32);nodes[:,0]=np.linspace(0,.5,41)
    rod=RopeBatch(nodes,events)
    backend=SimpleNamespace(__file__=str(m.BACKEND),RopeBatch=RopeBatch)
    clock=dict(run_uuid="run",env_id=0,episode_uid="ep1",reset_id=1,rod_uid="rod0",
        physics_step=2,rod_tick=0,timestamp_s=2/480,solver_time_s=0.,clock_valid=True)
    sim=SimpleNamespace(current_time_step_index=2,current_time=2/480)
    stream={"current":0}
    monkeypatch.setattr(torch.cuda,"current_stream",lambda device:stream["current"])
    monkeypatch.setattr(torch.cuda,"default_stream",lambda device:0)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda device:events.append("synchronize"))
    actual_as_tensor=torch.as_tensor
    class Tensor:
        def __init__(self,array):
            self.array=array;self.device=torch.device("cuda:0");self.dtype=torch.float32
        def detach(self):return self
        def clone(self):events.append("clone");return Tensor(self.array.copy())
        def cpu(self):events.append("cpu");return torch.from_numpy(self.array.copy())
    monkeypatch.setattr(torch,"as_tensor",lambda value,*a,**k:
        Tensor(value.array) if isinstance(value,CudaView) else actual_as_tensor(value,*a,**k))
    owner=m.MaterialDirectorReadback(nodes,device="cuda:0")
    def row():
        return dict(clock,node_positions=rod.p.copy(),rest_lengths_m=np.linalg.norm(np.diff(nodes,axis=0),axis=1),
            contact_valid=clock["rod_tick"]>0)
    def advance():
        clock.update(physics_step=clock["physics_step"]+1,rod_tick=clock["rod_tick"]+12,
            timestamp_s=(clock["physics_step"]+1)/480,solver_time_s=float(np.float32(rod.time+1/480)))
        sim.current_time_step_index=clock["physics_step"];sim.current_time=clock["timestamp_s"]
        rod.time=clock["solver_time_s"]
    args=dict(rope=rod,backend=backend,sim=sim,read_clock=lambda:dict(clock))
    return SimpleNamespace(m=m,owner=owner,rod=rod,backend=backend,sim=sim,clock=clock,row=row,
        advance=advance,args=args,nodes=nodes,events=events,stream=stream)


def test_locked_actual_binary_and_api_source_basis():
    m=mod();proof=m.source_basis()
    assert proof["backend_sha256"]=="92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0"
    assert proof["quaternion_order"]=="xyzw"
    assert proof["native_rest_lengths_getter_available"] is False
    assert proof["native_qualification"] is False
    root=m.BUILD/"source"
    bindings=(root/"csrc/bindings.cpp").read_text().split('py::class_<RopeBatch>(m, "RopeBatch")')[1]
    assert '.def("quaternions"' in bindings and '.def("rest_lengths"' not in bindings
    assert b"quaternions" in m.BACKEND.read_bytes()
    batch=(root/"csrc/batch.cu").read_text()
    body=batch.split("void RopeBatch::step_free(")[1].split("void RopeBatch::reset_envs")[0]
    assert body.index("SimAccess::finish")<body.index("sdf.end_step")<body.index("gather_nolock")
    assert "d_outPos, d_outVel, d_outQuat" in batch


def test_reset_then_completed_endpoint_uses_real_getters_without_steps(case):
    c=case
    reset=c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    assert c.events==["synchronize","positions","quaternions","clone","cpu","clone","cpu"]
    assert reset["identity"]==c.clock
    assert reset["segment_quaternions_xyzw"].shape==(1,40,4)
    np.testing.assert_array_equal(reset["padding_quaternion_xyzw"],[[2,3,4,5]])
    assert reset["native_rest_lengths_readback_m"] is None
    c.advance()
    theta=.2
    c.rod.q[0,20]=[np.sin(theta/2),0,0,np.cos(theta/2)]
    packet=c.owner.capture(c.row(),**c.args)
    assert packet["identity"]==c.clock
    assert packet["boundary"]=="completed_physics_endpoint"
    assert packet["solver_substeps_since_reset"]==12
    assert packet["material_reference_uid"]=="material-1"
    assert packet["native_qualification"] is packet["teacher_admission"] is False
    assert packet["r0_tangential_slip_m"] is packet["continuous_contact_s_m"] is None
    assert packet["new_sdk_steps"]==packet["new_solver_steps"]==0
    # Existing analytic marker consumes actual-order endpoint director arrays.
    from go2_pvcnn.control.dh116.tabletop_cosserat_material_marker_reference import make_marker,marker_position
    marker=make_marker(packet["constructor_rest_lengths_m"],segment_index=20,alpha=.5,
        offset_yz_m=[0,.005],material_reference_uid="material-1")
    point=marker_position(marker,packet["constructor_rest_lengths_m"],packet["node_positions_m"][0],
        packet["quaternions_xyzw_with_padding"][0])
    assert point[1]==pytest.approx(-.005*np.sin(theta),abs=1e-9)
    assert np.array_equal(reset["quaternions_xyzw_with_padding"][0,20],[0,0,0,1])
    c.rod.p[:]=999;c.rod.q[:]=999
    assert packet["node_positions_m"].max()<1
    assert packet["quaternions_xyzw_with_padding"][0,20,3]<1


@pytest.mark.parametrize("fault",["nondefault","wrong_backend","wrong_type","positions","rest","quaternion","shape","clock","sdk","solver_time","solver_error"])
def test_invalid_capture_poison_requires_fresh_actual_reset(case,fault):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args);c.advance()
    row=c.row()
    if fault=="nondefault":c.stream["current"]=7
    elif fault=="wrong_backend":c.backend.__file__="/wrong/_ropesim.so"
    elif fault=="wrong_type":c.backend.RopeBatch=type("Another",(),{})
    elif fault=="positions":row["node_positions"][0,0,0]+=.001
    elif fault=="rest":row["rest_lengths_m"][0]+=.001
    elif fault=="quaternion":c.rod.q[0,3]=0
    elif fault=="shape":c.rod.q=c.rod.q[:,:40]
    elif fault=="clock":c.clock["clock_valid"]=False
    elif fault=="sdk":c.sim.current_time_step_index+=1
    elif fault=="solver_time":c.rod.time+=.001
    elif fault=="solver_error":c.rod.last_error=2
    with pytest.raises((ValueError,RuntimeError)):c.owner.capture(row,**c.args)
    with pytest.raises(RuntimeError,match="fresh reset"):c.owner.capture(row,**c.args)


@pytest.mark.parametrize("change",["reset","episode","run","rod","environment","skip","repeat","tick"])
def test_sequence_identity_never_silently_rebinds(case,change):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    if change!="repeat":c.advance()
    field={"reset":"reset_id","episode":"episode_uid","run":"run_uuid","rod":"rod_uid","environment":"env_id","skip":"physics_step","tick":"rod_tick"}.get(change)
    if field:c.clock[field]=c.clock[field]+1 if isinstance(c.clock[field],int) else c.clock[field]+"changed"
    with pytest.raises((ValueError,RuntimeError)):c.owner.capture(c.row(),**c.args)


def test_clock_change_during_getter_rejected(case):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args);c.advance()
    getter=c.rod.quaternions
    def racing():
        value=getter();c.advance();return value
    c.rod.quaternions=racing
    with pytest.raises(ValueError,match="clock"):c.owner.capture(c.row(),**c.args)


def test_fresh_reset_rebinds_material_but_not_global_solver_time(case):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    c.advance();c.owner.capture(c.row(),**c.args)
    c.clock.update(reset_id=2,episode_uid="ep2",rod_tick=0)
    packet=c.owner.capture_reset(c.row(),material_reference_uid="material-2",**c.args)
    assert packet["identity"]["solver_time_s"]>0
    assert packet["identity"]["rod_tick"]==0
    with pytest.raises(ValueError):c.owner.capture_reset(c.row(),material_reference_uid="material-2",**c.args)


def test_adapter_does_not_retain_borrowed_native_owners(case):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    ref=weakref.ref(c.rod)
    c.args.clear();c.rod=None;c.row=None;c.advance=None
    gc.collect()
    assert ref() is None


def test_first_sample_requires_explicit_reset(case):
    with pytest.raises(RuntimeError,match="fresh reset"):case.owner.capture(case.row(),**case.args)


def test_no_cuda_or_native_import_at_module_import(tmp_path):
    import subprocess,sys,os
    code="""import sys
from go2_pvcnn.control.dh116.tabletop_material_director_readback import MaterialDirectorReadback
assert not any(x.split('.')[0] in ('pxr','isaaclab','isaacsim','ropesim') for x in sys.modules)
import torch
assert not torch.cuda.is_initialized()
print('CPU_ONLY')
"""
    r=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,env={**os.environ,"CUDA_VISIBLE_DEVICES":""})
    assert r.returncode==0,r.stderr
    assert r.stdout.strip()=="CPU_ONLY"


def test_identical_other_rope_instance_is_not_same_material_owner(case):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    c.advance()
    other=RopeBatch(c.nodes,[]);other.time=c.rod.time
    with pytest.raises(ValueError,match="same actual RopeBatch"):
        c.owner.capture(c.row(),**dict(c.args,rope=other))


def test_failed_reset_capture_allows_only_next_fresh_reset(case):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    c.clock.update(reset_id=2,episode_uid="ep2")
    c.rod.q[0,0]=0
    with pytest.raises(ValueError,match="unit"):
        c.owner.capture_reset(c.row(),material_reference_uid="material-2",**c.args)
    c.rod.q[0,0]=[0,0,0,1]
    with pytest.raises(ValueError,match="fresh"):
        c.owner.capture_reset(c.row(),material_reference_uid="material-2",**c.args)
    c.clock.update(reset_id=3,episode_uid="ep3")
    packet=c.owner.capture_reset(c.row(),material_reference_uid="material-3",**c.args)
    assert packet["identity"]["reset_id"]==3
    assert packet["identity"]["physics_step"]==2


@pytest.mark.parametrize("field",["physics_step","solver_time_s"])
def test_reset_after_failed_later_attempt_cannot_move_backwards(case,field):
    c=case;c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    c.advance();c.owner.capture(c.row(),**c.args)
    for _ in range(4):c.advance()
    c.clock.update(reset_id=2,episode_uid="ep2",rod_tick=0)
    c.rod.q[0,0]=0
    with pytest.raises(ValueError,match="unit"):
        c.owner.capture_reset(c.row(),material_reference_uid="material-2",**c.args)
    c.rod.q[0,0]=[0,0,0,1]
    c.clock.update(reset_id=3,episode_uid="ep3",physics_step=8,timestamp_s=8/480)
    if field=="physics_step":c.clock.update(physics_step=4,timestamp_s=4/480)
    else:c.clock["solver_time_s"]=float(np.float32(2/480))
    c.sim.current_time_step_index=c.clock["physics_step"];c.sim.current_time=c.clock["timestamp_s"]
    c.rod.time=c.clock["solver_time_s"]
    with pytest.raises(ValueError,match="monotonic"):
        c.owner.capture_reset(c.row(),material_reference_uid="material-3",**c.args)
