"""Raw capture retention is CPU evidence, never a new native contact sample."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from test_dh116_deformx_pair_capture import packet
from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import read_pair_capture
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture,validate_complete_capture
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest

def raw():
    p=packet()
    p.update(node_forces_on_rod_world_n=np.array([[0.,0.,3.]]),
        mesh_wrenches_on_mesh_world=np.array([[0.,0.,-1.,0.,-.5,0.],[0.,0.,-2.,0.,0.,0.]]))
    return p

def capture(p=None):
    p=raw() if p is None else p
    calls=[]
    def read():calls.append(1);return p
    result=read_complete_pair_capture(SimpleNamespace(contact_pair_samples=read),
        dt_s=1/480,physics_step=3,rod_tick=12,mesh_names=("left.finger14_link","table"))
    return p,calls,result

def test_one_atomic_read_and_exact_legacy_projection():
    p,calls,result=capture()
    original=read_pair_capture(SimpleNamespace(contact_pair_samples=lambda:p),dt_s=1/480,
        physics_step=3,rod_tick=12,mesh_names=("left.finger14_link","table"))
    assert calls==[1]
    assert {k:v for k,v in result.items() if k!="complete_capture"}==original
    saved=result["complete_capture"]
    assert {k:saved["raw"][k] for k in ("capacity","attempted","overflow")}==dict(capacity=8,attempted=2,overflow=False)
    assert saved["raw_sha256"]==_digest(p)
    assert validate_complete_capture(result)["complete"] is True
    assert not result["native_qualification"]

def test_snapshot_does_not_alias_mutable_solver_buffers():
    p,_,result=capture();expected=_digest(result)
    p["vectors"][0,8]=99
    p["indices"][0,0]=0
    p["capacity"]=1
    assert _digest(result)==expected
    assert validate_complete_capture(result)["complete"]

def test_historical_derived_pairs_cannot_invent_capacity_or_overflow():
    _,_,result=capture()
    del result["complete_capture"]
    with pytest.raises(ValueError,match="complete"):validate_complete_capture(result)

@pytest.mark.parametrize("edit",[
    lambda r:r["complete_capture"]["raw"].update(overflow=True),
    lambda r:r["complete_capture"]["raw"].update(attempted=1),
    lambda r:r["complete_capture"]["raw"].update(capacity=1),
    lambda r:r["complete_capture"]["raw"]["vectors"].__setitem__((0,8),99),
    lambda r:r["pairs"].pop(),
    lambda r:r["pairs"][0].update(mesh_name="right.finger14_link"),
    lambda r:r.update(physics_step=4),
    lambda r:r.update(rod_tick=24),
    lambda r:r["complete_capture"].update(mesh_names=["table","left.finger14_link"]),
    lambda r:r["complete_capture"].update(raw_sha256="0"*64),
    lambda r:r["complete_capture"].update(dt_s=1/60),
    lambda r:r["complete_capture"].update(version="unknown"),
])
def test_tamper_or_stale_boundary_rejected(edit):
    _,_,r=capture();edit(r)
    with pytest.raises(ValueError):validate_complete_capture(r)

@pytest.mark.parametrize("edit",[lambda p:p.update(overflow=True),
    lambda p:p.pop("capacity"),lambda p:p.update(capacity=1),
    lambda p:p["indices"].__setitem__(0,p["indices"][1])])
def test_incomplete_native_packet_never_becomes_complete(edit):
    p=raw();edit(p)
    with pytest.raises((ValueError,KeyError)):capture(p)

def test_no_contacts_is_known_only_when_actual_complete_packet_has_zero_attempts():
    p=raw()
    p.update(attempted=0,indices=np.empty((0,3),dtype=np.int64),
        vectors=np.empty((0,12),dtype=np.float32),
        node_forces_on_rod_world_n=np.zeros((1,3)),mesh_wrenches_on_mesh_world=np.zeros((2,6)))
    _,_,r=capture(p)
    checked=validate_complete_capture(r)
    assert checked["attempted"]==0 and checked["complete"]
    assert r["pairs"]==[]
