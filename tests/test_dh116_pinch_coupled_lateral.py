"""Coupled static pose search is CPU geometry, never a native contact claim."""
from copy import deepcopy
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts import solve_dh116_pinch_coupled_lateral as m

@pytest.fixture
def parent():
    return dict(side="right",base_pose_wxyz=[.5,-.2,.4,1.,0.,0.,0.],
        active_q_rad=[.3,.4,.6,1.,1.,1.],active_delta_rad=[0,.025,.025,0,0,0])

def test_decode_uses_original_absolute_reset_not_accumulated_seed(parent):
    before=deepcopy(parent); x=np.array([.1,1,1,.1,.2,.2,-.1])
    a=m.decode(parent,x);b=m.decode(parent,x)
    assert a==b and parent==before
    np.testing.assert_allclose(a["base_pose_wxyz"][:3],[.5006,-.197,.4002])
    np.testing.assert_allclose(Rotation.from_quat(np.array(a["base_pose_wxyz"])[[4,5,6,3]]).as_rotvec(),[0,0,-.003])
    assert a["active_q_rad"][0]==pytest.approx(.36)
    np.testing.assert_allclose(a["active_delta_rad"],[0,.03,.035,0,0,0])
    assert not a["native_qualification"] and not a["teacher_admission"]

@pytest.mark.parametrize("index,value",[(5,1.001),(6,-1.001),(0,-.01),(3,2),(4,np.nan)])
def test_bounds_and_original_finger_speed_cap_are_hard(parent,index,value):
    x=np.zeros(7);x[index]=value
    with pytest.raises(ValueError):m.decode(parent,x)

def test_model_exact_columns_29_calls_and_bounds():
    calls=[];lo=-np.ones(7);hi=np.ones(7);x=np.array([1.,-1,0,0,0,0,0])
    def query(v):
        assert np.all(v>=lo) and np.all(v<=hi);calls.append(v)
        return dict(labels=["linear"],values=[float(v@np.arange(1,8))])
    d=m.model(query,x,lo,hi,.2)
    assert len(calls)==29
    np.testing.assert_allclose(d["jacobians"][1],[np.arange(1,8)],atol=1e-10)
    assert d["supported"]

def test_contact_improvement_requires_full_nonlinear_candidate():
    calls=[]
    def query(x):
        calls.append(x.copy())
        return dict(labels=["initial:safe","endpoint:pad:contact_minus0p1mm"],
                    values=[1.,float(x[5]+x[6]-1)])
    trace=[];out=m.refine(query,np.zeros(7),-np.ones(7),np.ones(7),trace)
    assert len(calls)==2*30 and len(trace)==2
    assert out["accepted_directions"]==2
    assert all("actual" in t and t["assessment"]["hard_constraints_preserved"] for t in trace)
    assert out["selected_x"][5]==pytest.approx(.4) and out["selected_x"][6]==pytest.approx(.4)

def test_hard_collision_failure_cannot_replace_parent():
    def query(x):
        return dict(labels=["initial:safe","endpoint:pad:contact_minus0p1mm"],
                    values=[float(.001-100*x[5]**2),float(x[5]-1)])
    trace=[];out=m.refine(query,np.zeros(7),-np.ones(7),np.ones(7),trace)
    assert out["accepted_directions"]==0
    assert all(not t["assessment"]["hard_constraints_preserved"] for t in trace)

def test_direction_and_geometry_caps_are_explicit():
    assert m.DIM==7 and m.DIRECTIONS==2
    assert m.MAX_STATES==2*(2*(29*2+2)+33)==306
    b=m.trust.StateBudget([],maximum=m.MAX_STATES)
    for i in range(m.MAX_STATES):b.call(lambda x:x,i)
    with pytest.raises(m.trust.BudgetLimit):b.call(lambda x:x,0)

def test_timeout_retains_attempt_and_never_creates_native(tmp_path,monkeypatch):
    def run(project,result):
        result["state_records"]=[dict(status="interrupted")]
        raise m.BudgetExpired()
    monkeypatch.setattr(m,"run",run)
    assert m.main(["--output",str(tmp_path/"out")])==0
    r=json.loads((tmp_path/"out"/"result.json").read_text())
    assert r["status"]=="partial_timeout" and r["state_records"][0]["status"]=="interrupted"
    assert not r["native_qualification"] and not r["reset_proposal_created"]
