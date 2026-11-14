from copy import deepcopy
import itertools
import numpy as np
import pytest

def m():
    from go2_pvcnn.control.dh116 import tabletop_polyline_closest_support as s
    return s

def box(center=(0,0,0),half=.001):
    return np.array(list(itertools.product([-half,half],repeat=3)))+center

def test_four_variable_distance_proposal_uses_full_source_support_proof():
    v=box();p=np.array([[.006,.006,-.003],[.006,.006,.003]])
    r=m().refine(v,p,max_segments=1)
    assert r["lower_m"]>0 and r["lower_m"]<=np.sqrt(2)*.005-.005
    assert r["optimization_variables"]==4 and r["optimizer_calls"]==1
    assert m().validate(r,v,p)
    assert not r["actual_contact_measurement"] and not r["native_qualification"]

def test_true_overlap_is_not_made_clear_by_successful_optimizer():
    v=box();p=np.array([[0,0,-.003],[0,0,.003]])
    r=m().refine(v,p,max_segments=1)
    assert r["lower_m"]<0
    assert not r["clear"] and not r["native_qualification"]
    assert m().validate(r,v,p)

def test_budget_does_not_drop_unprocessed_segments():
    v=box();p=np.array([[0,0,-.003],[0,0,0],[0,0,.003]])
    r=m().refine(v,p,max_segments=1)
    assert r["optimizer_calls"]<=1 and len(r["segments"])==2
    assert not r["clear"] and m().validate(r,v,p)

@pytest.mark.parametrize("fault",["vertices","nodes","normal","bound","radius"])
def test_certificate_rechecks_original_full_shape_and_polyline(fault):
    v=box();p=np.array([[.006,.006,-.003],[.006,.006,.003]])
    r=m().refine(v,p,max_segments=1);vv=v.copy();pp=p.copy()
    if fault=="vertices":vv[0,0]+=.001
    elif fault=="nodes":pp[0,0]+=.001
    elif fault=="normal":r["segments"][0]["new_normal"]=[100,0,0]
    elif fault=="bound":r["lower_m"]+=.001
    else:r["rod_radius_m"]=.004
    with pytest.raises(ValueError):m().validate(r,vv,pp)

def test_translation_invariance_and_hull_order():
    v=box();p=np.array([[.006,.006,-.003],[.006,.006,.003]])
    a=m().refine(v,p,max_segments=1);shift=np.array([.43,-.2,.305])
    b=m().refine(v[::-1]+shift,p+shift,max_segments=1)
    assert a["lower_m"]==pytest.approx(b["lower_m"],abs=1e-10)
    assert m().validate(b,v[::-1]+shift,p+shift)

def test_failed_optimizer_cannot_upgrade_original_certificate(monkeypatch):
    from types import SimpleNamespace
    s=m()
    monkeypatch.setattr(s,"minimize",lambda *a,**k:SimpleNamespace(success=False,status=9,nit=64))
    v=box();p=np.array([[0,0,-.003],[0,0,.003]])
    r=s.refine(v,p,max_segments=1)
    assert r["lower_m"]==r["original_certificate"]["lower_m"]
    assert r["segments"][0]["new_normal"] is None and not r["clear"]
    assert s.validate(r,v,p)

@pytest.mark.parametrize("budget",[0,41,True,.5])
def test_invalid_proposal_budget_refused(budget):
    with pytest.raises(ValueError):m().refine(box(),np.array([[0,0,0],[0,1,0]]),max_segments=budget)

def test_invented_upper_witness_rejected():
    s=m();v=box();p=np.array([[.006,.006,-.003],[.006,.006,.003]])
    r=s.refine(v,p,max_segments=1)
    r["segments"][0]["proposal"]["point_world_m"]=[.02,.02,.02]
    with pytest.raises(ValueError):s.validate(r,v,p)
