import numpy as np
import pytest
from scripts.certify_dh116_tripod_tracking_envelope import (
    certify_box, shifted_plan, configuration_at, envelope_flags)
def test_clear_box_has_continuous_certificate_not_native_qualification():
    r=certify_box(lambda x:{"gap":2-x[0]-x[1]},{"gap":np.array([1.,1.])},
        np.zeros(2),np.full(2,.25),max_samples=100)
    assert r["proved"] and len(r["certified_boxes"])>0
    assert r["native_ready"] is False
def test_interior_failure_not_hidden_by_passing_endpoints_or_center():
    r=certify_box(lambda x:{"gap":(x[0]-.4)**2-.004},{"gap":np.array([1.2])},
        np.zeros(1),np.ones(1),max_samples=500)
    assert not r["proved"] and r["violations"]
def test_insufficient_budget_is_unresolved_not_passed():
    r=certify_box(lambda x:{"gap":.001},{"gap":np.array([1.])},
        np.zeros(1),np.ones(1),max_samples=1)
    assert not r["proved"] and r["unresolved_boxes"]
def test_nonfinite_or_missing_constraint_rejected():
    for fn in [lambda x:{"gap":float("nan")},lambda x:{"other":1.}]:
        with pytest.raises(ValueError):
            certify_box(fn,{"gap":np.array([1.])},np.zeros(1),np.ones(1),max_samples=5)
def parent():
    return dict(end_pose_wxyz=[.4,.1,.38,1.,0.,0.,0.],
        q0_rad=[.4,.3,.6,.7,.8,.9],q1_rad=[.36,.32,.64,.708,.8,.9])
def test_root_shift_is_same_for_start_and_end_not_extra_action():
    from scripts.prepare_dh116_tripod_fixed_orientation import compile_candidate
    p=shifted_plan(parent(),[.00005,0,.00015])
    old=compile_candidate(parent(),raise_m=.0015,steps=200)
    np.testing.assert_allclose(p["actions"],old["actions"],atol=1e-14)
    np.testing.assert_allclose(np.array(p["end_pose_wxyz"])[:3]-old["end_pose_wxyz"][:3],[.00005,0,.00015])
    assert p["native_ready"] is False
@pytest.mark.parametrize("offset",[[.001,0,0],[0,.0001,0],[0,0,float("nan")]])
def test_offset_search_cannot_expand_original_planning_scope(offset):
    with pytest.raises(ValueError):shifted_plan(parent(),offset)
def test_fractional_time_covers_continuum_not_only_sdk_samples():
    p=shifted_plan(parent(),[0,0,0])
    pose,q=configuration_at(p,100.5,[10.,0,5.,0,0,0],6.)
    np.testing.assert_allclose(q,np.array(p["q0_rad"])+np.array([.4525,.5025,.4775,.5025,.5025,.5025])*(np.array(p["q1_rad"])-p["q0_rad"]))
    assert pose[2]==pytest.approx(.3815-.0015*.4725)
def test_proof_is_conditional_not_future_contact_or_teacher():
    r=envelope_flags(True)
    assert r["conditional_kinematic_envelope_proved"] is True
    assert r["actual_tracking_qualified"] is r["native_ready"] is r["teacher_admission"] is False

def test_clipped_phase_radius_is_zero_after_every_delay_has_settled():
    from scripts.certify_dh116_tripod_tracking_envelope import phase_radii
    np.testing.assert_array_equal(phase_radii(
        np.array([216.,0.,0.]),np.array([224.,11.,6.])),[0.,0.])
def test_clipped_radius_covers_all_domain_corners_and_interior():
    from itertools import product
    from scripts.certify_dh116_tripod_tracking_envelope import phase_radii
    a=np.array([195.,0.,2.]);b=np.array([211.,11.,6.]);mid=(a+b)/2
    center=np.clip(mid[0]-mid[1:],0,200)
    radii=phase_radii(a,b)
    for bits in product((0.,.25,.5,.75,1.),repeat=3):
        x=a+np.array(bits)*(b-a)
        assert np.all(np.abs(np.clip(x[0]-x[1:],0,200)-center)<=radii+1e-12)
def test_clipped_certificate_can_prove_settled_tail_with_one_query():
    from scripts.certify_dh116_tripod_tracking_envelope import phase_radii
    bounds={"gap":np.array([2.,1.,1.])}
    r=certify_box(lambda x:{"gap":.001},bounds,np.array([216.,0.,0.]),
        np.array([224.,11.,6.]),max_samples=1,
        radius_bound=lambda a,b:{"gap":float(phase_radii(a,b).sum())})
    assert r["proved"] and len(r["samples"])==1

def test_certificate_cover_rejects_holes_duplicates_and_false_margin():
    from copy import deepcopy
    from scripts.certify_dh116_tripod_tracking_envelope import verify_box_certificate
    query=lambda x:{"gap":.2}
    bounds={"gap":np.array([1.])};lo=np.zeros(1);hi=np.ones(1)
    p=certify_box(query,bounds,lo,hi,max_samples=100)
    assert p["proved"] and len(p["certified_boxes"])>1
    verify_box_certificate(p,query,bounds,lo,hi)
    for fault in ("hole","overlap","margin","sample"):
        bad=deepcopy(p)
        if fault=="hole":bad["certified_boxes"].pop()
        elif fault=="overlap":bad["certified_boxes"].append(deepcopy(bad["certified_boxes"][0]))
        elif fault=="margin":bad["certified_boxes"][0]["minimum_margin_m"]+=1.
        else:bad["samples"][bad["certified_boxes"][0]["sample_index"]]["margin"]["gap"]+=1.
        with pytest.raises(ValueError):verify_box_certificate(bad,query,bounds,lo,hi)
