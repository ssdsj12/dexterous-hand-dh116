import copy
import numpy as np
import pytest
from scripts.solve_dh116_tripod_contact_closure import (
    CONFIG, decode, validate_config, physical_constraints, candidate_admission,
    bounded_search, BudgetExceeded)

def parent():
    return dict(end_pose_wxyz=[.53,.14,.385,1.,0.,0.,0.],
        q0_rad=[.45,.44,.7,.75,.7,.95],q1_rad=[.42,.46,.74,.765,.7,.95])
def test_decoding_keeps_fixed_orientation_inactive_fingers_and_original_limits():
    pose,q=decode(parent(),[.2,-.1,0,0,0,.1,.2,.3,.4])
    np.testing.assert_allclose(pose[:3],[.53008,.14,.38496])
    np.testing.assert_array_equal(pose[3:],[1,0,0,0])
    np.testing.assert_allclose(q,[.421,.462,.743,.769,.7,.95])
    for x in ([2,0,0,0,0,0,0,0,0],[0,0,float('nan'),0,0,0,0,0,0]):
        with pytest.raises(ValueError):decode(parent(),x)
def test_native_limits_unchanged_and_planning_constraints_more_conservative():
    validate_config(CONFIG)
    for field,value in [('native_required_gap_m',-.00031),('table_clearance_m',.002),
                        ('native_execution',True),('required_planning_gap_m',-.0004)]:
        cfg=copy.deepcopy(CONFIG);cfg[field]=value
        with pytest.raises(ValueError):validate_config(cfg)
def test_contact_prediction_cannot_bypass_non_target_guard():
    c=physical_constraints({'source:finger22_link':.003,'cooked:finger14_link':.003},
        {'source:finger22_link':-.00001,'cooked:finger14_link':-.0001},
        {n:-.000015 for n in ('finger14_link','finger23_link','finger33_link')},CONFIG)
    assert c['rod:source:finger22_link']<0
    a=candidate_admission(c)
    assert not a['static_constraints_passed']
    assert not a['native_ready'] and not a['actual_contact_observed'] and not a['teacher_admission']
def test_required_and_non_target_thresholds_are_distinct():
    c=physical_constraints({'source:finger22_link':.003,'cooked:finger14_link':.003},
        {'source:finger22_link':.00003,'cooked:finger14_link':-.0001},
        {n:-.000015 for n in ('finger14_link','finger23_link','finger33_link')},CONFIG)
    assert all(v>=0 for v in c.values())
    assert candidate_admission(c)['static_constraints_passed']
def test_all_three_supported_pad_targets_required():
    for pads in ({},dict(finger14_link=float('nan')),dict(finger14_link=-1.)):
        with pytest.raises(ValueError):physical_constraints({}, {}, pads, CONFIG)
def test_search_cannot_mark_numerical_optimizer_status_as_geometric_pass():
    def query(x):return {'x':np.asarray(x).tolist(),'constraints':{'impossible':-1.}}
    r=bounded_search(query,np.zeros(6),maximum_evaluations=80,maximum_seconds=2.)
    assert not r['best']['admission']['static_constraints_passed']
    assert not r['best']['admission']['native_ready']
def test_search_preserves_evaluation_budget_and_failed_samples():
    rows=[]
    def query(x):
        result={'x':np.asarray(x).tolist(),'constraints':{'gap':float(x[0]-.9)}}
        rows.append(result);return result
    r=bounded_search(query,np.zeros(6),maximum_evaluations=2,maximum_seconds=2.)
    assert r['termination']=='budget_exhausted'
    assert r['evaluations']==len(rows)==2
    assert r['best']['admission']['native_ready'] is False
def test_search_uses_complete_finite_constraint_identity():
    calls=[0]
    def query(x):
        calls[0]+=1
        return {'x':np.asarray(x).tolist(),'constraints':{'x' if calls[0]==1 else 'wrong':-1.}}
    with pytest.raises(ValueError):bounded_search(query,np.zeros(6),maximum_evaluations=20,maximum_seconds=2.)

def test_constant_reset_rotation_is_not_a_wrist_rotation_action():
    from scripts.prepare_dh116_tripod_fixed_orientation import compile_candidate
    x=np.zeros(9);x[2]=.25
    pose,q=decode(parent(),x)
    assert pose[3:] != pytest.approx(parent()['end_pose_wxyz'][3:])
    p=parent();p['end_pose_wxyz']=pose;p['q1_rad']=q
    plan=compile_candidate(p,raise_m=.0015,steps=200)
    # Same quaternion at both endpoints: changing reset orientation generates
    # no policy wrist rotation command in either hand.
    np.testing.assert_array_equal(plan['seed_pose_wxyz'][3:],plan['end_pose_wxyz'][3:])
    np.testing.assert_array_equal(np.asarray(plan['actions'])[:,18:24],0.)

def test_minimax_search_finds_feasible_contact_without_trading_guard_violations():
    def query(x):
        return {'x':np.asarray(x).tolist(),'constraints':{
            'contact':float((x[0]-.15)*.001),
            'clearance':float((.3-x[0])*.001)}}
    r=bounded_search(query,np.zeros(3),maximum_evaluations=100,maximum_seconds=3.)
    assert r['best']['admission']['static_constraints_passed']
    assert .15<=r['best']['x'][0]<=.3
