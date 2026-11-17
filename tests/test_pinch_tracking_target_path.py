"""Full original continuous path checks stay mandatory after endpoint search."""
from copy import deepcopy
import importlib
import pytest

@pytest.fixture
def m():return importlib.import_module("scripts.audit_dh116_pinch_tracking_target_path")

def test_auditor_namespace_never_mutates_historical_parent(m):
    old=(m.path.PARENT,m.path.PARENT_SHA,m.path.bow)
    audit=m.isolated_auditor('/temporary/candidate.json','a'*64,m.plateau_bow)
    assert str(audit.PARENT)=='/temporary/candidate.json'
    assert audit.bow is m.plateau_bow
    assert (m.path.PARENT,m.path.PARENT_SHA,m.path.bow)==old

def test_actual_candidate_requires_complete_observed_validation(m):
    r,g=m.load_candidate()
    assert r["qualification"]["empirical_endpoint_passed"]
    parent=m.parent_record(r,g)
    assert parent["sides"]["right"]["search"]["best"]["admission"]["static_constraints_passed"]
    assert parent["sides"]["left"]["search"]["best"]["admission"]["static_constraints_passed"] is False
    assert parent["native_ready"] is False
    r=deepcopy(r);del r["all_observed_validation"]["scenarios"]["212"]
    with pytest.raises(ValueError,match="complete"):m.parent_record(r,g)

def test_changed_candidate_bytes_are_rejected(m):
    with pytest.raises(ValueError,match="SHA"):m.load_candidate(expected_sha="0"*64)

def test_sample_or_endpoint_pass_never_grants_native_contact(m):
    assert m.interpret({"sides":{"right":{"static_path_passed":True}}})=={
        "static_path_passed":True,"tracking_qualified":False,"native_ready":False,"teacher_admission":False}
    assert m.interpret({"sides":{"right":{"static_path_passed":False}}})["static_path_passed"] is False


def test_plateau_uses_same_amplitude_and_maximum_segment_speed(m):
    import numpy as np
    old=m.path.bow
    points=np.linspace(0,1,21)
    y=np.array([m.plateau_bow(float(x),.0001) for x in points])
    previous=np.array([old(float(x),.0001) for x in points])
    assert y[0]==y[-1]==0
    assert y.max()<=.0001
    assert np.all(y>=previous-1e-15)
    assert np.max(np.abs(np.diff(y)))/(1/60)<=.00114+1e-15
    assert m.plateau_bow(.7,.0001)>old(.7,.0001)
    for fraction in (-.1,1.1):
        with pytest.raises(ValueError):m.plateau_bow(fraction,.0001)

def test_two_audit_namespaces_do_not_change_each_other(m):
    a=m.isolated_auditor('/a.json','a'*64,m.plateau_bow)
    b=m.isolated_auditor('/b.json','b'*64,m.path.bow)
    assert a.PARENT!=b.PARENT
    assert a.bow is m.plateau_bow
    assert b.bow is m.path.bow

def test_original_table_non_target_and_depth_guards_remain(m):
    a={'table_m':{'a':.0021},'non_target_rod_gap_m':{'b':0.},'pads':{'c':{'minimum_gap_m':-.0003}}}
    assert m.path_guards(a)
    for group,key in [('table_m','a'),('non_target_rod_gap_m','b')]:
        bad=deepcopy(a);bad[group][key]-=1e-10
        assert m.path_guards(bad) is False
    bad=deepcopy(a);bad['pads']['c']['minimum_gap_m']-=1e-10
    assert m.path_guards(bad) is False


def test_whole_hand_union_and_inter_sample_margin_are_not_centres(m):
    import numpy as np
    g={'source_colliders':{'a':{'vertices':np.array([[0.,0.,0.]])}},
       'cooked_pads':{'b':{'vertices':np.array([[3.,0.,0.]])}}}
    poses={'a':np.eye(4),'b':np.eye(4)}
    union=m.whole_hand(g,poses)
    assert union.shape==(2,3)
    assert union[:,0].max()==3.
    assert m.interhand_lower(.001,.1,101)==pytest.approx(.0005)
    with pytest.raises(ValueError):m.interhand_lower(.001,.1,1)


def test_old_plan_is_loaded_before_temporary_audit_parent(tmp_path,m,monkeypatch):
    real=m.targets.plan.load_pair;parent=m.path.PARENT
    real_boundary=m.targets.plan.expected_boundary;old_bow=m.path.bow
    def check_boundary(*args,**kwargs):
        assert m.path.bow is old_bow, 'historical command must not use candidate bow'
        return real_boundary(*args,**kwargs)
    monkeypatch.setattr(m.targets.plan,'expected_boundary',check_boundary)
    def check(*args,**kwargs):
        assert m.path.PARENT==parent, 'legacy source plan read inside temporary new parent'
        return real(*args,**kwargs)
    monkeypatch.setattr(m.targets.plan,'load_pair',check)
    result=m.run(tmp_path,m.PROFILE)
    assert result['observed_path']['samples']==232
    assert result['continuous_interhand']['samples']==1025
    assert result['native_ready'] is False
