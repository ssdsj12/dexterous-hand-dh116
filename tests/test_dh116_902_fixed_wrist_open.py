import itertools
import numpy as np
import pytest
from scripts.screen_dh116_902_fixed_wrist_open import compile_closure, reset_geometry
from go2_pvcnn.control.dh116.linear_coupling import expand

def test_closure_preserves_original_endpoint_and_all_24d_held_axes():
    start=np.array([.05,.01,.57,.69,.25,.25]);goal=start+np.array([.3,0,.1,.1,0,0])
    a,states=compile_closure(start,goal,windows=64)
    assert a.shape==(64,24) and states.shape==(512,6)
    held=list(range(6))+[7,10,11]+list(range(12,24))
    assert np.count_nonzero(a[:,held])==0 and np.max(abs(a))<=1
    q=start.astype(np.float32).astype(float)
    for step,expected in enumerate(states):
        q=(q+a[step//8,6:12]*.3/480).astype(np.float32).astype(float)
        np.testing.assert_array_equal(q,expected);expand(q)
    np.testing.assert_allclose(q,goal,atol=1e-6,rtol=0)

@pytest.mark.parametrize("windows",[0,True,1000,1])
def test_invalid_or_too_fast_closure_rejected(windows):
    with pytest.raises(ValueError):
        compile_closure([.05,.01,.57,.69,.25,.25],[.35,.01,.67,.79,.25,.25],windows=windows)

def shapes():
    from scripts.screen_dh116_902_fixed_wrist_open import REQUIRED_SHAPES
    v=np.array(list(itertools.product([.02,.03],[.02,.03],[.02,.03])))
    return {k:v.copy() for k in REQUIRED_SHAPES}

def test_reset_uses_entire_archive_and_every_original_shape():
    points=np.zeros((239,41,3));points[:,:,0]=np.linspace(-.1,.1,41)
    boxes=[(np.array([.1,.1,.1]),np.array([.11,.11,.11]))]*2
    row=reset_geometry(shapes(),points,boxes,table_top=-1.)
    assert row["passed"] and row["shape_count"]==22 and row["frame_count"]==239
    points[0,:,1:]=.025
    assert not reset_geometry(shapes(),points,boxes,table_top=-1.)["passed"]

@pytest.mark.parametrize("bad",["missing_shape","missing_frame","nan"])
def test_partial_geometry_never_admitted(bad):
    s=shapes();points=np.zeros((239,41,3));points[:,:,0]=np.linspace(-.1,.1,41)
    if bad=="missing_shape":s.pop(next(iter(s)))
    if bad=="missing_frame":points=points[1:]
    if bad=="nan":points[0,0,0]=float("nan")
    with pytest.raises(ValueError):reset_geometry(s,points,[(np.ones(3),np.ones(3)*2)]*2,table_top=-1.)


def test_candidate_cannot_self_authorize_a_different_source(tmp_path):
    import hashlib,json
    from scripts import screen_dh116_902_fixed_wrist_open as m
    fake=tmp_path/"fake.json";fake.write_text(json.dumps({"target":"different"}))
    candidate={"source_request":"fake.json","source_request_sha256":hashlib.sha256(fake.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):m.checked_source_request(candidate,tmp_path)

@pytest.mark.parametrize("bad",["gap","side","opposition","nan","missing"])
def test_endpoint_geometry_is_not_contact_when_evidence_fails(bad):
    from scripts.screen_dh116_902_fixed_wrist_open import endpoint_contact_candidate
    row={"pad_gaps_m":[-.00001]*3,"node_ids":[2,3,4],"opposition":[-.9,-.8]}
    if bad=="gap":row["pad_gaps_m"][2]=.00023
    if bad=="side":row["node_ids"][2]=30
    if bad=="opposition":row["opposition"][0]=.3
    if bad=="nan":row["opposition"][0]=float("nan")
    if bad=="missing":row.pop("node_ids")
    assert not endpoint_contact_candidate(row)

def test_endpoint_pass_is_only_geometric_candidate():
    from scripts.screen_dh116_902_fixed_wrist_open import endpoint_contact_candidate
    assert endpoint_contact_candidate({"pad_gaps_m":[-.00001]*3,"node_ids":[2,3,4],"opposition":[-.9,-.8]})
