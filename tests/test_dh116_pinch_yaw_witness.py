"""Geometry witnesses are upper bounds on source convex envelopes, not contacts."""
from itertools import product
import numpy as np
import pytest
from scripts import audit_dh116_pinch_yaw_witness as m

def cube():
    return np.array(list(product([-.01,.01],[-.01,.01],[-.01,.01])),dtype=float)

def test_feasible_point_pair_proves_convex_envelope_overlap_only():
    r=m.witness(cube(),np.array([[.012,-.1,0],[.012,0,0],[.012,.1,0.]]))
    assert r["gap_upper_m"]<0 and r["convex_envelope_overlap_witness"] is True
    assert r["native_contact_proven"] is r["geometry_clearance_proven"] is False
    assert r["full_polyline_closest_distance_proven"] is False
    assert r["source"]=="convex envelope of supplied source vertices; one actual polyline segment"

def test_positive_upper_bound_does_not_prove_clearance():
    r=m.witness(cube(),np.array([[.03,-.1,0],[.03,.1,0]]))
    assert r["gap_upper_m"]>0 and r["convex_envelope_overlap_witness"] is False
    assert r["geometry_clearance_proven"] is False

def test_chosen_segment_is_real_curved_polyline_not_chord():
    nodes=np.array([[.04,-.1,0],[.012,0,0],[.04,.1,0]])
    r=m.witness(cube(),nodes)
    i=r["segment_index"]
    a,b=nodes[i:i+2];point=np.array(r["closest"]["axis_point_w"])
    t=np.dot(point-a,b-a)/np.dot(b-a,b-a)
    assert -.000001<=t<=1.000001
    assert np.linalg.norm(point-a-t*(b-a))<1e-10

@pytest.mark.parametrize("bad",[
    np.array([[0,0,0]]),np.zeros((2,3)),np.array([[0,np.nan,0],[1,0,0]])])
def test_degenerate_or_nonfinite_polyline_rejected(bad):
    with pytest.raises(ValueError):m.witness(cube(),bad)

def test_error_cannot_be_saved_as_zero_gap(monkeypatch):
    def fail(*args):raise ValueError("unconverged")
    monkeypatch.setattr(m,"convex_patch_closest",fail)
    with pytest.raises(ValueError,match="unconverged"):
        m.witness(cube(),np.array([[.012,-.1,0],[.012,.1,0]]))


def test_unexpected_native_geometry_exception_preserves_failure(monkeypatch,tmp_path):
    def fail(project,result):
        result["records"]=[{"status":"error","shape":"source:finger13"}]
        raise RuntimeError("unexpected QP failure")
    monkeypatch.setattr(m,"evaluate",fail)
    assert m.main(["--output",str(tmp_path/"out")])==1
    import json
    saved=json.loads((tmp_path/"out"/"result.json").read_text())
    assert saved["status"]=="diagnostic_error"
    assert saved["records"][0]["status"]=="error"
    assert "RuntimeError" in saved["error"]
    assert not saved["native_qualification"]
