"""Offline finite-grid and fail-closed candidate tests."""
import importlib.util
import numpy as np
import pytest
MODULE="go2_pvcnn.control.dh116.tabletop_teacher_acquisition_geometry"
def api():
    assert importlib.util.find_spec(MODULE) is not None, "acquisition geometry utility missing"
    return __import__(MODULE,fromlist=["x"])
def test_grid_reports_unknown_outside_and_uses_actual_node_count():
    m=api();v=np.indices((3,3,3)).sum(0).astype(float)
    grid=dict(values=v,origin=np.zeros(3),spacing=1.)
    p=np.tile([.5,.5,.5],(41,1));p[-1]=[3,0,0]
    r=m.sample_nodes(grid,p,np.eye(4),.2)
    assert r["nodes_inside_grid"]==40 and r["nodes_outside_grid"]==1
    assert r["nearest_sdf_gap_m"]==pytest.approx(1.3)
    p[:]=[4,0,0]
    assert m.sample_nodes(grid,p,np.eye(4),.2)["nearest_sdf_gap_m"] is None
def test_rejects_hidden_interpolated_nodes():
    with pytest.raises(ValueError,match="41"):
        api().sample_nodes(dict(values=np.zeros((2,2,2)),origin=np.zeros(3),spacing=1.),np.zeros((42,3)),np.eye(4),.008)
def test_unknown_or_penetrating_required_pad_cannot_be_reachable():
    m=api()
    for gap in (None,-.00001,.001001):
        assert not m.required_reachable({"a":{"nearest_sdf_gap_m":gap}},["a"])
    assert m.required_reachable({"a":{"nearest_sdf_gap_m":.001}},["a"])
def test_missing_cooked_geometry_cannot_freeze_even_reachable():
    r=api().freeze_decision(True,True,False)
    assert not r["freeze_s0"] and not r["native_admission"]
    assert r["blockers"]

def test_failed_continuous_geometry_solver_remains_unknown():
    def failed():raise ValueError("solver did not converge")
    r=api().checked_geometry(failed)
    assert r["status"]=="unknown"
    assert r["intersects_capsule"] is None

def test_translation_does_not_accumulate_into_fk_root():
    original={"root":np.eye(4)}
    first=api().translated_poses(original,[.01,0,0])
    second=api().translated_poses(original,[.01,0,0])
    assert original["root"][0,3]==0
    assert first["root"][0,3]==second["root"][0,3]==.01

def test_actual_scene_radius_not_historical_candidate_radius():
    assert api().source_radius({"wire_radius_m":.005})==.005
    with pytest.raises(ValueError):api().source_radius({"wire_radius_m":float("nan")})

def test_grid_interpolation_keeps_small_optimizer_derivatives():
    grid=dict(values=np.indices((2,2,2))[0].astype(np.float32)*.01,origin=np.zeros(3),spacing=1.)
    p=np.tile([.5,.5,.5],(41,1));q=p.copy();q[:,0]+=1e-10
    assert api().sample_nodes(grid,q,np.eye(4),.001)["nearest_sdf_gap_m"]>api().sample_nodes(grid,p,np.eye(4),.001)["nearest_sdf_gap_m"]

def test_capsule_plane_bound_only_certifies_separated_segments():
    vertices=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    nodes=np.c_[np.full(41,3.),np.linspace(-2,2,41),np.zeros(41)]
    r=api().capsule_plane_bound(vertices,nodes,.5)
    assert r["intersects_capsule"] is False
    assert r["capsule_clearance_lower_bound_m"]==pytest.approx(1.5)
    nodes[:,0]=0
    assert api().capsule_plane_bound(vertices,nodes,.5)["intersects_capsule"] is None

def test_constructor_keeps_original_straight_nodes_separate_from_observed_curve():
    straight=np.c_[np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.368)]
    observed=straight.copy();observed[1:-1,2]-=.02
    r=api().proposal_nodes(straight,observed)
    assert np.array_equal(r["constructor_nodes"],straight)
    assert np.array_equal(r["observed_reference_nodes"],observed)
    assert not np.array_equal(r["constructor_nodes"],r["observed_reference_nodes"])

@pytest.mark.parametrize("field,value",[
    ("spacing",float("nan")),("spacing",0.),("spacing",-1.),
    ("origin",np.zeros(2)),("origin",np.array([0.,0.,float("nan")])),
    ("values",np.full((2,2,2),float("nan"))),
    ("values",np.zeros((1,2,2)))])
def test_grid_rejects_invalid_numeric_boundaries(field,value):
    grid=dict(values=np.zeros((2,2,2)),origin=np.zeros(3),spacing=1.)
    grid[field]=value
    with pytest.raises(ValueError):api().sample_nodes(grid,np.zeros((41,3)),np.eye(4),.005)

@pytest.mark.parametrize("radius",[float("nan"),float("inf"),0.,-.005])
def test_radius_must_be_finite_and_positive(radius):
    with pytest.raises(ValueError):
        api().sample_nodes(dict(values=np.zeros((2,2,2)),origin=np.zeros(3),spacing=1.),np.zeros((41,3)),np.eye(4),radius)

@pytest.mark.parametrize("pose",[
    np.diag([2.,1.,1.,1.]),np.diag([-1.,1.,1.,1.]),
    np.diag([1.,1.,1.,2.]),np.eye(4)+np.diag([0.,0.,0.,.1])])
def test_pose_must_be_proper_rigid_transform(pose):
    with pytest.raises(ValueError):
        api().sample_nodes(dict(values=np.zeros((2,2,2)),origin=np.zeros(3),spacing=1.),np.zeros((41,3)),pose,.005)

@pytest.mark.parametrize("invalid_side",["original","observed"])
def test_proposal_rejects_nonfinite_constructor_or_reference(invalid_side):
    original=np.zeros((41,3));observed=original.copy()
    (original if invalid_side=="original" else observed)[0,0]=float("nan")
    with pytest.raises(ValueError):api().proposal_nodes(original,observed)
