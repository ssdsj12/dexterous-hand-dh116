import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.solve_dh116_right_pinch_surface_registration import proper_registration, surface_seed, reflected_targets, EvaluationBudget

def test_proper_rigid_registration_recovers_known_transform():
    p=np.array([[0,0,0],[.012,.01,0]])
    n=np.array([[1,0,0],[-.8,0,.6]])
    r=Rotation.from_rotvec([.2,-.3,.4]).as_matrix();shift=np.array([.4,-.2,.3])
    result=proper_registration(p,n,p@r.T+shift,n@r.T)
    np.testing.assert_allclose(result["matrix"][:3,:3],r,atol=1e-12)
    np.testing.assert_allclose(result["matrix"][:3,3],shift,atol=1e-12)
    assert result["rms_m"]<1e-12 and np.linalg.det(result["matrix"][:3,:3])==pytest.approx(1)

def test_improper_mapping_never_returns_a_reflection_root():
    p=np.array([[0,0,0],[.012,.01,.003]])
    n=np.array([[1,0,0],[0,0,1.]])
    reflection=np.diag([1,-1,1])
    result=proper_registration(p,n,p@reflection,n@reflection)
    assert np.linalg.det(result["matrix"][:3,:3])==pytest.approx(1)
    assert result["rms_m"]>1e-5

@pytest.mark.parametrize("kind",["normal","shape","nan","collinear"])
def test_registration_rejects_undefined_geometry(kind):
    p=np.array([[0,0,0],[.01,.01,0]]);n=np.array([[1,0,0],[-1,0,0.]])
    if kind=="normal":n[0]*=2
    if kind=="shape":p=p[:1]
    if kind=="nan":p[0,0]=np.nan
    if kind=="collinear":p[1]=[.01,0,0]
    with pytest.raises(ValueError):proper_registration(p,n,p,n)

def test_locked_sdf_surface_seed_uses_real_zero_and_gradient():
    x=np.linspace(-.01,.01,21);values=np.broadcast_to(x[:,None,None],(21,21,21)).copy()
    g=dict(values=values,origin=np.array([-.01]*3),spacing=.001)
    r=surface_seed(g,[.0007,0,0],[1,0,0])
    assert abs(r["point_local_m"][0])<1e-7
    np.testing.assert_allclose(r["normal_local"],[1,0,0],atol=1e-6)
    assert r["native_contact_observed"] is False

def test_missing_locked_surface_cannot_be_invented():
    g=dict(values=np.ones((21,21,21)),origin=np.array([-.01]*3),spacing=.001)
    with pytest.raises(ValueError):surface_seed(g,[0,0,0],[1,0,0])

def test_reflected_directions_use_own_reset_node_and_are_not_mirrored_joints():
    nodes=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)
    pairs=[dict(mesh_name="left.finger14_link",node_index=35,substep=11,outward_normal_world=[-1,0,0]),
           dict(mesh_name="left.finger23_link",node_index=34,substep=11,outward_normal_world=[.8,0,-.6])]
    r=reflected_targets(pairs,nodes)
    assert r["node_indices"]==[5,6]
    np.testing.assert_allclose(r["points_world_m"][0],nodes[5]+[.00485,0,0],atol=1e-12)
    assert r["reference_only"] and r["teacher_success"] is False

@pytest.mark.parametrize("damage",["duplicate","side","node","substep"])
def test_target_identity_cannot_mix_pad_side_or_substep(damage):
    nodes=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)
    pairs=[dict(mesh_name="left.finger14_link",node_index=35,substep=11,outward_normal_world=[-1,0,0]),
           dict(mesh_name="left.finger23_link",node_index=34,substep=11,outward_normal_world=[1,0,0])]
    if damage=="duplicate":pairs.append(dict(pairs[0]))
    if damage=="side":pairs[0]["mesh_name"]="right.finger14_link"
    if damage=="node":pairs[0]["node_index"]=999
    if damage=="substep":pairs[0]["substep"]=10
    with pytest.raises(ValueError):reflected_targets(pairs,nodes)

def test_budget_enforced_before_fk_and_never_replenished():
    b=EvaluationBudget(2);assert b.consume()==1;assert b.consume()==2
    with pytest.raises(ValueError,match="budget"):b.consume()
    assert b.count==2

@pytest.mark.parametrize("clearance,passed",[(.0022,True),(-.001,False)])
def test_all_shapes_are_checked_even_when_pad_registration_is_exact(monkeypatch,clearance,passed):
    from scripts import solve_dh116_right_pinch_surface_registration as m
    names=[*m.PADS,*["other"+str(i) for i in range(15)]]
    cooked=[*m.PADS,*names[2:5]]
    def shape(name):
        z=.3+clearance if name=="other14" else .31
        return {"vertices":np.array([[0,0,z],[.01,0,z],[0,.01,z],[0,0,z+.01]])}
    g={"source_colliders":{n:shape(n) for n in names},"cooked_pads":{n:shape(n) for n in cooked}}
    monkeypatch.setattr(m.base,"fk_poses",lambda *args:{n:np.eye(4) for n in names})
    monkeypatch.setattr(m,"_separation",lambda *args:.001)
    monkeypatch.setattr(m.base,"sample_contact",lambda grid,*args:dict(minimum_gap_m=-.00015,
        gradient_norm_at_minimum=1,local_node_at_minimum=[0,0,0]))
    monkeypatch.setattr(m.base,"sample_grid",lambda grid,*args:(np.array([0]),np.array([grid]),np.array([True])))
    r=m.geometry_record(g,dict(zip(m.PADS,[[1,0,0],[-1,0,0]])),np.eye(4),[.2,.2,.2,1.15,1.15,1.15],np.zeros((41,3)))
    assert len(r["table_m"])==22 and len(r["non_target_rod_gap_m"])==18
    assert r["endpoint_passed"] is passed
    assert r["geometric_gates"]["native_ready"] is False
