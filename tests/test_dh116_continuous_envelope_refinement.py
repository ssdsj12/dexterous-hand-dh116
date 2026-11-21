"""Analytic support witnesses: no USD, assets, physics or optimizer-success admission."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts import audit_dh116_certified_segment_distance as cert
from scripts.screen_dh116_tripod_supported_reset import archived_tube
from scripts.verify_dh116_supported_distal_transfer import validate_live_continuous_hulls
from scripts.verify_dh116_power_shared_rod import check_supported_shape

METHOD = "dh116-continuous-envelope-support-v1"

def fixture(x=.0049):
    rot = Rotation.from_euler("y", 3.75, degrees=True).as_matrix()
    vertices = np.array([[a,b,c] for a in (x,x+.002)
                        for b in (-.015,.015) for c in (-.02,.02)]) @ rot.T
    nodes = np.c_[np.zeros(41), np.linspace(-.2,.2,41), np.zeros(41)][None]
    return vertices, nodes, rot

def test_opt_in_recovers_a_valid_plane_between_fast_directions():
    v,n,_ = fixture()
    result = cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    assert result["passed"] and result["method"] == METHOD
    assert result["fast_lower_m"] < -.00015
    assert -.000100001 <= result["lower_m"] <= -.000099999
    assert 0 < result["qp_calls"] <= 40

def test_last_frame_certificate_cannot_accept_earlier_occluded_envelope():
    v,n,rot=fixture()
    history=np.repeat(n,9,axis=0)
    history[:8] += .001*rot[:,0]
    result=cert.rod_envelope_clearance(v,history,max_penetration_m=.00015)
    assert not result["passed"] and result["lower_m"] < -.00015

@pytest.mark.parametrize("budget",[0,1])
def test_budget_exhaustion_rejects_unresolved_groups(budget):
    v,n,_=fixture()
    result=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015,max_qp_calls=budget)
    assert not result["passed"] and result["reason"] == "qp_budget_exhausted"
    assert result["qp_calls"] == budget

def test_failed_qp_is_not_a_geometric_pass(monkeypatch):
    def broken(*a,**kw): raise ValueError("uncertified")
    monkeypatch.setattr(cert,"segment_hull_bounds",broken)
    v,n,_=fixture()
    result=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    assert not result["passed"] and result["reason"] == "distance_certificate_failed"
    assert result["error"] == "uncertified"

@pytest.mark.parametrize("x",[.0047,-.001])
def test_true_collision_is_rejected_even_if_optimizer_succeeds(x):
    v,n,_=fixture(x)
    result=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    assert not result["passed"]

def test_translating_entire_geometry_preserves_proof():
    v,n,_=fixture()
    a=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    t=np.array([2.,-3.,.7])
    b=cert.rod_envelope_clearance(v+t,n+t,max_penetration_m=.00015)
    assert a["passed"] == b["passed"]
    assert abs(a["lower_m"]-b["lower_m"]) < 1e-10

@pytest.mark.parametrize("change",["nan","wrong_nodes","too_many_frames","looser_limit","budget41"])
def test_invalid_contract_cannot_enter_refinement(change):
    v,n,_=fixture(); options=dict(max_penetration_m=.00015)
    if change=="nan": n[0,0,0]=np.nan
    elif change=="wrong_nodes": n=n[:,:40]
    elif change=="too_many_frames": n=np.repeat(n,10,axis=0)
    elif change=="looser_limit": options["max_penetration_m"]=.000151
    else: options["max_qp_calls"]=41
    with pytest.raises(ValueError):
        cert.rod_envelope_clearance(v,n,**options)

def test_native_cooked_guard_requires_explicit_version_and_keeps_legacy_rejection():
    v,n,_=fixture(); hulls={"right.finger23_link":v}
    poses={"right.finger23_link":(np.zeros(3),np.array([0.,0.,0.,1.]))}
    with pytest.raises(ValueError,match="continuous"):
        validate_live_continuous_hulls(hulls,poses,n,max_penetration_m=.00015)
    audits={}
    validate_live_continuous_hulls(hulls,poses,n,max_penetration_m=.00015,
                                  envelope_refinement=METHOD,record_refinement=audits.update)
    assert audits["right.finger23_link"]["passed"]
    with pytest.raises(ValueError,match="version"):
        validate_live_continuous_hulls(hulls,poses,n,max_penetration_m=.00015,
                                      envelope_refinement=True)

def test_source_guard_refines_only_selected_pad_and_keeps_support_guard():
    v,n,_=fixture(); tube=archived_tube(n)
    assert check_supported_shape(v,[],tube,contact_shape=True,contact_penetration_m=.00015)
    logs=[]
    assert check_supported_shape(v,[],tube,contact_shape=True,contact_penetration_m=.00015,
        reference_nodes=n,envelope_refinement=METHOD,record_refinement=logs.append) is None
    assert logs[0]["passed"]
    fail=check_supported_shape(v,[],tube,contact_shape=False,reference_nodes=n,
                              envelope_refinement=METHOD)
    assert "rod_clearance" in fail["conditions"]
    boxes=[("support",np.array([-.1,-.1,-.1]),np.array([.1,.1,.1]))]
    fail=check_supported_shape(v,boxes,tube,contact_shape=True,contact_penetration_m=.00015,
                              reference_nodes=n,envelope_refinement=METHOD)
    assert "support_clearance" in fail["conditions"]

def test_refined_source_uses_full_nodes_not_unrelated_precomputed_tube():
    v,n,rot=fixture(); unrelated=archived_tube(n+np.array([0.,0.,10.]))
    history=np.repeat(n,9,axis=0); history[:8]+=.001*rot[:,0]
    fail=check_supported_shape(v,[],unrelated,contact_shape=True,contact_penetration_m=.00015,
                              reference_nodes=history,envelope_refinement=METHOD)
    assert "contact_rod_penetration" in fail["conditions"]

def test_refinement_code_is_a_frozen_contact_dependency():
    from scripts.screen_dh116_902_flexible_contact import contact_dependency_files
    assert "audit_dh116_certified_segment_distance.py" in [p.name for p in contact_dependency_files()]


def probe_fixture():
    from scripts.verify_dh116_supported_distal_transfer import SupportedDistalProbe
    v,n,_=fixture()
    name="right.finger23_link"
    binding={name:dict(side="right",body_index=1,body_name="finger23_link",prim_path="/World/right/finger23_link")}
    probe=SupportedDistalProbe(408,{name:[2]},binding,[(1,1,v)],grids={name:None},
        archive={},max_pad_penetration_m=.00015,continuous_cooked_hulls={name:v},
        envelope_refinement=METHOD)
    result=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    return probe,n,{name:result}

def test_probe_never_promotes_geometry_without_current_full_source_and_cooked_evidence():
    probe,n,a=probe_fixture()
    with pytest.raises(ValueError,match="envelope"):
        probe.consume_envelope(409)
    for patches in [dict(rod_source_sdk=407),dict(cooked={}),dict(source={})]:
        args=dict(sdk=409,rod_source_sdk=408,reference_nodes=n,source=a,cooked=a)
        args.update(patches)
        with pytest.raises(ValueError,match="envelope"):
            probe.bind_envelope(**args)
    probe.bind_envelope(sdk=409,rod_source_sdk=408,reference_nodes=n,source=a,cooked=a)
    with pytest.raises(ValueError,match="envelope"):
        probe.consume_envelope(410)
    record=probe.consume_envelope(409)
    assert record["observed_sdk_step"]==409 and record["rod_source_physics_step"]==408
    assert record["phase"]=="sdk_complete_rod_pending" and len(record["rod_nodes_sha256"])==64
    with pytest.raises(ValueError,match="envelope"):
        probe.consume_envelope(409)

def test_old_v1_request_cannot_silently_enable_refinement_or_66_windows():
    from scripts.screen_dh116_902_flexible_contact import checked_request,independent_entry
    request=dict(version="dh116-902-distal-contact-v1",grasp="tripod",contact_side="right",
                 control_windows=24,hold_windows=0,max_penetration_m=.00015,
                 source_sdk=408,source_outer_sha256="a"*64)
    checked_request(request,grasp="tripod",sdk=408,outer_sha="a"*64)
    for patch in [dict(envelope_refinement=METHOD),dict(control_windows=66)]:
        with pytest.raises(ValueError):
            checked_request(dict(request,**patch),grasp="tripod",sdk=408,outer_sha="a"*64)
    with pytest.raises(ValueError,match="refinement"):
        independent_entry("tripod",dict(entry_contract=request["version"],envelope_refinement=METHOD))


@pytest.mark.parametrize("cooked_failure",[False,True])
def test_real_measure_links_binds_post_sdk_envelope_and_observe_consumes_once(monkeypatch,cooked_failure):
    from types import SimpleNamespace
    from scripts import verify_dh116_power_shared_rod as shared
    from scripts import verify_dh116_supported_distal_transfer as transfer
    from go2_pvcnn.control.dh116 import tabletop_batch_geometry as geometry
    probe,n,_=probe_fixture(); v,_,_=fixture()
    name="right.finger23_link"
    native_poses=np.zeros((1,23,7));native_poses[:,:,6]=1.
    local_com=native_poses.copy()
    bindings={}
    for side in ("left","right"):
        names=["link"+str(i) for i in range(23)];names[1]="finger23_link"
        paths=["/World/"+side+"/"+x for x in names]
        view=SimpleNamespace(get_link_transforms=lambda:native_poses.copy(),
                             get_coms=lambda:local_com.copy(),link_paths=[paths])
        bindings[side]=SimpleNamespace(hand=SimpleNamespace(root_physx_view=view,body_names=names))
    meshes={name:dict(side="right",body_name="finger23_link",body_index=1,
                      prim_path="/World/right/finger23_link")}
    shapes=[(1,1,v)]
    monkeypatch.setattr(shared.native,"clock",lambda sim:dict(sdk_step=409))
    monkeypatch.setattr(geometry,"collision_clearance",lambda *a:(np.full((1,2),.1),None))
    monkeypatch.setattr(transfer,"validate_live_pad_gaps",lambda *a,**kw:None)
    support=dict(supports=[dict(name="far",center_m=[0.,0.,-1.],size_m=[.1,.1,.1])])
    if cooked_failure:
        def failed(*args,**kw):
            kw["record_refinement"]({name:dict(passed=False,reason="test_uncertified")})
            raise ValueError("cooked proof failed")
        monkeypatch.setattr(transfer,"validate_live_continuous_hulls",failed)
        logs=[]
        def record_guard(row):
            # Mirrors run_supported_action: callback owns the actual rod source tick.
            logs.append(dict(**row,rod_source_physics_step=408))
        with pytest.raises(ValueError,match="cooked proof"):
            shared.measure_links(None,bindings,meshes,shapes,sdk=409,support_scene=support,
                                 reference_nodes=n,distal_probe=probe,rod_source_sdk=408,
                                 record_guard=record_guard)
        assert logs[0]["cooked_envelope"][name]["reason"]=="test_uncertified"
        assert logs[0]["source_envelope"][name]["passed"]
        assert not probe._envelope_pending
        return
    # Pre-step read has no pending observation; do not pollute the next tick.
    shared.measure_links(None,bindings,meshes,shapes,sdk=409,support_scene=support,
                         reference_nodes=n,distal_probe=probe,rod_source_sdk=409)
    assert not probe._envelope_pending
    shared.measure_links(None,bindings,meshes,shapes,sdk=409,support_scene=support,
                         reference_nodes=n,distal_probe=probe,rod_source_sdk=408)
    assert set(probe._envelope_pending)=={409}
    monkeypatch.setattr(transfer,"measure_supported_distal_contact",
        lambda *a,**kw:dict(bilateral=False,wrong_side_contacts=[],
                           per_hand={s:dict(loaded=False) for s in ("left","right")}))
    def row(sdk):
        return dict(physics_step=sdk,reaction_source_physics_step=sdk-1,issued_composer_wrench={},
                    rod=dict(hand_wrenches_com_world={},rod_mesh_contact_pairs=dict(pairs=[])))
    probe.observe(row(408))
    evidence,reason=probe.observe(row(409))
    assert reason is None
    audit=evidence["continuous_envelope"]
    assert set(audit["source"])==set(audit["cooked"])=={name}
    assert audit["rod_source_physics_step"]==408 and audit["observed_sdk_step"]==409
    assert not probe.summary()["teacher_admission"]
    with pytest.raises(ValueError,match="envelope"):
        probe.observe(row(409))


def test_overflow_cannot_turn_empty_failed_group_list_into_pass():
    v,n,_=fixture()
    v[:]=np.finfo(float).max
    with np.errstate(over="ignore",invalid="ignore"):
        with pytest.raises(ValueError,match="finite"):
            cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)

def test_opt_in_contact_guard_requires_its_original_penetration_limit():
    v,n,_=fixture()
    with pytest.raises(ValueError,match="limit"):
        check_supported_shape(v,[],archived_tube(n),contact_shape=True,
                              reference_nodes=n,envelope_refinement=METHOD)


def bent_group_fixture(distance=.004853):
    # All original groups remain; group0 is a triangle whose closest support
    # feature is the endpoint chord, not either adjacent physical segment.
    rot=Rotation.from_euler("z",1.,degrees=True).as_matrix()
    v=np.array([[x,y,z] for x in (distance,distance+.002)
                for y in (-.0001,.0001) for z in (-.001,.001)])@rot.T
    n=np.c_[np.zeros(41),np.linspace(0.,.4,41),np.zeros(41)]
    n[:3]=[[0.,-.001,0.],[-.001,0.,0.],[0.,.001,0.]]
    return v,(n@rot.T)[None],rot


def test_whole_bent_convex_group_recovers_analytic_support_plane():
    v,n,_=bent_group_fixture()
    result=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    # Independent analytic closest plane is x=.004853 before rigid rotation:
    # the full group reaches x=0, y=0 via its endpoint convex combination.
    assert result["passed"]
    assert -.000147001<=result["lower_m"]<=-.000146999
    assert 2<result["qp_calls"]<=40
    full=[row for row in result["witnesses"] if row["segment"]=="full_group_convex_hull"]
    assert len(full)==1 and full[0]["source_node_count"]==3
    assert 0<=full[0]["group_upper_m"]-full[0]["support_lower_m"]<=1e-9


def test_whole_group_fallback_respects_shared_qp_budget_and_full_history():
    v,n,rot=bent_group_fixture()
    exhausted=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015,max_qp_calls=2)
    assert not exhausted["passed"] and exhausted["qp_calls"]==2
    assert exhausted["reason"]=="qp_budget_exhausted"
    history=np.repeat(n,9,axis=0)
    history[:8]+=.0002*rot[:,0]
    full=cert.rod_envelope_clearance(v,history,max_penetration_m=.00015)
    assert not full["passed"] and full["lower_m"]<-.00015
    assert full["witnesses"][-1]["source_node_count"]==27
    assert full["witnesses"][-1]["source_frames"]==list(range(9))


def test_whole_group_collision_cannot_use_safe_adjacent_segment_upper_bound():
    v,n,_=bent_group_fixture(.00484)
    result=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    assert not result["passed"] and result["qp_calls"]==3
    assert result["witnesses"][-1]["group_upper_m"]<-.00015

def test_uncertified_whole_group_cannot_enable_acceptance(monkeypatch):
    v,n,_=bent_group_fixture()
    monkeypatch.setattr(cert,"_group_envelope_bounds",lambda *a:None,raising=False)
    result=cert.rod_envelope_clearance(v,n,max_penetration_m=.00015)
    assert not result["passed"] and result["reason"]=="full_envelope_unresolved"
