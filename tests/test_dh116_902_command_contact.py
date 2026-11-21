"""Versioned actual-command contact route, without advancing Isaac."""
from copy import deepcopy
import importlib
import numpy as np
import pytest

def api():
    return importlib.import_module("scripts.screen_dh116_902_command_contact")

def request():
    return dict(version="dh116-902-command-contact-v2",grasp="tripod",contact_side="right",
        source_sdk=408,source_outer_sha256="a"*64,control_windows=66,hold_windows=0,
        max_penetration_m=.00015,support_reserve_m=.001,
        envelope_refinement="dh116-continuous-envelope-support-v1",
        index_delay_windows=16,temporary_relief_rad={"8":.001,"9":.0005},
        target=dict(pose_wxyz=[.4,-.12,.45,1.,0.,0.,0.],active_q_rad=[.3,.1,.7,.8,.25,.25]))

def test_versioned_recipe_preserves_original_limits_and_excludes_unbounded_budget():
    m=api();m.checked_request(request(),grasp="tripod",sdk=408,outer_sha="a"*64)
    for change in [dict(control_windows=67),dict(hold_windows=1),dict(max_penetration_m=.0002),
                   dict(support_reserve_m=0.),dict(envelope_refinement=True),dict(source_sdk=400),
                   dict(contact_side="left"),dict(temporary_relief_rad={"7":.001}),
                   dict(index_delay_windows=66),dict(version="dh116-902-distal-contact-v1")]:
        with pytest.raises(ValueError):
            m.checked_request(dict(request(),**change),grasp="tripod",sdk=408,outer_sha="a"*64)

def test_schedule_preserves_endpoint_held_hand_and_original_action_range():
    m=api();a=np.zeros((66,24));a[:,8]=.02;a[:,9]=.01;a[:,15]=.01
    b=m.apply_schedule(a,request())
    np.testing.assert_allclose(b.sum(0),a.sum(0),atol=1e-12)
    assert np.max(abs(b))<=1.
    assert np.all(b[:16,8]==0.)
    assert np.all(b[:,list(range(6))+list(range(12,15))+list(range(18,21))]==0.)
    with pytest.raises(ValueError):
        m.apply_schedule(np.ones((66,24)),request())

def passing_proof():
    row=dict(shape_count=22,noncontact_rod_clearance_m=.001,support_clearance_m=.0012,
             table_clearance_m=.01,contact_rod_clearance_m=-.0001,
             pad_gaps_m=[-.00005]*3,node_ids=[2,3,4],opposition=[-.8,-.9],
             all_envelopes_certified=True)
    return dict(version="dh116-902-command-contact-v2",grasp="tripod",contact_side="right",
        control_windows=66,before_sdk=408,after_sdk=936,
        scope="ideal_float32_command_tracking_against_archived_envelope",
        samples=[dict(ordinal=i+1,hand_rows={"right":deepcopy(row),
            "left":dict(shape_count=22,noncontact_rod_clearance_m=.001,
                        support_clearance_m=.0012,table_clearance_m=.01)}) for i in range(528)],
        endpoint_error=dict(position_m=0.,rotation_rad=0.,joint_rad=0.),
        teacher_admission=False,native_contact_verified=False)

def test_complete_proof_rejects_missing_steps_wrong_side_and_midpath_violation():
    m=api();p=passing_proof();assert m.route_passed(p,request())
    variants=[]
    a=deepcopy(p);a["samples"].pop(200);variants.append(a)
    a=deepcopy(p);a["samples"][100]["ordinal"]=100;variants.append(a)
    a=deepcopy(p);a["samples"][381]["hand_rows"]["right"]["support_clearance_m"]=.0005;variants.append(a)
    a=deepcopy(p);a["samples"][381]["hand_rows"]["right"]["all_envelopes_certified"]=False;variants.append(a)
    a=deepcopy(p);a["samples"][120]["hand_rows"]["left"]["noncontact_rod_clearance_m"]=.0002;variants.append(a)
    a=deepcopy(p);a["samples"][-1]["hand_rows"]["right"]["node_ids"]=[21,22,23];variants.append(a)
    a=deepcopy(p);a["samples"][-1]["hand_rows"]["right"]["pad_gaps_m"]=[.0001]*3;variants.append(a)
    a=deepcopy(p);a["endpoint_error"]["joint_rad"]=.01;variants.append(a)
    for a in variants: assert not m.route_passed(a,request())

def test_station_whitelist_is_bound_to_every_command_sample():
    m=api();p=passing_proof()
    assert m.stations(p)=={"right.finger14_link":[2],"right.finger23_link":[3],"right.finger33_link":[4]}
    p["samples"][200]["hand_rows"]["right"]["node_ids"]=[3,3,4]
    assert m.stations(p)["right.finger14_link"]==[2,3]

def test_sat_refinement_preserves_real_separation_without_changing_support():
    m=api()
    # A thin slanted box: AABB almost touches, although a separating diagonal has margin.
    from scipy.spatial.transform import Rotation
    v=np.array([[x,y,z] for x in (-.02,.02) for y in (-.001,.001) for z in (-.01,.01)])
    v=v@Rotation.from_euler("z",45,degrees=True).as_matrix().T
    v+=np.array([.03,.03,0.])
    box=(np.array([-.01,-.01,-.01]),np.array([.01,.01,.01]))
    a=m.support_bound(v,*box,reserve_m=.001)
    assert a["lower_m"] > .001 and a["legacy_lower_m"] > 0.
    b=m.support_bound(v-v.mean(0),*box,reserve_m=.001)
    assert b["lower_m"]<=0.


def test_existing_cpu_entry_dispatches_v2_without_rewriting_v1(tmp_path,monkeypatch):
    from scripts import screen_dh116_902_flexible_contact as old
    m=api();r=tmp_path/"request.json";p=tmp_path/"proof.json"
    import json
    r.write_text(json.dumps(request()));p.write_text("{}")
    monkeypatch.setattr(m,"prepare",lambda *a,**kw:("new-proof","actions","contact","files"))
    assert old.prepare(tmp_path/"receipt.json",r,expected_grasp="tripod")[0]=="new-proof"
    monkeypatch.setattr(m,"load_stage",lambda *a,**kw:("new-actions","contact","files"))
    assert old.load_stage(tmp_path/"receipt.json",r,p,"tripod",before_sdk=408)[0]=="new-actions"
    monkeypatch.setattr(m,"independent_entry",lambda task,contact: task=="tripod")
    assert old.independent_entry("tripod",dict(entry_contract=request()["version"]))

def test_float32_command_compilation_tracks_original_window_endpoints():
    m=api()
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    initial=np.array([.08079639077186584,.0788518488407135,.6594340801239014,
                      .8413524031639099,.2500137388706207,.25001564621925354])
    target=np.array([.34626287970199243,.00801263221802571,.676515080182161,
                     .7983430915979037,.25001397728919983,.25001591444015503])
    scale=8/480*ACTION_CONTRACT["joint_rate_rad_s"]
    base=np.zeros((66,24));base[:,6:12]=(target-initial)/(66*scale)
    original=m.apply_schedule(base,request())
    compiled=m.quantized_window_commands(original,initial)
    q=initial.copy();desired=initial.copy();uncorrected=initial.copy()
    for ideal,actual in zip(original,compiled):
        desired+=ideal[6:12]*scale
        for _ in range(8):
            q=(q+actual[6:12]*scale/8).astype(np.float32).astype(float)
            uncorrected=(uncorrected+ideal[6:12]*scale/8).astype(np.float32).astype(float)
        assert np.max(abs(q-desired)) < 1e-6
    assert np.max(abs(uncorrected-target)) > 1e-5
    assert np.max(abs(q-target)) < 1e-6
    assert np.array_equal(compiled[:,:6],original[:,:6])
    assert np.array_equal(compiled[:,12:],original[:,12:])
    assert np.max(abs(compiled))<=1.
    # Sub-ULP compensation cannot turn this into an arbitrary joint controller.
    assert np.max(abs(compiled-original))*scale < 1e-6

def test_float32_command_compiler_rejects_nonfinite_and_out_of_range():
    m=api();a=np.zeros((66,24));q=np.ones(6)*.5
    for bad in [np.full((66,24),np.nan),np.ones((66,24))*1.1,np.zeros((65,24))]:
        with pytest.raises(ValueError): m.quantized_window_commands(bad,q)
    with pytest.raises(ValueError):m.quantized_window_commands(a,np.full(6,np.nan))

def sealed_contact():
    m=api()
    from scripts.diagnose_dh116_teacher_initialization import digest
    p=passing_proof();req=request()
    archive=dict(physics_step=408,guard_failure=None,
        rod=dict(physics_step=408,rod_tick=(408-2)*12),sdk_hand_geometry=dict(physics_step=408))
    p.update(outer_sha256="a"*64,all_passed=True,archive_digest=digest(archive),
        stations=m.stations(p),continuous_cooked_hulls={"right."+pad:[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        for pad in m.PADS})
    return m._contact(p,req,archive)

@pytest.mark.parametrize("mutation",["clock","source","station","cooked","step","wrong_side"])
def test_runtime_entry_rejects_mixed_identity_proofs(mutation):
    m=api();c=sealed_contact();assert m.independent_entry("tripod",c)
    if mutation=="clock":c["archive"]["rod"]["physics_step"]=407
    if mutation=="source":c["command_request"]["source_outer_sha256"]="b"*64
    if mutation=="station":c["stations"]=dict(c["stations"],**{"right.finger14_link":[19]})
    if mutation=="cooked":c["continuous_cooked_hulls"]=dict(c["continuous_cooked_hulls"],**{"right.finger14_link":[[0.,0.,0.]]})
    if mutation=="step":c["command_proof"]["samples"].pop(400)
    if mutation=="wrong_side":c["contact_side"]="left"
    with pytest.raises(ValueError):m.independent_entry("tripod",c)

@pytest.mark.parametrize("field",["request_sha256","outer_sha256","archive_digest","source_sha256","action_sha256"])
def test_frozen_loader_rejects_changed_seals(tmp_path,monkeypatch,field):
    import json
    m=api()
    from scripts import screen_dh116_902_flexible_contact as old
    from scripts import verify_dh116_902_flexible_reset as reset
    from scripts.diagnose_dh116_teacher_initialization import digest
    c=sealed_contact();proof=c["command_proof"];raw=c["archive"]
    receipt=tmp_path/"outer.json";reqfile=tmp_path/"request.json";pf=tmp_path/"proof.json"
    receipt.write_text("{}")
    req=dict(request(),source_outer_sha256=reset.sha(receipt))
    reqfile.write_text(json.dumps(req))
    actions=np.zeros((66,24))
    monkeypatch.setattr(old,"boundary",lambda *a:([raw],[],(),408))
    monkeypatch.setattr(reset,"load_candidate",lambda *a:({}, {}, ()))
    monkeypatch.setattr(m,"compile_actions",lambda *a:actions)
    monkeypatch.setattr(m,"dependency_files",lambda:())
    proof.update(request_sha256=reset.sha(reqfile),outer_sha256=reset.sha(receipt),
                 archive_digest=digest(raw),source_sha256={},action_sha256=digest(actions.tolist()))
    pf.write_text(json.dumps(proof))
    assert m.load_stage(receipt,reqfile,pf,"tripod",before_sdk=408)[0].shape==(66,24)
    proof[field]={"stale":"b"*64} if field=="source_sha256" else "b"*64
    pf.write_text(json.dumps(proof))
    with pytest.raises(ValueError):m.load_stage(receipt,reqfile,pf,"tripod",before_sdk=408)

def test_v2_native_parent_import_is_clean_in_fresh_process():
    import os,sys,subprocess
    result=subprocess.run([sys.executable,"-B","-c",
        "from scripts import screen_dh116_902_command_contact; "
        "from scripts import screen_dh116_902_flexible_contact; "
        "from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary; "
        "assert_clean_import_boundary()"],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
        text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
