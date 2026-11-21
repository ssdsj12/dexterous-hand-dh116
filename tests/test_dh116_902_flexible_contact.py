import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.screen_dh116_902_flexible_contact import measured_fk_pose


def rigid(xyz, angles):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    m[:3, 3] = xyz
    return m


def test_native_link_offset_follows_new_link_not_world_axes():
    old = rigid([.5, .2, .48], [.4, -.7, 1.2])
    body_offset = rigid([.002, -.003, .001], [.1, .04, -.08])
    observed = old @ body_offset
    new = rigid([.499, .203, .478], [.7, -.6, .9])
    expected = new @ body_offset
    np.testing.assert_allclose(measured_fk_pose(observed, old, new), expected, atol=1e-12)
    wrong = observed @ np.linalg.inv(old) @ new
    assert np.linalg.norm(wrong[:3, 3] - expected[:3, 3]) > 1e-4


def test_measured_link_prediction_is_covariant_under_world_origin_change():
    old = rigid([.5, .2, .48], [.4, -.7, 1.2])
    observed = old @ rigid([.002, 0., 0.], [.1, .04, -.08])
    new = rigid([.499, .203, .478], [.7, -.6, .9])
    origin = rigid([2., -3., .7], [.2, .3, -.5])
    np.testing.assert_allclose(measured_fk_pose(origin @ observed, origin @ old, origin @ new),
                               origin @ measured_fk_pose(observed, old, new), atol=1e-12)


@pytest.mark.parametrize("bad", [np.full((4,4), np.nan), np.zeros((4,4)), np.eye(3)])
def test_measured_link_prediction_rejects_nonrigid_or_missing_pose(bad):
    with pytest.raises(ValueError, match="rigid"):
        measured_fk_pose(bad, np.eye(4), np.eye(4))


def test_contact_route_keeps_nontarget_table_and_support_guards():
    from scripts.screen_dh116_902_flexible_contact import contact_route_passed
    row = dict(shape_count=22, noncontact_rod_clearance_m=.0006,
               support_clearance_m=.0006, table_clearance_m=.01,
               pad_gaps_m=[-.00005, -.00005], node_ids=[35,35], opposition=[-.8],
               contact_rod_clearance_m=-.00006)
    rows = [dict(row) for _ in range(17)]
    assert contact_route_passed(rows, "left", pad_count=2, max_penetration_m=.00015)
    for key,value in [("noncontact_rod_clearance_m",.00024),
                      ("support_clearance_m",0.),("table_clearance_m",.0020),
                      ("pad_gaps_m",[-.00016,0.]),("node_ids",[20,21]),
                      ("opposition",[.1])]:
        bad=[dict(row) for _ in range(17)];bad[-1][key]=value
        assert not contact_route_passed(bad, "left", pad_count=2, max_penetration_m=.00015)


def test_fixed_wrist_command_uses_backend_target_instead_of_measured_joint():
    from scripts.screen_dh116_902_flexible_contact import fixed_wrist_commands
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    hands = {s:dict(joint_names=list(ACTIVE), active_indices=list(range(6)),
                    q=np.full((1,6),.3),backend_targets=np.full((1,6),.4))
             for s in ("left","right")}
    row=dict(action_readback=dict(post=dict(hands=hands)))
    commands=fixed_wrist_commands(row,{s:np.full(6,.41) for s in hands},windows=10)
    np.testing.assert_allclose(commands[:,:12],.2,atol=1e-12)
    np.testing.assert_array_equal(commands[:,12:],np.zeros((10,12)))
    with pytest.raises(ValueError,match="rate"):
        fixed_wrist_commands(row,{s:np.full(6,.7) for s in hands},windows=1)


def test_independent_contact_entry_requires_own_same_tick_archive_and_source():
    from scripts.screen_dh116_902_flexible_contact import independent_entry
    from scripts.diagnose_dh116_teacher_initialization import digest
    archive=dict(physics_step=408,guard_failure=None,
                 sdk_hand_geometry=dict(physics_step=408),
                 rod=dict(physics_step=408,rod_tick=4872))
    contact=dict(entry_contract="dh116-902-distal-contact-v1",grasp="tripod",
                 boundary_sdk=408,archive=archive,archive_digest=digest(archive))
    stations={f"right.{link}":[i+2] for i,link in enumerate(("finger14_link","finger23_link","finger33_link"))}
    hull=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)]).tolist()
    contact.update(contact_side="right",stations=stations,continuous_cooked_hulls={k:hull for k in stations},
                   max_pad_penetration_m=.00015)
    assert independent_entry("tripod",contact)
    for patch in (dict(continuous_cooked_hulls=None),dict(stations={}),
                  dict(contact_side="left"),dict(max_pad_penetration_m=.0002)):
        with pytest.raises(ValueError,match="continuous"):
            independent_entry("tripod",dict(contact,**patch))
    assert not independent_entry("tripod",dict(grasp="tripod"))
    for bad in (dict(contact,grasp="pinch"),dict(contact,boundary_sdk=400),
                dict(contact,archive_digest="0"*64),dict(contact,boundary_sdk=411)):
        with pytest.raises(ValueError,match="boundary"):
            independent_entry("tripod",bad)


def test_held_hand_uses_full_envelope_guard_even_when_other_hand_contacts():
    from scripts.screen_dh116_902_flexible_contact import held_route_passed
    row=dict(shape_count=22,rod_clearance_m=.0003,table_clearance_m=.01,support_clearance_m=.0003)
    rows=[dict(row) for _ in range(17)]
    assert held_route_passed(rows)
    rows[8]["rod_clearance_m"]=.00024
    assert not held_route_passed(rows)


def test_contact_route_rejects_between_node_penetration_despite_node_contact():
    from scripts.screen_dh116_902_flexible_contact import contact_route_passed
    row=dict(shape_count=22, noncontact_rod_clearance_m=.0006,
        support_clearance_m=.0006, table_clearance_m=.01,
        pad_gaps_m=[-.00005]*3,node_ids=[3,2,4],opposition=[-.87,-.98],
        contact_rod_clearance_m=-.0022249075)
    rows=[dict(row) for _ in range(17)]
    assert not contact_route_passed(rows,"right",pad_count=3,max_penetration_m=.00015)
    for missing in (None,float("nan"),float("-inf")):
        for v in rows:v["contact_rod_clearance_m"]=missing
        assert not contact_route_passed(rows,"right",pad_count=3,max_penetration_m=.00015)
    for v in rows:v.pop("contact_rod_clearance_m")
    assert not contact_route_passed(rows,"right",pad_count=3,max_penetration_m=.00015)


def test_contact_request_and_station_identity_are_bound_to_reviewed_side():
    from scripts.screen_dh116_902_flexible_contact import checked_request, checked_stations
    request=dict(version="dh116-902-distal-contact-v1",grasp="tripod",contact_side="right",
        control_windows=12,hold_windows=6,max_penetration_m=.00015,
        source_sdk=408,source_outer_sha256="a"*64)
    checked_request(request,grasp="tripod",sdk=408,outer_sha="a"*64)
    for patch in (dict(source_sdk=432),dict(source_outer_sha256="b"*64),
                  dict(contact_side="other"),dict(max_penetration_m=.00016),
                  dict(control_windows=True),dict(hold_windows=13),dict(version="legacy")):
        with pytest.raises(ValueError):
            checked_request(dict(request,**patch),grasp="tripod",sdk=408,outer_sha="a"*64)
    samples=[dict(node_ids=[3,2,4]) for _ in range(17)]
    stations={"right.finger14_link":[3],"right.finger23_link":[2],"right.finger33_link":[4]}
    assert checked_stations(samples,"right","tripod",stations)==stations
    for bad in (dict(stations,**{"right.finger33_link":[21]}),
                {key.replace("right","left"):v for key,v in stations.items()}):
        with pytest.raises(ValueError):
            checked_stations(samples,"right","tripod",bad)


def test_live_selected_source_shape_cannot_hide_continuous_rod_penetration():
    from scripts.verify_dh116_power_shared_rod import check_supported_shape
    from scripts.screen_dh116_tripod_supported_reset import archived_tube
    shape=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)])
    nodes=np.c_[np.linspace(-.25,.25,41),np.full(41,.004),np.zeros(41)]
    failure=check_supported_shape(shape,[],archived_tube(nodes[None]),
        contact_shape=True,contact_penetration_m=.00015)
    assert failure["conditions"]==["contact_rod_penetration"]
    assert failure["rod_gap_lower_m"]<-.00015


def test_live_cooked_continuous_guard_detects_missed_segment_between_nodes():
    from scripts.verify_dh116_supported_distal_transfer import validate_live_continuous_hulls
    shape=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)])
    nodes=np.c_[np.linspace(-.25,.25,41)+.00625,np.full(41,.004),np.zeros(41)]
    poses={"right.finger33_link":(np.zeros(3),np.array([0.,0.,0.,1.]))}
    with pytest.raises(ValueError,match="continuous"):
        validate_live_continuous_hulls({"right.finger33_link":shape},poses,nodes[None],max_penetration_m=.00015)
    nodes[:,1]=.0062
    validate_live_continuous_hulls({"right.finger33_link":shape},poses,nodes[None],max_penetration_m=.00015)


def test_contact_stage_roundtrip_and_original_command_isolation(tmp_path, monkeypatch):
    import json
    from scripts import screen_dh116_902_flexible_contact as contact
    from scripts import verify_dh116_902_flexible_reset as reset
    from scripts.diagnose_dh116_teacher_initialization import digest
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    pose=[.5,.1,.4,1.,0.,0.,0.]
    hands={s:dict(joint_names=list(ACTIVE),active_indices=list(range(6)),
        q=np.full((1,6),.3),backend_targets=np.full((1,6),.4)) for s in ("left","right")}
    row=dict(physics_step=408,guard_failure=None,
        sdk_hand_geometry=dict(physics_step=408),rod=dict(physics_step=408,rod_tick=4872),
        action_readback=dict(post=dict(hands=hands)),
        wrist_projection=dict(projected_root_poses=np.array([[pose,pose]],dtype=float),
                              projected_angles=np.zeros((1,2,3))))
    raw=encode_owned(row)
    config=dict(hands={s:dict(pose_wxyz=pose) for s in hands})
    receipt=tmp_path/"outer.json";receipt.write_text("{}")
    request=dict(version="dh116-902-distal-contact-v1",grasp="tripod",contact_side="right",
        source_sdk=408,source_outer_sha256=reset.sha(receipt),max_penetration_m=.00015,
        control_windows=12,hold_windows=6,target=dict(pose_wxyz=pose,active_q_rad=[.41]*6))
    request_path=tmp_path/"request.json";request_path.write_text(json.dumps(request))
    right=dict(shape_count=22,noncontact_rod_clearance_m=.0006,contact_rod_clearance_m=-.0001,
        support_clearance_m=.001,table_clearance_m=.01,node_ids=[3,2,4],
        pad_gaps_m=[-.00005]*3,opposition=[-.8,-.9])
    left=dict(shape_count=22,rod_clearance_m=.0006,support_clearance_m=.001,table_clearance_m=.01)
    samples={"right":[dict(right) for _ in range(17)],"left":[dict(left) for _ in range(17)]}
    stations=contact.checked_stations(samples["right"],"right","tripod")
    hull=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)]).tolist()
    actions=contact.stage_actions(config,row,request)
    proof=dict(version=request["version"],grasp="tripod",contact_side="right",before_sdk=408,
        control_windows=12,hold_windows=6,request_sha256=reset.sha(request_path),
        outer_sha256=reset.sha(receipt),all_passed=True,samples=samples,stations=stations,
        continuous_cooked_hulls={k:hull for k in stations},action_sha256=digest(actions.tolist()))
    proof_path=tmp_path/"proof.json";proof_path.write_text(json.dumps(proof))
    monkeypatch.setattr(contact,"boundary",lambda *args:([raw],[],[],408))
    monkeypatch.setattr(reset,"load_candidate",lambda grasp:({},config,[]))
    tail,binding,files=contact.load_stage(receipt,request_path,proof_path,"tripod",before_sdk=408)
    assert json.loads(json.dumps(binding))==binding
    assert tail.shape==(18,24)
    np.testing.assert_array_equal(tail[:,:6],0.)
    np.testing.assert_array_equal(tail[:,12:],0.)
    np.testing.assert_array_equal(tail[-6:],0.)
    np.testing.assert_allclose(tail[:12,6:12],1/6)
    for key,value in (("contact_side","left"),("hold_windows",7),
                       ("stations",{k:[18] for k in stations}),
                       ("continuous_cooked_hulls",{}),("action_sha256","0"*64)):
        broken=dict(proof,**{key:value});proof_path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            contact.load_stage(receipt,request_path,proof_path,"tripod",before_sdk=408)


def test_contact_parent_import_preserves_clean_kit_boundary():
    import os,subprocess,sys
    result=subprocess.run([sys.executable,"-c",
        "from scripts.screen_dh116_902_flexible_contact import load_stage;"
        "from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary;"
        "assert_clean_import_boundary()"],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
