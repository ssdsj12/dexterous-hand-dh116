import importlib
import numpy as np
import pytest
def module():
    return importlib.import_module("scripts.audit_dh116_902_contact_guard")

def test_segment_penetration_is_not_replaced_by_positive_node_clearance():
    m=module()
    box=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)])
    nodes=np.column_stack([np.zeros(41),np.linspace(-.3,.3,41)+.0075,np.zeros(41)])
    result=m.geometry_bounds(box,nodes)
    assert result["segment_min_upper_m"]<-.00015
    assert result["node_min_lower_m"]>0
    assert result["inter_node_collision_without_node_contact"] is True

def test_geometry_bounds_rigid_transform_invariance():
    from scipy.spatial.transform import Rotation
    m=module();box=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)])
    nodes=np.column_stack([np.ones(41)*.009,np.linspace(-.3,.3,41),np.zeros(41)])
    original=m.geometry_bounds(box,nodes)
    rot=Rotation.from_euler("xyz",[.2,.4,-.5]).as_matrix();shift=np.array([1.2,-.6,.4])
    moved=m.geometry_bounds(box@rot.T+shift,nodes@rot.T+shift)
    assert abs(original["segment_min_lower_m"]-moved["segment_min_lower_m"])<1e-8

def test_guard_audit_refuses_unsealed_or_forced_outer(tmp_path):
    import json
    m=module();p=tmp_path/"outer.json"
    for outer in [dict(exit_code=0,forced_exit=False),dict(exit_code=1,forced_exit=True),
                  dict(exit_code=1,forced_exit=False,debugger_inferior_signal="SIGSEGV")]:
        p.write_text(json.dumps(outer))
        with pytest.raises(ValueError):m.audit(p,"tripod_geometry_guard_476.json")

def sealed_case(tmp_path, *, failures=True):
    import json, hashlib
    box=[[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)]
    nodes=np.column_stack([np.zeros(41),np.linspace(-.3,.3,41)+.0075,np.zeros(41)]).tolist()
    item=dict(side="right",body_name="finger33_link",link_prim="/World/DeformXDualHand/rightHand/finger33_link",
              pose_world_wxyz=[0,0,0,1,0,0,0],source_vertices_local_m=box,conditions={})
    guard=dict(observed_sdk_step=4,rod_source_physics_step=3,current_sdk_clock=dict(sdk_step=4),
               rod_node_positions_world_m=[nodes],failures=[item] if failures else [])
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
    q=expand([.2]*6)
    hand=dict(joint_names=list(JOINTS),active_indices=[JOINTS.index(x) for x in ACTIVE],
              q=[q.tolist()],qd=[np.zeros(11).tolist()],backend_targets=[np.zeros(11).tolist()])
    previous=dict(physics_step=3,guard_failure=None,action_readback=dict(post=dict(hands={"left":hand,"right":hand})),
        rod=dict(physics_step=3,rod_tick=12,node_positions=[nodes],hand_wrenches_com_world={
            s:dict(forces=[[0,0,0]],torques=[[0,0,0]]) for s in ("left","right")},
            rod_mesh_contact_pairs=dict(pairs=[])))
    # Cooked hull displaced; same source-only violation must not be called actual contact.
    cooked=(np.asarray(box)+[.03,0,0]).tolist()
    plan=dict(distal_contact=dict(continuous_cooked_hulls={"right.finger33_link":cooked},
        command_proof=dict(continuous_cooked_hulls={"right.finger33_link":cooked})))
    native=tmp_path/"native";native.mkdir()
    artifacts={"tripod_geometry_guard_4.json":guard,"tripod_control_3.json":previous,"frozen_plan.json":plan}
    hashes={}
    for name,data in artifacts.items():
        p=native/name;p.write_text(json.dumps(data));hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=tmp_path/"outer.json";receipt.write_text(json.dumps(dict(exit_code=1,forced_exit=False,exceeded_budget=False,
        debugger_status_verified=True,debugger_inferior_signal=None,debugger_inferior_exit_code=1,artifact_sha256=hashes)))
    return receipt

def test_sealed_source_rejection_is_not_cooked_contact(tmp_path):
    result=module().audit(sealed_case(tmp_path),"tripod_geometry_guard_4.json")
    f=result["failures"][0]
    assert f["geometry"]["certified_contact_limit_violation"]
    assert f["cooked_geometry"]["segment_min_lower_m"]>0
    assert f["source_only_rejection_not_cooked_contact"] is True
    assert result["teacher_admission"] is False
    assert result["tracking_source_sdk"]==3

def test_empty_failures_cannot_produce_successful_audit(tmp_path):
    with pytest.raises(ValueError,match="nonempty"):
        module().audit(sealed_case(tmp_path,failures=False),"tripod_geometry_guard_4.json")

def test_tampered_native_artifact_rejected(tmp_path):
    receipt=sealed_case(tmp_path)
    with (tmp_path/"native/tripod_control_3.json").open("a") as f:f.write(" ")
    with pytest.raises(ValueError,match="SHA"):
        module().audit(receipt,"tripod_geometry_guard_4.json")
