from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116 import tabletop_measured_closure_geometry as m


@pytest.fixture(scope="module")
def owner():
    from pathlib import Path
    return m.MeasuredClosureGeometry(Path(__file__).resolve().parents[1])


@pytest.fixture
def case(owner):
    from scripts.prepare_dh116_dual_pinch_readback_profile import load_native
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding,_capture
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
    rows,initial,_=load_native();exchange=deepcopy(rows[-1]["exchange"])
    clock={k:exchange["rod"][k] for k in m.CLOCK_FIELDS}
    sim=SimpleNamespace(current_time_step_index=clock["physics_step"],current_time=clock["timestamp_s"])
    bindings=[];hands=[]
    for side in m.SIDES:
        old=exchange["action_readback"]["post"]["hands"][side];g=initial[side]
        h=SimpleNamespace(joint_names=old["joint_names"],body_names=g["actual_body_names"],device="cpu")
        h.q=old["q"].copy();h.qd=np.zeros_like(h.q);h.targets=old["backend_targets"].copy()
        h.transforms=np.array(g["actual_link_poses_xyzw"],dtype=np.float32)
        h.velocities=np.zeros((1,23,6),dtype=np.float32)
        h.root=np.array(old["native_anchor_pose_w"][:,[0,1,2,4,5,6,3]],dtype=np.float32)
        h.data=SimpleNamespace(joint_pos=torch.from_numpy(h.q),joint_vel=torch.from_numpy(h.qd),
            joint_pos_target=torch.from_numpy(h.targets),root_pose_w=torch.from_numpy(old["native_anchor_pose_w"]))
        h.root_physx_view=SimpleNamespace(prim_paths=old["prim_paths"],link_paths=g["actual_link_paths"],
            get_dof_positions=lambda h=h:h.q,get_dof_velocities=lambda h=h:h.qd,
            get_dof_position_targets=lambda h=h:h.targets,get_dof_types=lambda old=old:old["dof_types"],
            get_root_transforms=lambda h=h:h.root,get_link_transforms=lambda h=h:h.transforms,
            get_link_velocities=lambda h=h:h.velocities)
        bindings.append(HandBinding(side,h,{clock["env_id"]:old["prim_paths"][0]},MountView(h)));hands.append(h)
    read=lambda:deepcopy(clock)
    ident=dict(clock,policy_step=exchange["policy_step"],substep=exchange["substep"])
    exchange["action_readback"]["post"]=_capture(bindings,(clock["env_id"],),lambda:[ident])
    return exchange,sim,bindings,read,hands,clock


def capture(owner,case):
    row,sim,bindings,read,*_=case
    return owner.capture(row,sim=sim,bindings=bindings,read_clock=read)


def test_real_source_geometry_cpu_sdk_double_roundtrip(owner,case):
    p=capture(owner,case)
    assert p["passed"] and p["terminal_reason"] is None
    assert all(len(s["table_clearance_m"])==22 for s in p["sides"].values())
    assert owner.validate(deepcopy(p),case[0])["passed"]
    assert p["new_sdk_steps"]==0 and not p["native_qualification"]
    before=p["raw"]["hands"]["left"]["link_poses_xyzw"].copy()
    case[4][0].transforms+=1
    np.testing.assert_array_equal(before,p["raw"]["hands"]["left"]["link_poses_xyzw"])


def test_subfloor_actual_source_returns_terminal_packet(owner,case):
    h=case[4][0];i=h.body_names.index("finger22_link")
    h.transforms[0,i,2]-=.0001
    p=capture(owner,case)
    assert not p["passed"] and any("table" in x for x in p["failure_reasons"])
    assert p["terminal_reason"]=="closure_geometry_guard"
    assert p["raw"]["hands"]["left"]["link_poses_xyzw"][0,i,2]==h.transforms[0,i,2]


def test_actual_non_target_rod_overlap_fails_closed(owner,case):
    h=case[4][1];i=h.body_names.index("finger13_link")
    h.transforms[0,i,:3]=[.43,-.2,.305]
    p=capture(owner,case)
    assert not p["passed"] and any("non_target" in x for x in p["failure_reasons"])


def test_required_sdf_outside_all_nodes_rejected(owner,case):
    h=case[4][0];i=h.body_names.index("finger14_link");h.transforms[0,i,0]+=1.
    p=capture(owner,case)
    assert any("sdf_coverage" in x for x in p["failure_reasons"])


def test_required_negative_sdf_is_not_zero_contact_gate():
    assert m.required_sdf_valid(dict(minimum_gap_m=-.001,gradient_norm_at_minimum=1.,in_lattice_nodes=[4]))
    assert not m.required_sdf_valid(dict(minimum_gap_m=None,gradient_norm_at_minimum=None,in_lattice_nodes=[]))


@pytest.mark.parametrize("fault",["clock","order","foreign_link","nan_pose","stale_q"])
def test_invalid_actual_source_refused(owner,case,fault):
    h=case[4][0]
    if fault=="clock":case[5]["physics_step"]+=1
    if fault=="order":h.body_names=list(reversed(h.body_names))
    if fault=="foreign_link":h.root_physx_view.link_paths[0][0]="/Wrong/base_link"
    if fault=="nan_pose":h.transforms[0,0,0]=np.nan
    if fault=="stale_q":h.q[0,0]+=.001
    with pytest.raises(ValueError):capture(owner,case)


def test_wrist_actual_speed_guard_returns_failure(owner,case):
    h=case[4][1];i=h.body_names.index("base_link");h.velocities[0,i,0]=.061
    p=capture(owner,case)
    assert any("wrist" in x for x in p["failure_reasons"])


def test_clock_advance_during_pose_capture_refused(owner,case):
    h=case[4][1]
    def drift():
        case[5]["physics_step"]+=1
        return h.transforms
    h.root_physx_view.get_link_transforms=drift
    with pytest.raises(ValueError):capture(owner,case)


@pytest.mark.parametrize("fault",["summary","qualified","rod","raw_pose"])
def test_disk_replay_rejects_summary_or_source_tampering(owner,case,fault):
    p=capture(owner,case)
    if fault=="summary":p["sides"]["left"]["minimum_table_clearance_m"]+=1
    if fault=="qualified":p["native_qualification"]=True
    if fault=="rod":p["raw"]["rod_nodes_m"][0,0]+=.001
    if fault=="raw_pose":p["raw"]["hands"]["left"]["link_poses_xyzw"][0,0,2]+=.001
    with pytest.raises(ValueError):owner.validate(p,case[0])


def test_disk_replay_rejects_changed_same_row_pad_sdf_binding(owner,case):
    p=capture(owner,case)
    case[0]["rod"]["mesh_bindings"]["left.finger14_link"]["sdf_sha256"]="0"*64
    with pytest.raises(ValueError):owner.validate(p,case[0])


def test_frozen_profile_nine_states_with_real_source_and_sdk_doubles(owner,case):
    from pathlib import Path
    from scripts.prepare_dh116_dual_pinch_readback_profile import load_native,hand_state
    from scripts.audit_dh116_pinch_initial_pose import load_geometry,pose_matrix
    from scripts.prepare_dh116_pinch_closure_margin import fk_poses,relative_pose
    from scipy.spatial.transform import Rotation
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS,expand
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _capture
    root=Path(__file__).resolve().parents[1]
    profile_path=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-readback-profile-20261113-dxcb62it/run02/result.json")
    import hashlib
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest()=="75ec5cee02eca0676d15db8eb07d00273c2b50b40b7dc8d7d16d09d18e17f803"
    saved=json.loads(profile_path.read_text());rows,initial,_=load_native();models={}
    for side in m.SIDES:
        g=load_geometry(root,side=side);pose0,q0,_=hand_state(rows[0]["exchange"]["action_readback"]["pre"]["hands"][side])
        old=fk_poses(g,pose0,q0);init=initial[side]
        actual={n:pose_matrix(np.r_[v[:3],v[6],v[3:6]]) for n,v in zip(init["actual_body_names"],init["actual_link_poses_xyzw"][0])}
        models[side]=(g,old,actual)
    exchange,sim,bindings,read,hands,clock=case
    for index in range(9):
        clock["physics_step"]=10+index;clock["rod_tick"]=96+12*index
        clock["timestamp_s"]=(10+index)/480;clock["solver_time_s"]=(8+index)/480
        sim.current_time_step_index=clock["physics_step"];sim.current_time=clock["timestamp_s"]
        exchange["physics_step"]=clock["physics_step"]
        for k in m.CLOCK_FIELDS:exchange["rod"][k]=clock[k]
        exchange["policy_step"]=1 if index else 0;exchange["substep"]=index-1 if index else 7
        for side,h in zip(m.SIDES,hands):
            d=saved["sides"][side];q=np.array(d["samples"][index]["active_q_rad"])
            g,old,actual=models[side];pred=fk_poses(g,d["post_base_pose_fk_wxyz"],q)
            for name,value in zip(JOINTS,expand(q)):h.q[0,h.joint_names.index(name)]=value
            for j,name in enumerate(m.ACTIVE):h.targets[0,h.joint_names.index(name)]=q[j]
            for name in g["source_colliders"]:
                matrix=relative_pose(actual[name],old[name],pred[name]);i=h.body_names.index(name)
                h.transforms[0,i,:3]=matrix[:3,3];h.transforms[0,i,3:]=Rotation.from_matrix(matrix[:3,:3]).as_quat()
        ident=dict(clock,policy_step=exchange["policy_step"],substep=exchange["substep"])
        exchange["action_readback"]["post"]=_capture(bindings,(clock["env_id"],),lambda:[ident])
        packet=capture(owner,case)
        assert packet["passed"],packet["failure_reasons"]
        assert owner.validate(packet,exchange)["passed"]


def test_guard_failure_row_is_json_lossless_and_required_contact_not_rejected(owner,case):
    h=case[4][0];name="finger14_link";i=h.body_names.index(name)
    grid=owner._grids["left"][name]
    local=np.array(grid["origin"])+float(grid["spacing"])*np.array(np.unravel_index(np.argmin(grid["values"]),grid["values"].shape))
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_quat(h.transforms[0,i,3:])
    node=case[0]["rod"]["node_positions"][0,36]
    h.transforms[0,i,:3]=node-rotation.apply(local)
    packet=capture(owner,case)
    sdf=packet["sides"]["left"]["required_pad_sdf"][name]
    assert sdf["minimum_gap_m"]<0 and m.required_sdf_valid(sdf)
    assert not any("sdf_coverage" in s for s in packet["failure_reasons"])
    decoded=m.decode_owned(json.loads(m._bytes(packet)))
    assert m._bytes(owner.validate(decoded,case[0]))==m._bytes(packet)


def test_module_import_does_not_cross_pre_app_native_boundary():
    import os,subprocess,sys
    code="""
import sys,importlib.abc
denied={'pxr','PhysxSchema','omni','isaacsim','isaaclab','ropesim','warp'}
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in denied:raise RuntimeError('forbidden pre-App import '+fullname)
sys.meta_path.insert(0,Deny())
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import MeasuredClosureGeometry
assert not any(n.split('.')[0] in denied for n in sys.modules)
"""
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),timeout=30)
    assert result.returncode==0,result.stderr
