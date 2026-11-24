from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_dh116_deformable_bimanual_step import setup,begin
from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedSourceIdentity
from go2_pvcnn.control.dh116.tabletop_tensor_control import TensorControl
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _capture
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets

def prepare():
    hands,bindings,clocks,read,old=setup()
    pre=_capture(bindings,(3,8),read)
    state={k:torch.stack([torch.tensor(pre["hands"][s][k]) for s in ("left","right")],1)
           for k in ("q","qd","backend_targets")}
    roots=torch.stack([torch.tensor(pre["hands"][s]["mount_anchor_pose_w"]) for s in ("left","right")],1)
    state["root_pose_wxyz"]=roots
    batch=SimpleNamespace(sides={s:SimpleNamespace(joint_names=h.joint_names) for s,h in zip(("left","right"),hands)})
    control=TensorControl(batch,roots,{"left.pad":dict(side="left",body_index=0),"right.pad":dict(side="right",body_index=0)},("left.pad","right.pad","table"))
    stamp=SharedSourceIdentity(1,2,(3,8),"run",("a","b"),("rod3","rod8"),(1,1),(0,0),2,2/480,1/480,1/480,2/480,4.,"a"*64,"cpu_fixture")
    return old,bindings,read,state,roots,control,stamp

def test_tensor_control_prepares_original_all24_and_wrist_projection_exactly():
    old,bindings,read,state,roots,control,stamp=prepare()
    action=torch.linspace(-1,1,48,dtype=torch.float64).reshape(2,24)
    zero=torch.zeros(2,2,3,dtype=torch.float64);clearance=torch.full((2,2),.0022,dtype=torch.float64)
    request=old.begin(bindings,action,read_clock=read,palms=zero,angles=zero)
    expected=project_policy_wrist_targets(root_seed_w=roots,measured_base_w=roots,
        common_lift_offset_m=torch.zeros(2,dtype=torch.float64),measured_clearance_m=clearance,
        palm_offsets_m=zero,requested_palm_offsets_m=request["requested_palm_offsets_m"],
        angles_rad=zero,requested_angles_rad=request["requested_angles_rad"],
        allowed=torch.ones(2,dtype=torch.bool),live=torch.ones(2,dtype=torch.bool))
    actual=control.prepare(action,state,clearance,roots,stamp=stamp)
    for s in (0,1):
        torch.testing.assert_close(actual["expected"][s][:,:6],request["requested_finger_targets_rad"][:,6*s:6*s+6].float(),rtol=0,atol=0)
    for k,v in expected.items():
        if isinstance(v,torch.Tensor):torch.testing.assert_close(actual["projection"][k],v,rtol=0,atol=0)

@pytest.mark.parametrize("fault",["target","clock","reset","rod_tick"])
def test_tensor_control_finish_rejects_same_original_faults(fault):
    control,stamp,state,tr,w,_=submission_fixture()
    control.submit(tr,stamp=stamp,wrenches=w,state=state,reaction_source=reaction_identity(stamp,w))
    for s in (0,1):state["backend_targets"][:,s,control.write_ids[s]]=tr["expected"][s]
    after=replace(stamp,sdk_step=3,rod_ticks=(12,12),solver_time_s=4+1/480)
    if fault=="target":state["backend_targets"][0,0,control.write_ids[0][0]]+=1e-5
    if fault=="clock":after=replace(after,sdk_step=4)
    if fault=="reset":after=replace(after,reset_ids=(2,1))
    if fault=="rod_tick":after=replace(after,rod_ticks=(24,12))
    with pytest.raises(ValueError):control.finish(tr,state,stamp=after)

def submission_fixture():
    _,_,_,state,roots,control,stamp=prepare()
    writes=[]
    for side in ("left","right"):
        control.batch.sides[side].hand=SimpleNamespace(permanent_wrench_composer=SimpleNamespace(reset=lambda: writes.append("reset")))
        control.batch.sides[side].set_wrenches=lambda **kw: writes.append(kw)
        control.batch.sides[side].set_joint_position_target=lambda *args,**kw: writes.append("target")
    control.batch.write_data_to_sim=lambda: writes.append("write")
    state["link_pose_xyzw"]=torch.zeros(2,2,1,7,dtype=torch.float64)
    state["body_com_pose_w"]=torch.zeros(2,2,1,7,dtype=torch.float64)
    transaction=control.prepare(torch.zeros(2,24),state,torch.full((2,2),.003),roots,stamp=stamp)
    return control,stamp,state,transaction,torch.zeros(2,3,6,dtype=torch.float64),writes

@pytest.mark.parametrize("fault",["over_force","nan_force","nan_torque","nan_com","nan_link","wrong_shape"])
def test_reaction_fault_is_rejected_before_any_native_write(fault):
    control,stamp,state,tr,w,writes=submission_fixture()
    if fault=="over_force":w[0,0,0]=30+1e-7
    if fault=="nan_force":w[0,0,0]=float("nan")
    if fault=="nan_torque":w[0,0,3]=float("nan")
    if fault=="nan_com":state["body_com_pose_w"][0,0,0,0]=float("nan")
    if fault=="nan_link":state["link_pose_xyzw"][0,0,0,0]=float("nan")
    if fault=="wrong_shape":w=w[:1]
    with pytest.raises(ValueError):control.submit(tr,stamp=stamp,wrenches=w,state=state,reaction_source=reaction_identity(stamp,w))
    assert not writes

def test_sdk_transaction_requires_one_submit_before_one_finish():
    control,stamp,state,tr,w,writes=submission_fixture()
    after=replace(stamp,sdk_step=3,rod_ticks=(12,12),solver_time_s=4+1/480)
    with pytest.raises(ValueError):control.finish(tr,state,stamp=after)
    control,stamp,state,tr,w,writes=submission_fixture()
    control.submit(tr,stamp=stamp,wrenches=w,state=state,reaction_source=reaction_identity(stamp,w))
    count=len(writes)
    with pytest.raises(ValueError):control.submit(tr,stamp=stamp,wrenches=w,state=state,reaction_source=reaction_identity(stamp,w))
    assert len(writes)==count

def reaction_identity(stamp,w):
    return dict(source_solver_id=stamp.parent_solver_id,parent_num_envs=stamp.parent_num_envs,
        run_uuid=stamp.run_uuid,source_physics_step=stamp.sdk_step,applied_physics_step=stamp.sdk_step+1,
        rod_ticks=stamp.rod_ticks,episode_uids=stamp.episode_uids,reset_ids=stamp.reset_ids,
        solver_time_s=stamp.solver_time_s,mesh_wrenches=w)

@pytest.mark.parametrize("fault",["source_solver_id","source_physics_step","applied_physics_step","reset_ids","buffer"])
def test_reaction_cannot_be_relabelled_from_a_different_owner_or_reset(fault):
    control,stamp,state,tr,w,writes=submission_fixture()
    source=reaction_identity(stamp,w)
    if fault=="buffer":source["mesh_wrenches"]=w.clone()
    elif fault=="reset_ids":source[fault]=(2,1)
    else:source[fault]+=1
    with pytest.raises(ValueError):
        control.submit(tr,stamp=stamp,wrenches=w,state=state,reaction_source=source)
    assert not writes

@pytest.mark.parametrize("key",["q","qd","backend_targets","root_pose_wxyz"])
def test_nonfinite_actual_prestate_never_reaches_native_writes(key):
    control,stamp,state,tr,w,writes=submission_fixture()
    control.invalidate()
    state[key][0,0,0]=float("nan")
    with pytest.raises(ValueError):
        control.prepare(torch.zeros(2,24),state,torch.full((2,2),.003),state["root_pose_wxyz"],stamp=stamp)
    assert not writes

def test_mutated_nonfinite_command_is_rejected_before_native_write():
    control,stamp,state,tr,w,writes=submission_fixture()
    tr["expected"][0][0,0]=float("nan")
    with pytest.raises(ValueError):
        control.submit(tr,stamp=stamp,wrenches=w,state=state,reaction_source=reaction_identity(stamp,w))
    assert not writes

def test_distinct_nonzero_lane_wrenches_match_original_com_shift():
    from scripts.verify_dh116_deformx_rigid_bridge import wrench_at_com
    control,stamp,state,tr,w,writes=submission_fixture()
    w[0,0]=torch.tensor([.5,-.2,1.,.03,.02,-.01],dtype=torch.float64)
    w[1,0]=torch.tensor([-.4,.3,.8,-.01,.04,.03],dtype=torch.float64)
    w[:,1]=-w[:,0]
    state["link_pose_xyzw"][...,0]=48.125
    state["body_com_pose_w"][...,0]=48.13500000023
    state["body_com_pose_w"][...,1]=.003
    control.submit(tr,stamp=stamp,wrenches=w,state=state,reaction_source=reaction_identity(stamp,w))
    for s,issued in enumerate(v for v in writes if isinstance(v,dict)):
        expected=torch.tensor(np.stack([wrench_at_com(w[i,s].numpy(),
            state["link_pose_xyzw"][i,s,0,:3].numpy(),
            state["body_com_pose_w"][i,s,0,:3].numpy()) for i in (0,1)]))
        torch.testing.assert_close(issued["forces"][:,0],expected[:,:3].float(),atol=0,rtol=0)
        torch.testing.assert_close(issued["torques"][:,0],expected[:,3:].float(),atol=0,rtol=0)
        assert not torch.equal(issued["forces"][0],issued["forces"][1])
