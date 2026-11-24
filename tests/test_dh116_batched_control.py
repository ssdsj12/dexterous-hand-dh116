"""Batched command/snapshot parity; fixture SDK only."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_dh116_deformable_bimanual_step import Hand,advance
from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import BimanualStepConnection,HandBinding,DT,_capture


def fixture(n=3):
    groups=[];clocks=[]
    for i in range(n):
        hands=[Hand(s,(i,)) for s in ("left","right")]
        groups.append(tuple(HandBinding(s,h,{i:h.root_physx_view.prim_paths[0]},MountView(h))
                            for s,h in zip(("left","right"),hands)))
        clocks.append(dict(run_uuid="fixture",env_id=i,episode_uid="ep"+str(i),reset_id=1,
            rod_uid="rod"+str(i),physics_step=2,rod_tick=0,policy_step=0,substep=0,
            timestamp_s=2*DT,solver_time_s=4.,clock_valid=True))
    return groups,clocks


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_batch_prepare_matches_independent_full_24d_and_capture_reuse(device):
    if device.startswith("cuda") and not torch.cuda.is_available():pytest.skip("CUDA")
    from go2_pvcnn.control.dh116.tabletop_batched_control import BatchedControl
    groups,clocks=fixture()
    pool=BatchedControl(groups,lambda i:deepcopy(clocks[i]),device=device)
    actions=torch.linspace(-1,1,72,dtype=torch.float64).reshape(3,24)
    palms=torch.zeros(3,2,3,dtype=torch.float64);angles=palms.clone()
    batch=pool.prepare(actions,palms,angles)
    for i,group in enumerate(groups):
        old=BimanualStepConnection(env_ids=(i,),solver_substeps=12)
        oracle=old.begin(group,actions[i:i+1],read_clock=lambda i=i:[deepcopy(clocks[i])],
                         palms=palms[i:i+1],angles=angles[i:i+1])
        for field in ("requested_finger_targets_rad","requested_palm_offsets_m","requested_angles_rad"):
            torch.testing.assert_close(batch[i]._request[field],oracle[field],rtol=0,atol=1e-10)
    # submit performs no extra capture, and finish reads one new physical state.
    for i,c in enumerate(batch):
        root=torch.stack([b.hand.data.root_pose_w[0] for b in groups[i]])[None]
        c.submit(groups[i],read_clock=lambda i=i:[deepcopy(clocks[i])],projected_root_poses=root,
                 projected_angles=angles[i:i+1],projection_source="fixture")
    before=pool.capture_count
    assert before==3
    advance(clocks)
    for i,c in enumerate(batch):c.finish(groups[i],read_clock=lambda i=i:[deepcopy(clocks[i])])
    assert pool.capture_count==before+3
    pool.prepare(actions,palms,angles)
    assert pool.capture_count==before+3
    clocks[1]["episode_uid"]="reset";clocks[1]["reset_id"]+=1;clocks[1]["rod_tick"]=0
    pool.prepare(actions,palms,angles)
    assert pool.capture_count==before+4

@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_batch_wrist_and_reaction_match_original_128_and_reject_nonfinite(device):
    if device.startswith("cuda") and not torch.cuda.is_available():pytest.skip("CUDA")
    from go2_pvcnn.control.dh116.tabletop_batched_control import BatchReactions,batch_projection
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import route_world_wrenches,PADS
    rng=np.random.default_rng(11);n=128
    bindings=[];measured=[]
    keys=[s+"."+p for s in ("left","right") for p in PADS]
    # Reuse exact original binding validator fixture.
    def original_bindings():
        return {s+"."+p:dict(side=s,body_name=p,body_index=j,prim_path="/World/"+s+"Hand/"+p,
                    source_usd_sha256="a"*64,cooked_geometry_sha256="b"*64,sdf_sha256="c"*64)
                for s in ("left","right") for j,p in enumerate(PADS)}
    for i in range(n):
        names=original_bindings()
        poses={k:(rng.normal(size=3),np.array([0.,0.,0.,1.])) for k in keys}
        centers={k:poses[k][0]+rng.normal(size=3)*.03 for k in keys}
        bindings.append(names);measured.append((poses,centers))
    w=torch.as_tensor(rng.normal(size=(n,11,6)),device=device,dtype=torch.float64)
    names=keys+["table"]
    batch=BatchReactions(bindings,names,measured,device=device)
    rows,tensors=batch.prepare(w,measured)
    for i in range(n):
        old,_=route_world_wrenches({k:w[i,j].cpu().numpy() for j,k in enumerate(names)},
                                   bindings[i],*measured[i])
        for side in ("left","right"):
            assert rows[i][side]["body_ids"]==old[side]["body_ids"]
            for field in ("forces","torques"):
                np.testing.assert_allclose(rows[i][side][field],old[side][field],rtol=0,atol=1e-12)
    bad=w.clone();bad[5,2,0]=float("nan")
    with pytest.raises(ValueError):batch.prepare(bad,measured)
    bad=w.clone();bad[6,3,0]=31.
    with pytest.raises(ValueError,match="30 N"):batch.prepare(bad,measured)
    root=torch.zeros(n,2,7,dtype=torch.float64);root[...,3]=1;root[...,2]=.5
    base=root.clone();palms=torch.zeros(n,2,3,dtype=torch.float64)
    angles=palms.clone();requested=palms+torch.tensor(rng.normal(size=(n,2,3))*.0001)
    clearance=torch.tensor(rng.uniform(.0019,.004,(n,2)),dtype=torch.float64)
    connections=[SimpleNamespace(_request=dict(requested_palm_offsets_m=requested[i:i+1],
                                               requested_angles_rad=angles[i:i+1])) for i in range(n)]
    actual=batch_projection(connections,root,base,clearance,palms,angles,device=device)
    for i in range(n):
        old=project_policy_wrist_targets(root_seed_w=root[i:i+1],measured_base_w=base[i:i+1],
             common_lift_offset_m=torch.zeros(1,dtype=torch.float64),measured_clearance_m=clearance[i:i+1],
             palm_offsets_m=palms[i:i+1],requested_palm_offsets_m=requested[i:i+1],
             angles_rad=angles[i:i+1],requested_angles_rad=angles[i:i+1],
             allowed=torch.ones(1,dtype=torch.bool),live=torch.ones(1,dtype=torch.bool))
        for key,value in old.items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(actual[i][key],value,rtol=0,atol=1e-12)
            else:assert actual[i][key]==value

@pytest.mark.parametrize("fault",["clock","nan","view","reset"])
def test_batch_rejects_original_guard_or_identity_faults(fault):
    from go2_pvcnn.control.dh116.tabletop_batched_control import BatchedControl
    from go2_pvcnn.control.dh116.linear_coupling import MIMIC,ACTIVE,active_limits
    groups,clocks=fixture(1);pool=BatchedControl(groups,lambda i:deepcopy(clocks[i]),device="cpu")
    actions=torch.zeros(1,24,dtype=torch.float64);palms=torch.zeros(1,2,3,dtype=torch.float64)
    connections=pool.prepare(actions,palms,palms)
    hand=groups[0][0].hand
    if fault in ("clock","reset"):
        clocks[0]["physics_step" if fault=="clock" else "reset_id"]+=1
        root=torch.stack([b.hand.data.root_pose_w[0] for b in groups[0]])[None]
        with pytest.raises(ValueError):
            connections[0].submit(groups[0],read_clock=lambda:[deepcopy(clocks[0])],
                 projected_root_poses=root,projected_angles=palms,projection_source="fixture")
        return
    if fault=="nan":actions[0,0]=float("nan")
    elif fault=="view":hand.root_physx_view=deepcopy(hand.root_physx_view)
    elif fault=="mimic":hand.data.joint_pos[0,hand.joint_names.index(MIMIC[0][0])]+=.2
    elif fault=="joint_limit":hand.data.joint_pos[0,hand.joint_names.index(ACTIVE[0])]=active_limits()[1][0]+.2
    with pytest.raises((ValueError,RuntimeError)):
        pool.prepare(actions,palms,palms)

@pytest.mark.parametrize("fault",["mimic","joint_limit"])
def test_bad_measured_joints_keep_original_projection_and_original_hard_guard(fault):
    from go2_pvcnn.control.dh116.tabletop_batched_control import BatchedControl
    from go2_pvcnn.control.dh116.linear_coupling import MIMIC,ACTIVE,active_limits
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
    groups,clocks=fixture(1)
    hand=groups[0][0].hand
    if fault=="mimic":hand.measured[0,hand.joint_names.index(MIMIC[0][0])]+=.2
    else:hand.measured[0,hand.joint_names.index(ACTIVE[0])]=active_limits()[1][0]+.2
    action=torch.ones(1,24,dtype=torch.float64);palm=torch.zeros(1,2,3,dtype=torch.float64)
    old=BimanualStepConnection(env_ids=(0,),solver_substeps=12)
    oracle=old.begin(groups[0],action,read_clock=lambda:[deepcopy(clocks[0])],palms=palm,angles=palm)
    actual=BatchedControl(groups,lambda i:deepcopy(clocks[i]),device="cpu").prepare(action,palm,palm)[0]
    torch.testing.assert_close(actual._request["requested_finger_targets_rad"],
                               oracle["requested_finger_targets_rad"],rtol=0,atol=1e-12)
    for packet in (old._pre,actual._pre):
        row=packet["hands"]["left"];q=row["q"][0];names=row["joint_names"]
        residuals=[q[names.index(f)]-ratio*q[names.index(m)]-offset for f,m,ratio,offset in MIMIC]
        assert joint_safety_failure(q[[names.index(k) for k in ACTIVE]],np.stack(active_limits(),-1),residuals)

def test_owned_transaction_cannot_submit_into_another_lane():
    from go2_pvcnn.control.dh116.tabletop_batched_control import BatchedControl
    groups,clocks=fixture(2);palms=torch.zeros(2,2,3,dtype=torch.float64)
    pool=BatchedControl(groups,lambda i:deepcopy(clocks[i]),device="cpu")
    c=pool.prepare(torch.zeros(2,24,dtype=torch.float64),palms,palms)[0]
    root=torch.stack([b.hand.data.root_pose_w[0] for b in groups[0]])[None]
    with pytest.raises(ValueError,match="owned lane"):
        c.submit(groups[1],read_clock=lambda:[deepcopy(clocks[0])],
                 projected_root_poses=root,projected_angles=palms[:1],projection_source="fixture")
    assert not any(b.hand.writes for b in groups[1])

@pytest.mark.parametrize("fault",["nonfinite","quaternion","shape"])
def test_full_batch_link_source_rejects_noncontact_corruption(fault):
    from go2_pvcnn.control.dh116.tabletop_batched_control import checked_links
    poses=np.zeros((128,2,23,7));poses[...,6]=1
    if fault=="nonfinite":poses[127,1,22,0]=float("nan")
    elif fault=="quaternion":poses[127,1,22,6]=1.001
    else:poses=poses[:,:,:22]
    with pytest.raises(ValueError):checked_links(poses,128)
