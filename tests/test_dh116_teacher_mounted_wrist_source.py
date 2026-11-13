"""Mounted-wrist CPU API fixtures; no Isaac or optimizer."""
from copy import deepcopy
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import JOINTS
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
from go2_pvcnn.control.dh116.tabletop_teacher_preaction_readback import require_teacher_common
from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import (
    capture_authored_reset_seed, capture_mounted_hand, adapt_mounted_hands,
    validate_mounted_observation, require_mounted_teacher_common,
)


def clock(step=10):
    return dict(run_uuid="fixture",env_id=8,episode_uid="ep8",reset_id=1,rod_uid="rod8",
        physics_step=step,rod_tick=(step-2)*12,policy_step=1,substep=0,timestamp_s=step/480,
        solver_time_s=4+(step-2)/480,clock_valid=True,phase="pre_action_before_target_write")


def world():
    roots={"left":[.1,.2,.3,1.,0.,0.,0.],"right":[-.1,.2,.3,1.,0.,0.,0.]}
    origin=np.array([[1.2,-.4,0.],[9.,9.,0.]])
    xy=np.array([[.01,.02],[-.01,-.02]])
    seed=np.array([roots[s] for s in ("left","right")])[None].repeat(2,0)
    seed[:,:,:3]+=origin[:,None]
    seed[:,:,:2]+=xy
    return SimpleNamespace(root_seed=torch.tensor(seed),origins=torch.tensor(origin),initial_offset=.04,
        w=SimpleNamespace(layout={"hand_roots":roots},seed={"root_xy":xy.tolist(),"root_offset":.04}))


def hand(side):
    names=list(reversed(JOINTS+WRIST_LINEAR+WRIST_ANGULAR))
    q=np.full((2,17),.2)
    for name,value in zip(WRIST_ANGULAR,(.2,-.3,.4)):q[:,names.index(name)]=value
    qd=np.full((2,17),.03)
    types=np.array([[1 if n in WRIST_LINEAR else 0 for n in names]]*2)
    seed=world().root_seed.numpy()
    hi=0 if side=="left" else 1
    pose=np.zeros((2,2,7));pose[...,3]=1.
    pose[:,1,:3]=seed[:,hi,:3]+[.005,.01,.02]
    pose[:,1,3:]=[-.5,.5,-.5,.5]
    velocity=np.zeros((2,2,6));velocity[:,1]=[.05,.1,-.15,.3,.6,.9]
    sdk_pose=pose[..., [0,1,2,4,5,6,3]].copy()
    paths=[f"/World/envs/env_{e}/{side}Hand" for e in (8,3)]
    h=SimpleNamespace(joint_names=names,body_names=["other","base_link"])
    h.data=SimpleNamespace(joint_pos=torch.tensor(q),joint_vel=torch.tensor(qd),
        body_link_pose_w=torch.tensor(pose),body_lin_vel_w=torch.tensor(velocity[...,:3]),
        body_ang_vel_w=torch.tensor(velocity[...,3:]),root_pose_w=torch.full((2,7),99.),
        body_link_vel_w=torch.full((2,2,6),88.),body_com_pose_w=torch.full((2,2,7),77.))
    h.sdk_q=q.copy();h.sdk_qd=qd.copy();h.sdk_pose=sdk_pose;h.sdk_velocity=velocity.copy();h.types=types
    h.root_physx_view=SimpleNamespace(prim_paths=paths,
        link_paths=[[p+"/other",p+"/base_link"] for p in paths],
        get_dof_positions=lambda:h.sdk_q,get_dof_velocities=lambda:h.sdk_qd,
        get_dof_types=lambda:h.types,get_link_transforms=lambda:h.sdk_pose,
        get_link_velocities=lambda:h.sdk_velocity)
    return h


def seed_packet():
    return capture_authored_reset_seed(world(),read_identity=lambda:clock(2),data_env_index=0)


def capture(h,side="left",seed=None,read_identity=clock,expected_articulation_path=None):
    return capture_mounted_hand(h,MountView(h),side=side,read_identity=read_identity,
        data_env_index=0,expected_hand_container_path=f"/World/envs/env_8/{side}Hand",
        expected_articulation_path=expected_articulation_path or f"/World/envs/env_8/{side}Hand",
        reset_seed=seed_packet() if seed is None else seed)


def test_exact_existing_observe_prefix_with_raw_wxyz_com_velocity_and_authored_seed(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_dexhands_feedback
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import observe
    packets=[capture(hand(side),side) for side in ("left","right")]
    result=adapt_mounted_hands(packets,expected_identity=clock())
    seed=torch.tensor(packets[0]["reset_seed"]["root_seed_w"],dtype=torch.float64)[None]
    m=dict(wrist_pose=torch.tensor([[p["body_link_pose_w"] for p in packets]],dtype=torch.float64),
           wrist_velocity=torch.tensor([[p["body_lin_vel_w"]+p["body_ang_vel_w"] for p in packets]],dtype=torch.float64),
           wrist_joint_angles=torch.tensor([[[p["q"][p["joint_names"].index(n)] for n in WRIST_ANGULAR]
                                             for p in packets]],dtype=torch.float64),q=torch.zeros((1,12),dtype=torch.float64))
    t=SimpleNamespace(m=m,free_wrist_world=SimpleNamespace(root_seed=seed,
        wrist_angles=torch.zeros((1,2,3)),palms=torch.zeros((1,2,3))),
        command=torch.zeros((1,12)),low=torch.zeros(12),high=torch.ones(12),
        rotation_targets=torch.zeros((1,2,3)),actions=torch.zeros((1,24)),surface_force_sharing=False)
    monkeypatch.setattr(tabletop_dexhands_feedback,"geometry_features",lambda m:torch.zeros((1,20)))
    expected=observe(t)[0,:32].numpy()
    np.testing.assert_array_equal(result["values"][153:185],expected)
    assert result["values"][156] == -.5  # raw WXYZ; no sign canonicalization
    np.testing.assert_allclose(result["values"][167:173],[1,2,-3,.3,.6,.9])
    assert result["valid"].sum() == 66
    assert validate_mounted_observation(result)["unknown_channels"] == 305
    with pytest.raises(ValueError,match="305"):require_mounted_teacher_common(result)
    with pytest.raises(ValueError,match="contract"):require_teacher_common(result)


def test_only_one_mounted_hand_never_synthesizes_the_other():
    result=adapt_mounted_hands([capture(hand("left"))],expected_identity=clock())
    assert result["valid"].sum() == 33
    assert np.flatnonzero(result["valid"][153:185]).tolist() == list(range(7))+list(range(14,20))+list(range(26,29))
    assert np.isnan(result["values"][~result["valid"]]).all()


@pytest.mark.parametrize("edit",[
    lambda h:h.joint_names.__setitem__(0,"invented"),
    lambda h:h.body_names.__setitem__(1,"wrong_palm"),
    lambda h:h.types.__setitem__((0,h.joint_names.index("wrist_x")),0),
    lambda h:h.sdk_q.__setitem__((0,h.joint_names.index("wrist_roll")),.8),
    lambda h:h.sdk_qd.__setitem__((0,0),99),
    lambda h:h.sdk_pose.__setitem__((0,1,0),99),
    lambda h:h.sdk_velocity.__setitem__((0,1,0),99),
    lambda h:h.data.body_lin_vel_w.__setitem__((0,1,0),float("nan")),
    lambda h:h.root_physx_view.link_paths[0].__setitem__(1,"/World/other/base_link"),
])
def test_rejects_wrong_names_units_link_or_stale_cache_before_admission(edit):
    h=hand("left");edit(h)
    with pytest.raises((ValueError,AttributeError)):capture(h)


def test_fixed_base_does_not_get_wrist_placeholders():
    h=hand("left")
    h.joint_names=list(JOINTS)
    with pytest.raises(ValueError):capture(h)


def test_seed_uses_authored_root_not_reset_offset_anchor_or_measured_palm():
    seed=seed_packet()
    np.testing.assert_array_equal(seed["root_seed_w"],world().root_seed.numpy()[0])
    bad=world();bad.root_seed[:,:,2]+=.04
    with pytest.raises(ValueError,match="authored"):capture_authored_reset_seed(
        bad,read_identity=lambda:clock(2),data_env_index=0)
    forged=deepcopy(seed);forged["root_seed_w"][0][2]+=.02
    with pytest.raises(ValueError,match="seed"):capture(hand("left"),seed=forged)


@pytest.mark.parametrize("edit",[
    lambda p:p["identity"].update(episode_uid="stale"),
    lambda p:p["identity"].update(reset_id=0),
    lambda p:p.update(source="measured_initial_palm"),
])
def test_seed_cannot_be_reused_across_reset_or_relabelled_as_measurement(edit):
    seed=seed_packet();edit(seed)
    with pytest.raises(ValueError):capture(hand("left"),seed=seed)


def test_invalid_or_advancing_shared_clock_rejected():
    bad=clock();bad["clock_valid"]=False
    with pytest.raises(ValueError,match="clock"):capture(hand("left"),read_identity=lambda:bad)
    reads=iter([clock(),clock(11)])
    with pytest.raises(ValueError,match="changed"):capture(hand("left"),read_identity=lambda:next(reads))


def test_output_is_owned_and_validator_reconstructs_mask_and_values():
    h=hand("left");packet=capture(h)
    saved=deepcopy(packet)
    h.data.body_link_pose_w[:]=99;h.sdk_q[:]=99
    assert packet==saved
    result=adapt_mounted_hands([packet],expected_identity=clock())
    result["values"][153]=99
    with pytest.raises(ValueError,match="reconstruct"):validate_mounted_observation(result)
    result=adapt_mounted_hands([packet],expected_identity=clock())
    result["values"][~result["valid"]]=0;result["valid"][:]=True
    with pytest.raises(ValueError,match="reconstruct"):require_mounted_teacher_common(result)


def test_cross_hand_seed_and_clock_identity_must_match():
    left=capture(hand("left"));right=capture(hand("right"),"right")
    right["identity"]["rod_tick"]+=12
    with pytest.raises(ValueError,match="identity"):adapt_mounted_hands([left,right],expected_identity=clock())
    right=capture(hand("right"),"right")
    right["reset_seed"]["root_seed_w"][0][0]+=.1
    with pytest.raises(ValueError):adapt_mounted_hands([left,right],expected_identity=clock())


def test_finite_native_values_cannot_mark_overflowed_normalization_valid():
    h=hand("left")
    h.data.body_link_pose_w[0,1,0]=1e308
    h.sdk_pose[0,1,0]=1e308
    packet=capture(h)
    with np.errstate(over="ignore",invalid="ignore"):
        with pytest.raises(ValueError,match="normalization"):
            adapt_mounted_hands([packet],expected_identity=clock())


def test_validation_rejects_changed_execution_counters():
    result=adapt_mounted_hands([capture(hand("left"))],expected_identity=clock())
    result["optimizer_updates"]=1
    with pytest.raises(ValueError,match="reconstruct"):
        validate_mounted_observation(result)


def test_mounted_root_joint_view_and_sibling_base_link_are_separate_identities():
    h=hand("left")
    h.root_physx_view.prim_paths=[p+"/wrist_fixed" for p in h.root_physx_view.prim_paths]
    packet=capture(h,expected_articulation_path="/World/envs/env_8/leftHand/wrist_fixed")
    assert packet["articulation_path"] == "/World/envs/env_8/leftHand/wrist_fixed"
    assert packet["hand_container_path"] == "/World/envs/env_8/leftHand"
    assert packet["body_path"] == "/World/envs/env_8/leftHand/base_link"
    assert validate_mounted_observation(adapt_mounted_hands(
        [packet],expected_identity=clock()))["valid_channels"] == 33



@pytest.mark.parametrize("edit,expected_root",[
    (lambda h:h.root_physx_view.prim_paths.__setitem__(0,"/World/envs/env_3/leftHand/wrist_fixed"),
     "/World/envs/env_8/leftHand/wrist_fixed"),
    (lambda h:h.root_physx_view.prim_paths.__setitem__(0,"/World/envs/env_8/rightHand/wrist_fixed"),
     "/World/envs/env_8/rightHand/wrist_fixed"),
    (lambda h:h.root_physx_view.link_paths[0].__setitem__(0,"/World/envs/env_8/leftHandExtra/other"),
     "/World/envs/env_8/leftHand"),
    (lambda h:h.root_physx_view.link_paths[0].__setitem__(1,"/World/envs/env_8/leftHand/wrist_fixed/base_link"),
     "/World/envs/env_8/leftHand"),
    (lambda h:h.root_physx_view.prim_paths.__setitem__(0,"/World/envs/env_8/leftHand/../leftHand"),
     "/World/envs/env_8/leftHand/../leftHand"),
])
def test_articulation_and_container_identity_mismatches_are_rejected(edit,expected_root):
    h=hand("left");edit(h)
    with pytest.raises(ValueError,match="path"):
        capture(h,expected_articulation_path=expected_root)


def test_source_replay_rejects_old_or_conflated_path_packet():
    packet=capture(hand("left"))
    packet["version"]="dh116-teacher-mounted-preaction-source-v1"
    with pytest.raises(ValueError,match="source"):
        adapt_mounted_hands([packet],expected_identity=clock())
    packet=capture(hand("left"))
    packet["articulation_path"]="/World/envs/env_8/rightHand/wrist_fixed"
    with pytest.raises(ValueError,match="path"):
        adapt_mounted_hands([packet],expected_identity=clock())
