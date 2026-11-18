"""Direct-target contracts; synthetic tests are not grasp qualification."""
import numpy as np
import pytest
import torch
from gymnasium import spaces
from go2_pvcnn.control.dh116.tabletop_direct import joint_target, contact_quality
from go2_pvcnn.control.dh116.tabletop_policy import BoundedMeanPolicy
from go2_pvcnn.control.dh116.linear_coupling import active_limits


def test_joint_targets_use_fixed_envelope_and_coupled_limits():
    lo, hi = (np.tile(v, 2) for v in active_limits())
    center = (lo+hi)/2
    np.testing.assert_allclose(joint_target(center, np.ones(12), .08), center+.08)
    np.testing.assert_allclose(joint_target(lo, -np.ones(12), .08), lo)
    with pytest.raises(ValueError): joint_target(center, np.ones(12)*1.01, .08)


def test_contact_quality_prefers_balanced_pinch_to_one_large_force():
    one = contact_quality([1.,0.,0.,0.])
    four = contact_quality([.1]*4)
    assert four > contact_quality([.1,.1,0.,0.]) > one
    assert contact_quality([0.]*4) == 0
    assert contact_quality([30.]*4) == four  # excessive load earns no extra contact reward


def test_bounded_policy_cannot_emit_unbounded_gaussian_means(tmp_path):
    policy = BoundedMeanPolicy(spaces.Box(-np.inf,np.inf,(104,),dtype=np.float32),
        spaces.Box(-1.,1.,(12,),dtype=np.float32), lambda _: 3e-4, log_std_init=-1.2)
    with torch.no_grad():
        policy.action_net[0].bias.fill_(10.)
        obs=torch.zeros((3,104))
        distribution=policy.get_distribution(obs).distribution
        assert torch.max(torch.abs(distribution.mean)) <= 1
        assert torch.isfinite(distribution.log_prob(torch.zeros((3,12)))).all()
    policy.save(tmp_path/'policy.pt')
    restored=BoundedMeanPolicy.load(tmp_path/'policy.pt')
    np.testing.assert_allclose(policy.predict(np.zeros(104),deterministic=True)[0],
                               restored.predict(np.zeros(104),deterministic=True)[0])


def test_palm_adjustment_is_per_hand_bounded_and_freezes():
    from go2_pvcnn.control.dh116.tabletop_direct import adjust_palms
    result=adjust_palms(np.zeros(2), [1,0,0,0], .005, 'close', .01, allowed=True)
    assert result[0] == 0 and result[1] == pytest.approx(-.00001)
    np.testing.assert_allclose(adjust_palms(result,[0]*4,.005,'hold',1.,allowed=True),result)
    np.testing.assert_allclose(adjust_palms(result,[0]*4,.005,'close',1.,allowed=False),result)
    assert adjust_palms(np.zeros(2),[0]*4,.0021,'close',1.,allowed=True).tolist()==[0.,0.]


def test_close_seed_preserves_translated_object_and_named_velocity():
    from go2_pvcnn.control.dh116.tabletop_direct import snapshot_state
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS, expand
    lo,hi=active_limits();q=expand((lo+hi)/2)
    roots=np.array([[.1,-.1,.5,1,0,0,0],[.1,.1,.5,1,0,0,0]])
    layout={'hand_roots':{'left':roots[0].copy(),'right':roots[1].copy()}}
    moved=roots.copy();moved[:,0]+=.017;moved[:,2]+=.005
    row=dict(phase='close',root_offset_m=.005,object_relative=[.02,0,-.1],object_quat=[1,0,0,0],object_velocity=[.004,0,0,0,.6,0],hands=[])
    for root in moved:
        row['hands'].append(dict(root_pose=root.tolist(),joints_rad=dict(zip(JOINTS,q.tolist())),joint_vel_rad_s=list(np.arange(11)*.001)))
    state=snapshot_state(row,layout)
    np.testing.assert_allclose(state['root_xy'],[.017,0])
    np.testing.assert_allclose(state['wire_pose'][:3],[.137,0,.405])
    assert state['joint_velocities'][0][4]==pytest.approx(.004)
    assert 'forces' not in state
    row['hands'][0]['root_pose'][0]+=.01
    with pytest.raises(ValueError): snapshot_state(row,layout)


def test_object_grasp_mode_attempts_lift_without_four_pad_contact():
    from go2_pvcnn.control.dh116.tabletop_task import TabletopTask
    from go2_pvcnn.control.dh116.tabletop_control import curriculum_complete
    task=TabletopTask(phase='close',grasp_mode='object_stability')
    m=dict(approach_error_m=.004,pad_forces_n=[.2,0,.2,0],clearance_m=.005,
           object_bottom_m=.30,object_speed_m_s=0.,slip_m=0.,max_force_n=.2)
    for _ in range(600): task.update(1/240,**m)
    assert task.phase=='lift' and not task.success
    assert not curriculum_complete('close',task)  # an attempt is not grasp proof
    # A stationary object on the table never qualifies.
    for _ in range(200): task.update(1/240,**m)
    assert not task.success
    m.update(object_bottom_m=.335,object_speed_m_s=.001,slip_m=.004)
    for _ in range(130): task.update(1/240,**m)
    assert task.success  # no four-pad prerequisite


def test_object_grasp_mode_rejects_slipping_lift_as_success():
    from go2_pvcnn.control.dh116.tabletop_task import TabletopTask
    task=TabletopTask(phase='lift',grasp_mode='object_stability')
    m=dict(approach_error_m=.004,pad_forces_n=[0]*4,clearance_m=.005,
           object_bottom_m=.335,object_speed_m_s=0.,slip_m=.03,max_force_n=0.)
    for _ in range(180):task.update(1/240,**m)
    assert not task.success
def test_recovery_target_reduces_signed_coupling_error_without_changing_other_fingers():
    from go2_pvcnn.control.dh116.tabletop_direct import coupling_recovery_target
    from go2_pvcnn.control.dh116.linear_coupling import active_limits, MIMIC
    low, high = active_limits()
    q = np.tile((low+high)/2, 2)
    target = q + .01
    errors = np.zeros(10)
    errors[6] = -.016  # right finger22 follower is behind its master
    restored = coupling_recovery_target(target, q, errors)
    ratio = MIMIC[1][2]
    assert restored[8] == pytest.approx(q[8] + errors[6]/ratio)
    assert restored[8] < q[8]
    np.testing.assert_array_equal(np.delete(restored, 8), np.delete(target, 8))
    errors[6] = .016
    assert coupling_recovery_target(target, q, errors)[8] > q[8]
    errors[6] = -.003
    np.testing.assert_array_equal(coupling_recovery_target(target, q, errors), target)


@pytest.mark.parametrize('index', range(10))
@pytest.mark.parametrize('sign', [-1., 1.])
def test_recovery_covers_each_mimic_and_clips_to_coupled_limits(index, sign):
    from go2_pvcnn.control.dh116.tabletop_direct import coupling_recovery_target
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC
    low, high = (np.tile(v, 2) for v in active_limits())
    _, master, ratio, _ = MIMIC[index % 5]
    joint = (index // 5)*6 + ACTIVE.index(master)
    q = (low+high)/2
    target = q.copy()
    errors = np.zeros(10)
    errors[index] = sign*.012
    expected = target.copy()
    expected[joint] = q[joint] + errors[index]/ratio
    np.testing.assert_allclose(coupling_recovery_target(target, q, errors), expected)
    q[joint] = high[joint] if sign > 0 else low[joint]
    expected[joint] = q[joint]
    np.testing.assert_allclose(coupling_recovery_target(target, q, errors), expected)
