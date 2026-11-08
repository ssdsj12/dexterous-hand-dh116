"""CPU contracts and geometry checks. These are not native grasp trials."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, coupling_matrix
from go2_pvcnn.control.dh116.tabletop_bimanual_physical_action_adapter import (
    BimanualPhysicalActionAdapter, HandPhysicalKinematics, PAD_BODIES,
    _full_angles, audit_native_snapshot,
)


def model(side):
    """Known non-mirror synthetic tree, exclusively a CPU fixture."""
    edges = []
    parent = 'base_link'
    for index, name in enumerate(JOINTS):
        child = PAD_BODIES[index - 8] if index >= 8 else name + '_body'
        t0, t1 = np.eye(4), np.eye(4)
        t0[:3, 3] = [.007, .002 * (index % 2), .004]
        axis = np.array([0., 0., 1.])
        if side == 'right' and name == 'finger12':
            axis = Rotation.from_rotvec([.16, 0., 0.]).apply(axis)
        edges.append(dict(name=name, parent=parent, child=child,
                          t0=t0.tolist(), t1=t1.tolist(), axis=axis.tolist()))
        parent = child
    pads = {body: dict(center_local_m=[.001, .002, .003],
                       reference_outward_normal_local=[0., 1., 0.]) for body in PAD_BODIES}
    return HandPhysicalKinematics(side, 'base_link', edges, pads,
                                  ('a' if side == 'left' else 'b') * 64)


def models():
    return {side: model(side) for side in ('left', 'right')}


def poses():
    q = Rotation.from_euler('XYZ', [.4, -.2, .5]).as_quat()
    return dict(left=[.1, .2, .3, q[3], *q[:3]], right=[-.3, .4, .2, 1., 0., 0., 0.])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_analytic_full_and_coupled_pad_jacobians_match_finite_difference(side):
    hand = model(side)
    q6 = np.array([.4, .3, .7, .5, .6, .8])
    root = poses()[side]
    audit = hand.audit_jacobians(q6, root)
    assert audit['position_error_m_per_rad'].max() < 2e-10
    assert audit['normal_error_per_rad'].max() < 1e-9
    q11 = _full_angles(q6)
    ref = hand.pad_kinematics(q11, root)
    for i in range(11):
        delta = np.eye(11)[i] * 1e-6
        plus = hand.pad_kinematics(q11 + delta, root)
        minus = hand.pad_kinematics(q11 - delta, root)
        np.testing.assert_allclose((plus['points_w_m'] - minus['points_w_m']) / 2e-6,
                                   ref['position_jacobian_full'][..., i], atol=2e-10, rtol=0)
        np.testing.assert_allclose((plus['normals_w'] - minus['normals_w']) / 2e-6,
                                   ref['normal_jacobian_full'][..., i], atol=1e-9, rtol=0)


def test_source_right_finger12_axis_is_used_instead_of_left_mirror_assumption():
    hands = models()
    q = _full_angles([.4, .3, .7, .5, .6, .8])
    root = [0., 0., 0., 1., 0., 0., 0.]
    left = hands['left'].pad_kinematics(q, root)
    right = hands['right'].pad_kinematics(q, root)
    assert np.linalg.norm(left['points_w_m'] - right['points_w_m']) > 1e-3
    assert np.linalg.norm(left['position_jacobian_active'] - right['position_jacobian_active']) > 1e-3


def test_measured_passive_angles_remain_distinct_from_affine_prediction():
    hand = model('right')
    active = [.4, .3, .7, .5, .6, .8]
    full = _full_angles(active)
    root = poses()['right']
    before = hand.pad_kinematics(full, root)['points_w_m']
    full[JOINTS.index('finger13')] += .05
    after = hand.pad_kinematics(full, root)['points_w_m']
    assert np.linalg.norm(before - after) > 1e-3
    np.testing.assert_allclose(hand.fk_active(active, root)[hand.pads[0][0]],
                               hand.fk_full(_full_angles(active), root)[hand.pads[0][0]], atol=1e-14)


def test_original_physical_contract_roundtrip_has_no_reflection_or_joint_scaling():
    adapter = BimanualPhysicalActionAdapter(models())
    rng = np.random.default_rng(1165)
    base = poses()
    palm = {side: np.array(pose, copy=True) for side, pose in base.items()}
    palm['right'][:3] += [.01, -.02, .03]
    for _ in range(30):
        action = rng.uniform(-1., 1., 24)
        decoded = adapter.decode(action, base, palm, [[.1, .2, -.3], [-.2, .1, .3]], wrist_seed_poses_w=base)
        np.testing.assert_allclose(adapter.to_sdk(decoded), action, atol=2e-16, rtol=0)
        for hi, side in enumerate(('left', 'right')):
            np.testing.assert_allclose(decoded[side]['finger_velocity_rad_s'], action[hi * 6:(hi + 1) * 6] * .3)
            np.testing.assert_allclose(decoded[side]['full_joint_velocity_rad_s'],
                                       coupling_matrix() @ decoded[side]['finger_velocity_rad_s'])


def test_runtime_palm_frame_and_intrinsic_rotation_axes_are_explicit():
    adapter = BimanualPhysicalActionAdapter(models())
    pose = [0., 0., 0., 1., 0., 0., 0.]
    q = Rotation.from_euler('Z', 90, degrees=True).as_quat()
    palm = [0., 0., 0., q[3], *q[:3]]
    action = np.zeros(24)
    action[12] = 1.
    action[18] = 1.
    decoded = adapter.decode(action, dict(left=pose, right=pose), dict(left=palm, right=pose))
    np.testing.assert_allclose(decoded['left']['translation_palm_m_s'], [0., -.01, 0.], atol=1e-16)
    np.testing.assert_allclose(decoded['left']['angular_velocity_world_rad_s'], [.5, 0., 0.])
    np.testing.assert_allclose(decoded['left']['base_from_palm'][:3, :3], Rotation.from_quat(q).as_matrix())


def test_sdk_named_joint_mapping_is_independent_of_articulation_order():
    adapter = BimanualPhysicalActionAdapter(models())
    action = np.zeros(24)
    action[7] = .4
    decoded = adapter.decode(action, poses(), poses())['right']
    names = ['wrist_z'] + list(reversed(JOINTS))
    velocity = adapter.sdk_joint_velocity(decoded, names)
    assert velocity[names.index('finger12')] == pytest.approx(.12)
    assert velocity[names.index('finger13')] == pytest.approx(.12 * MIMIC[0][2])
    assert velocity[0] == 0.


def test_authored_wrist_seed_cannot_be_inferred_from_tracking_lag():
    adapter = BimanualPhysicalActionAdapter(models())
    with pytest.raises(ValueError, match='authored wrist seed'):
        adapter.decode(np.zeros(24), poses(), poses(), [[.1, 0., 0.], [0., 0., 0.]])


def test_existing_or_mismatched_checkpoint_cannot_silently_enable_adapter():
    adapter = BimanualPhysicalActionAdapter(models())
    for contract in ({}, None, {'physical_action_adapter': {'version': 'legacy'}}):
        with pytest.raises(ValueError, match='explicit matching'):
            adapter.adapt_for_checkpoint(np.zeros(24), poses(), poses(), contract)
    contract = {'physical_action_adapter': adapter.contract()}
    np.testing.assert_array_equal(adapter.adapt_for_checkpoint(np.zeros(24), poses(), poses(), contract), np.zeros(24))
    assert contract['physical_action_adapter']['source_asset_sha256']['left'] != contract['physical_action_adapter']['source_asset_sha256']['right']
    assert len(contract['physical_action_adapter']['adapter_source_sha256']) == 64
    assert len(contract['physical_action_adapter']['sha256']) == 64


def test_native_snapshot_distinguishes_measured_passive_and_coupled_fk():
    hands, state = models(), []
    for side in ('left', 'right'):
        hand = hands[side]
        full = _full_angles([.4, .3, .7, .5, .6, .8])
        full[JOINTS.index('finger13')] += .02
        all_poses = hand.fk_full(full, poses()[side])
        names = list(all_poses)
        link_poses = []
        for body in names:
            t = all_poses[body]
            q = Rotation.from_matrix(t[:3, :3]).as_quat()
            link_poses.append([*t[:3, 3], q[3], *q[:3]])
        state.append(SimpleNamespace(joint_names=list(JOINTS), body_names=names,
                     data=SimpleNamespace(joint_pos=full[None], body_link_pose_w=np.array(link_poses)[None])))
    rows = audit_native_snapshot(SimpleNamespace(hands=state), hands, [0])
    assert len(rows) == 2
    for row in rows:
        assert row['env_id'] == 0
        assert max(row['pad_position_error_measured_q_m3']) < 1e-12
        assert max(row['pad_position_error_coupled_q_m3']) > 1e-4
        assert row['passive_coupling_residual_rad5'][0] == pytest.approx(.02)
        assert row['palm_link_name'] == 'base_link'
        assert row['fd_position_jacobian_max_m_per_rad'] < 1e-9


@pytest.mark.parametrize('bad_action', [np.full(24, np.nan), np.full(24, 1.1), np.zeros(23)])
def test_invalid_actions_are_rejected_without_clamping(bad_action):
    with pytest.raises(ValueError):
        BimanualPhysicalActionAdapter(models()).decode(bad_action, poses(), poses())
