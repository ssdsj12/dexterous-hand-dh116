"""CPU API doubles only: no native launch and no physical qualification."""
from copy import deepcopy
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView, local_translation
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import (
    BimanualStepConnection, HandBinding, DT,
)


class Hand:
    def __init__(self, side, order, mounted=True):
        self.joint_names = list(reversed(JOINTS + (WRIST_LINEAR + WRIST_ANGULAR if mounted else ())))
        self.body_names = ['base_link']
        self.writes = []
        self.backend = torch.zeros((len(order), len(self.joint_names)))
        types = torch.tensor([1 if name in WRIST_LINEAR else 0 for name in self.joint_names])
        self.types = types.repeat(len(order), 1)
        q = self.backend.double().clone()
        for i, env in enumerate(order):
            full = expand(np.full(6, .25 + env * .001))
            for name, v in zip(JOINTS, full):
                q[i, self.joint_names.index(name)] = v
        root = torch.tensor([[.5, 0., .4, np.sqrt(.5), 0., 0., np.sqrt(.5)]],
                            dtype=torch.float64).repeat(len(order), 1)
        self.data = SimpleNamespace(joint_pos=q, joint_vel=q*.01,
                                   joint_pos_target=q.float().clone(), root_pose_w=root)
        self.backend[:] = self.data.joint_pos_target
        self.measured = q.clone()
        self.velocity = q*.01
        self.root_native = root[:, [0, 1, 2, 4, 5, 6, 3]].clone()
        self.root_physx_view = SimpleNamespace(
            prim_paths=[f'/World/envs/env_{e}/{side}Hand' for e in order],
            get_dof_types=lambda: self.types,
            get_dof_positions=lambda: self.measured,
            get_dof_velocities=lambda: self.velocity,
            get_root_transforms=lambda: self.root_native,
            get_dof_position_targets=lambda: self.backend)
        self.device = 'cpu'

    def set_joint_position_target(self, values, joint_ids):
        self.writes.append((list(joint_ids), values.clone()))
        self.data.joint_pos_target[:, joint_ids] = values

    def write_data_to_sim(self):
        self.backend[:] = self.data.joint_pos_target


def setup(mounted=True):
    env_ids = (3, 8)
    hands = [Hand('left', (8, 3), mounted), Hand('right', (3, 8), mounted)]
    bindings = [HandBinding(side, hand, {e: f'/World/envs/env_{e}/{side}Hand' for e in env_ids},
                            MountView(hand) if mounted else None)
                for side, hand in zip(('left', 'right'), hands)]
    clock = [dict(run_uuid='run-a', env_id=e, episode_uid=f'episode-{e}', reset_id=1,
                  rod_uid=f'rod-{e}', physics_step=2, rod_tick=0, policy_step=0, substep=0,
                  timestamp_s=2*DT, solver_time_s=4., clock_valid=True) for e in env_ids]
    read = lambda: deepcopy(clock)
    connection = BimanualStepConnection(env_ids=env_ids, solver_substeps=12)
    return hands, bindings, clock, read, connection


def begin(connection, bindings, read, action=None):
    action = torch.zeros((2, 24), dtype=torch.float64) if action is None else action
    return connection.begin(bindings, action, read_clock=read,
                            palms=torch.zeros((2, 2, 3)), angles=torch.zeros((2, 2, 3)))


def submit(connection, bindings, read, request):
    poses = torch.tensor([.5, 0., .4, 1., 0., 0., 0.], dtype=torch.float64).repeat(2, 2, 1)
    poses[..., :3] += request['requested_palm_offsets_m']
    return connection.submit(bindings, read_clock=read, projected_root_poses=poses,
                             projected_angles=request['requested_angles_rad'],
                             projection_source='CPU fixture; not collision qualification')


def advance(clock):
    for item in clock:
        item['physics_step'] += 1
        item['rod_tick'] += 12
        item['timestamp_s'] += DT
        item['solver_time_s'] += DT


def test_routes_all_24_groups_by_actual_instance_and_joint_name():
    hands, bindings, clock, read, connection = setup()
    action = torch.linspace(-1, 1, 48, dtype=torch.float64).reshape(2, 24)
    request = begin(connection, bindings, read, action)
    assert not any(h.writes for h in hands)
    assert request['action_contract']['size'] == 24
    for i, group in enumerate(('left_fingers', 'right_fingers', 'left_world_translation',
                              'right_world_translation', 'left_intrinsic_rotation',
                              'right_intrinsic_rotation')):
        bounds = ((0, 6), (6, 12), (12, 15), (15, 18), (18, 21), (21, 24))[i]
        assert np.array_equal(request['action_groups'][group], action[:, slice(*bounds)].numpy())
    submitted = submit(connection, bindings, read, request)
    for hi, hand in enumerate(hands):
        permutation = [1, 0] if hi == 0 else [0, 1]
        active = [hand.joint_names.index(name) for name in ACTIVE]
        assert torch.allclose(hand.backend[:, active].double(),
            request['requested_finger_targets_rad'][permutation, 6*hi:6*hi+6], atol=3e-8)
        root = submitted['projected_root_poses'][permutation, hi]
        expected = local_translation(root, hand.data.root_pose_w)
        ids = [hand.joint_names.index(name) for name in WRIST_LINEAR]
        assert torch.allclose(hand.backend[:, ids].double(), expected, atol=1e-8)
    advance(clock)
    packet = connection.finish(bindings, read_clock=read)
    assert packet['new_physics_steps_executed_by_adapter'] == 0
    assert packet['observed_sdk_step_delta'] == 1
    assert packet['qualified_for_teacher_ppo'] is False
    assert packet['pre']['hands']['left']['row_indices'] == [1, 0]
    assert packet['pre']['hands']['left']['q'].shape == (2, 17)
    assert packet['post']['hands']['left']['backend_source'] == 'physx_backend_get_dof_position_targets'
    assert np.array_equal(packet['pre']['hands']['left']['q'][:, :],
                          hands[0].data.joint_pos[[1, 0]].numpy())
    with pytest.raises(ValueError, match='submitted'):
        connection.finish(bindings, read_clock=read)


@pytest.mark.parametrize('fault', ['same_hand', 'swapped_paths', 'duplicate_env', 'missing_joint',
                                   'wrong_type', 'missing_backend', 'changed_clock'])
def test_pre_rejects_wrong_instance_mapping_or_unavailable_source(fault):
    hands, bindings, clock, read, connection = setup()
    if fault == 'same_hand':
        bindings[1] = HandBinding('right', hands[0], bindings[1].env_paths, None)
    elif fault == 'swapped_paths':
        hands[1].root_physx_view.prim_paths.reverse()
        hands[1].root_physx_view.prim_paths[0] = hands[0].root_physx_view.prim_paths[0]
    elif fault == 'duplicate_env':
        bindings[1] = HandBinding('right', hands[1], {3: bindings[1].env_paths[3]}, bindings[1].mount)
    elif fault == 'missing_joint':
        hands[0].joint_names[0] = 'invented_joint'
    elif fault == 'wrong_type':
        hands[0].types[:, hands[0].joint_names.index('finger11')] = 1
    elif fault == 'missing_backend':
        def fail():
            raise RuntimeError('no backend')
        hands[0].root_physx_view.get_dof_position_targets = fail
    else:
        def unstable():
            clock[0]['physics_step'] += 1
            clock[0]['timestamp_s'] += DT
            return deepcopy(clock)
        read = unstable
    with pytest.raises(ValueError):
        begin(connection, bindings, read)
    assert not any(h.writes for h in hands)


@pytest.mark.parametrize('key,change', [('reset_id', 1), ('physics_step', 1), ('rod_tick', 1),
                                      ('solver_time_s', .01), ('policy_step', 1)])
def test_post_requires_same_episode_and_exact_sdk_solver_clock(key, change):
    hands, bindings, clock, read, connection = setup()
    request = begin(connection, bindings, read)
    submit(connection, bindings, read, request)
    advance(clock)
    clock[0][key] += change
    if key == 'physics_step':
        clock[0]['timestamp_s'] += DT
    with pytest.raises(ValueError):
        connection.finish(bindings, read_clock=read)
    with pytest.raises(ValueError, match='submitted'):
        connection.finish(bindings, read_clock=read)


def test_cache_cannot_mask_backend_target_mismatch():
    hands, bindings, clock, read, connection = setup()
    request = begin(connection, bindings, read)
    submit(connection, bindings, read, request)
    hands[1].backend[:, hands[1].joint_names.index('finger11')] += .01
    advance(clock)
    with pytest.raises(ValueError, match='backend target'):
        connection.finish(bindings, read_clock=read)


def test_unmounted_hands_accept_fingers_and_reject_wrist_before_any_write():
    hands, bindings, clock, read, connection = setup(mounted=False)
    action = torch.zeros((2, 24), dtype=torch.float64)
    action[:, 12] = .1
    with pytest.raises(ValueError, match='mount'):
        begin(connection, bindings, read, action)
    assert not any(h.writes for h in hands)
    request = begin(connection, bindings, read)
    connection.submit(bindings, read_clock=read)
    advance(clock)
    assert connection.finish(bindings, read_clock=read)['post']['hands']['left']['q'].shape == (2, 11)


def test_validation_all_hands_before_first_write_and_poison_after_write_failure():
    hands, bindings, clock, read, connection = setup()
    request = begin(connection, bindings, read)
    bindings[1].mount.hand = hands[0]
    with pytest.raises(ValueError, match='mount'):
        submit(connection, bindings, read, request)
    assert not any(h.writes for h in hands)
    bindings[1].mount.hand = hands[1]
    def fail():
        raise RuntimeError('write failed')
    hands[1].write_data_to_sim = fail
    with pytest.raises(RuntimeError, match='write failed'):
        submit(connection, bindings, read, request)
    with pytest.raises(ValueError, match='prepared'):
        submit(connection, bindings, read, request)


def test_owned_evidence_does_not_retain_native_handles():
    hands, bindings, clock, read, connection = setup()
    refs = [weakref.ref(h) for h in hands]
    request = begin(connection, bindings, read)
    submit(connection, bindings, read, request)
    advance(clock)
    result = connection.finish(bindings, read_clock=read)
    saved = result['post']['hands']['left']['backend_targets'].copy()
    hands[0].backend[:] = -100
    assert np.array_equal(result['post']['hands']['left']['backend_targets'], saved)
    del hands, bindings
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_replacement_articulation_cannot_reuse_same_paths_and_values():
    hands, bindings, clock, read, connection = setup()
    request = begin(connection, bindings, read)
    replacement = Hand('right', (3, 8))
    bindings[1] = HandBinding('right', replacement, bindings[1].env_paths, MountView(replacement))
    with pytest.raises(ValueError, match='instance'):
        submit(connection, bindings, read, request)
    assert not any(h.writes for h in hands)
    assert not replacement.writes


def test_missing_projection_fails_before_writes():
    hands, bindings, clock, read, connection = setup()
    begin(connection, bindings, read)
    with pytest.raises(ValueError, match='projection'):
        connection.submit(bindings, read_clock=read)
    assert not any(h.writes for h in hands)


def test_unstable_submission_clock_poisoned_after_actual_writes():
    hands, bindings, clock, read, connection = setup()
    request = begin(connection, bindings, read)
    original_write = hands[1].write_data_to_sim
    def bad_write():
        original_write()
        advance(clock)
    hands[1].write_data_to_sim = bad_write
    with pytest.raises(ValueError, match='inside target submission'):
        submit(connection, bindings, read, request)
    with pytest.raises(ValueError, match='prepared'):
        submit(connection, bindings, read, request)


def test_actual_sdk_q_and_qd_are_independent_of_stale_isaaclab_caches():
    hands, bindings, clock, read, connection = setup()
    hands[0].data.joint_pos[:] = 999.
    hands[0].data.joint_vel[:] = 999.
    request = begin(connection, bindings, read)
    submit(connection, bindings, read, request)
    advance(clock)
    result = connection.finish(bindings, read_clock=read)
    packet = result['pre']['hands']['left']
    assert np.array_equal(packet['q'], hands[0].measured[[1, 0]].numpy())
    assert packet['measured_source'] == 'physx_backend_get_dof_positions/get_dof_velocities'


def test_stale_mount_anchor_cache_is_rejected_before_writes():
    hands, bindings, clock, read, connection = setup()
    hands[0].root_native[:, 0] += .001
    with pytest.raises(ValueError, match='anchor'):
        begin(connection, bindings, read)
    assert not any(h.writes for h in hands)


@pytest.mark.parametrize('value', [False, None, 1])
def test_runtime_invalid_clock_cannot_pass_after_solver_partial_failure(value):
    hands, bindings, clock, read, connection = setup()
    clock[0]['clock_valid'] = value
    with pytest.raises(ValueError, match='valid'):
        begin(connection, bindings, read)
    assert not any(h.writes for h in hands)
