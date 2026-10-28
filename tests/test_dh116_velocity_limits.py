from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_load_parameters import physics_load_readback


def hand_readback(instances=3):
    # Deliberately use a different ordering from either JOINTS or PhysX.
    names = list(reversed(JOINTS))
    active = np.array([name in ACTIVE for name in names])
    values = dict(
        get_dof_stiffnesses=np.tile(np.where(active, 10., 0.), (instances, 1)),
        get_dof_dampings=np.tile(np.where(active, .2, 0.), (instances, 1)),
        get_dof_max_forces=np.tile(np.where(active, .2, 0.), (instances, 1)),
        get_dof_max_velocities=np.tile(np.where(active, .3, 2.97), (instances, 1)),
        get_dof_limits=np.zeros((instances, 11, 2)),
        get_masses=np.ones((instances, 2)),
        get_inertias=np.ones((instances, 2, 9)),
        get_coms=np.zeros((instances, 2, 7)),
    )
    view = SimpleNamespace(**{key: lambda key=key: torch.tensor(values[key], dtype=torch.float32)
                              for key in values})
    return SimpleNamespace(joint_names=names, body_names=['base', 'finger'], root_physx_view=view), values


def test_candidate_readback_verifies_all_instances_by_joint_name():
    hand, values = hand_readback()
    report = physics_load_readback([hand], passive_velocity_limit_rad_s=2.97)
    assert report['hands'][0]['instances'] == 3
    assert report['hands'][0]['all_instance_velocity_limits_verified'] is True
    assert len(report['hands'][0]['velocity_limits_rad_s']) == 1


@pytest.mark.parametrize('joint,value', [('finger13', 3.), ('finger11', .31),
                                        ('finger52', float('nan')), ('finger12', float('inf'))])
def test_candidate_readback_rejects_fault_in_unsampled_last_clone(joint, value):
    hand, values = hand_readback()
    values['get_dof_max_velocities'][-1, hand.joint_names.index(joint)] = value
    with pytest.raises(ValueError, match='velocity'):
        physics_load_readback([hand], passive_velocity_limit_rad_s=2.97)


@pytest.mark.parametrize('shape', [(11,), (1, 11), (3, 10), (3, 11, 1)])
def test_candidate_readback_rejects_velocity_shape_mismatch(shape):
    hand, values = hand_readback()
    values['get_dof_max_velocities'] = np.full(shape, .3)
    with pytest.raises(ValueError, match='velocity'):
        physics_load_readback([hand], passive_velocity_limit_rad_s=2.97)


def test_candidate_readback_rejects_wrong_named_joint_set():
    hand, _ = hand_readback()
    hand.joint_names[hand.joint_names.index('finger13')] = 'unknown_passive'
    with pytest.raises(ValueError, match='velocity'):
        physics_load_readback([hand], passive_velocity_limit_rad_s=2.97)


def test_legacy_readback_does_not_enable_new_velocity_validation():
    hand, values = hand_readback()
    values['get_dof_max_velocities'][-1] = 123.
    report = physics_load_readback([hand])
    assert 'all_instance_velocity_limits_verified' not in report['hands'][0]


@pytest.mark.parametrize('mode', ['joint_target', 'adaptive_grasp'])
def test_passive_candidate_is_explicit_and_default_is_unchanged(mode):
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import passive_velocity_limit
    assert passive_velocity_limit(mode) == 3.
    assert passive_velocity_limit(mode, 2.97) == 2.97


def test_other_control_modes_keep_original_passive_cap():
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import passive_velocity_limit
    assert passive_velocity_limit('residual', 2.97) == .3


@pytest.mark.parametrize('value', [0., 2.99, 3.01, float('nan'), float('inf'), True, '2.97'])
def test_candidate_rejects_unreviewed_or_invalid_caps(value):
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import passive_velocity_limit
    with pytest.raises(ValueError, match='passive velocity'):
        passive_velocity_limit('adaptive_grasp', value)


def test_headroom_metadata_does_not_claim_guaranteed_measured_velocity_bound():
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import VELOCITY_HEADROOM_CONTRACT
    assert VELOCITY_HEADROOM_CONTRACT['passive_velocity_limit_rad_s'] == 2.97
    assert VELOCITY_HEADROOM_CONTRACT['nominal_guard_limit_rad_s'] == 3.
    assert VELOCITY_HEADROOM_CONTRACT['guaranteed_measured_velocity_bound'] is False
