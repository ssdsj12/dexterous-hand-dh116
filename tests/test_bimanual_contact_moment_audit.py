import importlib.util
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import JOINTS


path = Path(__file__).resolve().parents[1] / 'tools' / 'audit_bimanual_joint_tracking.py'
spec = importlib.util.spec_from_file_location('contact_moment_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class KnownAxisModel:
    side = 'right'
    source_sha256 = 'a' * 64

    def _fk(self, q11, base):
        assert q11.shape == (11,)
        assert base.shape == (7,)
        axes = {'finger12': (np.zeros(3), np.array([0., 0., 1.]))}
        ancestors = {'/finger14_link': ('finger12',), '/finger23_link': ('finger21',),
            '/finger33_link': ('finger31',), '/finger43_link': ('finger41',), '/finger53_link': ('finger51',)}
        return {}, axes, ancestors


def packet():
    return dict(identity={'env_id': 2, 'episode_uid': 'run:2:0', 'physics_step': 9},
        full_joint_names=[list(JOINTS), list(JOINTS)], right_full_position_rad=[0.] * 11,
        actual_base_link_pose_w=[[0., 0., 0., 1., 0., 0., 0.]] * 2,
        actual_base_link_pose_valid=[True, True],
        contacts=dict(effective_pad_measurement_valid=True,
            effective_pad_load_n=[[1.] * 5, [1.] * 5],
            effective_pad_force_on_hand_w_n=[[[0., 1., 0.]] * 5] * 2,
            effective_pad_points_w_m=[[[1., 0., 0.]] * 5] * 2,
            effective_pad_point_valid=[[True] * 5] * 2))


def test_normal_moment_uses_only_descendant_pads_and_actual_force_sign():
    result = module.reconstruct_normal_contact_moment(packet(), KnownAxisModel())
    assert result['valid']
    assert result['normal_pad_moment_nm'] == pytest.approx(1.)
    assert len(result['contributing_pads']) == 1
    assert result['friction_moment_nm'] is None
    assert result['rejected_contact_moment_nm'] is None
    assert result['solver_or_motor_torque_nm'] is None
    assert result['exact_manifold_moment'] is False


def test_missing_or_invalid_contact_centroid_does_not_become_zero_torque():
    row = packet()
    row['contacts']['effective_pad_point_valid'][1][0] = False
    result = module.reconstruct_normal_contact_moment(row, KnownAxisModel())
    assert not result['valid']
    assert result['normal_pad_moment_nm'] is None
    assert result['reason'] == 'relevant_positive_contact_point_invalid'


def test_legacy_evidence_without_measured_base_pose_remains_unavailable():
    row = packet()
    del row['actual_base_link_pose_w']
    result = module.reconstruct_normal_contact_moment(row, KnownAxisModel())
    assert not result['valid']
    assert result['reason'] == 'measured_base_pose_unavailable'
