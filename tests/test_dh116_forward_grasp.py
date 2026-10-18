import numpy as np
import pytest

from go2_pvcnn.control.dh116.forward_grasp import forward_pose_report


def pose():
    t = np.eye(4)
    t[:3, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    return t


def test_forward_down_and_wrist_behind():
    r = forward_pose_report(pose(), [.3, 0, .4], [.43, 0, .308])
    assert r['passed']
    assert r['finger_error_rad'] == 0
    assert r['palm_error_rad'] == 0
    assert r['wrist_rear_margin_m'] == pytest.approx(.13)


@pytest.mark.parametrize('axis', [1, 2])
def test_wrong_direction_rejected(axis):
    t = pose()
    t[:3, axis] *= -1
    t[:3, 0] *= -1  # preserve a proper rotation
    assert not forward_pose_report(t, [.3, 0, .4], [.43, 0, .308])['passed']


@pytest.mark.parametrize('x', [.43, .5])
def test_wrist_not_behind_rejected(x):
    assert not forward_pose_report(pose(), [x, 0, .4], [.43, 0, .308])['passed']


@pytest.mark.parametrize('degrees,expected', [(14.99, True), (15.01, False)])
def test_direction_threshold(degrees, expected):
    a = np.deg2rad(degrees)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = pose()
    t[:3, :3] = rz @ t[:3, :3]
    assert forward_pose_report(t, [.3, 0, .4], [.43, 0, .308])['passed'] is expected


@pytest.mark.parametrize('kind', ['nan', 'reflection', 'shape', 'bottom', 'wrist', 'grasp'])
def test_malformed_inputs_rejected(kind):
    t = pose(); wrist = [.3, 0, .4]; grasp = [.43, 0, .308]
    if kind == 'nan': t[0, 3] = np.nan
    if kind == 'reflection': t[:3, 0] *= -1
    if kind == 'shape': t = t[:3]
    if kind == 'bottom': t[3, 0] = 1
    if kind == 'wrist': wrist = [0, 0]
    if kind == 'grasp': grasp = [np.inf, 0, 0]
    with pytest.raises(ValueError):
        forward_pose_report(t, wrist, grasp)
