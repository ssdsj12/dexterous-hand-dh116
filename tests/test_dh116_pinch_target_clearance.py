import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import pose_matrix
from scripts.audit_dh116_pinch_target_clearance import target_clearances


def test_wrist_rotation_can_cross_table_floor_with_safe_translation():
    # Point 25 mm ahead of the wrist. A pitch of 0.02 rad drops it 0.5 mm.
    actual = np.array([0., 0., .0024, 1., 0., 0., 0.])
    target = actual.copy()
    target[3:] = np.roll(Rotation.from_rotvec([0., .02, 0.]).as_quat(), 1)
    q = np.zeros(6)
    vertices = np.array([[.025, 0., 0.]])
    values = target_clearances(vertices, 0., pose_matrix(actual), actual, q, target, q,
                               lambda pose, angles: pose_matrix(pose))
    assert values["measured_m"] == pytest.approx(.0024)
    assert values["translation_only_m"] == pytest.approx(.0024)
    assert values["fingers_only_m"] == pytest.approx(.0024)
    assert values["combined_m"] == pytest.approx(.0024 - .025 * np.sin(.02))
    assert values["wrist_only_m"] == pytest.approx(values["combined_m"])
    assert values["combined_m"] < .0021 < values["translation_only_m"]


def test_relative_fk_keeps_measured_link_offset_and_passive_state():
    base = np.array([0., 0., .004, 1., 0., 0., 0.])
    source = pose_matrix(base)
    source[2, 3] += .0005

    def fk(pose, q):
        mat = pose_matrix(pose)
        mat[2, 3] -= q[0] * .01
        return mat

    values = target_clearances(np.zeros((1, 3)), 0., source, base, np.zeros(6),
                               base, np.array([.1, 0., 0., 0., 0., 0.]), fk)
    assert values["measured_m"] == pytest.approx(.0045)
    assert values["fingers_only_m"] == pytest.approx(.0035)
    assert values["combined_m"] == pytest.approx(.0035)
