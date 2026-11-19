import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_replay_transplants_observed_world_link_motion_and_does_not_mutate_reset():
    from scripts.audit_dh116_pinch_recentered_tracking import transplant_link_motion

    old = np.array([[.2, .3, .4, 0., 0., 0., 1.]])
    updated = np.array([[.4, .1, .5, *Rotation.from_euler("x", 10, degrees=True).as_quat()]])
    actual = old.copy()
    actual[0, :3] += [.001, 0., -.0001]
    actual[0, 3:] = Rotation.from_euler("y", 5, degrees=True).as_quat()
    replay = transplant_link_motion(old, actual, updated)
    np.testing.assert_allclose(replay[0, :3], [.401, .1, .4999])
    np.testing.assert_allclose(Rotation.from_quat(replay[0, 3:]).as_matrix(),
        (Rotation.from_euler("y", 5, degrees=True) * Rotation.from_euler("x", 10, degrees=True)).as_matrix())
    assert updated[0, 2] == .5
    with pytest.raises(ValueError, match="complete"):
        transplant_link_motion(old, actual, updated[:0])


def test_link_drop_cannot_pass_original_clearance_floor():
    from scripts.audit_dh116_pinch_recentered_tracking import _table_minimum, transplant_link_motion
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

    geometry = {
        "reference": {"reference_layout": {"table_center": [0., 0., -.01],
                                            "table_size": [1., 1., .02]}},
        "source_colliders": {"finger22_link": {"vertices": np.zeros((1, 3))}},
        "cooked_pads": {},
    }
    origin = np.array([[0., 0., CLEARANCE_M + 50e-6, 0., 0., 0., 1.]])
    measured = origin.copy()
    measured[0, 2] -= 60e-6
    replay = transplant_link_motion(origin, measured, origin)
    clearance, kind, link = _table_minimum(geometry, ["finger22_link"], replay)
    assert (kind, link) == ("source_colliders", "finger22_link")
    assert clearance - CLEARANCE_M == pytest.approx(-10e-6)
