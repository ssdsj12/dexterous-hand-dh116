import numpy as np
import pytest

from scripts import prepare_dh116_pinch_wrist_first as wrist_first


def test_stale_drives_are_included_in_the_first_phase():
    measured = np.zeros(12)
    issued = measured.copy()
    goal = measured.copy()
    issued[1:3], goal[1:3] = [.003, .004], [.008, .01]
    issued[6:9], goal[6:9] = [.0002, .00001, 0.], [.0001, 0., 0.]
    issued[9:12], goal[9:12] = [0., .012, -.0001], [0., .01, 0.]
    envelope = wrist_first.phase_envelope(measured, issued, goal, release_progress=.55)
    lo, hi = envelope["angle_first"]
    np.testing.assert_allclose(hi, [.375, .4, 2., 1.2])
    assert np.all(lo == 0.)
    lo, hi = envelope["wrist_first"]
    np.testing.assert_allclose(lo, [0., 0., 0., .55])
    np.testing.assert_allclose(hi, [.375, .4, 2., 1.02])
    lo, hi = envelope["finger_release"]
    np.testing.assert_allclose(lo, [0., 0., .55, .55])
    np.testing.assert_allclose(hi, [1., 1., 1.02, 1.02])
    np.testing.assert_allclose(envelope["wrist_residuals"][0], [0., .00001, 0.], atol=1.e-12)


def test_due_wrist_motion_and_late_finger_with_no_cross_group_changes():
    measured = np.zeros(12)
    proposed = measured.copy()
    proposed[1] = .0075
    proposed[8] = -.00012
    proposed[10] = .009
    rotate, translate, load = wrist_first.right_control_windows(measured, proposed)
    assert np.all(rotate[:, :21] == 0.)
    assert np.all(translate[:, :15] == 0.)
    assert np.all(translate[:, 18:] == 0.)
    np.testing.assert_allclose(translate[:, 17], [-.72, 0.])
    assert np.all(load[:, 12:] == 0.)
    assert np.all(load[:, :6] == 0.)
    assert rotate[0, 22] == 1. and rotate[1, 22] > 0.
    np.testing.assert_allclose(load[:, 7], [1., .5])


def test_rejects_reversed_closing_progress_and_degenerate_wrist():
    measured = np.zeros(12)
    goal = measured.copy()
    goal[1:3], goal[9:] = [.009, .008], [0., .01, 0.]
    with pytest.raises(ValueError, match="wrist"):
        wrist_first.phase_envelope(measured, measured, goal, release_progress=.55)
    goal[8] = .0001
    issued = goal.copy()
    issued[1] = -.003
    with pytest.raises(ValueError, match="closing"):
        wrist_first.phase_envelope(measured, issued, goal, release_progress=.55)



def test_first_phase_does_not_force_the_full_downward_wrist_travel():
    a = np.zeros(12)
    goal = a.copy()
    goal[1:3] = .008
    goal[8] = -.0002
    goal[10] = .01
    issued = goal * .4
    env = wrist_first.phase_envelope(a, issued, goal, release_progress=.55)
    assert env["angle_first"][1][2] == .4
    assert env["wrist_first"][1][2] == .65
    assert env["finger_release"][0][2] == .55
    assert env["finger_release"][1][2] >= .65

def test_independent_wrist_table_bound_uses_z_of_live_alignment():
    from scipy.spatial.transform import Rotation
    from scripts.prepare_dh116_pinch_wrist_first import independent_translation_bounds

    rotation = Rotation.from_euler("y", 5., degrees=True).as_matrix()
    offsets = np.array([.0004, -.0005, .0002])
    ordinary, table = independent_translation_bounds(rotation, offsets)
    np.testing.assert_array_equal(ordinary, np.abs(offsets))
    np.testing.assert_allclose(table, np.abs((rotation @ np.diag(offsets))[2]))
    assert table[1] == 0. and table[0] > 0.
    # The exact table sensitivity for every independent world-axis increment
    # is measured after the per-link alignment, including tilted links.
    measured = Rotation.from_euler("XYZ", [.01, -.15, .02]).as_matrix()
    _, table = independent_translation_bounds(measured, offsets)
    for scale in ([1., -.3, .4], [-.1, .7, -.2]):
        actual = abs((measured @ (np.array(scale) * offsets))[2])
        assert actual <= np.abs(scale) @ table + 1e-15


def test_independent_wrist_bounds_reject_non_rigid_transform():
    from scripts.prepare_dh116_pinch_wrist_first import independent_translation_bounds
    with pytest.raises(ValueError, match="rigid"):
        independent_translation_bounds(np.diag([1., 2., 1.]), np.ones(3))
