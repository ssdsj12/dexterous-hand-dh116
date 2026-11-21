import numpy as np
import pytest

from scripts.screen_dh116_power_retreat_then_open import SAMPLES, certify_intervals


def test_power_retreat_certificate_subtracts_between_sample_motion_per_shape():
    values = np.tile([.001, .002], (SAMPLES, 3, 1))
    travel = np.array([.032, .0064])
    low = certify_intervals(values, travel)
    np.testing.assert_allclose(low, .00075)
    values[SAMPLES//2, 0, 1] = -.001
    assert certify_intervals(values, travel)[0] == pytest.approx(-.00105)
    with pytest.raises(ValueError):
        certify_intervals(values[:-1], travel)
    with pytest.raises(ValueError):
        certify_intervals(values, [-.001, .01])


def test_power_schedule_never_opens_during_retreat_and_never_moves_wrists_during_opening():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
    from scripts.verify_dh116_power_retreat_then_open import compile_schedule

    low, high = active_limits()
    q = (low + high) / 2
    requested = q.copy()
    requested[1:] -= .1
    joint = dict(names=list(ACTIVE), target=q[None, :])
    row = dict(native_joint_state={side: joint for side in ('left','right')},
               hand_link_poses={side+'.base_link': (np.array([.5,.1,.4]), None) for side in ('left','right')})
    proof = dict(hands={side: dict(opened_active_rad=requested,
                retreat_pose_wxyz=[.504,.1,.4,1.,0.,0.,0.]) for side in ('left','right')})
    actions = compile_schedule(row, proof)
    assert actions.shape == (80, 24)
    np.testing.assert_array_equal(np.flatnonzero(actions[0]), [12,15])
    np.testing.assert_array_equal(actions[32:,[0,6,*range(12,24)]], 0.)
    np.testing.assert_allclose(np.sum(actions[:32,12], axis=0)*.01/60, .004)
    np.testing.assert_allclose(np.sum(actions[32:,1], axis=0)*.3/60, -.1)
