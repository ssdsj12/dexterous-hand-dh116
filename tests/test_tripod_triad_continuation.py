import numpy as np
import pytest


def test_tri_pad_direction_improves_both_pose_models_and_respects_group_rates():
    from scripts.audit_dh116_tripod_triad_continuation import RIGHT_CHANNELS, triad_direction

    derivative = np.zeros((6, len(RIGHT_CHANNELS)))
    derivative[0, 1] = derivative[3, 1] = -1.
    derivative[1, 2] = derivative[4, 2] = -2.
    derivative[2, 3] = derivative[5, 3] = -1.
    result = triad_direction(derivative)
    assert result is not None
    assert np.all(derivative @ result[RIGHT_CHANNELS] < 0.)
    np.testing.assert_array_equal(result[:6], 0.)
    np.testing.assert_array_equal(result[12:15], 0.)
    np.testing.assert_array_equal(result[18:21], 0.)
    assert np.linalg.norm(result[15:18]) <= 1.
    assert np.linalg.norm(result[21:24]) <= 1.


def test_tri_pad_direction_rejects_a_conflict_between_pose_models():
    from scripts.audit_dh116_tripod_triad_continuation import RIGHT_CHANNELS, triad_direction

    derivative = np.zeros((6, len(RIGHT_CHANNELS)))
    derivative[:3, 1] = -1.
    derivative[3:, 1] = 1.
    assert triad_direction(derivative) is None


def test_prefix_certificate_distinguishes_exact_upper_failure_from_unresolved():
    from scripts.audit_dh116_tripod_triad_continuation import exact_start_deficit

    sample = dict(threshold_m=-.00015, reserve_m=.0004375,
                  failed_segment=dict(group=1, segment=2, upper_m=.0002752))
    assert exact_start_deficit(sample) == pytest.approx(.0000123)
    sample["failed_segment"]["upper_m"] = .00029
    assert exact_start_deficit(sample) is None
