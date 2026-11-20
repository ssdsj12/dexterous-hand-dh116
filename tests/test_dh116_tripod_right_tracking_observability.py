import numpy as np
import pytest

from scripts.audit_dh116_tripod_right_tracking_observability import axis_profile, classify_direction


def test_no_command_excitation_does_not_create_a_lag_bound():
    result = axis_profile(np.full(8, .3), np.full(8, .299))
    assert result["motion_steps"] == 0
    assert result["apparent_delay_ticks"] is None
    assert classify_direction(.005, result) == "no_motion_observed"


def test_same_axis_history_only_supports_the_observed_direction():
    target = np.linspace(.31, .32, 11)
    result = axis_profile(target, target - .0005)
    assert result["motion_steps"] == 10
    assert result["apparent_delay_ticks"] == pytest.approx([.5, .5])
    assert classify_direction(-.005, result) == "direction_not_observed"
    assert classify_direction(.005, result) == "historical_direction_only"
    assert classify_direction(0., result) == "stationary_in_candidate"
