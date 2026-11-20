import pytest

from scripts.audit_dh116_tripod_right_seed_corridor import tracking_stress_margin


def test_nominal_tripod_clearance_can_fail_measured_tracking_stress():
    assert tracking_stress_margin(.000030837, .000053518) == pytest.approx(-.000022681)
    assert tracking_stress_margin(.000080, .000053518) > 0


@pytest.mark.parametrize("peak", [-1., float("inf"), float("nan")])
def test_tracking_stress_requires_a_finite_peak_displacement(peak):
    with pytest.raises(ValueError, match="finite"):
        tracking_stress_margin(.000030837, peak)
