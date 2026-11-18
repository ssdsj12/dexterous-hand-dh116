import pytest

from go2_pvcnn.control.dh116.recovery_planner import diagnose_lift


def observation(**changes):
    value = dict(
        safety_ok=True,
        contact_valid=True,
        tracking_error_m=0.001,
        slip_m=0.0005,
        support_force_n=1.2,
        required_support_force_n=1.0,
        station_rise_m=[0.01, 0.01],
        min_surface_clearance_m=0.0,
    )
    value.update(changes)
    return value


def test_diagnose_lift_prioritizes_safety_abort():
    result = diagnose_lift(**observation(safety_ok=False, slip_m=0.01))
    assert result == {"failure_class": "SAFETY_ABORT", "action": "STOP"}


@pytest.mark.parametrize(
    ("changes", "failure", "action"),
    [
        ({"tracking_error_m": 0.02}, "TRACKING_FAIL", "STOP"),
        ({"contact_valid": False}, "CONTACT_LOST", "SAFE_LOWER"),
        ({"slip_m": 0.003}, "SLIP", "INCREASE_PRELOAD"),
        ({"support_force_n": 0.5}, "SUPPORT_INSUFFICIENT", "REPLAN_GRASP"),
        ({"station_rise_m": [0.04, 0.01], "min_surface_clearance_m": 0.0}, "PARTIAL_LIFT", "ADJUST_LIFT"),
    ],
)
def test_diagnose_lift_classifies_measured_failure(changes, failure, action):
    result = diagnose_lift(**observation(**changes))
    assert result["failure_class"] == failure
    assert result["action"] == action


def test_clearance_and_rise_are_not_declared_success_without_full_evidence():
    result = diagnose_lift(**observation(station_rise_m=[0.04, 0.04], min_surface_clearance_m=0.003))
    assert result["failure_class"] == "OK"
    assert result["action"] == "CONTINUE"
