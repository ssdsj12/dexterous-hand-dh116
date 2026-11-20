import pytest

from scripts.audit_dh116_teacher_window_wall import audit_window_wall


def _inputs():
    phase = dict(source_binding=100, pre_action=200, physical_control=800, post_action=300)
    trace = [dict(phase="control_window_wall", version="dh116-control-window-wall-v1",
                  scope="owner_source_unqualified", control_index=0, phase_ns=phase,
                  physical_observer_ns=500, elapsed_ns=sum(phase.values()), sdk_steps_delivered=3,
                  transition_recorded=True, terminated=True, truncated=False, quarantined=False,
                  terminal_reason="post_step_source_clearance:right",
                  native_qualification=False, teacher_admission=False)]
    activities = [("decision", {"collection_token": "step-0"})]
    for step in range(3, 6):
        activities.append(("delivered_exchange", {
            "physics_step": step, "guard_failure": "post_step_source_clearance:right" if step == 5 else None,
        }))
    activities.extend([("settlement", {"reward": {
        "token": "step-0", "terminated": True, "truncated": False,
    }}), ("transition_recorded", {"collection_token": "step-0", "rollout_length": 1})])
    journal = [dict(sequence=i, operation=operation, payload=payload,
                    scope="owner_source_unqualified") for i, (operation, payload) in enumerate(activities)]
    return trace, journal


def test_terminal_window_is_paired_to_original_physics_and_rollout():
    trace, journal = _inputs()
    result = audit_window_wall(trace, journal)
    assert result["measured_sdk_steps"] == 3
    assert result["terminal_sdk_step"] == 5
    assert result["terminal_reason"] == "post_step_source_clearance:right"
    assert result["complete_windows"] == 0
    assert result["median_complete_ns"] is None


@pytest.mark.parametrize("damage", ["invalid_ns", "wrong_tick", "wrong_guard", "extra_trace", "wrong_terminal"])
def test_ledger_and_timing_disagreements_fail_closed(damage):
    trace, journal = _inputs()
    if damage == "invalid_ns":
        trace[0]["physical_observer_ns"] = 801
    if damage == "wrong_tick":
        journal[2]["payload"]["physics_step"] = 7
    if damage == "wrong_guard":
        trace[0]["terminal_reason"] = "post_step_source_clearance:left"
    if damage == "extra_trace":
        trace.append(dict(trace[0]))
    if damage == "wrong_terminal":
        trace[0]["terminated"] = False
    with pytest.raises(ValueError):
        audit_window_wall(trace, journal)
