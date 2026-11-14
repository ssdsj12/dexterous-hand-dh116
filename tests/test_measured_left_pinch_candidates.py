"""Current moving-left source is required; CPU geometry never admits native use."""
import importlib
import numpy as np
import pytest

def api():return importlib.import_module("scripts.solve_dh116_measured_left_pinch")

@pytest.fixture(scope="module")
def packet():return api().load_packet()

def test_actual_left_motion_source_not_held_left(packet):
    assert packet["executed_left_actions_zero"] is False
    assert packet["left_motion_observed"] is True
    assert packet["source_sha256"]["receipt"]==api().source.RECEIPT_SHA
    assert [s["identity"]["physics_step"] for s in packet["scenarios"]]==list(range(211,235))
    assert all(s["source_hand"]=="left" and s["empirical_only"] for s in packet["scenarios"])
    assert not packet["native_ready"] and not packet["teacher_admission"]

def test_source_correlation_and_current_plan(packet):
    from scripts import prepare_dh116_left_pinch_approach as original
    p=original.load_pair();end=original.expected_boundary(p,232)
    np.testing.assert_array_equal(packet["anchor"]["q_rad"],end["left_q_rad"])
    assert packet["original_candidate"]==p["left"]["candidate"]
    for s in packet["scenarios"]:
        np.testing.assert_allclose(np.asarray(packet["anchor"]["q_rad"])+s["active_error_rad"],
            s["measured_active_q_rad"],atol=1e-14,rtol=0)
        assert s["joint_identity"]==s["identity"]==s["rod_identity"]

def test_cpu_budget_and_no_execution_contract():
    c=api().CONFIG
    assert c["maximum_evaluations"]==600 and c["maximum_search_seconds"]==60.
    assert c["physics_steps"]==c["ppo_updates"]==0
    assert c["historical_errors_are_predictions_for_new_actions"] is True
    assert c["native_execution"] is False

def test_missing_left_motion_cannot_be_promoted(packet):
    from copy import deepcopy
    bad=deepcopy(packet);bad["left_motion_observed"]=False
    with pytest.raises(ValueError):api().validate_packet(bad)

def test_cross_episode_scenarios_rejected(packet):
    from copy import deepcopy
    bad=deepcopy(packet);bad["scenarios"][0]["identity"]["episode_uid"]="other"
    with pytest.raises(ValueError):api().validate_packet(bad)
