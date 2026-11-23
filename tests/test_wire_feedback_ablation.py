from copy import deepcopy

import pytest
import torch

from test_native_teacher_model import backend_profile
from test_wire_feedback_teacher import _fixture, _packet

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest


def _compare(model, obs, rows, *, trusted=None):
    from go2_pvcnn.control.dh116.tabletop_wire_feedback_ablation import compare_wire_feedback_actions

    trusted = rows if trusted is None else trusted
    return compare_wire_feedback_actions(
        model, obs, rows, base_observation_sha256=_digest(obs),
        clocks=[row["clock"] for row in trusted],
        pre_action_sha256s=[row["pre_action_sha256"] for row in trusted],
    )


@pytest.mark.parametrize("grasp", ["pinch", "tripod", "power"])
def test_force_value_isolated_from_validity_and_reported_in_canonical_groups(backend_profile, grasp):
    _, obs, model = _fixture(backend_profile, grasp)
    # Deterministic diagnostic weights isolate geometry, force validity and force value.
    with torch.no_grad():
        hidden, _, output = model.actor_feedback
        hidden.weight.zero_()
        hidden.bias.zero_()
        hidden.weight[0, 0] = 1.   # Left palm x in geometry.
        hidden.weight[1, 74] = 1.  # Left first pad validity.
        hidden.weight[2, 15] = 1.  # Left first pad mean normal load.
        output.weight[12, 0] = 1. # Left wrist translation.
        output.weight[0, 1] = 1.  # Left finger response to available measurement.
        output.weight[1, 2] = 1.  # Left finger response to measured load.
    samples = [_packet(0, .4), _packet(1, 0.)]
    original = deepcopy(samples)
    model_hash = _digest(model.state_dict())
    result = _compare(model, obs, samples)
    loaded, zero = result["rows"]
    assert result["grasp"] == grasp and result["weights_sha256"] == model_hash
    assert result["scope"] == "same_state_cpu_policy_diagnostic"
    assert result["generalization_qualified"] is False
    assert samples == original and _digest(model.state_dict()) == model_hash and model.training
    assert loaded["absolute_max_by_group"]["geometry"]["left_translation"] > 0
    assert loaded["absolute_max_by_group"]["force_availability"]["left_fingers"] > 0
    assert loaded["absolute_max_by_group"]["force_value"]["left_fingers"] > 0
    assert loaded["absolute_max_by_group"]["force_value"]["right_fingers"] == 0
    assert zero["absolute_max_by_group"]["force_value"]["left_fingers"] == 0
    assert zero["absolute_max_by_group"]["force_availability"]["left_fingers"] > 0
    assert loaded["reaction_valid"][0] == [True, False, False, False, False, False]


def test_missing_force_stays_unavailable_and_stale_clock_is_rejected(backend_profile):
    _, obs, model = _fixture(backend_profile)
    rows = [_packet(0, 0., available=False), _packet(1, .2)]
    missing = _compare(model, obs, rows)["rows"][0]
    assert not any(valid for hand in missing["reaction_valid"] for valid in hand)
    assert missing["correction24"]["force_value"] == [0.] * 24
    assert missing["correction24"]["force_availability"] == [0.] * 24
    altered = deepcopy(rows)
    altered[1]["source_window"]["end_sdk_step"] -= 1
    with pytest.raises(ValueError, match="preceding control interval"):
        _compare(model, obs, altered, trusted=rows)
