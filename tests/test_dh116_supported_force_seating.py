import numpy as np
import pytest

from scripts.verify_dh116_supported_force_seating import validate_seating_schedule, select_seating_command


def evidence(index=.06, middle=.03, loaded=False, total=.32):
    pads = {"thumb": .11, "index": index, "middle": middle}
    return {"per_hand": {"right": {"loaded": loaded, "per_substep": [
        {"per_pad": {pad: {"normal_load_n": force} for pad, force in pads.items()}, "total_normal_load_n": total}
        for _ in range(12)]}}}


def test_only_two_screened_fingers_can_change_at_fixed_wrist():
    command = np.zeros(24)
    command[[8, 9]] = .06
    validate_seating_schedule(np.stack((command, np.zeros(24))), {8: "index", 9: "middle"}, max_travel_rad=.003)
    issued, limits = select_seating_command(command, evidence(index=.10), {8: "index", 9: "middle"})
    assert limits == {"index": .10, "middle": .03}
    assert np.flatnonzero(issued).tolist() == [9]
    continued, _ = select_seating_command(command, evidence(index=.09, middle=.09, total=.28), {8: "index", 9: "middle"})
    np.testing.assert_array_equal(continued, command)
    np.testing.assert_array_equal(select_seating_command(command, evidence(loaded=True), {8: "index", 9: "middle"})[0],
                                  np.zeros(24))


@pytest.mark.parametrize("slot", [1, 7, 10, 12, 16, 21])
def test_tentative_seating_rejects_any_other_action_channel(slot):
    command = np.zeros((2, 24))
    command[:, slot] = .01
    with pytest.raises(ValueError, match="finger-only"):
        validate_seating_schedule(command, {8: "index", 9: "middle"}, max_travel_rad=.003)


def test_budget_and_same_sample_force_completeness():
    command = np.zeros((11, 24))
    command[:, 9] = .06
    with pytest.raises(ValueError, match="travel"):
        validate_seating_schedule(command, {9: "middle"}, max_travel_rad=.003)
    reading = evidence()
    reading["per_hand"]["right"]["per_substep"].pop()
    with pytest.raises(ValueError, match="12"):
        select_seating_command(command[0], reading, {9: "middle"})
