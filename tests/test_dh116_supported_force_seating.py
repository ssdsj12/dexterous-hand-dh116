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


def test_pinch_seating_uses_only_the_right_index_and_its_own_receipt(monkeypatch, tmp_path):
    from scripts import verify_dh116_supported_force_seating as seating

    repo = tmp_path
    request = repo / "configs/dh116_tasks/flexible_pinch_right_force_seating_v1.json"
    request.parent.mkdir(parents=True)
    import json
    request.write_text(json.dumps(dict(version="dh116-pinch-right-force-seating-v1",control_boundary_sdk=656,
        maximum_finger_travel_rad=.0012,planned_index_action=.03,dose_windows=8,settle_windows=16)))
    monkeypatch.setattr(seating.measured, "ROOT", repo)
    sealed = {"physics_step": 656, "source": "Pinch original right"}
    receipt = repo / "receipt.json"

    def archive(path, *, contract, sdk, task):
        assert path == receipt
        assert sdk == 656 and task == "pinch"
        assert contract.name == "dh116_teacher_pinch_right_first_contact_native_v1.json"
        return [sealed], [], (receipt,)

    monkeypatch.setattr(seating.measured, "archive", archive)
    commands, plan, sources = seating.load_pinch_seating(receipt)
    assert commands.shape == (24, 24)
    assert np.flatnonzero(np.any(commands,axis=0)).tolist() == [8]
    np.testing.assert_allclose(commands[:8,8], .03)
    np.testing.assert_array_equal(commands[8:], 0.)
    assert plan["archive"] == sealed and plan["pad_by_slot"] == {"8":"index"} and receipt in sources
