"""Check the sealed Tripod replay interlocks without starting Isaac."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def replay():
    from scripts.verify_dh116_tripod_902_approach import TripodReplayGate

    archive = os.environ.get("DH116_TRIPOD_902_OUTER")
    if not archive:
        pytest.skip("sealed 902 trajectory kept in the external evidence directory")
    native = Path(archive).parent / "native"
    parked = json.loads((native / "tripod_exchange_240.json").read_bytes())
    anchor = json.loads((native / "tripod_control_552.json").read_bytes())
    return TripodReplayGate(parked, anchor), parked, anchor


def test_parked_boundary_rejects_wrench_loss_before_issuing_first_action(replay):
    gate, parked, _ = replay
    wrong = deepcopy(parked)
    wrong["rod"]["external_support_wrenches_world"].pop("support_0")
    with pytest.raises(ValueError, match="240"):
        gate.check_parked(wrong)
    assert gate.completed == ()
    gate.check_parked(parked)
    assert gate.completed == (240,)


def test_first_modified_window_requires_exact_preceding_anchor(replay):
    gate, parked, anchor = replay
    gate.check_parked(parked)
    for mutate in (
        lambda row: row["action_readback"]["post"]["hands"]["right"].pop("qd"),
        lambda row: row["rod"].update(rod_tick=6599),
        lambda row: row["sdk_hand_geometry"].update(physics_step=551),
        lambda row: row["issued_composer_wrench"].pop("right"),
    ):
        wrong = deepcopy(anchor)
        mutate(wrong)
        with pytest.raises(ValueError, match="552"):
            gate.observe(wrong)
        assert gate.completed == (240,)
    with pytest.raises(ValueError, match="552"):
        gate.observe(dict(anchor, physics_step=553))
    with pytest.raises(ValueError, match="552"):
        gate.before_window(552)
    gate.observe(anchor)
    gate.before_window(552)
    assert gate.completed == (240, 552)


def test_new_windows_keep_timing_and_precontact_scope(replay):
    gate, parked, anchor = replay
    gate.check_parked(parked)
    gate.observe(anchor)
    later = deepcopy(anchor)
    later["physics_step"] = 553
    later["rod"]["physics_step"] = 553
    later["rod"]["rod_tick"] += 12
    later["reaction_source_physics_step"] = 552
    later["sdk_hand_geometry"]["physics_step"] = 553
    with pytest.raises(ValueError, match="guard or reaction clock"):
        gate.observe(dict(later,guard_failure="right:mimic_error"))
    later["rod"]["rod_mesh_contact_pairs"]["pairs"].append({"mesh_name": "right.finger24_link"})
    with pytest.raises(ValueError, match="precontact"):
        gate.observe(later)

def test_each_candidate_step_retains_both_supports(replay):
    gate, parked, anchor = replay
    gate.check_parked(parked)
    gate.observe(anchor)
    row = deepcopy(anchor)
    row["physics_step"] = 553
    row["reaction_source_physics_step"] = 552
    row["rod"]["physics_step"] = 553
    row["rod"]["rod_tick"] += 12
    row["sdk_hand_geometry"]["physics_step"] = 553
    pairs = row["rod"]["rod_mesh_contact_pairs"]["pairs"]
    assert {pair["mesh_name"] for pair in pairs} == {"support_0", "support_1"}
    row["rod"]["rod_mesh_contact_pairs"]["pairs"] = [pair for pair in pairs if pair["mesh_name"] == "support_0"]
    with pytest.raises(ValueError, match="support"):
        gate.observe(row)
    row["rod"]["rod_mesh_contact_pairs"]["pairs"] = pairs
    for sdk in range(553, 569):
        row["physics_step"] = sdk
        row["reaction_source_physics_step"] = sdk - 1
        row["rod"]["physics_step"] = sdk
        row["rod"]["rod_tick"] = (sdk - 2) * 12
        row["sdk_hand_geometry"]["physics_step"] = sdk
        if sdk == 561:
            gate.before_window(560)
        gate.observe(row)
    gate.finish()
