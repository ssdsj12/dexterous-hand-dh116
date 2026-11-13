"""Pair-derived diagnostics must not become a slip or R0 certificate."""
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_contact_diagnostics import summarize_contact_pairs
from test_dual_hand_rod_runtime import runtime, reset


def sample(monkeypatch):
    rod, base, _, poses, centers = runtime(monkeypatch, pairs=True)
    reset(rod, poses)
    return rod.step(poses, centers, physics_step=3)


def test_real_slot_identity_and_node_material_station_are_separate(monkeypatch):
    row = sample(monkeypatch)
    result = summarize_contact_pairs(row, grasp="pinch")
    assert result["hands"]["left"]["per_pad"]["finger14_link"]["mean_normal_force_n"] == 1.
    assert result["hands"]["right"]["per_pad"]["finger14_link"]["mean_normal_force_n"] == 2.
    assert result["hands"]["left"]["per_pad"]["finger14_link"]["loaded_node_rest_stations_m"] == [0.]
    assert result["simultaneous_required_pad_substeps"] == 0
    assert result["table_sdf_loaded"] is False
    assert result["external_support"] is None
    assert result["material_anchor_slip"] is None
    assert result["training_reward"] is None and not result["native_qualified"]


def test_unknown_capture_keeps_counts_unknown(monkeypatch):
    row = sample(monkeypatch)
    row["rod_mesh_contact_pairs"] = None
    row["pair_capture_status"] = "unavailable"
    result = summarize_contact_pairs(row, grasp="tripod")
    assert result["table_sdf_loaded"] is None
    assert result["simultaneous_required_pad_substeps"] is None
    assert result["hands"] is None


@pytest.mark.parametrize("kind", ["clock", "slot", "duplicate", "node", "substep"])
def test_malformed_pair_is_rejected(monkeypatch, kind):
    row = sample(monkeypatch)
    packet = row["rod_mesh_contact_pairs"]
    if kind == "clock":
        packet["physics_step"] += 1
    elif kind == "slot":
        packet["pairs"][0]["mesh_name"] = "right.finger14_link"
    elif kind == "duplicate":
        packet["pairs"].append(dict(packet["pairs"][0]))
    elif kind == "node":
        packet["pairs"][0]["node_index"] = 1000
    else:
        packet["pairs"][0]["substep"] = 1000
    with pytest.raises(ValueError):
        summarize_contact_pairs(row, grasp="power")


def test_same_substep_overlap_is_not_inferred_across_substeps(monkeypatch):
    row = sample(monkeypatch)
    pairs = row["rod_mesh_contact_pairs"]["pairs"]
    template = next(p for p in pairs if p["mesh_index"] == 0)
    for slot, substep in ((1, 0), (7, 1)):
        pair = dict(template, mesh_index=slot, mesh_name=row["mesh_names"][slot], substep=substep)
        pairs.append(pair)
        row["distributed_contact"][0, 0, 2] += 1/12
        row["named_wrenches"][pair["mesh_name"]][2] -= 1/12
    summary = summarize_contact_pairs(row, grasp="pinch")
    assert summary["hands"]["left"]["all_required_loaded_substeps"] == 1
    assert summary["hands"]["right"]["all_required_loaded_substeps"] == 1
    assert summary["simultaneous_required_pad_substeps"] == 0
