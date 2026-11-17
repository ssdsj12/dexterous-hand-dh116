"""Necessary-condition negatives; fixtures never qualify physical grasp."""
from copy import deepcopy
from pathlib import Path
import pytest
from go2_pvcnn.control.dh116.tabletop_r0_effective_pad_source import (
    negative_side_result, FrozenEffectivePadSource, EffectivePadR0Diagnostic,
)
from scripts.audit_dh116_effective_right_tracking import RECEIPT, RECEIPT_SHA, SCOPE
ROOT = Path(__file__).resolve().parents[1]

def test_empty_complete_capture_proves_negative_not_unknown():
    r = negative_side_result([], complete=True, rod_uid="rod")
    assert r["value"]["pads"] == [] and r["value"]["opposed"] is False
    assert r["known_no_effective_contact"]
    assert all(r["value"][k] is None for k in ("material_reference_uid", "material_s_m", "tangential_slip_m"))
    assert not r["positive_effective_contact_qualified"]

def test_multiple_rejected_pad_collisions_are_measured_zero():
    r = negative_side_result([False, False, False], complete=True, rod_uid="rod")
    assert r["value"]["pads"] == [] and r["raw_pair_count"] == 3

@pytest.mark.parametrize("rows", [[True], [False, True], [True, True]])
def test_region_candidate_is_not_positive_effective_contact(rows):
    r = negative_side_result(rows, complete=True, rod_uid="rod")
    assert r["value"] is None and not r["known_no_effective_contact"]
    assert r["reason"] == "region_candidate_requires_depth_and_effective_contact_source"

@pytest.mark.parametrize("complete", [False, None, 1])
def test_incomplete_capture_never_becomes_zero(complete):
    with pytest.raises(ValueError):
        negative_side_result([], complete=complete, rod_uid="rod")

@pytest.mark.parametrize("rows", [[None], [1], ["False"]])
def test_missing_patch_classification_is_not_false(rows):
    with pytest.raises(ValueError):
        negative_side_result(rows, complete=True, rod_uid="rod")

@pytest.fixture(scope="module")
def source():
    return FrozenEffectivePadSource(ROOT, RECEIPT, expected_receipt_sha256=RECEIPT_SHA,
                                    expected_scope=SCOPE)

def test_real_last_step_recomputes_original_region(source):
    x = deepcopy(source.exchanges[-1])
    # A saved derived annotation is not the authority.
    x["closure_geometry"]["pad_region_diagnostic"] = {"untrusted": True}
    # Full row identity rejects changing even its derived evidence.
    with pytest.raises(ValueError, match="frozen exchange"):
        source.capture(x)
    r = source.capture(source.exchanges[-1])
    assert r["identity"]["physics_step"] == 234
    assert r["hands"]["right"]["raw_pair_count"] > 0
    assert r["hands"]["right"]["known_no_effective_contact"]
    assert r["hands"]["left"]["known_no_effective_contact"]
    assert not r["teacher_admission"]

@pytest.mark.parametrize("mutation", ["pairs", "clock", "side", "aggregate"])
def test_changed_native_sample_is_rejected(source, mutation):
    x = deepcopy(source.exchanges[-1])
    if mutation == "pairs": x["rod"]["rod_mesh_contact_pairs"]["pairs"].pop()
    elif mutation == "clock": x["closure_geometry"]["raw"]["identity"]["timestamp_s"] += .01
    elif mutation == "side": x["rod"]["mesh_bindings"]["right.finger14_link"]["side"] = "left"
    else: x["rod"]["named_wrenches"]["table"][0] += .1
    with pytest.raises(ValueError, match="frozen exchange"): source.capture(x)

def test_original_r0_unknowns_are_preserved_except_proven_contact(source):
    x = source.exchanges[-1]
    observer = object.__new__(EffectivePadR0Diagnostic)
    observer._grasp = "pinch"
    observer.effective_pad_source = source
    m, diag, na = observer._measurements({"exchange": x},
        x["action_readback"]["post"]["identity"][0], None)
    assert m["right"]["value"]["pads"] == []
    assert m["left"]["value"]["opposed"] is False
    assert "left" not in na and "right" not in na
    assert m["stable"]["value"] is None
    assert "stable" in na and "material_anchor_slip" in na
    assert diag["effective_pad_negative_source"]["teacher_admission"] is False

def test_wrong_receipt_cannot_admit_any_negative_source():
    with pytest.raises(ValueError, match="receipt"):
        FrozenEffectivePadSource(ROOT, RECEIPT, expected_receipt_sha256="0"*64,
                                 expected_scope=SCOPE)
