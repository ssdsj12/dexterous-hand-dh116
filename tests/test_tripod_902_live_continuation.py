"""Bind the original live Tripod SDK 568 before selecting any new command."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from scripts import verify_dh116_tripod_902_approach as approach


@pytest.fixture
def live():
    path = os.environ.get("DH116_TRIPOD_902_LIVE_OUTER")
    if not path:
        pytest.skip("sealed native Tripod 902 result is external")
    root = Path(path).parent / "native"

    def row(stem, sdk):
        return json.loads((root / f"tripod_{stem}_{sdk}.json").read_bytes())

    return row


def test_bound_live_anchor_mismatch_blocks_sdk_569_before_issue(live):
    anchor = live("control", 568)
    gate = approach.TripodReplayGate(live("exchange", 240), live("control", 552),
                                     continuation_anchor=anchor)
    gate.check_parked(live("exchange", 240))
    for sdk in range(552, 568):
        gate.observe(live("control", sdk))
    incorrect = deepcopy(anchor)
    incorrect["action_readback"]["post"]["hands"]["right"]["native_anchor_pose_w"]["__dh116_array__"][0][0] += .001
    with pytest.raises(ValueError, match="568"):
        gate.observe(incorrect)
    with pytest.raises(ValueError, match="568"):
        gate.before_window(568)
    gate.observe(anchor)
    gate.before_window(568)
    assert gate.completed == (240, 552, 568)
    with pytest.raises(ValueError, match="576"):
        gate.finish()


def test_new_bounded_window_finishes_without_reclassifying_contact(live):
    original = live("control", 568)
    gate = approach.TripodReplayGate(live("exchange", 240), live("control", 552),
                                     continuation_anchor=original)
    gate.check_parked(live("exchange", 240))
    for sdk in range(552, 569):
        gate.observe(live("control", sdk))
    gate.before_window(568)
    row = deepcopy(original)
    for sdk in range(569, 577):
        row["physics_step"] = sdk
        row["reaction_source_physics_step"] = sdk - 1
        row["rod"]["physics_step"] = sdk
        row["rod"]["rod_tick"] = (sdk - 2) * 12
        row["sdk_hand_geometry"]["physics_step"] = sdk
        if sdk == 569:
            with_non_support = deepcopy(row)
            with_non_support["rod"]["rod_mesh_contact_pairs"]["pairs"][0]["mesh_name"] = "right.finger24_link"
            with pytest.raises(ValueError, match="precontact"):
                gate.observe(with_non_support)
        gate.observe(row)
    gate.finish()
    assert gate.completed == (240, 552, 568, 576)
    with pytest.raises(ValueError, match="576"):
        gate.before_window(576)


def test_live_tail_only_binds_exact_manifest_and_positive_right_command(live, tmp_path):
    receipt = Path(os.environ["DH116_TRIPOD_902_LIVE_OUTER"])
    proof_path = os.environ.get("DH116_TRIPOD_902_LIVE_PROOF")
    if not proof_path:
        pytest.skip("sealed next-window proof is external")
    prefix = json.loads((receipt.parent / "native" / "frozen_plan.json").read_bytes())["tripod_approach"]
    action, row, files = approach.load_live_continuation(receipt, Path(proof_path), prefix)
    assert len(action) == 24 and not any(action[k] for k in (*range(6), *range(12, 15), *range(18, 21)))
    assert row["physics_step"] == 568 and len(files) >= 5
    changed = deepcopy(prefix)
    changed[-1][-1] += .001
    with pytest.raises(ValueError, match="prefix"):
        approach.load_live_continuation(receipt, Path(proof_path), changed)
    tampered = tmp_path / "proof.json"
    tampered.write_bytes(Path(proof_path).read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA|hash|digest"):
        approach.load_live_continuation(receipt, tampered, prefix)
