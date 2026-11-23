"""CPU tests for exact-source hands-clear 902 Pinch withdrawal."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import screen_dh116_902_pinch_withdrawal as withdrawal
from scripts import verify_dh116_902_pinch_withdrawal as native


def test_withdrawal_action_preserves_both_index_slots_and_tiny_original_wrist_values():
    trial = np.zeros(24)
    trial[[2, 8]] = -.75
    trial[23] = 1e-15
    np.testing.assert_array_equal(withdrawal.checked_withdrawal(trial), trial)
    trial[3] = .01
    with pytest.raises(ValueError, match="index-only"):
        withdrawal.checked_withdrawal(trial)


def test_window_cannot_start_without_the_real_parked_anchor():
    gate = native.PinchWithdrawalGate(None, {}, None)
    with pytest.raises(ValueError, match="240"):
        gate.before_window(240)
    with pytest.raises(ValueError, match="240"):
        gate.before_window(432)


def test_real_first_leg_preserves_the_exact_r6_action_and_clearance():
    sources = [os.environ.get(k) for k in ("DH116_PINCH_ALIGNMENT_RECEIPT", "DH116_PINCH_R7_R6")]
    if not all(sources):
        pytest.skip("sealed R6 command and original native archive are external")
    report = withdrawal.screen(*map(Path, sources))
    assert [row["sdk"] for row in report["rows"]] == list(range(433, 497))
    assert report["minimum_full_rod_clearance_m"] > .00025
    assert report["minimum_cross_hand_m"] > .25
    baseline = np.asarray(report["nonparticipating_baseline_f32"])
    for row in report["rows"]:
        np.testing.assert_array_equal(np.asarray(row["submitted_fingers_f32"])[:, [3, 4, 5, 9, 10, 11]], baseline)
    assert report["phase"] == "hands_clear_index_withdrawal"
    assert report["later_phases_authorized"] is False


def test_original_prefix_replays_and_gate_rejects_one_binary32_target_bit():
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned
    from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR, WRIST_LINEAR
    from scripts.bake_dh116_deformx_pad_sdfs import checked_json

    required = [os.environ.get(k) for k in ("DH116_PINCH_ALIGNMENT_RECEIPT", "DH116_PINCH_R7_R6")]
    if not all(required):
        pytest.skip("sealed prefix and withdrawal sources are external")
    receipt, r6 = map(Path, required)
    outer = json.loads(receipt.read_bytes())
    folder = receipt.parent / "native"
    sha = outer["artifact_sha256"]

    def source(name):
        return checked_json(folder / name, sha[name])

    proof = withdrawal.screen(receipt, r6)
    parked = source("pinch_exchange_240.json")
    archive = {sdk: source(f"pinch_control_{sdk}.json") for sdk in range(425, 433)}
    prefix = source("frozen_plan.json")["action_schedule"]
    gate = native.PinchWithdrawalGate(parked, archive, dict(
        actions=[*prefix, *([proof["action24"]] * 8)], expected_tail_rows=proof["rows"]))
    gate.check_parked(parked)
    for sdk in range(241, 433):
        if (sdk - 241) % 8 == 0:
            gate.before_window(sdk - 1)
        gate.observe(source(f"pinch_control_{sdk}.json"))
    gate.before_window(432)

    # Synthetic single next row for this gate's I/O checks, not a native rollout.
    trial = decode_owned(archive[432])
    trial["physics_step"] = trial["rod"]["physics_step"] = trial["sdk_hand_geometry"]["physics_step"] = 433
    trial["rod"]["rod_tick"] = (433 - 2) * 12
    trial["substep"] = 0
    trial["reaction_source_physics_step"] = 432
    trial["action_readback"]["request"]["action24"] = np.asarray(proof["action24"])[None]
    next_row = proof["rows"][0]
    for i, side in enumerate(("left", "right")):
        hand = trial["action_readback"]["post"]["hands"][side]
        hand["backend_targets"][:, hand["active_indices"]] = np.asarray(next_row["submitted_fingers_f32"])[:, i*6:i*6+6]
        slots = [hand["joint_names"].index(n) for n in WRIST_LINEAR + WRIST_ANGULAR]
        hand["backend_targets"][:, slots] = np.asarray(next_row["submitted_wrists_f32"])[:, i]
    post_right = trial["action_readback"]["post"]["hands"]["right"]
    slot = post_right["active_indices"][2]
    before = np.float32(post_right["backend_targets"][0, slot])
    post_right["backend_targets"][0, slot] = np.nextafter(before, np.float32(np.inf))
    with pytest.raises(ValueError, match="binary32 backend target differs at SDK 433"):
        gate.observe(encode_owned(trial))
    post_right["backend_targets"][0, slot] = before
    gate.observe(encode_owned(trial))
    with pytest.raises(ValueError, match="did not complete"):
        gate.finish()
