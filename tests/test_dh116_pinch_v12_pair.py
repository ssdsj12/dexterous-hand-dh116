import json
from pathlib import Path

import pytest

from scripts.prepare_dh116_pinch_v12_pair import PINCH_ARCHIVES, prepare_pair

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119")
ATTEMPTS = Path("/home/xk/.dh116-native-control-11-17/native-attempts")


@pytest.fixture
def real_inputs():
    parent = json.loads((EVIDENCE / "dc7-pinch-v10-bound-candidate.json").read_bytes())
    ledgers = {}
    for run in PINCH_ARCHIVES:
        intent, = (ATTEMPTS / f"teacher.pinch-ppo-fresh-smoke32-{run}").glob("*/*/intent.json")
        ledgers[run] = Path(json.loads(intent.read_bytes())["output"]) / "training_events.jsonl"
    return parent, ledgers


def test_candidate_preserves_right_hand_and_rod_and_passes_both_fk_checks(real_inputs):
    original, ledgers = real_inputs
    pair, result = prepare_pair(original, ledgers, project=ROOT)
    assert pair["right"] == original["right"] and pair["rod_seed"] == original["rod_seed"]
    assert pair["reset_seed"]["sha256"] != original["reset_seed"]["sha256"]
    assert pair["left"]["path"]["geometry_preparation_passed"] is True
    assert pair["left"]["candidate"]["strict_endpoint_passed"] is True
    assert pair["left_continuous_clearance_m"] >= .0021 + pair["left_extra_clearance_m"]
    for side in ("left", "right"):
        assert sum(row["samples"] for row in result["replayed"][side].values()) == 139
    assert result["baseline"]["left"]["v11"]["minimum_actual_m"] < .0021
    assert .00231 < result["minimum_replayed_m"]["left"] < .00232
    assert .00217 < result["minimum_replayed_m"]["right"] < .00218
    assert pair["native_qualified"] is False and pair["teacher_admission"] is False
    assert result["contact_or_tracking_validated"] is False


def test_swapping_same_task_archives_cannot_relabel_motion(real_inputs):
    original, ledgers = real_inputs
    ledgers["v9"], ledgers["v10"] = ledgers["v10"], ledgers["v9"]
    with pytest.raises(ValueError, match="pinned input bytes differ"):
        prepare_pair(original, ledgers, project=ROOT)
