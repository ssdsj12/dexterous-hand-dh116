import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_dh116_pinch_seeded_actions import audit


ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = Path("/home/xk/.dh116-native-control-11-17/native-attempts")
JOURNAL_SHA = "002f76be32d57ea269e3f9e737cabe0de0a61aedafbede97b2f5d966df23119f"


def test_rebuilds_nine_real_action_samples_in_declared_seed_order():
    intent, = (ATTEMPTS / "teacher.pinch-ppo-fresh-smoke32-v11").glob("*/*/intent.json")
    journal = Path(json.loads(intent.read_bytes())["output"]) / "training_events.jsonl"
    config = json.loads((ROOT / "configs/dh116_execution/pinch-actual-smoke32-core-v11.json").read_bytes())
    pin = config["teacher_contract"]["training_config"]
    result = audit(journal, JOURNAL_SHA, Path(pin["path"]), pin["sha256"], project=ROOT)
    assert len(result["actions"]) == 9
    assert result["replay_max_abs"] < 1e-7
    assert result["weights_sha256"] == "0b432671a36bc9c7a435b4a1ccaa54efabb7a82b962f4ee161f7c6b27c224571"
    assert result["native_qualification"] is False and result["ppo_updates"] == 0


def test_rejects_mismatched_raw_bytes_before_parsing(tmp_path):
    raw = tmp_path / "journal.jsonl"
    raw.write_bytes(b"not-json")
    with pytest.raises(ValueError, match="SHA"):
        audit(raw, hashlib.sha256(b"different").hexdigest(), raw, "0" * 64, project=ROOT)
