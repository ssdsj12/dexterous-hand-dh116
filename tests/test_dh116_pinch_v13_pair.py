import json
from pathlib import Path

from scripts import prepare_dh116_pinch_v13_pair as v13


EVIDENCE = Path.home() / ".local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119"


def test_right_candidate_keeps_left_rod_and_native_boundaries():
    parent = json.loads((EVIDENCE / "dc10-pinch-v12-reset-native/pair.json").read_bytes())
    pair = v13.prepare_pair(EVIDENCE / "dc10-pinch-v12-reset-native/pair.json",
        EVIDENCE / "ag1900-pinch-v12-index-seed-reserve.json",
        EVIDENCE / "ag1900-pinch-four-history-replay-index-reserve.json")
    assert pair["left"] == parent["left"]
    assert pair["rod_seed"] == parent["rod_seed"]
    assert pair["wrist_clearance_floor_m"] == parent["wrist_clearance_floor_m"] == .0021
    assert pair["reset_seed"]["sha256"] != parent["reset_seed"]["sha256"]
    assert pair["right"]["candidate"]["active_delta_rad"][1] < parent["right"]["candidate"]["active_delta_rad"][1]
    assert pair["right_continuous_clearance_m"] >= .0021
    assert pair["native_qualified"] is pair["grasp_qualified"] is pair["teacher_admission"] is False
