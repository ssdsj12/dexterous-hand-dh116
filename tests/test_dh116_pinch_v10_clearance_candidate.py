import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.prepare_dh116_pinch_v10_clearance_candidate import prepare_candidate

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119")
ATTEMPTS = Path("/home/xk/.dh116-native-control-11-17/native-attempts")
LEDGER_SHA = {
    "v9": "24cf6d9c3bdc574ec09cb80c26b4f7cf6275af49af25a411b94294c257e17593",
    "v10": "fa7e6300ac5a48804a00b4f9e0b30bbda010e88168bf2114cb0850d18bbdb878",
}


def test_candidate_keeps_the_same_rod_and_left_hand_and_survives_both_replays():
    parent = json.loads((EVIDENCE / "dc6-pinch-right-pair-verified.json").read_bytes())
    ledgers = {}
    for run, sha in LEDGER_SHA.items():
        package = ATTEMPTS / f"teacher.pinch-ppo-fresh-smoke32-{run}"
        intent, = package.glob("*/*/intent.json")
        output = Path(json.loads(intent.read_bytes())["output"])
        ledgers[run] = (output / "training_events.jsonl", sha)

    candidate, evidence = prepare_candidate(parent, ledgers, project=ROOT)
    assert candidate["left"] == parent["left"] and candidate["rod_seed"] == parent["rod_seed"]
    assert candidate["reset_seed"]["sha256"] != parent["reset_seed"]["sha256"]
    assert candidate["right"]["candidate"]["strict_endpoint_passed"] is True
    assert candidate["right"]["path"]["geometry_preparation_passed"] is True
    assert evidence["baseline"]["v10"]["minimum_target_m"] < CLEARANCE_M
    assert evidence["candidate_replay"]["v9"]["samples"] == 48
    assert evidence["candidate_replay"]["v10"]["samples"] == 19
    assert 0.00218 < evidence["minimum_replay_clearance_m"] < 0.00220
    assert evidence["native_qualification"] is False and evidence["teacher_admission"] is False
