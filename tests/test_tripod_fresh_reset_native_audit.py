"""Audit the actual exited fresh17 run; no new App, CUDA step or qualification."""
import hashlib
import importlib
import json
from pathlib import Path
import pytest

RUN = Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ref902/fresh-supported-tripod-reset-native-v2")


def module():
    name = "scripts.audit_dh116_tripod_fresh_reset"
    assert importlib.util.find_spec(name) is not None, "missing independent fresh Tripod native receipt audit"
    return importlib.import_module(name)


def test_real_fresh17_native_receipt_matches_sources_state_geometry_and_natural_exit():
    m = module()
    outer = RUN/"outer.json"
    report = m.audit_run(RUN,receipt=outer,expected_sha256=hashlib.sha256(outer.read_bytes()).hexdigest())
    assert report["native_reset_component_verified"] is True
    assert report["sdk_steps"] == 2 and report["episode_resets"] == 1
    assert report["control_steps"] == report["solver_steps"] == report["ppo_updates"] == 0
    assert report["common_valid_dimensions"] == 66 and report["common_unknown_dimensions"] == 305
    assert report["source_shapes"] == 34 and report["cooked_envelopes"] == 10
    assert report["minimum_support_gap_m"] > .005
    assert report["grasp_qualified"] is report["teacher_admission"] is False


@pytest.mark.parametrize("damage",["forced_exit","boolean_exit_code"])
def test_exit_failure_cannot_be_promoted_by_rehashing_the_outer_receipt(tmp_path,damage):
    m = module()
    data = json.loads((RUN/"outer.json").read_bytes())
    if damage == "forced_exit":
        data["forced_exit"] = True
    else:
        data["exit_code"] = False
    receipt = tmp_path/"forged-outer.json"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="natural exit"):
        m.audit_run(RUN,receipt=receipt,expected_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
