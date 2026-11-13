"""Read only the already acquired 16-row ledger; no SDK doubles or new stepping."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def archive():
    from scripts.audit_dh116_dual_pinch_closure_readback import load_archive
    return load_archive()


def test_actual_archive_command_and_response_are_separate(archive):
    from scripts.audit_dh116_dual_pinch_closure_readback import summarize
    rows, context = archive
    report = summarize(rows, context)
    assert report["physical_samples"] == 16
    assert report["new_physics_steps"] == report["optimizer_updates"] == 0
    assert report["native_qualification"] is report["grasp_qualified"] is False
    assert report["hand_table_contacts"] is None
    assert report["raw_pairs_by_mesh"] == {"table": 7872}
    assert report["clock"]["sdk_steps"] == list(range(3, 19))
    assert report["clock"]["rod_ticks"] == list(range(12, 193, 12))
    joint = report["windows"][1]["hands"]["left"]["active_joints"]["finger12"]
    assert joint["backend_target_delta_rad"] > 0
    assert joint["actual_q_delta_rad"] != joint["backend_target_delta_rad"]
    assert report["geometry"]["left"]["other_pad_sdf"] == {
        "finger33_link": None, "finger43_link": None, "finger53_link": None}
    assert len(report["geometry"]["right"]["table_clearance_m"]) == 22


@pytest.mark.parametrize("damage", [
    "missing_physical", "missing_end", "swapped_windows", "episode", "rod_tick",
    "action", "geometry_clock", "geometry_q", "geometry_raw_hash",
    "source_identity", "named_joint", "nonfinite", "table_missing",
    "geometry_minimum", "sdf_minimum", "fake_table_zero",
])
def test_refuses_incomplete_or_mismatched_saved_rows(archive, damage):
    from scripts.audit_dh116_dual_pinch_closure_readback import summarize
    rows, context = deepcopy(archive)
    indices = [i for i, e in enumerate(rows) if e.get("operation") == "physical"]
    ex = rows[indices[8]]["payload"]["exchange"]
    if damage == "missing_physical": del rows[indices[-1]]
    elif damage == "missing_end": rows.pop()
    elif damage == "swapped_windows":
        rows[indices[0]], rows[indices[8]] = rows[indices[8]], rows[indices[0]]
    elif damage == "episode": ex["rod"]["episode_uid"] += ":wrong"
    elif damage == "rod_tick": ex["rod"]["rod_tick"] += 12
    elif damage == "action": ex["action_readback"]["request"]["action24"][0, 1] *= -1
    elif damage == "geometry_clock": ex["closure_geometry"]["raw"]["identity"]["physics_step"] -= 1
    elif damage == "geometry_q": ex["closure_geometry"]["raw"]["joint_readback"]["hands"]["left"]["q"][0, 6] += .01
    elif damage == "geometry_raw_hash": ex["closure_geometry"]["raw_sha256"] = "0" * 64
    elif damage == "source_identity": ex["closure_geometry"]["sources"]["left"]["source_sha256"]["usd"] = "0" * 64
    elif damage == "named_joint": ex["action_readback"]["post"]["hands"]["left"]["active_indices"].reverse()
    elif damage == "nonfinite": ex["action_readback"]["post"]["hands"]["right"]["qd"][0, 6] = np.nan
    elif damage == "table_missing": del ex["closure_geometry"]["sides"]["right"]["table_clearance_m"]["cooked:finger53_link"]
    elif damage == "geometry_minimum": ex["closure_geometry"]["sides"]["left"]["minimum_table_clearance_m"] += .1
    elif damage == "sdf_minimum": ex["closure_geometry"]["sides"]["left"]["required_pad_sdf"]["finger14_link"]["minimum_gap_m"] = -1
    elif damage == "fake_table_zero": ex["hand_table_contacts"]["total_contacts"] = 0
    with pytest.raises(ValueError):
        summarize(rows, context)


def test_external_receipt_digest_rejected_before_artifact_read(tmp_path):
    from scripts.audit_dh116_dual_pinch_closure_readback import load_archive
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        load_archive(receipt=receipt)


def test_artifact_binding_mismatch_rejected(tmp_path, archive):
    from scripts.audit_dh116_dual_pinch_closure_readback import load_archive, RECEIPT
    data = json.loads(RECEIPT.read_text())
    data["artifact_sha256"]["physical.jsonl"] = "0" * 64
    # Preserve actual audit location, exercising the digest check before parsing.
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA|audit"):
        load_archive(receipt=receipt, receipt_sha=hashlib.sha256(receipt.read_bytes()).hexdigest())


def test_exclusive_output_refuses_without_replay(tmp_path, monkeypatch):
    import scripts.audit_dh116_dual_pinch_closure_readback as audit
    target = tmp_path / "receipt.json"
    target.write_text("preserve")
    monkeypatch.setattr(audit, "load_archive", lambda **kwargs: pytest.fail("must refuse before replay"))
    with pytest.raises(SystemExit):
        audit.main(["--output", str(target)])
    assert target.read_text() == "preserve"


def test_cli_cpu_only_and_deterministic_payload(tmp_path, archive):
    from scripts.audit_dh116_dual_pinch_closure_readback import summarize
    output = tmp_path / "result.json"
    run = subprocess.run([sys.executable, "-m", "scripts.audit_dh116_dual_pinch_closure_readback",
                          "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    saved = json.loads(output.read_text())
    assert saved["native_imports"] == []
    assert saved["summary"] == summarize(*archive)
