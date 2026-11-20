from copy import deepcopy
import hashlib

import pytest

from scripts.run_dh116_work_packages import Repository


PACKAGE = "teacher.pinch-real-smoke-a"
RECEIPT_SHA = "a" * 64


@pytest.fixture
def archive(tmp_path, monkeypatch):
    repo = Repository(tmp_path)
    checkpoint = tmp_path / "teacher_clean.pt"
    checkpoint.write_bytes(b"already byte-audited test checkpoint")
    current = {
        "source_sha256": "b" * 64,
        "semantic_sha256": "c" * 64,
        "reset_sha256": "d" * 64,
        "reward_sha256": "e" * 64,
        "backend_sha256": "f" * 64,
        "target_sha256": "9" * 64,
        "limits": {"controls": 8},
    }
    producer = {
        "executor": "native_bounded",
        "status": "passed",
        "last_result": {"result_sha256": RECEIPT_SHA, "diagnostic_verified": True,
                        "native_qualification": False, "teacher_admission": False},
    }
    meta = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "policy_weights_sha256": "4" * 64,
        "rng_state_sha256": "6" * 64,
        "counters": {"environment_steps": 8, "updates": 1},
        "policy_version": 1,
        "accepted_minibatch_updates": 2,
        "physical_state_saved": False,
        "restart_requires_verified_reset": True,
        "identity": {"training_contract": {
            "scope": "owner_source_unqualified",
            "config_sha256": current["semantic_sha256"],
            "source_sha256": {"worker_source_closure": current["source_sha256"]},
            **{key: current[key] for key in ("reset_sha256", "reward_sha256", "target_sha256")},
        }},
    }
    evidence = {
        "actual_setup_source_verified": True,
        "cap_source_verified": True,
        "physical_to_model_snapshot_binding_verified": True,
        "checkpoint_audit": meta,
        "setup_evidence": {"episode_uid": "fresh-episode-A"},
    }
    summary = {"package": PACKAGE, "receipt_sha256": RECEIPT_SHA, "tested_head": "1" * 40,
               "diagnostic_verified": True, "native_qualification": False, "teacher_admission": False}
    monkeypatch.setattr(repo, "registry", lambda: {"packages": {PACKAGE: producer}})
    monkeypatch.setattr(repo, "native_recipe", lambda package: {"spec": {"version": 2, "teacher_contract": current}})
    calls = []

    def fresh(name, sha):
        calls.append((name, sha))
        return summary, evidence, tmp_path

    monkeypatch.setattr(repo, "_verified_actual_fresh_diagnostic", fresh, raising=False)
    consumer = {"version": 2, "teacher_contract": deepcopy(current)}
    return repo, checkpoint, producer, meta, evidence, consumer, calls


def test_resume_origin_reopens_same_a_bytes_without_grasp_promotion(archive):
    repo, path, _, meta, _, consumer, calls = archive
    origin = repo.read_actual_resume_origin(PACKAGE, RECEIPT_SHA, resume_spec=consumer)
    assert calls == [(PACKAGE, RECEIPT_SHA)]
    assert origin["checkpoint_path"] == str(path)
    assert origin["producer_episode_uid"] == "fresh-episode-A"
    assert origin["checkpoint_sha256"] == meta["checkpoint_sha256"]
    assert origin["policy_weights_sha256"] == meta["policy_weights_sha256"]
    assert origin["rng_state_sha256"] == meta["rng_state_sha256"]
    assert origin["loaded_policy_version"] == 1 and origin["loaded_accepted_minibatch_updates"] == 2
    assert origin["loaded_environment_steps"] == 8 and origin["loaded_updates"] == 1
    assert origin["origin_scope"] == "actual_resume_diagnostic_unqualified"
    assert origin["diagnostic_verified"] is True
    assert all(origin[key] is False for key in
               ("native_qualification", "native_ready", "grasp_qualified", "teacher_admission"))


@pytest.mark.parametrize("changed", ["source_sha256", "semantic_sha256", "reset_sha256", "budget"])
def test_changed_resume_contract_fails_before_opening_old_a(archive, changed):
    repo, _, _, _, _, consumer, calls = archive
    if changed == "budget":
        consumer["teacher_contract"]["limits"]["controls"] = 32
    else:
        consumer["teacher_contract"][changed] = "5" * 64
    with pytest.raises(ValueError, match="semantic or source"):
        repo.read_actual_resume_origin(PACKAGE, RECEIPT_SHA, resume_spec=consumer)
    assert calls == []


@pytest.mark.parametrize("corruption", ["checkpoint bytes", "CPU fixture", "no Reset", "missing episode UID", "invalid RNG digest", "not published"])
def test_incomplete_origins_do_not_enter_resume(archive, corruption):
    repo, path, published, meta, evidence, consumer, _ = archive
    if corruption == "checkpoint bytes":
        path.write_bytes(b"tampered")
    elif corruption == "CPU fixture":
        meta["identity"]["training_contract"]["scope"] = "cpu_fixture"
    elif corruption == "no Reset":
        evidence["actual_setup_source_verified"] = False
    elif corruption == "missing episode UID":
        evidence["setup_evidence"].clear()
    elif corruption == "invalid RNG digest":
        meta["rng_state_sha256"] = "0" * 64
    else:
        published["last_result"]["diagnostic_verified"] = False
    with pytest.raises(ValueError):
        repo.read_actual_resume_origin(PACKAGE, RECEIPT_SHA, resume_spec=consumer)
