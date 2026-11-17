"""Portable CPU checks by default; private saved evidence requires explicit opt-in.

Set DH116_RETAINED_PINCH_PRIVATE_TESTS=1 only on the original evidence host.
These optional replays never spend a native budget.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_retained_dh116_pinch_v3 as replay
from scripts import run_dh116_pinch_fast_evidence_work_packages as legacy
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

@pytest.fixture
def evidence():
    clock = dict(setup_steps=2, measured_steps=232, total_steps=234)
    audit = dict(lifecycle_qualified=True, reset_mesh_qualified=True,
                 substep_contact_frame_qualified=True,
                 same_clock_health_getter_qualified=True, physics_steps=234,
                 measurement_steps=232, ppo_updates=0,
                 qualification_scope=replay.entry.SCOPE)
    numeric = dict(numeric_state_unchanged=True, actual_samples=2,
                   new_sdk_steps=0, new_solver_steps=0)
    receipt = dict(duration_seconds=161.766509436,
                   qualification_scope=replay.entry.SCOPE,
                   native_qualification=False, tested_head="5eac3ac40ffaaae69a4458b7838d4dca9e806f04")
    generic = dict(physics_clock=clock,
                   artifact_sha256={"physical.jsonl": "2a99a771962c977cd3f0f652456d2bf9c2a852d49c361758ac0789ff20cf0bf9"})
    return receipt, generic, audit, numeric

def test_retained_result_cannot_rewrite_failed_receipt_or_claim_ppo(evidence):
    saved = deepcopy(evidence)
    result = replay.assemble_verified_sources(*evidence)
    assert evidence == saved
    assert result["verified_saved_physical_sources"] is True
    assert result["original_scheduler_status"] == "failed"
    for flag in ("native_qualification", "teacher_admission", "same_rod_grasp_qualified"):
        assert result[flag] is False
    assert result["ppo_updates"] == result["new_physics_steps"] == 0
    assert result["original_receipt_sha256"] == replay.RECEIPT_SHA
    assert result["independent_physical_audit"] == evidence[2]

@pytest.mark.parametrize("key", ["lifecycle_qualified", "reset_mesh_qualified",
    "substep_contact_frame_qualified", "same_clock_health_getter_qualified"])
@pytest.mark.parametrize("bad", [False, None, 1])
def test_missing_physical_qualification_is_rejected(evidence, key, bad):
    evidence[2][key] = bad
    with pytest.raises(ValueError, match="physical predicate"):
        replay.assemble_verified_sources(*evidence)

@pytest.mark.parametrize("block,key,bad", [
    (0,"duration_seconds",181), (0,"duration_seconds",0),
    (0,"native_qualification",True),
    (0,"qualification_scope","different"),
    (2,"physics_steps",235), (2,"measurement_steps",231),
    (2,"ppo_updates",1),
    (2,"qualification_scope","different"),
    (3,"numeric_state_unchanged",False), (3,"actual_samples",1),
    (3,"new_sdk_steps",1), (3,"new_solver_steps",1),
])
def test_changed_budget_clock_numeric_reset_or_scope_is_rejected(evidence, block, key, bad):
    evidence[block][key] = bad
    with pytest.raises(ValueError):
        replay.assemble_verified_sources(*evidence)

def test_changed_sdk_clock_is_rejected(evidence):
    evidence[1]["physics_clock"]["measured_steps"] = 231
    with pytest.raises(ValueError, match="clock"):
        replay.assemble_verified_sources(*evidence)

def test_consumed_sha_attribute_failure_is_reproduced_without_physics(monkeypatch, tmp_path):
    """The same real wrapper rebinding fails after its inherited audit returned."""
    numeric_wrapper = legacy.prior
    monkeypatch.setattr(numeric_wrapper, "_PREVIOUS_AUDIT", lambda *a: {"lifecycle_qualified": False})
    (tmp_path / replay.entry.NUMERIC_ARTIFACT).write_text("{}")
    with bindings((numeric_wrapper, "entry", replay.entry)):
        with pytest.raises(AttributeError, match="has no attribute 'sha'"):
            numeric_wrapper.audit_outcome(tmp_path, tmp_path/"outer.json", "explicit")

def test_tampered_original_receipt_rejected_before_any_native_or_expensive_replay(tmp_path):
    receipt = tmp_path/"receipt.json"
    receipt.write_text(json.dumps(dict(native_qualification=True)))
    with pytest.raises(ValueError, match="SHA"):
        replay.verify_original_attempt(tmp_path, receipt)

def test_original_receipt_symlink_rejected(tmp_path):
    target = tmp_path/"real.json"; target.write_text("{}")
    path = tmp_path/"receipt.json"; path.symlink_to(target)
    with pytest.raises(ValueError, match="regular original receipt"):
        replay.verify_original_attempt(tmp_path,path)

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f8-pinch-fast-evidence-native/"
    "f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/"
    "attempt-2ynpy5si/receipt.json")

PRIVATE_OPT_IN_ENV = "DH116_RETAINED_PINCH_PRIVATE_TESTS"

@pytest.fixture
def private_saved_evidence():
    """Keep private 621 MB historical audits outside the portable quick suite."""
    if os.environ.get(PRIVATE_OPT_IN_ENV) != "1":
        pytest.skip("private saved evidence is opt-in: set "+PRIVATE_OPT_IN_ENV+"=1")
    if not RECEIPT.is_file():
        pytest.skip("original private receipt is unavailable")
    receipt=json.loads(RECEIPT.read_bytes());output=Path(receipt["output"])
    required=[RECEIPT.parent/name for name in ("intent.json","outer_exit.json","native.log")]
    required.extend(output/name for name in receipt["artifact_sha256"])
    missing=[path for path in required if not path.is_file()]
    if missing:pytest.skip("private historical artifact is unavailable: "+str(missing[0]))
    recorded=json.loads((output/replay.entry.NUMERIC_ARTIFACT).read_bytes())
    registry=replay.scheduler.Repository(ROOT).registry()
    spec=json.loads((ROOT/registry["packages"][replay.PACKAGE]["command_spec"]).read_bytes())
    external=set(recorded["sources"]["sources"])|set(spec.get("inputs",{}))
    missing=[name for name in external if not Path(name).is_file()]
    if missing:pytest.skip("private historical source is unavailable: "+missing[0])


def outer_fixture():
    debug=dict(debugger_status_verified=True,debugger_backtrace_captured=False,
        debugger_inferior_exit_code=0,debugger_inferior_signal=None)
    outer=dict(duration_seconds=161.766509436,outer_exit_code=0,
        native_lock_observed=True,log_sha256="d"*64,**debug)
    return dict(outer,exit_code=0),outer,dict(budget=dict(timeout_seconds=180)),debug,"d"*64

@pytest.mark.parametrize("damage",["outer_exit","receipt_exit","signal","inferior_exit",
    "unverified","duration","nan_duration","lock","log","debug_mismatch"])
def test_outer_failure_not_waived(damage):
    assert hasattr(replay,"verify_outer_evidence"),"explicit outer verifier missing"
    receipt,outer,spec,debug,actual=outer_fixture()
    if damage=="outer_exit":outer["outer_exit_code"]=1
    elif damage=="receipt_exit":receipt["exit_code"]=1
    elif damage=="signal":debug["debugger_inferior_signal"]=11
    elif damage=="inferior_exit":debug["debugger_inferior_exit_code"]=1
    elif damage=="unverified":debug["debugger_status_verified"]=False
    elif damage=="duration":outer["duration_seconds"]=181
    elif damage=="nan_duration":outer["duration_seconds"]=float("nan")
    elif damage=="lock":outer["native_lock_observed"]=False
    elif damage=="log":actual="f"*64
    else:receipt["debugger_inferior_exit_code"]=2
    with pytest.raises(ValueError,match="natural exit"):
        replay.verify_outer_evidence(receipt,outer,spec,debug,actual)

def test_outer_complete_fields_pass():
    assert hasattr(replay,"verify_outer_evidence"),"explicit outer verifier missing"
    assert replay.verify_outer_evidence(*outer_fixture()) is True

@pytest.mark.parametrize("artifact",["physical.jsonl","sdk_clock.jsonl",
    "mesh_reset_state.json","mesh_reset_numeric_state.json"])
@pytest.mark.usefixtures("private_saved_evidence")
def test_real_artifact_mismatch_rejected_in_memory(monkeypatch,artifact):
    from scripts import run_dh116_work_packages as scheduler
    before=RECEIPT.read_bytes();receipt=json.loads(before)
    damaged=Path(receipt["output"])/artifact;original=scheduler.file_hash
    monkeypatch.setattr(scheduler,"file_hash",
        lambda path:"f"*64 if Path(path)==damaged else original(path))
    with pytest.raises(ValueError):replay.verify_original_attempt(ROOT,RECEIPT)
    assert RECEIPT.read_bytes()==before

@pytest.mark.usefixtures("private_saved_evidence")
def test_real_debugger_signal_rejected(monkeypatch):
    from scripts import run_dh116_work_packages as scheduler
    original=scheduler.debugger_status
    def status(path):
        return dict(original(path),debugger_inferior_exit_code=None,debugger_inferior_signal=11)
    monkeypatch.setattr(scheduler,"debugger_status",status)
    with pytest.raises(ValueError,match="natural exit"):
        replay.verify_original_attempt(ROOT,RECEIPT)

@pytest.mark.usefixtures("private_saved_evidence")
def test_complete_saved_sources_preserve_all_original_artifacts(monkeypatch):
    from scripts import run_dh116_work_packages as scheduler
    before=RECEIPT.read_bytes();receipt=json.loads(before);output=Path(receipt["output"])
    watched={RECEIPT:scheduler.file_hash(RECEIPT)}
    watched.update({p:scheduler.file_hash(p) for p in output.iterdir() if p.is_file()})
    actual=replay.audit_retained_numeric;calls=[]
    def audited(output,outer,cleanup):
        calls.append((output,outer,cleanup))
        return actual(output,outer,cleanup)
    monkeypatch.setattr(replay,"audit_retained_numeric",audited)
    result=replay.verify_saved_run(ROOT,RECEIPT)
    assert calls==[(output,RECEIPT.parent/"outer_exit.json","explicit")]
    assert result["qualification_scope"]=="retained_pinch_v3_saved_physical_source_reaudit_only"
    assert result["new_solver_steps"]==0
    assert result["original_artifacts_unchanged"] is True
    assert result["original_receipt_unchanged"] is True
    assert len(result["replay_code_config_sha256"])==64
    assert result["verified_saved_physical_sources"] is True
    assert result["original_scheduler_status"]=="failed"
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert result["new_physics_steps"]==result["ppo_updates"]==0
    assert result["replay_source_sha256"],"new replay source identity missing"
    assert result["original_receipt_sha256"]==replay.RECEIPT_SHA
    assert RECEIPT.read_bytes()==before
    assert all(scheduler.file_hash(p)==sha for p,sha in watched.items())

import shutil

@pytest.mark.parametrize("case",["success","failure","missing","corrupt","numeric_changed"])
@pytest.mark.usefixtures("private_saved_evidence")
def test_real_fast_numeric_context_preserves_actual_parent_audit(tmp_path,case):
    assert hasattr(replay,"audit_retained_numeric"),"stable retained numeric audit missing"
    receipt=json.loads(RECEIPT.read_bytes())
    original=Path(receipt["output"]);output=tmp_path/"native"
    shutil.copytree(original,output)
    assert output.resolve().is_relative_to(tmp_path.resolve())
    assert output.resolve()!=original.resolve()
    artifact=output/replay.entry.NUMERIC_ARTIFACT
    if case=="failure":
        path=output/"result.json";value=json.loads(path.read_bytes())
        value["failure"]="injected parent lifecycle failure"
        path.write_text(json.dumps(value)+"\n")
    elif case=="missing":artifact.unlink()
    elif case=="corrupt":artifact.write_bytes(b"{")
    elif case=="numeric_changed":
        value=json.loads(artifact.read_bytes());value["byte_equal"]=False
        artifact.write_text(json.dumps(value)+"\n")
    numeric_wrapper=legacy.prior
    with replay.producer_profile(ROOT,original),replay.entry.identity_scope(),bindings(
        (numeric_wrapper,"entry",replay.entry),(replay.parent,"entry",replay.entry),
        (replay.entry.operation,"VERSION",replay.entry.VERSION),
        (replay.entry.operation,"SCOPE",replay.entry.SCOPE)):
        assert numeric_wrapper.entry is replay.entry
        assert not hasattr(numeric_wrapper.entry,"numeric")
        assert not hasattr(numeric_wrapper.entry.prior,"sha")
        result=replay.audit_retained_numeric(output,RECEIPT.parent/"outer_exit.json","explicit")
    assert result["native_qualification"] is result["teacher_admission"] is False
    if case=="success":
        assert result["lifecycle_qualified"] is True,result.get("reason")
        assert result["numeric_state_audit"]["numeric_state_unchanged"] is True
        assert len(result["numeric_state_artifact_sha256"])==64
    else:
        assert result["lifecycle_qualified"] is False,result
        assert result["numeric_state_audit"] is None
        assert result.get("reason") or result.get("numeric_sealing_error")
    if case=="failure":
        assert len(result["numeric_state_artifact_sha256"])==64
    if case=="missing":
        assert result["numeric_state_artifact_sha256"] is None

@pytest.mark.parametrize("parent_success,missing_attribute",[(True,"numeric"),(False,"sha")])
@pytest.mark.usefixtures("private_saved_evidence")
def test_consumed_wrapper_two_real_context_errors_remain_reproducible(
        tmp_path,parent_success,missing_attribute):
    receipt=json.loads(RECEIPT.read_bytes())
    original=Path(receipt["output"]);output=tmp_path/"native"
    shutil.copytree(original,output)
    if not parent_success:
        path=output/"result.json";value=json.loads(path.read_bytes())
        value["failure"]="injected parent lifecycle failure"
        path.write_text(json.dumps(value)+"\n")
    numeric_wrapper=legacy.prior
    with replay.producer_profile(ROOT,original),replay.entry.identity_scope(),bindings(
        (numeric_wrapper,"entry",replay.entry),(replay.parent,"entry",replay.entry),
        (replay.entry.operation,"VERSION",replay.entry.VERSION),
        (replay.entry.operation,"SCOPE",replay.entry.SCOPE)):
        with pytest.raises(AttributeError,match="has no attribute '"+missing_attribute+"'"):
            numeric_wrapper.audit_outcome(output,RECEIPT.parent/"outer_exit.json","explicit")


@pytest.mark.parametrize("artifact", ["scripts/verify_dh116_pinch_fast_evidence.py", "configs/dh116_execution/pinch_fast_evidence_native_v3.json"])
@pytest.mark.usefixtures("private_saved_evidence")
def test_frozen_recipe_source_tamper_rejected(monkeypatch, artifact):
    if artifact.startswith("configs/"):
        artifact = replay.scheduler.Repository(ROOT).registry()["packages"][replay.PACKAGE]["command_spec"]
    damaged = ROOT / artifact
    original = replay.scheduler.file_hash
    monkeypatch.setattr(replay.scheduler, "file_hash", lambda p: "f"*64 if Path(p)==damaged else original(p))
    if artifact.startswith("configs/"):
        # native_recipe hashes command bytes directly; mutate the independently recomputed recipe result.
        old = replay.scheduler.Repository.native_recipe
        monkeypatch.setattr(replay.scheduler.Repository, "native_recipe",
            lambda self, package: dict(old(self, package), command_spec_sha256="f"*64))
    with pytest.raises(ValueError, match="approval or source identity"):
        replay.verify_original_attempt(ROOT, RECEIPT)


@pytest.mark.usefixtures("private_saved_evidence")
def test_real_clock_tamper_rejected_without_rewriting_source(monkeypatch):
    original = replay.scheduler.validate_native_artifacts
    def changed(spec, output):
        result = original(spec, output)
        result["physics_clock"]["total_steps"] = 235
        return result
    monkeypatch.setattr(replay.scheduler, "validate_native_artifacts", changed)
    with pytest.raises(ValueError, match="SDK ledger"):
        replay.verify_original_attempt(ROOT, RECEIPT)


@pytest.mark.parametrize("damage", ["duplicate", "symlink", "tamper"])
def test_replay_source_inventory_rejects_ambiguous_or_changed_files(tmp_path, damage):
    assert hasattr(replay, "replay_sources"), "explicit replay source inventory missing"
    (tmp_path/"source.py").write_text("source = 1\n")
    names = ["source.py"]
    frozen={"source.py":replay.scheduler.file_hash(tmp_path/"source.py")}
    if damage == "duplicate": names *= 2
    elif damage == "symlink":
        (tmp_path/"alias.py").symlink_to(tmp_path/"source.py");names=["alias.py"]
        frozen={"alias.py":frozen["source.py"]}
    else: (tmp_path/"source.py").write_text("source = 2\n")
    with pytest.raises(ValueError): replay.replay_sources(tmp_path, names, expected=frozen)


def test_replay_source_inventory_hashes_regular_source(tmp_path):
    assert hasattr(replay, "replay_sources"), "explicit replay source inventory missing"
    path=tmp_path/"source.py";path.write_text("source = 1\n")
    assert replay.replay_sources(tmp_path,["source.py"]) == {"source.py":replay.scheduler.file_hash(path)}


@pytest.mark.parametrize("damage", ["existing", "symlink", "parent_symlink", "project", "relative"])
def test_output_envelope_rejects_nonexclusive_or_nonprivate_path(tmp_path, damage):
    assert hasattr(replay, "write_report"), "exclusive private report writer missing"
    project=tmp_path/"repo";project.mkdir()
    outside=tmp_path/"private";outside.mkdir()
    output=outside/"reaudit.json"
    if damage=="existing":output.write_text("preserved")
    elif damage=="symlink":
        target=outside/"target.json";target.write_text("preserved");output.symlink_to(target)
    elif damage=="parent_symlink":
        alias=tmp_path/"alias";alias.symlink_to(outside,target_is_directory=True);output=alias/"reaudit.json"
    elif damage=="project":output=project/"reaudit.json"
    else:output=Path("reaudit.json")
    with pytest.raises((ValueError,FileExistsError)):
        replay.write_report(project,output,{"verified_saved_physical_sources":True})
    if damage in ("existing","symlink"):assert output.read_text()=="preserved"


def test_output_envelope_writes_exclusive_private_json(tmp_path):
    assert hasattr(replay, "write_report"), "exclusive private report writer missing"
    project=tmp_path/"repo";project.mkdir();output=tmp_path/"reaudit.json"
    result={"original_scheduler_status":"failed","native_qualification":False}
    replay.write_report(project,output,result)
    assert json.loads(output.read_bytes())==result
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises((ValueError,FileExistsError)):replay.write_report(project,output,{})
