import ast
import copy
import hashlib
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module():
    from scripts import audit_dh116_pinch_segment_postexit_v2 as m
    return m

def test_artifact_validator_changes_only_scope_binding_and_clock_accessor():
    m=module()
    old=ast.parse((ROOT/"scripts/dh116_pinch_margin_contact_audit.py").read_text())
    new=ast.parse((ROOT/"scripts/audit_dh116_pinch_segment_postexit_v2.py").read_text())
    oldfn=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=="validate_artifacts")
    newfn=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="validate_artifacts")
    for n in ast.walk(oldfn):
        if isinstance(n,ast.ImportFrom):
            for alias in n.names:
                if alias.name=="prepare_dh116_pinch_margin_contact":alias.name="prepare_dh116_pinch_segment_contact"
        if isinstance(n,ast.Attribute) and n.attr=="sdk_clock_sample" and isinstance(n.value,ast.Name) and n.value.id=="operation":
            n.value.id="margin"
    assert ast.dump(oldfn,include_attributes=False)==ast.dump(newfn,include_attributes=False)

def test_consumed_receipt_is_immutable_and_old_audit_reason_retained():
    m=module();r=m.load_locked()
    assert r["native_qualification"] is False and r["outer_exit_code"]==0
    assert "result exact guarded closure execution scope differs" in r["reason"]
    assert m.sha(m.RECEIPT)==m.RECEIPT_SHA

@pytest.mark.parametrize("damage",["receipt","outer","ledger","source_config"])
def test_changed_existing_artifact_refused(monkeypatch,damage):
    m=module();old=m.sha
    target={"receipt":m.RECEIPT,"outer":m.RECEIPT.parent/"outer_exit.json","ledger":m.NATIVE/"physical.jsonl","source_config":m.EXECUTED_ROOT/m.margin_audit.reader.CONFIG}[damage]
    monkeypatch.setattr(m,"sha",lambda p:"0"*64 if Path(p)==target else old(p))
    with pytest.raises(ValueError):m.load_locked()

def test_same_existing_native_234_rows_pass_repaired_full_audit_without_steps():
    m=module();before={p:m.sha(p) for p in (m.RECEIPT,m.NATIVE/"physical.jsonl",m.NATIVE/"sdk_clock.jsonl")}
    r=m.reaudit()
    assert r["lifecycle_qualified"] is True,r.get("reason")
    assert r["physics_steps"]==234 and r["measurement_steps"]==232
    assert r["new_native_steps"]==0 and r["new_ppo_updates"]==0
    assert r["qualification_scope"]==m.SCOPE
    assert r["original_scheduler_qualification"] is False
    assert r["teacher_admission"] is False and r["grasp_qualified"] is False
    assert all(m.sha(p)==h for p,h in before.items())
