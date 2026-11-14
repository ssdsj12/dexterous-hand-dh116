"""A seal is pending evidence, never a qualified physical audit."""
import ast
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import pytest

NAME="scripts.dh116_deferred_physical_audit"
def api():
    assert importlib.util.find_spec(NAME) is not None,"two-phase physical audit missing"
    return importlib.import_module(NAME)

def ledger(tmp_path):
    path=tmp_path/"physical.jsonl"
    path.write_bytes(b'{"header":true}\n{"operation":"physical"}\n')
    return path

def test_seal_keeps_pending_and_sha_without_claiming_physics(tmp_path):
    m=api();path=ledger(tmp_path);before=path.read_bytes()
    row=m.seal_pending(path)
    assert row["status"]=="pending_post_exit"
    assert row["sha256"]==hashlib.sha256(before).hexdigest()
    assert row["bytes"]==len(before) and row["jsonl_lines"]==2
    assert row["native_qualification"] is row["teacher_admission"] is False
    assert row["full_audit_performed"] is False
    assert path.read_bytes()==before
    assert m.validate_seal(path,row)==row

@pytest.mark.parametrize("fault",["truncated","empty","name","symlink"])
def test_seal_rejects_bad_file_boundary(tmp_path,fault):
    m=api();path=ledger(tmp_path)
    if fault=="truncated":path.write_bytes(b'{"x":1}')
    elif fault=="empty":path.write_bytes(b"")
    elif fault=="name":path=path.rename(tmp_path/"other.jsonl")
    else:
        target=path.rename(tmp_path/"target.jsonl");path.symlink_to(target)
    with pytest.raises(ValueError):m.seal_pending(path)

@pytest.mark.parametrize("fault",["sha","bytes","lines","qualified","status","append"])
def test_any_changed_seal_or_raw_file_rejected(tmp_path,fault):
    m=api();path=ledger(tmp_path);row=m.seal_pending(path)
    if fault=="sha":row["sha256"]="0"*64
    elif fault=="bytes":row["bytes"]+=1
    elif fault=="lines":row["jsonl_lines"]+=1
    elif fault=="qualified":row["native_qualification"]=True
    elif fault=="status":row["status"]="passed"
    else:
        with path.open("ab") as stream:stream.write(b"{}\n")
    with pytest.raises(ValueError):m.validate_seal(path,row)

def test_original_post_exit_checks_retained_except_producer_summary():
    m=api()
    from scripts import validate_dh116_effective_right_contact as original
    old=inspect.getsource(original.validate_artifacts)
    line='    original.require(journal==acquisition["journal"],"producer summary differs from independent journal replay")'
    assert old.count(line)==1
    expected=old.replace(line,'    validate_seal(native_artifact(output,"physical.jsonl"),acquisition["journal"])')
    assert ast.dump(ast.parse(inspect.getsource(m.validate_artifacts)))==ast.dump(ast.parse(expected))

def test_natural_exit_checked_before_any_ledger_audit(tmp_path,monkeypatch):
    m=api();events=[]
    def failed(*args):events.append("outer");raise ValueError("outer failed")
    monkeypatch.setattr(m.original,"_outer",failed)
    from scripts import prepare_dh116_effective_right_native as op
    monkeypatch.setattr(op,"inspect_interval",lambda *a,**k:events.append("inspector"))
    with pytest.raises(ValueError,match="outer failed"):
        m.validate_artifacts(tmp_path,tmp_path/"outer.json","explicit",profile_sha256="a"*64)
    assert events==["outer"]

def test_a_pending_seal_does_not_make_invalid_json_a_valid_audit(tmp_path):
    m=api();path=ledger(tmp_path)
    path.write_bytes(b"malformed-but-complete-line\n")
    marker=m.seal_pending(path)
    assert marker["full_audit_performed"] is False
    from scripts.prepare_dh116_effective_right_native import inspect_interval
    with pytest.raises(Exception):inspect_interval(path)
