"""Real file SHA drift proofs for a shared native runtime attempt."""
import hashlib
import importlib
import json

import pytest

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def setup(tmp_path,monkeypatch):
    mod=importlib.import_module("go2_pvcnn.control.dh116.tabletop_runtime_source_audit")
    monkeypatch.setattr(mod,"SOURCE_NAMES",("runtime.py",))
    project=tmp_path/"project";project.mkdir()
    source=project/"runtime.py";source.write_text("value = 1\n")
    output=tmp_path/"native";output.mkdir()
    attempt=tmp_path/"attempt.json"
    attempt.write_text(json.dumps(dict(source_sha256={"runtime.py":sha(source)})))
    return mod,project,source,attempt,output

def test_actual_source_seal_survives_until_preclose_without_touching_files(tmp_path,monkeypatch):
    mod,project,source,attempt,output=setup(tmp_path,monkeypatch)
    audit=mod.SourceFreezeAudit(project=project,attempt=attempt,output=output)
    audit.check_startup()
    assert audit.before_close() is True
    report=json.loads((output/"source_freeze_audit.json").read_text())
    assert report["admissible"] is True and report["drift"]==[]
    assert report["attempt_sha256"]==sha(attempt)
    assert report["source_sha256"]=={"runtime.py":sha(source)}

@pytest.mark.parametrize("phase",["startup","preclose","attempt"])
def test_source_or_attempt_mutation_fails_and_preserves_drift(tmp_path,monkeypatch,phase):
    mod,project,source,attempt,output=setup(tmp_path,monkeypatch)
    audit=mod.SourceFreezeAudit(project=project,attempt=attempt,output=output)
    if phase=="startup":
        source.write_text("value = 2\n")
        with pytest.raises(ValueError,match="source freeze"):
            audit.check_startup()
    else:
        audit.check_startup()
        if phase=="attempt":
            attempt.write_text("{}")
        else:
            source.write_text("value = 2\n")
        with pytest.raises(ValueError,match="source freeze"):
            audit.before_close()
    report=json.loads((output/"source_freeze_audit.json").read_text())
    assert report["admissible"] is False and report["drift"]

def test_missing_source_inventory_cannot_claim_release(tmp_path,monkeypatch):
    mod,project,source,attempt,output=setup(tmp_path,monkeypatch)
    attempt.write_text("{}")
    with pytest.raises(ValueError,match="source"):
        mod.SourceFreezeAudit(project=project,attempt=attempt,output=output)


def test_both_native_producers_can_bind_the_shared_source_audit():
    import inspect
    from scripts.audit_dh116_isaac_batch_scene import _run
    inspect.signature(_run).bind(None,None,3,2,None,coupled=True,runtime_source_audit=True)
