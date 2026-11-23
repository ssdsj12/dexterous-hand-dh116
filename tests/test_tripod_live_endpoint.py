import hashlib
import json

import pytest


def test_live_endpoint_audit_rejects_a_changed_ledger_artifact(tmp_path):
    from scripts.audit_dh116_tripod_live_endpoint import sealed_inventory

    directory = tmp_path / "native"
    directory.mkdir()
    step = directory / "tripod_control_568.json"
    step.write_text("{}")
    outer = tmp_path / "outer.json"
    outer.write_text(json.dumps(dict(exit_code=0, debugger_inferior_exit_code=0, debugger_status_verified=True,
        debugger_inferior_signal=None, forced_exit=False, exceeded_budget=False,
        artifact_sha256={step.name: hashlib.sha256(step.read_bytes()).hexdigest()})))
    digest = hashlib.sha256(outer.read_bytes()).hexdigest()
    assert tuple(sealed_inventory(tmp_path, digest)) == (step.name,)
    step.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash|digest|SHA"):
        sealed_inventory(tmp_path, digest)
