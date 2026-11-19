import hashlib
import json

import pytest


@pytest.mark.parametrize("damage", ["none", "policy", "native_flag", "bool_step", "source_bytes"])
def test_owned_update_is_an_exact_unqualified_copy(damage):
    from scripts.audit_dh116_pinch_owned_update import validate_owned_copy

    acquisition = dict(physical={"start_sdk_step": 2, "end_sdk_step": 258},
                       policy_weights_sha256="a" * 64, physics_rows=256, updates=1)
    raw = json.dumps({"acquisition": acquisition}).encode()
    pin = hashlib.sha256(raw).hexdigest()
    owned = dict(acquisition, process_exit_verified=False, native_qualification=False, teacher_admission=False)
    if damage == "policy":
        owned["policy_weights_sha256"] = "b" * 64
    elif damage == "native_flag":
        owned["native_qualification"] = True
    elif damage == "bool_step":
        owned["updates"] = True
    elif damage == "source_bytes":
        raw += b" "
    if damage == "none":
        payload = json.dumps(owned).encode()
        assert validate_owned_copy(raw, payload, pin) == hashlib.sha256(payload).hexdigest()
    else:
        with pytest.raises(ValueError):
            validate_owned_copy(raw, json.dumps(owned).encode(), pin)


def test_owned_update_readback_requires_verified_scheduler_receipt(tmp_path):
    from scripts.audit_dh116_pinch_owned_update import PACKAGE, audit_owned_update

    life_dir = tmp_path / "lifecycle"
    life_dir.mkdir()
    acquisition = {"updates": 1, "physics_rows": 256}
    lifecycle = json.dumps({"acquisition": acquisition}).encode()
    (life_dir / "result.json").write_bytes(lifecycle)
    payload = dict(acquisition, process_exit_verified=False, native_qualification=False, teacher_admission=False)
    owned = tmp_path / "owned_update.json"
    owned.write_text(json.dumps(payload))
    receipt = {"output": str(tmp_path),
               "lifecycle_sha256": {"result.json": hashlib.sha256(lifecycle).hexdigest()}}
    expected_receipt = "a" * 64

    class Source:
        verified = False

        def verify_actual_diagnostic(self, name, expected):
            assert name == PACKAGE and expected == expected_receipt
            self.verified = True
            return {"diagnostic_verified": True, "native_qualification": False}

        def registry(self):
            return {"packages": {PACKAGE: {}}}

        def native_execution_identity(self, _package):
            return {"attempt_fingerprint": "b" * 64}

        def _read_native_receipt_bytes(self, name, *, attempt_fingerprint, expected):
            assert self.verified and name == PACKAGE and expected == expected_receipt
            assert attempt_fingerprint == "b" * 64
            return tmp_path / "receipt.json", json.dumps(receipt).encode()

    result = audit_owned_update(Source(), expected_receipt)
    assert result["owned_update_sha256"] == hashlib.sha256(owned.read_bytes()).hexdigest()
    assert result["diagnostic_verified"] is True and result["native_qualification"] is False
    owned.unlink()
    with pytest.raises(FileNotFoundError):
        audit_owned_update(Source(), expected_receipt)
