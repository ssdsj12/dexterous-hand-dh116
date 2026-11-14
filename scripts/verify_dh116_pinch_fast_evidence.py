#!/usr/bin/env python3
"""Same Pinch physical replay with the byte-equivalent evidence encoder."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts import verify_dh116_mesh_reset_numeric_pinch as prior
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_physical_evidence_encoding as codec

ENTRY = "scripts/verify_dh116_pinch_fast_evidence.py"
WRAPPER = "scripts/run_dh116_pinch_fast_evidence_work_packages.py"
VERSION = "dh116-pinch-fast-evidence-v3"
SCOPE = "same_numeric_mesh_reset_pair_v2_health_2_setup_232_fast_evidence_only"
WORKER_ENV = "DH116_PINCH_FAST_EVIDENCE_WORKER_V3"
FILES = (*prior.FILES, ENTRY, WRAPPER,
         "src/go2_pvcnn/control/dh116/tabletop_physical_evidence_encoding.py",
         "tests/test_physical_evidence_encoding.py",
         "tests/test_pinch_fast_evidence_worker.py")
PARENT_PACKAGE, PARENT_SCOPE = prior.PARENT_PACKAGE, prior.PARENT_SCOPE
PARENT_RECEIPT, PARENT_SHA = prior.PARENT_RECEIPT, prior.PARENT_SHA
PROFILE_SHA = prior.PROFILE_SHA
NUMERIC_ARTIFACT = prior.NUMERIC_ARTIFACT
operation = prior.operation
_PREFLIGHT, _EXECUTE, _MAIN = prior.preflight, prior.execute, prior.main
TIMEOUT_RECEIPT = Path(
    "/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f8-mesh-reset-numeric-pinch-native/"
    "c48203afab2a7cb17ca2aaa672a4096ea380c20f85d1146d414a117930d84247/"
    "attempt-iwm7090o/receipt.json"
)
TIMEOUT_SHA = "bfbfa1415d816794d92e20e33d8ecb8acb677f4581713ff3c78d864e98dc7a4c"
TIMEOUT_AUDIT_SHA = "aa466fba99a72ed157034169290ef9025d63e2f7151b4e2d8e26c245c996aec9"


@contextmanager
def identity_scope():
    with bindings((prior, "ENTRY", ENTRY), (prior, "WRAPPER", WRAPPER),
                  (prior, "VERSION", VERSION), (prior, "WORKER_ENV", WORKER_ENV),
                  (prior, "FILES", FILES), (prior, "SCOPE", SCOPE)):
        yield


def timeout_sources():
    sha, read = prior.prior.sha, prior.prior.read
    if TIMEOUT_RECEIPT.is_symlink() or sha(TIMEOUT_RECEIPT) != TIMEOUT_SHA:
        raise ValueError("retained timeout receipt changed")
    receipt = read(TIMEOUT_RECEIPT)
    if (receipt["outer_exit_code"] != 0 or receipt["debugger_inferior_exit_code"] != 0
            or receipt["debugger_inferior_signal"] is not None or receipt["duration_seconds"] <= 180
            or receipt["native_execution_performed"] is not True
            or receipt["native_qualification"] is not False):
        raise ValueError("retained timeout outcome differs")
    audit = TIMEOUT_RECEIPT.with_name("mesh-reset-numeric-pinch-post-exit-audit.json")
    if audit.is_symlink() or sha(audit) != TIMEOUT_AUDIT_SHA:
        raise ValueError("retained timeout audit changed")
    return {str(TIMEOUT_RECEIPT): TIMEOUT_SHA, str(audit): TIMEOUT_AUDIT_SHA}


def preflight(project, parent_path=PARENT_RECEIPT, parent_sha=PARENT_SHA):
    with identity_scope():
        previous = _PREFLIGHT(project, parent_path, parent_sha)
    sources = deepcopy(previous["sources"])
    sources["external_sources"].update(timeout_sources())
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(previous, sources=sources, code_config_sha256=digest,
                timed_out_source_receipt_sha256=TIMEOUT_SHA)


def execute(args, project, contract):
    with identity_scope(), codec.encoding_scope():
        return _EXECUTE(args, project, contract)


def main():
    with identity_scope(), bindings((prior, "preflight", preflight), (prior, "execute", execute)):
        return _MAIN()


if __name__ == "__main__":
    main()
