#!/usr/bin/env python3
"""Pinch native diagnostic with owned numerical copies of the rod Reset state."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts import verify_dh116_mesh_reset_health_projected_pinch as prior
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_mesh_reset_numeric_state as numeric

ENTRY = "scripts/verify_dh116_mesh_reset_numeric_pinch.py"
WRAPPER = "scripts/run_dh116_mesh_reset_numeric_pinch_work_packages.py"
VERSION = "dh116-mesh-reset-numeric-pinch-v2"
SCOPE = "numeric_mesh_reset_pair_v2_health_2_setup_232_diagnostic_only"
WORKER_ENV = "DH116_MESH_RESET_NUMERIC_PINCH_WORKER_V2"
NUMERIC_ARTIFACT = "mesh_reset_numeric_state.json"
FILES = (*prior.FILES, ENTRY, WRAPPER,
         "src/go2_pvcnn/control/dh116/tabletop_mesh_reset_numeric_state.py",
         "tests/test_mesh_reset_numeric_state.py",
         "tests/test_mesh_reset_numeric_worker.py")
FAILED_RECEIPT = Path(
    "/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f8-mesh-reset-health-projected-pinch-native/"
    "4918ab0a9bb51fc20c0bbcc0bd070fafc7c846dac9a4579a170b1af30cd34601/"
    "attempt-kjuzues5/receipt.json"
)
FAILED_SHA = "6c97bc0cdb03c71f74c7f5a3c0ec350a9c52f64bf622451bf896737d73025027"
FAILED_RESET_SHA = "e2f5302ed2dabbb3916b438fb7e3c74bfaa6f7c96e1b4ab5133fdf6794983e41"
operation = prior.operation
PROFILE_SHA = prior.PROFILE_SHA
_PREFLIGHT, _EXECUTE, _MAIN = prior.preflight, prior.execute, prior.main
_RESET_TYPE = operation.evidence.make_reset_type
PARENT_PACKAGE, PARENT_SCOPE = prior.PARENT_PACKAGE, prior.PARENT_SCOPE
PARENT_RECEIPT, PARENT_SHA = prior.PARENT_RECEIPT, prior.PARENT_SHA


@contextmanager
def identity_scope():
    with bindings((prior, "ENTRY", ENTRY), (prior, "WRAPPER", WRAPPER),
                  (prior, "VERSION", VERSION), (prior, "WORKER_ENV", WORKER_ENV),
                  (prior, "FILES", FILES), (operation, "VERSION", VERSION),
                  (operation, "SCOPE", SCOPE)):
        yield


def _verify_failed_source():
    if FAILED_RECEIPT.is_symlink() or prior.sha(FAILED_RECEIPT) != FAILED_SHA:
        raise ValueError("consumed Reset attempt receipt changed")
    failure = prior.read(FAILED_RECEIPT)
    if (failure["outer_exit_code"] != 1 or failure["debugger_inferior_signal"] is not None
            or failure["native_execution_performed"] is not True
            or failure["native_qualification"] is not False):
        raise ValueError("expected failed Reset diagnostic receipt")
    reset = Path(failure["output"]) / "mesh_reset_state.json"
    if reset.is_symlink() or prior.sha(reset) != FAILED_RESET_SHA:
        raise ValueError("consumed Reset readback evidence changed")
    return {str(FAILED_RECEIPT): FAILED_SHA, str(reset): FAILED_RESET_SHA}


def preflight(project, parent_path=PARENT_RECEIPT, parent_sha=PARENT_SHA):
    with identity_scope():
        previous = _PREFLIGHT(project, parent_path, parent_sha)
    sources = deepcopy(previous["sources"])
    sources["external_sources"].update(_verify_failed_source())
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(previous, sources=sources, code_config_sha256=digest,
                failed_reset_receipt_sha256=FAILED_SHA)


def make_reset_type(project, prepared, output):
    reset_type = _RESET_TYPE(project, prepared, output)
    artifact = Path(output) / NUMERIC_ARTIFACT

    class NumericMeasuredReset(reset_type):
        def run(self, sim, rod, bindings, *, owner_resets):
            try:
                with numeric.numeric_reset_state_scope(
                    sim=sim, rod=rod,
                    record=lambda audit: operation.evidence._write(artifact, audit),
                ):
                    return super().run(sim, rod, bindings, owner_resets=owner_resets)
            finally:
                sim = rod = bindings = owner_resets = None

    return NumericMeasuredReset


def execute(args, project, contract):
    with identity_scope(), bindings((operation.evidence, "make_reset_type", make_reset_type)):
        return _EXECUTE(args, project, contract)


def main():
    with identity_scope(), bindings((prior, "preflight", preflight), (prior, "execute", execute)):
        return _MAIN()


if __name__ == "__main__":
    main()
