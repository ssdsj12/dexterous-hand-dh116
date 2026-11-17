#!/usr/bin/env python3
"""Independently re-open numerical Reset evidence after process exit."""
from pathlib import Path
from scripts import run_dh116_mesh_reset_health_work_packages as prior
from scripts import verify_dh116_mesh_reset_numeric_pinch as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

AUDIT_SOURCES = tuple(dict.fromkeys((*prior.AUDIT_SOURCES, *entry.FILES)))
AUDIT_NAME = "mesh-reset-numeric-pinch-post-exit-audit.json"
_PREVIOUS_AUDIT = prior.audit_outcome


def audit_outcome(output, outer_exit, cleanup):
    with entry.identity_scope():
        result = _PREVIOUS_AUDIT(output, outer_exit, cleanup)
    output = Path(output)
    artifact = output / entry.NUMERIC_ARTIFACT
    numeric_audit = None
    if result["lifecycle_qualified"]:
        packet = entry.operation._read(artifact)
        reset = entry.operation._read(output / entry.operation.evidence.RESET_FILE)
        numeric_audit = entry.numeric.validate_numeric_reset_audit(
            packet, reset_receipt={**reset["readback"]["completed_reset_receipt"],
                                   "mesh_state_readback": {key: value for key, value in reset["readback"].items()
                                                           if key != "completed_reset_receipt"}},
        )
    return dict(result, numeric_state_audit=numeric_audit,
                numeric_state_artifact_sha256=entry.prior.sha(artifact) if artifact.is_file() else None,
                qualification_scope=entry.SCOPE)


def main():
    # The inherited scheduler re-enters its own script in the worker subprocess.
    with bindings((prior, "__file__", str(Path(__file__).resolve())),
                  (prior, "entry", entry), (prior, "AUDIT_SOURCES", AUDIT_SOURCES),
                  (prior, "AUDIT_NAME", AUDIT_NAME), (prior, "audit_outcome", audit_outcome)):
        return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
