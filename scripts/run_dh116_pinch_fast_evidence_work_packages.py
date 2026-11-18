#!/usr/bin/env python3
"""Reopen full numerical Reset and contact evidence from the faster writer."""
from pathlib import Path
from scripts import run_dh116_mesh_reset_numeric_pinch_work_packages as prior
from scripts import verify_dh116_pinch_fast_evidence as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

AUDIT_SOURCES = tuple(dict.fromkeys((*prior.AUDIT_SOURCES, *entry.FILES)))
AUDIT_NAME = "pinch-fast-evidence-post-exit-audit.json"
_PREVIOUS_AUDIT = prior.audit_outcome


def audit_outcome(output, outer_exit, cleanup):
    with entry.identity_scope():
        result = _PREVIOUS_AUDIT(output, outer_exit, cleanup)
    return dict(result, qualification_scope=entry.SCOPE)


def main():
    with bindings((prior, "__file__", str(Path(__file__).resolve())),
                  (prior, "entry", entry), (prior, "AUDIT_SOURCES", AUDIT_SOURCES),
                  (prior, "AUDIT_NAME", AUDIT_NAME), (prior, "audit_outcome", audit_outcome)):
        return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
