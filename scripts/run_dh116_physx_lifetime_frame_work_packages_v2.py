#!/usr/bin/env python3
"""Reuse frozen transport with one new explicit scope and stopped-thread capture."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

from scripts import run_dh116_physx_lifetime_work_packages as base
from scripts import probe_dh116_physx_lifetime_frame_v2 as entry
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands

AUDIT_SOURCES=tuple(dict.fromkeys((*base.AUDIT_SOURCES,
    entry.ENTRY,entry.WRAPPER,"scripts/dh116_physx_lifetime_frame_acquisition_v2.py",
    "scripts/dh116_physx_lifetime_debugger_stopped.py")))
_original_cleanup_case=base.cleanup_case

def cleanup_case(spec):
    case=_original_cleanup_case(spec)
    if case!="explicit" or spec["validator"]["equals"].get("version")!=entry.VERSION:
        raise ValueError("frame-v2 accepts only its explicit scoped result")
    return case

@contextmanager
def configured():
    with entry.bindings(
        (base,"__file__",str(Path(__file__).resolve())),
        (base,"ENTRY",entry.ENTRY),(base,"WRAPPER",entry.WRAPPER),
        (base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","physx-lifetime-frame-post-exit-audit-v2.json"),
        (base,"AUDIT_SOURCES",AUDIT_SOURCES),(base,"cleanup_case",cleanup_case),
        (base,"render_gdb_commands",render_gdb_commands),(base,"audit_outcome",entry.audit_outcome)):
        yield

def main():
    # Frozen main routes child __file__ to this sibling too; its root remains unchanged.
    with configured():
        return base.main()

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
        print(f"DH116 lifetime frame-v2 scheduler: {error}",file=sys.stderr)
        raise SystemExit(1)
