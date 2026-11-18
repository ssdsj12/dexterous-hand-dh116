#!/usr/bin/env python3
"""Dedicated scheduler for a single ten-step dual Pinch diagnostic."""

from pathlib import Path

from scripts import run_dh116_physx_lifetime_work_packages as base
from scripts import verify_dh116_dual_pinch_interval as entry
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

AUDIT_SOURCES = tuple(dict.fromkeys((*base.AUDIT_SOURCES,*entry.FILES,
                                    "scripts/dh116_physx_lifetime_debugger_stopped.py")))
original_case = base.cleanup_case


def cleanup_case(spec):
    case = original_case(spec)
    argv = spec["argv"]
    flag = "--approved-code-config-sha256"
    budget = dict(physics_steps=10,updates=0,environments=1,gpu_index=3,timeout_seconds=180)
    if (case != "explicit" or any(spec["budget"].get(key) != value for key,value in budget.items())
            or spec["validator"]["equals"].get("version") != entry.VERSION
            or argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv)
            or not base.scheduler.SHA256.fullmatch(argv[argv.index(flag)+1])
            or spec["validator"]["equals"].get("code_config_sha256") != argv[argv.index(flag)+1]):
        raise ValueError("exact 10 step dual Pinch scope/worker approval required")
    return case


def audit_outcome(output,outer_exit,cleanup):
    from scripts.validate_dh116_dual_pinch_interval import audit_outcome as audit
    return audit(output,outer_exit,cleanup)


def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","dual-pinch-interval-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
