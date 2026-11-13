#!/usr/bin/env python3
"""Dedicated scheduler for the exact no-lift 2+8+8 dual Pinch diagnostic."""

from pathlib import Path

from scripts import run_dh116_physx_lifetime_work_packages as base
from scripts import run_dh116_dual_pinch_work_packages_v4 as previous
from scripts import verify_dh116_dual_pinch_closure_interval as entry
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

AUDIT_SOURCES = tuple(dict.fromkeys((*previous.AUDIT_SOURCES, *entry.FILES)))
base_case = base.cleanup_case


def cleanup_case(spec):
    case = base_case(spec)
    args = spec["argv"]
    flag = "--approved-code-config-sha256"
    budget = dict(physics_steps=18, updates=0, environments=1, gpu_index=3, timeout_seconds=180)
    if (case != "explicit" or any(spec["budget"].get(key) != val for key, val in budget.items())
            or spec["validator"]["equals"].get("version") != entry.VERSION
            or args.count(flag) != 1 or args.index(flag) + 1 >= len(args)
            or not base.scheduler.SHA256.fullmatch(args[args.index(flag) + 1])
            or spec["validator"]["equals"].get("code_config_sha256") != args[args.index(flag) + 1]
            or spec.get("dependency_receipt") != {
                "package": entry.PARENT_PACKAGE, "qualification_scope": entry.PARENT_SCOPE,
            }):
        raise ValueError("exact dual Pinch closure recipe and parent receipt required")
    return case


def audit_outcome(output, outer_exit, cleanup):
    from scripts.validate_dh116_dual_pinch_closure_interval import audit_outcome as audit

    return audit(output, outer_exit, cleanup, profile_sha256=entry.PROFILE_SHA)


def main():
    with bindings(
        (base, "__file__", str(Path(__file__).resolve())),
        (base, "ENTRY", entry.ENTRY), (base, "WRAPPER", entry.WRAPPER),
        (base, "VERSION", entry.VERSION), (base, "WORKER_ENV", entry.WORKER_ENV),
        (base, "AUDIT_NAME", "dual-pinch-closure-post-exit-audit.json"),
        (base, "AUDIT_SOURCES", AUDIT_SOURCES), (base, "cleanup_case", cleanup_case),
        (base, "render_gdb_commands", render_gdb_commands), (base, "audit_outcome", audit_outcome),
    ):
        return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
