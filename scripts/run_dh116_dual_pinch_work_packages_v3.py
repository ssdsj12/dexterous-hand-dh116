#!/usr/bin/env python3
"""Dedicated scheduler successor for the owned scene preparation handoff."""

from scripts import run_dh116_dual_pinch_work_packages as prior
from scripts import verify_dh116_dual_pinch_interval_v2 as previous
from scripts import verify_dh116_dual_pinch_interval_v3 as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

AUDIT_SOURCES = tuple(dict.fromkeys((*prior.AUDIT_SOURCES,*previous.FILES,*entry.FILES)))


def main():
    with bindings((prior,"entry",entry),(prior,"__file__",__file__),(prior,"AUDIT_SOURCES",AUDIT_SOURCES)):
        return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
