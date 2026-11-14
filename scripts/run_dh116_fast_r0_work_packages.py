#!/usr/bin/env python3
"""Single-attempt 234-step recipe with independent unchanged R0 exit audit."""
from pathlib import Path
import json
from scripts import run_dh116_live_r0_work_packages as previous
from scripts import verify_dh116_fast_r0_reset as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))

def cleanup_case(spec):
    with bindings((previous,"entry",entry)):
        return previous.cleanup_case(spec)

def check_codec_evidence(output):
    marker=json.loads((Path(output)/"codec_scope.json").read_bytes())
    if marker!=dict(version=entry.CODEC_VERSION,binding_restored=True,new_physics_steps=0,ppo_updates=0):
        raise ValueError("exact owned-codec scope and restoration evidence required")
    return marker

def audit_outcome(output,outer_exit,cleanup):
    result=previous.audit_outcome(output,outer_exit,cleanup)
    result["qualification_scope"]=entry.SCOPE
    if result.get("lifecycle_qualified"):
        result["codec_scope"]=check_codec_evidence(output)
    return result

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","fast-r0-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()
if __name__=="__main__":raise SystemExit(main())
