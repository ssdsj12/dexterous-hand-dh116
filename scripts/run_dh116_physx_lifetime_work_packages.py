#!/usr/bin/env python3
"""Independently gated lifecycle cases with child reentry and post-exit audit."""
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
try:
    from scripts import run_dh116_work_packages as scheduler
    from scripts.dh116_physx_lifetime_debugger import render_gdb_commands
    from scripts.validate_dh116_physx_lifetime import audit_outcome
except ModuleNotFoundError:
    import run_dh116_work_packages as scheduler
    from dh116_physx_lifetime_debugger import render_gdb_commands
    from validate_dh116_physx_lifetime import audit_outcome

VERSION="dh116-physx-lifetime-probe-v1"
ENTRY="scripts/probe_dh116_physx_lifetime.py"
WRAPPER="scripts/run_dh116_physx_lifetime_work_packages.py"
WORKER_ENV="DH116_PHYSX_LIFETIME_WORKER_V1"
AUDIT_NAME="physx-lifetime-post-exit-audit.json"
AUDIT_SOURCES=(WRAPPER,ENTRY,"scripts/dh116_physx_lifetime_acquisition.py",
               "scripts/dh116_physx_lifetime_debugger.py","scripts/validate_dh116_physx_lifetime.py")

def cleanup_case(spec):
    if spec["entrypoint"]!=ENTRY or not set(AUDIT_SOURCES)<=set(spec["source_files"]):
        raise ValueError("lifetime recipe must freeze the dedicated entry, capture, acquisition and audit")
    args=spec["argv"]
    if args.count("--cleanup")!=1 or any(a.startswith("--cleanup=") for a in args):
        raise ValueError("one explicit cleanup case required")
    index=args.index("--cleanup")+1
    case=args[index] if index<len(args) else None
    if (case not in ("registry","explicit") or spec["validator"]["equals"].get("case")!=case
            or spec.get("debugger")!="gdb_on_segv"):
        raise ValueError("lifetime case/capture identity differs")
    return case

def write_debugger(base_wrapper,attempt,result):
    argv=json.loads((attempt/"launch_argv.json").read_text())["argv"]
    if argv.count("--output")!=1 or argv[-1]=="--output":
        raise ValueError("capture must bind exactly one output")
    output=Path(argv[argv.index("--output")+1])
    old=scheduler.GDB_COMMANDS
    scheduler.GDB_COMMANDS=render_gdb_commands(output)
    try:
        return base_wrapper(attempt,result)
    finally:
        scheduler.GDB_COMMANDS=old

class LifetimeRepository(scheduler.Repository):
    def _lifetime_audit(self,path,spec,output):
        return dict(version="dh116-physx-lifetime-post-exit-v1",case=cleanup_case(spec),
            validator_source_sha256={name:scheduler.file_hash(self.root/name) for name in AUDIT_SOURCES},
            evidence=audit_outcome(output,path/"outer_exit.json",cleanup_case(spec)))

    def _native_child(self,path,spec,output,result,*,bound=None):
        case=cleanup_case(spec)
        previous=os.environ.get(WORKER_ENV)
        os.environ[WORKER_ENV]=json.dumps(dict(version=VERSION,case=case,output=str(output.resolve()),
            entry_sha256=scheduler.file_hash(self.root/ENTRY),
            wrapper_sha256=scheduler.file_hash(self.root/WRAPPER)),sort_keys=True)
        try:
            super()._native_child(path,spec,output,result,bound=bound)
        finally:
            if previous is None:os.environ.pop(WORKER_ENV,None)
            else:os.environ[WORKER_ENV]=previous
        generic=result.get("native_qualification") is True
        result["native_qualification"]=False
        try:
            audit=self._lifetime_audit(path,spec,output)
            audit_path=path/AUDIT_NAME
            scheduler.atomic_write(audit_path,scheduler.json_text(audit))
            result.update(lifetime_post_exit_audit_sha256=scheduler.file_hash(audit_path),
                lifetime_validator_source_sha256=audit["validator_source_sha256"])
            qualified=generic and audit["evidence"].get("lifecycle_qualified") is True
            result["native_qualification"]=qualified
            if not qualified:
                result.update(blocker_kind="failed_native",
                    reason="lifetime outcome is diagnostic only: "+audit["evidence"].get("reason",result.get("reason","generic gate failed")))
        except Exception as error:
            result.update(native_qualification=False,blocker_kind="failed_native",
                reason=f"independent lifetime audit rejected evidence: {type(error).__name__}: {error}")

    def _verified_native_receipt(self,name,scope,recipe,expected):
        path=super()._verified_native_receipt(name,scope,recipe,expected)
        if recipe["spec"]["entrypoint"]!=ENTRY:
            return path
        receipt=json.loads(path.read_text()); audit_path=path.parent/AUDIT_NAME
        if scheduler.file_hash(audit_path)!=receipt.get("lifetime_post_exit_audit_sha256"):
            raise ValueError("lifetime audit missing or changed")
        fresh=self._lifetime_audit(path.parent,recipe["spec"],Path(receipt["output"]))
        if (json.loads(audit_path.read_text())!=fresh or
            receipt.get("lifetime_validator_source_sha256")!=fresh["validator_source_sha256"] or
            fresh["evidence"].get("lifecycle_qualified") is not True):
            raise ValueError("independently reopened lifetime evidence changed or failed")
        return path

    def start_native(self,name,approval):
        cleanup_case(self.native_recipe(self.registry()["packages"][name])["spec"])
        return super().start_native(name,approval)

def main():
    entry=Path(__file__).resolve()
    old_entry,old_repo,old_debug=scheduler.__file__,scheduler.Repository,scheduler.native_debugger_wrapper
    if entry.parent!=Path(old_entry).resolve().parent:
        raise RuntimeError("dedicated worker and shared scheduler must be sibling entries")
    scheduler.__file__,scheduler.Repository=str(entry),LifetimeRepository
    scheduler.native_debugger_wrapper=partial(write_debugger,old_debug)
    try:
        return scheduler.main()
    finally:
        scheduler.__file__,scheduler.Repository,scheduler.native_debugger_wrapper=old_entry,old_repo,old_debug

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
        print(f"DH116 lifetime scheduler: {error}",file=sys.stderr)
        raise SystemExit(1)
