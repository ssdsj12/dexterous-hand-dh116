#!/usr/bin/env python3
"""Qualify only the free-Pinch candidate through its independent post-exit audit.

The shared scheduler remains unchanged. Its worker must re-enter this sibling
entrypoint so the additional receipt gate also runs in the child process.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from scripts import run_dh116_work_packages as scheduler
    from scripts.verify_dh116_deformx_pinch_contact import validate_pinch_artifacts
except ModuleNotFoundError:
    import run_dh116_work_packages as scheduler
    from verify_dh116_deformx_pinch_contact import validate_pinch_artifacts

ENTRY = "scripts/verify_dh116_deformx_pinch_contact.py"
WRAPPER = "scripts/run_dh116_pinch_work_packages.py"
RUNTIME = "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py"
AUDIT_NAME = "pinch-post-exit-audit.json"
WORKER_ENV = "DH116_PINCH_WORKER_V1"
AUDIT_SOURCES = (WRAPPER, ENTRY, RUNTIME)


def candidate_sha(spec):
    if spec["entrypoint"] != ENTRY:
        raise ValueError("Pinch scheduler accepts only the free-Pinch native entrypoint")
    if not set(AUDIT_SOURCES) <= set(spec["source_files"]):
        raise ValueError("Pinch recipe must freeze wrapper, validator and runtime sources")
    args=spec["argv"]
    flag="--approved-code-config-sha256"
    if args.count(flag)!=1 or any(value.startswith(flag+"=") for value in args):
        raise ValueError("Pinch recipe must bind one explicit candidate code/config SHA")
    index=args.index(flag)+1
    if index>=len(args) or not scheduler.SHA256.fullmatch(args[index]):
        raise ValueError("Pinch candidate SHA is invalid")
    if spec["validator"]["equals"].get("code_config_sha256")!=args[index]:
        raise ValueError("Pinch result and command candidate SHA differ")
    return args[index]


class PinchRepository(scheduler.Repository):
    def _pinch_audit(self,path,spec,output):
        expected=candidate_sha(spec)
        evidence=validate_pinch_artifacts(output,path/"outer_exit.json",expected)
        return {"version":"dh116-pinch-post-exit-audit-v1",
                "candidate_code_config_sha256":expected,
                "validator_source_sha256":{name:scheduler.file_hash(self.root/name) for name in AUDIT_SOURCES},
                "evidence":evidence}

    def _native_child(self,path,spec,output,result,*,bound=None):
        expected=candidate_sha(spec)  # Reject an unfrozen audit path before any native launch.
        previous=os.environ.get(WORKER_ENV)
        os.environ[WORKER_ENV]=json.dumps({
            "version":"dh116-pinch-worker-v1",
            "wrapper_sha256":scheduler.file_hash(self.root/WRAPPER),
            "candidate_code_config_sha256":expected,"output":str(output.resolve())},
            sort_keys=True,separators=(",",":"))
        try:
            super()._native_child(path,spec,output,result,bound=bound)
        finally:
            if previous is None:
                os.environ.pop(WORKER_ENV,None)
            else:
                os.environ[WORKER_ENV]=previous
        if result.get("native_qualification") is not True:
            return
        # Generic clocks/artifacts have passed. Revoke provisionally before the
        # independent multi-pad audit; the base worker writes its receipt later.
        result["native_qualification"]=False
        try:
            audit=self._pinch_audit(path,spec,output)
            audit_path=path/AUDIT_NAME
            scheduler.atomic_write(audit_path,scheduler.json_text(audit))
            result.update(pinch_post_exit_audit_sha256=scheduler.file_hash(audit_path),
                          pinch_validator_source_sha256=audit["validator_source_sha256"],
                          native_qualification=True)
        except Exception as error:
            result.pop("pinch_post_exit_audit_sha256",None)
            result.pop("pinch_validator_source_sha256",None)
            result.update(native_qualification=False,blocker_kind="failed_native",
                          reason=f"independent Pinch post-exit audit rejected evidence: {type(error).__name__}: {error}")

    def _verified_native_receipt(self,name,scope,recipe,expected):
        path=super()._verified_native_receipt(name,scope,recipe,expected)
        if recipe["spec"]["entrypoint"] != ENTRY:
            return path  # Existing prerequisite receipts retain their original validation.
        receipt=json.loads(path.read_text(encoding="utf-8"))
        audit_path=path.parent/AUDIT_NAME
        if scheduler.file_hash(audit_path)!=receipt.get("pinch_post_exit_audit_sha256"):
            raise ValueError("Pinch receipt lacks an unchanged independent post-exit audit")
        audit=json.loads(audit_path.read_text(encoding="utf-8"))
        fresh=self._pinch_audit(path.parent,recipe["spec"],Path(receipt["output"]))
        if audit!=fresh or receipt.get("pinch_validator_source_sha256")!=fresh["validator_source_sha256"]:
            raise ValueError("Pinch audit sources or independently reopened native evidence changed")
        return path

    def start_native(self,name,approval):
        candidate_sha(self.native_recipe(self.registry()["packages"][name])["spec"])
        return super().start_native(name,approval)


def main(argv=None):
    entry=Path(__file__).resolve()
    original_entry,original_repository=scheduler.__file__,scheduler.Repository
    if entry.parent != Path(original_entry).resolve().parent:
        raise RuntimeError("Pinch entry and shared scheduler must share the scripts directory")
    scheduler.__file__,scheduler.Repository=str(entry),PinchRepository
    try:
        return scheduler.main() if argv is None else scheduler.main(argv)
    finally:
        scheduler.__file__,scheduler.Repository=original_entry,original_repository


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
        print(f"DH116 Pinch scheduler: {error}",file=sys.stderr)
        raise SystemExit(1)
