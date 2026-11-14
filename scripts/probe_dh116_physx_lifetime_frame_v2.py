#!/usr/bin/env python3
"""Single explicit frame-detachment experiment; exact dedicated approval required."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

from scripts import probe_dh116_physx_lifetime as probe
from scripts import dh116_physx_lifetime_acquisition as original_acquisition
from scripts import validate_dh116_physx_lifetime as original_validator
from scripts.dh116_physx_lifetime_frame_acquisition_v2 import (
    build_native_detached, NATIVE_LOCALS, ORIGINAL_SOURCE_SHA256)

VERSION="dh116-physx-lifetime-frame-v2"
ENTRY="scripts/probe_dh116_physx_lifetime_frame_v2.py"
WRAPPER="scripts/run_dh116_physx_lifetime_frame_work_packages_v2.py"
WORKER_ENV="DH116_PHYSX_LIFETIME_FRAME_WORKER_V2"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@contextmanager
def bindings(*changes):
    """Process-local rebinding only; restore every frozen module on all paths."""
    previous=[(module,name,getattr(module,name)) for module,name,value in changes]
    try:
        for module,name,value in changes:
            setattr(module,name,value)
        yield
    finally:
        for module,name,value in reversed(previous):
            setattr(module,name,value)

def require_worker(args,project):
    try:
        marker=json.loads(os.environ[WORKER_ENV])
    except (KeyError,ValueError):
        raise RuntimeError("use the dedicated frame-v2 worker with exact approval") from None
    expected=dict(version=VERSION,case="explicit",output=str(args.output.resolve()),
                  entry_sha256=sha(project/ENTRY),wrapper_sha256=sha(project/WRAPPER))
    if (args.cleanup!="explicit" or marker!=expected
            or os.environ.get("DH116_PHYSX_LIFETIME_GDB")!="1"):
        raise RuntimeError("dedicated frame-v2 worker/capture identity differs")

def audit_outcome(output,outer_exit,cleanup):
    """Reuse every v1 disk/outer gate and add actual v2 scope/capture markers."""
    output=Path(output)
    with bindings((original_validator,"VERSION",VERSION)):
        result=original_validator.audit_outcome(output,outer_exit,cleanup)
    try:
        frame=json.loads((output/"build_frame_detached.json").read_text())
        if frame!=dict(version="dh116-build-frame-detachment-v2",
            clearing_completed=True,operation_raised=False,cleared_locals=list(NATIVE_LOCALS),
            native_release_verified=False):
            raise ValueError("builder detachment record incomplete or construction failed")
        capture=json.loads((output/"gdb_finalize.json").read_text())
        count=capture.get("stopped_thread_count")
        if type(count) is not int or count<1:
            raise ValueError("new stopped-thread GDB capture missing")
        if cleanup!="explicit":
            raise ValueError("frame-v2 permits only explicit cleanup")
        result["frame_detachment_audit"]=dict(
            marker_sha256=sha(output/"build_frame_detached.json"),stopped_thread_count=count,
            original_acquisition_source_sha256=ORIGINAL_SOURCE_SHA256,
            native_release_verified=False)
    except (OSError,ValueError,TypeError,KeyError) as error:
        result["lifecycle_qualified"]=False
        result["reason"]="; ".join(filter(None,(result.get("reason"),
            f"frame-v2 audit: {type(error).__name__}: {error}")))
    return result

def main():
    project=Path(__file__).resolve().parents[1]
    if sha(project/"scripts/dh116_physx_lifetime_acquisition.py")!=ORIGINAL_SOURCE_SHA256:
        raise RuntimeError("frozen original acquisition source changed")
    with bindings((probe,"VERSION",VERSION),(probe,"CASES",("explicit",)),
                  (probe,"require_worker",require_worker),
                  (original_acquisition,"build_native",build_native_detached)):
        return probe.main()

if __name__=="__main__":
    main()
