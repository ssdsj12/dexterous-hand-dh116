#!/usr/bin/env python3
"""Frozen corrected-pose Pinch candidate; only its dedicated worker may execute."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

from scripts.verify_dh116_deformx_pinch_contact import default_arguments,preflight as old_preflight
from scripts.verify_dh116_deformx_pair_capture_v2 import validate_artifacts as validate_calibration
from scripts.dh116_pinch_acquisition_v2 import VERSION,execute
from scripts.dh116_pinch_evidence_v2 import serial

ENTRY="scripts/verify_dh116_pinch_candidate_v2.py"
WRAPPER="scripts/run_dh116_pinch_work_packages_v2.py"
WORKER_ENV="DH116_PINCH_WORKER_V2"
PAIR_ROOT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python")
PAIR_RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-deformx-pair-capture-v2-native/94d855ce6be8700816a85a8e84d41a87fe74b52454bf4f8591661e932c6d08c6/attempt-nobp3d5x/receipt.json")
PAIR_RECEIPT_SHA="7af3e7d7358ea58c2d4be656c889d162835830ccbe1db20f36c457a5cfe51ba2"
PAIR_CODE="94d855ce6be8700816a85a8e84d41a87fe74b52454bf4f8591661e932c6d08c6"
PAIR_SPEC="77d0290a350ba6b877991b9831da8fb4fb62441481e1866412935fc6d28af594"
POSE="configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json"
NEW_SOURCES=(
    ENTRY,WRAPPER,"scripts/dh116_pinch_acquisition_v2.py","scripts/dh116_pinch_evidence_v2.py",
    "scripts/validate_dh116_pinch_v2.py",
    "src/go2_pvcnn/control/dh116/tabletop_pinch_initial_geometry_v2.py",
    "src/go2_pvcnn/control/dh116/tabletop_tripod_cooked_geometry.py",
    "scripts/audit_dh116_pinch_initial_pose.py",
    "scripts/verify_dh116_deformx_pair_capture.py","scripts/verify_dh116_deformx_pair_capture_v2.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformx_pair_capture.py",POSE,
    "tests/test_dh116_pinch_v2.py",
    "configs/dh116_execution/deformx_mesh_pair_capture_calibration_v2.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(project):
    defaults=default_arguments(project)
    contract,cfg=old_preflight(defaults,project)
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    geometry=load_geometry(project,side="left")
    pose=json.loads((project/POSE).read_text())
    if (pose["source_sha256"]!=geometry["source_sha256"] or pose["side"]!="left"
            or pose["active_joint_names"]!=["finger11","finger12","finger21","finger31","finger41","finger51"]
            or pose["approach"]["move_steps"]!=cfg["action"]["move_steps"]
            or pose["approach"]["velocity_rad_s"]!=cfg["action"]["thumb_velocity_normalized"]*.3):
        raise ValueError("corrected pose differs from its locked sources/action scope")
    cfg=deepcopy(cfg);cfg["initial_pose"]=pose
    if sha(PAIR_RECEIPT)!=PAIR_RECEIPT_SHA:
        raise ValueError("qualified pair receipt bytes changed")
    receipt=json.loads(PAIR_RECEIPT.read_text())
    if (receipt.get("native_qualification") is not True or receipt.get("outer_exit_code")!=0
            or receipt.get("native_lock_observed") is not True
            or receipt.get("code_config_sha256")!=PAIR_CODE
            or receipt.get("command_spec_sha256")!=PAIR_SPEC):
        raise ValueError("exact successful pair calibration prerequisite required")
    outer=json.loads((PAIR_RECEIPT.parent/"outer_exit.json").read_text())
    if (any(outer.get(k)!=receipt.get(k) for k in ("outer_exit_code","native_lock_observed","log_sha256","duration_seconds"))
            or not 0<=outer["duration_seconds"]<=90
            or sha(PAIR_RECEIPT.parent/"native.log")!=outer["log_sha256"]):
        raise ValueError("calibration independent outer evidence differs")
    output=Path(receipt["output"])
    validate_calibration(output)
    for name,digest in receipt["artifact_sha256"].items():
        if sha(output/name)!=digest:
            raise ValueError("calibration artifact differs from its qualified receipt")
    sources=deepcopy(contract["sources"])
    prior=json.loads((project/"configs/dh116_execution/physx_lifetime_frame_explicit_v2.json").read_text())
    for name in (*prior["source_files"],*NEW_SOURCES):
        sources["local_sources"][name]=sha(project/name)
    # Preserve original backend audit provenance; record the actually loaded pair extension separately.
    pair_spec=json.loads((project/"configs/dh116_execution/deformx_mesh_pair_capture_calibration_v2.json").read_text())
    for name,expected in pair_spec["inputs"].items():
        if sha(name)!=expected:raise ValueError("qualified pair backend input changed")
        sources["external_sources"][name]=expected
    for path in [PAIR_RECEIPT,PAIR_RECEIPT.parent/"outer_exit.json",PAIR_RECEIPT.parent/"native.log",
                 *[output/name for name in receipt["artifact_sha256"]]]:
        sources["external_sources"][str(path)]=sha(path)
    for name,expected in prior["inputs"].items():
        if sha(name)!=expected:raise ValueError("lifetime input changed")
        sources["external_sources"][name]=expected
    sdk=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics")
    for relative in ("omni.usd.schema.physx/bin/libphysxSchema.so",
                     "omni.usd.schema.physx/pxr/PhysxSchema/_physxSchema.so",
                     "omni.physx/bin/libomni.physx.plugin.so",
                     "omni.physx/omni/physx/bindings/_physx.cpython-310-x86_64-linux-gnu.so"):
        sources["external_sources"][str(sdk/relative)]=sha(sdk/relative)
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    contract={**contract,"version":VERSION,"code_config_sha256":fingerprint,"sources":sources,
        "pose_candidate_sha256":sha(project/POSE),"pair_calibration_receipt_sha256":sha(PAIR_RECEIPT),
        "pair_backend_root":str(PAIR_ROOT),"pair_calibration_code_config_sha256":PAIR_CODE,
        "native_execution_performed":False,"grasp_qualified":False,"teacher_admission":False}
    return contract,cfg


def require_worker(args,project):
    try:marker=json.loads(os.environ[WORKER_ENV])
    except (KeyError,ValueError):
        raise ValueError("dedicated Pinch v2 worker required") from None
    expected=dict(version=VERSION,case="explicit",output=str(args.output.resolve()),
        entry_sha256=sha(project/ENTRY),wrapper_sha256=sha(project/WRAPPER))
    if (marker!=expected or os.environ.get("CUDA_VISIBLE_DEVICES")!="3"
            or os.environ.get("DH116_PHYSX_LIFETIME_GDB")!="1"):
        raise ValueError("Pinch v2 worker/GPU/capture identity differs")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup",choices=["explicit"],required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--steps",type=int,default=240)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approved-code-config-sha256")
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    if args.steps!=240:parser.error("exactly 240 total SDK steps required")
    if args.execute_native:
        if args.output is None or not args.headless or args.device!="cuda:0":
            parser.error("fresh output/headless/cuda:0 required")
        require_worker(args,project)
    contract,cfg=preflight(project)
    if not args.execute_native:
        print(json.dumps(serial(contract),indent=2));return
    if args.approved_code_config_sha256!=contract["code_config_sha256"]:
        parser.error("current candidate code/config approval differs")
    args.output.mkdir(parents=True,exist_ok=False)
    args.ropesim_python=PAIR_ROOT
    sys.path.insert(0,str(PAIR_ROOT))
    execute(args,contract,cfg,project)


if __name__=="__main__":
    main()
