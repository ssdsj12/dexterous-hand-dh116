#!/usr/bin/env python3
"""Read-only verification of the saved Pinch v3 physical artifacts.

The scheduler receipt stays failed. This command independently verifies its
physical data and numerical Reset after the v3 post-exit wrapper error.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

from scripts import run_dh116_mesh_reset_health_work_packages as parent
from scripts import run_dh116_work_packages as scheduler
from scripts import verify_dh116_pinch_fast_evidence as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as profile
from go2_pvcnn.control.dh116 import tabletop_mesh_reset_numeric_state as stable_numeric

PACKAGE = "flex.f8-pinch-fast-evidence-native"
RECEIPT_SHA = "6bc124165c6cd907e84ac013b07c9f9df45b2ee1cd4f9499253b42d924adc4ef"
AUDIT_ERROR = ("independent lifetime audit rejected evidence: AttributeError: "
               "module 'scripts.verify_dh116_mesh_reset_numeric_pinch' has no attribute 'sha'")


def verify_outer_evidence(receipt,outer,spec,debug,actual_log_sha256):
    duration=outer.get("duration_seconds")
    keys=("duration_seconds","outer_exit_code","native_lock_observed","log_sha256")
    if (type(duration) not in (int,float) or not math.isfinite(duration)
        or not 0<duration<=spec["budget"]["timeout_seconds"]
        or any(outer.get(k)!=receipt.get(k) for k in keys)
        or any(type(row.get("outer_exit_code")) is not int or row["outer_exit_code"]!=0
            or row.get("native_lock_observed") is not True for row in (outer,receipt))
        or type(receipt.get("exit_code")) is not int or receipt["exit_code"]!=0
        or not isinstance(actual_log_sha256,str) or not scheduler.SHA256.fullmatch(actual_log_sha256)
        or actual_log_sha256!=outer.get("log_sha256")
        or debug.get("debugger_status_verified") is not True
        or type(debug.get("debugger_inferior_exit_code")) is not int
        or debug["debugger_inferior_exit_code"]!=0
        or debug.get("debugger_inferior_signal") is not None
        or any(row.get(k)!=v for row in (outer,receipt) for k,v in debug.items())):
        raise ValueError("actual wall time, native lock or natural exit differs")
    return True


def verify_original_attempt(project, receipt_path):
    path = Path(receipt_path)
    if path.resolve(strict=True) != path or path.name != "receipt.json":
        raise ValueError("explicit regular original receipt required")
    if scheduler.file_hash(path) != RECEIPT_SHA:
        raise ValueError("original failed receipt SHA differs")
    receipt = json.loads(path.read_bytes())
    if (receipt["native_qualification"] is not False or receipt["reason"] != AUDIT_ERROR
            or receipt["qualification_scope"] != entry.SCOPE
            or receipt["native_execution_performed"] is not True):
        raise ValueError("unexpected original scheduler outcome")

    repo = scheduler.Repository(project)
    recipe = repo.native_recipe(repo.registry()["packages"][PACKAGE])
    frozen = {key: recipe[key] for key in ("command_spec_sha256", "code_config_sha256")}
    intent = json.loads(scheduler.native_artifact(path.parent, "intent.json").read_bytes())
    output = Path(receipt["output"])
    if (intent["package"] != PACKAGE or intent["head"] != receipt["tested_head"]
            or intent["output"] != str(output) or output.resolve(strict=True) != output
            or any(intent[key] != value or receipt[key] != value for key, value in frozen.items())
            or intent["approval"]["packages"].get(PACKAGE) != frozen):
        raise ValueError("saved scheduler approval or source identity differs")

    outer_path = scheduler.native_artifact(path.parent, "outer_exit.json")
    outer = json.loads(outer_path.read_bytes())
    native_log = scheduler.native_artifact(path.parent, "native.log")
    debug = scheduler.debugger_status(native_log)
    verify_outer_evidence(receipt, outer, recipe["spec"], debug, scheduler.file_hash(native_log))

    generic = scheduler.validate_native_artifacts(recipe["spec"], output)
    if any(receipt[key] != value for key, value in generic.items()):
        raise ValueError("physical artifact hashes or SDK ledger differ")
    return receipt, output, outer_path, generic


def verify_numerical_reset(output):
    artifact=scheduler.native_artifact(output,entry.NUMERIC_ARTIFACT)
    reset=entry.operation._read(scheduler.native_artifact(
        output,entry.operation.evidence.RESET_FILE))
    readback=reset["readback"]
    receipt={**readback["completed_reset_receipt"],
        "mesh_state_readback":{k:v for k,v in readback.items()
                              if k!="completed_reset_receipt"}}
    return stable_numeric.validate_numeric_reset_audit(
        entry.operation._read(artifact),reset_receipt=receipt)


@contextmanager
def producer_profile(project, output):
    """Use the producer checkout path only after proving the local source bytes."""
    producer = Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/integration")
    if producer.resolve(strict=True) != producer:
        raise ValueError("original producer checkout unavailable")
    local = profile.source_basis()
    expected = {**local["local_sources"], profile.CONFIG: profile.CONFIG_SHA}
    for name, sha in expected.items():
        if any(scheduler.file_hash(root / name) != sha for root in (Path(project), producer)):
            raise ValueError("producer source relocation bytes differ: " + name)
    with bindings((profile, "ROOT", producer)):
        original = profile.source_basis()
        recorded = entry.operation._read(output / entry.NUMERIC_ARTIFACT)["sources"]
        if recorded != original:
            raise ValueError("recorded original producer source profile differs")
        yield


def audit_retained_numeric(output,outer_exit,cleanup):
    output=Path(output)
    with entry.identity_scope(),bindings(
        (entry.operation,"VERSION",entry.VERSION),
        (entry.operation,"SCOPE",entry.SCOPE),(parent,"entry",entry)):
        result=parent.audit_outcome(output,outer_exit,cleanup)
    result=dict(result);numeric=None;digest=None
    try:
        artifact=scheduler.native_artifact(output,entry.NUMERIC_ARTIFACT)
        digest=scheduler.file_hash(artifact)
        if result.get("lifecycle_qualified") is True:
            numeric=verify_numerical_reset(output)
    except (OSError,ValueError,KeyError,TypeError,IndexError,RuntimeError,AttributeError) as error:
        result.update(lifecycle_qualified=False,
            numeric_sealing_error=type(error).__name__+": "+str(error))
    return dict(result,numeric_state_audit=numeric,numeric_state_artifact_sha256=digest,
        qualification_scope=entry.SCOPE,native_qualification=False,
        teacher_admission=False,grasp_qualified=False,contact_success_qualified=False)


def verify_saved_run(project,receipt_path):
    project=Path(project)
    names=tuple(dict.fromkeys(("scripts/verify_retained_dh116_pinch_v3.py",
        "tests/test_retained_pinch_v3.py", *parent.AUDIT_SOURCES,*entry.FILES)))
    sources=replay_sources(project,names)
    receipt,output,outer,generic=verify_original_attempt(project,receipt_path)
    with producer_profile(project,output):
        audit=audit_retained_numeric(output,outer,"explicit")
    numeric=audit["numeric_state_audit"]
    result=assemble_verified_sources(receipt,generic,audit,numeric)
    replay_sources(project,names,expected=sources)
    unchanged=scheduler.file_hash(Path(receipt_path))==RECEIPT_SHA
    if not unchanged:raise ValueError("original failed receipt changed during replay")
    _,after_output,_,after_generic=verify_original_attempt(project,receipt_path)
    if after_output!=output or after_generic!=generic:
        raise ValueError("original artifacts changed during replay")
    result.update(replay_source_sha256=sources,
        qualification_scope="retained_pinch_v3_saved_physical_source_reaudit_only",
        original_receipt_unchanged=unchanged,
        original_artifacts_unchanged=True,
        replay_code_config_sha256=hashlib.sha256(scheduler.json_text(sources).encode()).hexdigest(),
        native_execution_performed=False,new_solver_steps=0)
    return result


def assemble_verified_sources(receipt, generic, audit, numeric):
    """Seal independently verified source data without promoting the failed run."""
    required = ("lifecycle_qualified", "reset_mesh_qualified", "substep_contact_frame_qualified",
                "same_clock_health_getter_qualified")
    if any(audit.get(key) is not True for key in required):
        raise ValueError("original independent physical predicate failed: " + str(audit.get("reason")))
    if numeric["numeric_state_unchanged"] is not True:
        raise ValueError("numerical Reset proof differs")
    clock = generic["physics_clock"]
    if any(clock.get(key) != value for key, value in
           dict(setup_steps=2, measured_steps=232, total_steps=234).items()):
        raise ValueError("original SDK clock differs")
    if (receipt["native_qualification"] is not False
            or not 0 < receipt["duration_seconds"] <= 180
            or receipt["qualification_scope"] != entry.SCOPE
            or audit["qualification_scope"] != entry.SCOPE
            or audit["physics_steps"] != 234 or audit["measurement_steps"] != 232
            or audit["ppo_updates"] != 0):
        raise ValueError("original budget, update or scope differs")
    if (numeric["actual_samples"] != 2 or numeric["new_sdk_steps"] != 0
            or numeric["new_solver_steps"] != 0):
        raise ValueError("original numerical Reset adds steps or lacks snapshots")
    return dict(version="dh116-pinch-v3-retained-physical-reaudit-v1",
                original_receipt_sha256=RECEIPT_SHA, original_scheduler_status="failed",
                original_lifetime_audit_error=AUDIT_ERROR, native_qualification=False,
                teacher_admission=False, same_rod_grasp_qualified=False, ppo_updates=0,
                new_physics_steps=0, new_solver_steps=0, verified_saved_physical_sources=True,
                grasp_qualified=False,
                natural_exit_seconds=receipt["duration_seconds"], numeric_reset=numeric,
                original_generic_artifacts=generic, independent_physical_audit=audit)


def replay_sources(project,names,*,expected=None):
    """Seal one unambiguous inventory of regular local replay dependencies."""
    names=tuple(names)
    if len(names)!=len(set(names)):
        raise ValueError("duplicate replay source")
    sources={name:scheduler.file_hash(scheduler.native_artifact(Path(project),name)) for name in names}
    if expected is not None and sources!=expected:
        raise ValueError("replay source identity changed")
    return sources


def validate_report_path(project,output):
    project=Path(project).resolve(strict=True);output=Path(output)
    if (not output.is_absolute() or output.resolve()!=output
            or output.is_relative_to(project) or output.exists() or output.is_symlink()
            or not output.parent.is_dir()):
        raise ValueError("fresh private output outside the project required")
    return output


def write_report(project,output,result):
    """Create a private, exclusive envelope without touching the scheduler receipt."""
    output=validate_report_path(project,output)
    descriptor=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,"w",encoding="utf-8") as stream:
        stream.write(scheduler.json_text(result))
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validate_report_path(root,args.output)
    result = verify_saved_run(root, args.receipt)
    write_report(root,args.output,result)


if __name__ == "__main__":
    main()
