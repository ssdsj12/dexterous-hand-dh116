#!/usr/bin/env python3
"""Replay the SHA-pinned actual dual Pinch v4 interval through unchanged R0."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import (
    PhysicalR0Diagnostic,decode_owned,encode_owned,replay_diagnostic_bundle,
)
from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import R0_TERMS,load_flexible_grasp_task

VERSION="dh116-actual-dual-v4-r0-diagnostic-replay-v1"
LEDGER=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-dual-pinch-zero-v4-20261113-utbAE0/native-runs/flex.f7-dual-pinch-zero-interval-v4-native/attempt-fz50kogx/native/physical.jsonl")
LEDGER_SHA="ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c"
BACKEND=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so")
BACKEND_SHA="92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0"
PROFILE_PINS={
    "tests/test_dh116_deformable_control_step.py":"404882b72c987cdbc8fa36baf26fc8c5a374552493ddf9887192814d745c0942",
    "configs/dh116_training/deformable_task_budget_examples_v2.json":"94f12f9bcc0c2e7aacd833a4b9470a0d2446cb0e15fda0769335e4a6ff743d6a",
    "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json":"8b4f7f8d3fb96c12b9794a7c5daaa2b65f9231699348d5635044bd3f5ac75d63",
}


def _bytes(value):
    return json.dumps(encode_owned(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def validate_operations(rows):
    expected=["setup","setup","reset","initial_geometry","initial_geometry","begin"]+["physical"]*8+["end"]
    if (len(rows)!=16 or [row.get("operation") for row in rows[1:]]!=expected
            or rows[0].get("physics_budget_end")!=10 or rows[0].get("solver_substeps")!=12):
        raise ValueError("exact original 2+8 journal operation sequence required")


def load_native_rows(ledger=LEDGER):
    ledger=Path(ledger)
    raw=ledger.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=LEDGER_SHA:
        raise ValueError("actual dual v4 ledger SHA differs")
    inspection=inspect_evidence(ledger)
    if inspection["record_sha256"]!=LEDGER_SHA:
        raise ValueError("ledger SHA changed during sequence inspection")
    rows=[decode_owned(json.loads(line)) for line in raw.splitlines()]
    validate_operations(rows)
    if (inspection["measured_physics_rows"]!=8 or inspection["complete_control_windows"]!=1
            or inspection["settled_control_windows"]!=1 or inspection["unsettled_control_window"]
            or inspection["terminal_reason"] is not None):
        raise ValueError("one complete nonterminal recorded native interval required")
    return rows,inspection


def replay_native_interval(ledger=LEDGER):
    """Consume only real reset/begin/8 physical/end events, with no auxiliary fill."""
    rows,inspection=load_native_rows(ledger)
    for rel,expected in PROFILE_PINS.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError("unchanged diagnostic fixture/task source required: "+rel)
    task=load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT)
    if any(value is not None for value in task["reward"]["coefficients"].values()):
        raise ValueError("frozen task coefficients must remain unknown")
    # Same explicit CPU fixture as tests/test_dh116_deformable_control_step.py:16-22.
    # This 360-control template is reducer capacity, not this run's native budget.
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_text())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2
    budget["physics"]["solver_substeps_per_physics_step"]=12
    observer=PhysicalR0Diagnostic(task,"pinch",env_id=0,budget_config=budget,
        diagnostic_coefficients={name:float(i+1) for i,name in enumerate(R0_TERMS)},
        backend_artifact=BACKEND,backend_sha256=BACKEND_SHA,
        radius_sample={"value":None,"source":None})
    for entry in rows[1:]:
        operation,payload=entry["operation"],entry["payload"]
        if operation in ("reset","physical"):
            observer.record(payload)  # No invented auxiliary measurement.
        elif operation=="begin":
            observer.begin(**payload)
        elif operation=="end":
            observer.end(payload)
        # Setup and initial geometry were independently inspected, not replayed
        # as physical samples, reset operations, or additional R0 evaluations.
    report=observer.report()
    bundle=observer.bundle()
    roundtrip=decode_owned(json.loads(_bytes(bundle)))
    repeated=replay_diagnostic_bundle(roundtrip)
    if _bytes(repeated)!=_bytes(report):
        raise ValueError("strict JSON bundle replay did not reproduce the same R0 report")
    control=report["controls"][0]
    receipt=control["control_receipt"]
    physical=[entry["payload"] for entry in rows if entry.get("operation")=="physical"]
    pairs=[p for event in physical for p in event["exchange"]["rod"]["rod_mesh_contact_pairs"]["pairs"]]
    hand_table=[event["hand_table_contacts"] for event in physical]
    summary=dict(version=VERSION,ledger_path=str(Path(ledger)),ledger_sha256=LEDGER_SHA,
        backend_artifact=str(BACKEND),backend_sha256=BACKEND_SHA,profile_source_sha256=PROFILE_PINS,
        journal=inspection,strict_json_bundle_replay_equal=True,
        bundle_sha256=hashlib.sha256(_bytes(bundle)).hexdigest(),
        report_sha256=hashlib.sha256(_bytes(report)).hexdigest(),
        recorded_control_windows=report["recorded_control_windows"],
        settled_control_windows=report["settled_control_windows"],
        setup_steps=2,physical_samples=8,
        physical_steps=[event["exchange"]["physics_step"] for event in physical],
        rod_ticks=[event["exchange"]["rod"]["rod_tick"] for event in physical],
        original_r0_evaluations_per_replay_pass=8,
        coefficient_scope="existing_CPU_fixture_only_not_physical_reward",
        budget_scope="existing_CPU_reducer_capacity_not_native_run_authorization",
        cpu_template_control_capacity=budget["approach_control_steps"]+budget["hold_control_steps"],
        radius_sample=dict(value=None,source=None),
        radius_na_reason="actual ledger has no native radius readback; configured radius is not substituted",
        auxiliary_measurements_supplied=False,
        terms={name:{k:term[k] for k in ("unit","coefficient","valid_samples","expected_samples","raw_sum","contribution_sum")}
               for name,term in receipt["terms"].items()},
        blocked_terms=receipt["blocked_terms"],
        raw_hand_pair_samples=sum(p["mesh_name"].startswith(("left.","right.")) for p in pairs),
        raw_rod_table_pair_samples=sum(p["mesh_name"]=="table" for p in pairs),
        hand_table_source_available=all(row["available"] is True for row in hand_table),
        no_hand_table_contact_proven=False,
        hand_table_unavailable_reasons=sorted({row["reason"] for row in hand_table if row["available"] is False}),
        phases=[row["r0"]["phase"] for row in control["physical_rows"]],
        body_net_lift_m=[row["r0"]["body_net_lift_m"] for row in control["physical_rows"]],
        min_node_net_lift_m=[row["r0"]["min_node_net_lift_m"] for row in control["physical_rows"]],
        prohibited_support_latched=[row["r0"]["prohibited_support_latched"] for row in control["physical_rows"]],
        terminated=receipt["terminated"],truncated=receipt["truncated"],success_candidate=receipt["success_candidate"],
        diagnostic_reward=None,training_reward=None,new_physics_steps=0,optimizer_updates=0,
        native_qualification=False,teacher_admission=False,full_371d_bound=False,
        r0_native_reset_complete=False,frozen_task_mutated=False)
    return report,bundle,summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger",type=Path,default=LEDGER)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bundle-output",type=Path,required=True)
    args=parser.parse_args(argv)
    outputs=(args.output,args.bundle_output)
    if len({p.resolve() for p in (*outputs,args.ledger)})!=3:
        parser.error("input, output and bundle output must be distinct")
    if any(p.exists() or p.is_symlink() for p in outputs):
        parser.error("output already exists; preserve all prior evidence")
    report,bundle,summary=replay_native_interval(args.ledger)
    native_imports=sorted(n for n in sys.modules if n.split(".")[0] in ("omni","isaaclab","isaacsim","ropesim","warp"))
    if native_imports:
        raise RuntimeError("native import crossed the CPU replay boundary")
    for p in outputs:p.parent.mkdir(parents=True,exist_ok=True)
    with args.bundle_output.open("xb") as stream:stream.write(_bytes(bundle)+b"\n")
    receipt=dict(summary=summary,report=encode_owned(report),native_imports=native_imports,
        bundle_file_sha256=hashlib.sha256(args.bundle_output.read_bytes()).hexdigest())
    with args.output.open("x") as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(physical_samples=8,settled_control_windows=1,diagnostic_reward=None,
        training_reward=None,new_physics_steps=0,teacher_admission=False,output=str(args.output))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
