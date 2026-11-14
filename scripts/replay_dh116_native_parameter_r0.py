#!/usr/bin/env python3
"""Offline compare original and opt-in R0 on the already completed parameter task.

Never starts Isaac, advances physics, injects missing measurements or reuses a
native budget. The historical receipt's qualification applies only to its
18-step endpoint/lifetime scope, not this reward or any Teacher.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import (
    PhysicalR0Diagnostic,decode_owned,encode_owned,
)
from go2_pvcnn.control.dh116.tabletop_deformable_r0_applicability import ApplicableR0ControlStep,VERSION
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import R0_TERMS,load_flexible_grasp_task
from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot

RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-native-parameter-task-native/bc73ea4aba22065b14830640743d21f1133eee28bedd0821fa191320f3a8ceb3/attempt-qxt8edbr/receipt.json")
RECEIPT_SHA="f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901"
BACKEND=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu/build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so")
BACKEND_SHA="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"

def read_pinned(path,sha):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("owned regular source file required")
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError("source SHA differs: "+str(path))
    return raw

def check_radius_packet(packet,rod):
    """Validate actual host-metadata radius with raw/derived/source/clock binding."""
    identity=packet["identity"]
    for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick",
              "timestamp_s","solver_time_s"):
        if identity[k]!=rod[k]:
            raise ValueError("radius packet and actual rod identity differ")
    raw=packet["raw_native_parameters"]
    verified=validate_snapshot(raw,expected_num_envs=1,expected_nodes=41,
                               expected_time_s=identity["solver_time_s"])
    if encode_owned(verified)!=encode_owned(packet["native_parameters"]):
        raise ValueError("native derived parameter fields differ")
    if not np.array_equal(packet["node_positions_m"],rod["node_positions"]):
        raise ValueError("radius packet rod positions differ")
    expected=dict(value=float(raw["contact_radius_m"]),source=dict(
        api="RopeBatch.parameter_snapshot.contact_radius_m",provenance="native_host_metadata",
        backend_sha256=BACKEND_SHA,identity=identity,uncertainty=None,
        hardware_measured=False,task_physics_qualified=False))
    if packet["reward_radius_sample"]!=expected:
        raise ValueError("raw radius/provenance differs")
    return expected

class _ResetBoundApplicableR0(ApplicableR0ControlStep):
    """Only the explicit historical-replay mode binds this recorded baseline."""
    def __init__(self,*args,reset_approach,**kwargs):
        super().__init__(*args,**kwargs)
        self._recorded_reset_approach=deepcopy(reset_approach)

    def reset(self,reset_hooks,read_after_reset):
        return super().reset(reset_hooks,read_after_reset,
                             reset_approach=deepcopy(self._recorded_reset_approach))


def replay(*,approach_reader=None,include_reset_approach=False):
    if include_reset_approach and approach_reader is None:
        raise ValueError("measured reset requires the actual geometry source")
    receipt=json.loads(read_pinned(RECEIPT,RECEIPT_SHA))
    if (receipt["qualification_scope"]!="exact_2_setup_plus_16_native_parameter_task_endpoints_only"
            or receipt["outer_exit_code"]!=0 or receipt["debugger_inferior_signal"] is not None
            or receipt["physics_clock"]["total_steps"]!=18):
        raise ValueError("expected completed historical endpoint task")
    root=Path(receipt["output"])
    rows=[decode_owned(json.loads(line)) for line in
          read_pinned(root/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"]).splitlines()]
    reset=json.loads(read_pinned(root/"director_reset.json",receipt["artifact_sha256"]["director_reset.json"]))
    p0=decode_owned(reset["payload"])
    reset_event=next(r["payload"] for r in rows if r.get("operation")=="reset")
    radius=check_radius_packet(p0,reset_event["reset"]["rod_reset"])
    physical=[r["payload"] for r in rows if r.get("operation")=="physical"]
    if len(physical)!=16:raise ValueError("expected sixteen measured endpoints")
    for event in physical:
        sample=check_radius_packet(event["exchange"]["material_director"],event["exchange"]["rod"])
        if sample["value"]!=radius["value"]:
            raise ValueError("episode contact radius changed")
    # Scalar source remains a string for the original R0 contract, but the full
    # raw provenance and every endpoint identity remain in the output below.
    source="native_host_metadata:RopeBatch.parameter_snapshot.contact_radius_m:"+BACKEND_SHA
    task=load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT)
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_text())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2
    budget["physics"]["solver_substeps_per_physics_step"]=12
    weights={name:float(i+1) for i,name in enumerate(R0_TERMS)}
    reports={}
    # Compute each actual source once; the two opt-in modes consume the same
    # owned endpoint values. No reset or first-step substitution is performed.
    approach_samples=([approach_reader.capture(event["exchange"]) for event in physical]
                      if approach_reader is not None else [])
    reset_approach_sample=(approach_reader.capture_reset(reset_event["reset"],p0)
                           if include_reset_approach else None)
    modes=("original","applicability_v2") + (("applicability_v2_with_measured_approach",) if approach_reader is not None else ())
    if include_reset_approach:
        modes+=("applicability_v2_with_measured_reset_approach",)
    for mode in modes:
        measured_index=0
        observer=PhysicalR0Diagnostic(task,"pinch",env_id=0,budget_config=budget,
            diagnostic_coefficients=weights,backend_artifact=BACKEND,backend_sha256=BACKEND_SHA,
            radius_sample=dict(value=radius["value"],source=source))
        if mode.startswith("applicability_v2"):
            # Explicitly isolated offline observer; no monkeypatch/default path.
            adapted=deepcopy(task);adapted["backend"]["extension_sha256"]=BACKEND_SHA
            reducer=(_ResetBoundApplicableR0 if mode=="applicability_v2_with_measured_reset_approach"
                     else ApplicableR0ControlStep)
            extra=({"reset_approach":reset_approach_sample["reset_approach"]}
                   if mode=="applicability_v2_with_measured_reset_approach" else {})
            observer._reducer=reducer(adapted,"pinch",env_id=0,budget_config=budget,
                                      diagnostic_coefficients=weights,**extra)
        for entry in rows[1:]:
            operation,payload=entry["operation"],entry["payload"]
            if operation in ("reset","physical"):
                auxiliary=None
                if operation=="physical" and mode in (
                        "applicability_v2_with_measured_approach",
                        "applicability_v2_with_measured_reset_approach"):
                    auxiliary=deepcopy(approach_samples[measured_index]["auxiliary"])
                    measured_index+=1
                observer.record(payload,auxiliary=auxiliary)
            elif operation=="begin":observer.begin(**payload)
            elif operation=="end":observer.end(payload)
        reports[mode]=observer.report()
    controls={mode:[r["control_receipt"] for r in report["controls"]] for mode,report in reports.items()}
    if any(len(v)!=2 or any(r is None or r["physical_samples"]!=8 for r in v) for v in controls.values()):
        raise ValueError("two exact original control intervals must be replayed")
    pairs=[p for event in physical for p in event["exchange"]["rod"]["rod_mesh_contact_pairs"]["pairs"]]
    return dict(
        version=VERSION,scope="offline_replay_of_existing_native_evidence",
        receipt_sha256=RECEIPT_SHA,ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],
        radius_provenance=radius,parameter_packets_validated=17,
        coefficient_scope="existing_CPU_fixture_only_not_training_weights",
        measured_physical_rows=16,controls=controls,
        per_step=[dict(identity={k:r["r0"][k] for k in ("env_id","episode_uid","reset_id","physics_step","rod_tick","timestamp_s")},
            terms=r["r0"]["terms"],blocked_terms=r["r0"]["blocked_terms"])
            for c in reports[modes[-1]]["controls"] for r in c["physical_rows"]],
        approach_samples=approach_samples,reset_approach_sample=reset_approach_sample,
        missing_sources=[*(["measured_reset_approach_distance"] if not include_reset_approach else []),
                         *(["per_step_approach_distance"] if approach_reader is None else []),
                         "qualified_stability_predicate","positive_bilateral_opposition",
                         "fixed_material_anchor_and_tangential_slip","hand_table_contact_source"],
        raw_hand_rod_pair_samples=sum(p["mesh_name"].startswith(("left.","right.")) for p in pairs),
        raw_table_pair_samples=sum(p["mesh_name"]=="table" for p in pairs),
        source_task_qualification=receipt["qualification_scope"],r0_native_reset_complete=False,
        new_physics_steps=0,optimizer_updates=0,training_reward=None,
        native_qualification=False,teacher_admission=False)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--measured-approach",action="store_true",help="opt-in actual measured Pinch SDF source, no reset fill")
    parser.add_argument("--measured-reset-approach",action="store_true",
        help="also derive the baseline from actual recorded reset getter buffers; requires --measured-approach")
    args=parser.parse_args(argv)
    if args.measured_reset_approach and not args.measured_approach:
        parser.error("--measured-reset-approach requires --measured-approach")
    if args.output.exists() or args.output.is_symlink():parser.error("preserve existing output")
    reader=None
    if args.measured_approach:
        from go2_pvcnn.control.dh116.tabletop_r0_measured_approach import MeasuredPinchApproach
        reader=MeasuredPinchApproach(ROOT)
    result=replay(approach_reader=reader,include_reset_approach=args.measured_reset_approach)
    native=[n for n in sys.modules if n.split(".")[0] in ("omni","isaacsim","isaaclab","ropesim","warp")]
    if native:raise RuntimeError("offline boundary crossed: "+str(native))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(encode_owned(dict(result,native_imports=native)),stream,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(args.output),new_physics_steps=0,teacher_admission=False,
        blocked={m:[r["blocked_terms"] for r in c] for m,c in result["controls"].items()})))
    return 0

if __name__=="__main__":raise SystemExit(main())
