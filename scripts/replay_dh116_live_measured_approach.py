#!/usr/bin/env python3
"""Read-only replay of a qualified bounded ledger; not another Isaac run."""
from pathlib import Path
import argparse,hashlib,json,os,time
from collections import Counter
from copy import deepcopy
from scripts.prepare_dh116_live_measured_approach import MeasuredApproachObserver,VERSION
from scripts.validate_dh116_complete_pair_r0 import events
from scripts.profile_dh116_complete_pair_r0_ledger import sha
from scripts import replay_dh116_native_parameter_r0 as radius
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest,encode_owned
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope

def compare_row(baseline,candidate):
    """All physical facts and all other R0 term histories must be unchanged."""
    old,new=deepcopy(baseline),deepcopy(candidate)
    measured=new["pair_diagnostics"].pop("measured_approach")
    if old["measurements"]["approach_distance_m"]!={"value":None,"source":None}:
        raise ValueError("comparison source already contains approach")
    for row in (old,new):
        row["measurements"].pop("approach_distance_m")
        row["unavailable_sources"].pop("approach_distance_m",None)
        r=row["r0"]
        for key in ("diagnostic_reward","blocked_terms"):r.pop(key)
        r["terms"].pop("approach_progress");r["term_stats"].pop("approach_progress")
    if _digest(old)!=_digest(new):
        raise ValueError("non-approach physical/R0 values or source histories changed")
    return measured

def run(receipt,expected_receipt_sha256,expected_physical_sha256,output,project):
    receipt,output,project=map(Path,(receipt,output,project))
    if sha(receipt)!=expected_receipt_sha256:raise ValueError("receipt SHA differs")
    r=json.loads(receipt.read_bytes())
    if r.get("native_qualification") is not True or r.get("outer_exit_code")!=0:
        raise ValueError("qualified source receipt required")
    source=Path(r["output"])
    if sha(source/"physical.jsonl")!=expected_physical_sha256:raise ValueError("physical SHA differs")
    originals={n:sha(source/n) for n in ("physical.jsonl","director_reset.json","live_r0.jsonl")}
    reset=next(e["payload"] for e in events(source/"physical.jsonl") if e.get("operation")=="reset")
    packet=next(events(source/"director_reset.json"))["payload"]
    actual=radius.check_radius_packet(packet,reset["reset"]["rod_reset"])
    task=load_flexible_grasp_task(project/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",project)
    budget=json.loads((project/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2;budget["physics"]["solver_substeps_per_physics_step"]=12
    obj=MeasuredApproachObserver(task,"pinch",env_id=0,budget_config=budget,
        diagnostic_coefficients={k:float(i+1) for i,k in enumerate(R0_TERMS)},
        backend_artifact=radius.BACKEND,backend_sha256=radius.BACKEND_SHA,
        radius_sample=dict(value=actual["value"],source="native_parameter_snapshot:"+_digest(actual)),
        project=project,parameter_packet=packet)
    output.mkdir(parents=True,exist_ok=False)
    source_rows=(e["payload"]["r0_row"] for e in events(source/"live_r0.jsonl") if e["operation"]=="physical")
    counts=Counter();blocked=Counter();origins=set();start=time.perf_counter();first=last=None
    with codec_scope(),(output/"replayed_r0.jsonl").open("x") as stream:
        def emit(op,payload):
            stream.write(json.dumps(encode_owned(dict(operation=op,payload=payload)),allow_nan=False)+"\n")
        init=obj.complete_physical_reset(reset,reset_approach=None);emit("reset",init)
        for event in events(source/"physical.jsonl"):
            op,p=event.get("operation"),event.get("payload")
            if op=="begin":obj.begin(**p)
            elif op=="physical":
                obj.record(p);row=obj._rows[-1]
                original=next(source_rows,None)
                if original is None:raise ValueError("missing original R0 sample")
                measured=compare_row(original,row)
                counts["physical_rows"]+=1
                counts["valid_approach_samples"]+=row["r0"]["terms"]["approach_progress"]["valid"]
                origins.update(row["r0"]["term_stats"]["approach_progress"]["source_identities"])
                if first is None:first=measured
                last=measured
                emit("physical",row)
            elif op=="end":
                result=obj.end(p);counts["control_windows"]+=1
                blocked.update(result["control_receipt"]["blocked_terms"])
                if result["training_reward"] is not None or result["teacher_admission"]:
                    raise ValueError("diagnostic unexpectedly acquired training admission")
                emit("end",result)
        if next(source_rows,None) is not None:raise ValueError("extra original R0 sample")
        stream.flush();os.fsync(stream.fileno())
    if counts["physical_rows"]!=232 or counts["control_windows"]!=29:
        raise ValueError("expected complete 232 measurements/29 controls")
    if originals!={n:sha(source/n) for n in originals}:raise ValueError("read-only evidence changed")
    def distance(m):return m["auxiliary"]["samples"]["approach_distance_m"]["value"]
    result=dict(version=VERSION,scope="offline_qualified_ledger_replay_not_new_native",source_receipt_sha256=expected_receipt_sha256,
        source_artifacts=originals,counts=dict(counts),blocked_control_terms=dict(blocked),
        approach_source_identities=len(origins),reset_distance_m=distance(init["measured_approach_contract"]["actual_reset_measurement"]),
        first_distance_m=distance(first),last_distance_m=distance(last),all_non_approach_samples_exactly_equal=True,
        elapsed_cpu_wall_s=time.perf_counter()-start,output_sha256=sha(output/"replayed_r0.jsonl"),
        new_physics_steps=0,ppo_updates=0,teacher_admission=False,native_qualification=False,training_reward=None)
    with (output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ("receipt","expected-receipt-sha256","expected-physical-sha256","output","project"):
        p.add_argument("--"+name,required=True)
    print(json.dumps(run(**vars(p.parse_args())),sort_keys=True))
if __name__=="__main__":main()
