#!/usr/bin/env python3
"""Exact offline replay of a pinned failed-budget native ledger, no SDK launch."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,os,time
from scripts import profile_dh116_complete_pair_r0_ledger as frozen
from scripts.validate_dh116_complete_pair_r0 import events
from scripts import replay_dh116_native_parameter_r0 as parameters
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest,encode_owned
from go2_pvcnn.control.dh116.tabletop_live_effective_pad_source import CompletePairLiveR0Observer
from go2_pvcnn.control.dh116.tabletop_r0_measurement_origin import OriginLiveR0Observer,FIELDS,restore_measurements
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope
PHYSICAL_SHA="4d2b457e7e0c581c0aed139b25e2531f1554bada54ef7b836825c833b0536298"
ROOT=Path(__file__).resolve().parents[1]

def equal(a,b,label):
    if _digest(a)!=_digest(b):raise ValueError(label+" differs")

def reward_sources(row,mapping):
    out=deepcopy(row)
    def source(s):
        return None if s is None else "|".join(mapping.get(t,t) for t in s.split("|"))
    for term in out["terms"].values():
        if "source" in term:term["source"]=source(term["source"])
        if "sources" in term:term["sources"]=sorted({source(s) for s in term["sources"]})
    for term in out.get("term_stats",{}).values():
        term["source_identities"]=sorted({source(s) for s in term["source_identities"]})
    return out

def compare_row(old,new,mapping):
    old=deepcopy(old);new=deepcopy(new)
    p=new["pair_diagnostics"].pop("measurement_origin_provenance")
    restored=restore_measurements(new["measurements"],p)
    equal(old["measurements"],restored,"original measurements")
    for name,s in p["samples"].items():
        if s["origin"] is not None:
            previous=mapping.setdefault(s["original_source"],s["origin"])
            if previous!=s["origin"]:raise ValueError("one raw sample mapped to conflicting origin")
            old["measurements"][name]["source"]=s["origin"]
    old["r0"]=reward_sources(old["r0"],mapping)
    equal(old,new,"all R0 values, validity, diagnostics and original guard")

def observer(cls,reset,radius):
    task=load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT)
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2
    budget["physics"]["solver_substeps_per_physics_step"]=12
    obj=cls(task,"pinch",env_id=0,budget_config=budget,
        diagnostic_coefficients={k:float(i+1) for i,k in enumerate(R0_TERMS)},
        backend_artifact=parameters.BACKEND,backend_sha256=parameters.BACKEND_SHA,
        radius_sample=dict(value=radius["value"],source="native_parameter_snapshot:"+_digest(radius)))
    obj.complete_physical_reset(reset,reset_approach=None)
    return obj

def replay(cls,path,output):
    reset=next(r["payload"] for r in events(path/"physical.jsonl") if r.get("operation")=="reset")
    packet=next(events(path/"director_reset.json"))["payload"]
    radius=parameters.check_radius_packet(packet,reset["reset"]["rod_reset"])
    start=time.perf_counter();obj=observer(cls,reset,radius);n=controls=0
    stats=Counter();unique=set()
    with Path(output).open("x") as f:
        def emit(op,payload):
            encoded=json.dumps(encode_owned(dict(operation=op,payload=payload)),allow_nan=False)
            f.write(encoded+"\n");stats["serialized_bytes"]+=len(encoded.encode())+1
        emit("reset",obj.initialization)
        for entry in events(path/"physical.jsonl"):
            op=entry.get("operation");payload=entry.get("payload")
            if op=="begin":obj.begin(**payload)
            elif op=="physical":
                obj.record(payload);row=obj._rows[-1];n+=1
                emit(op,row)
                for term in row["r0"]["term_stats"].values():
                    stats["source_occurrences"]+=len(term["source_identities"])
                    unique.update(term["source_identities"])
                stats["r0_compact_bytes"]+=len(json.dumps(encode_owned(row["r0"]),separators=(",",":"),allow_nan=False).encode())
            elif op=="end":
                emit(op,obj.end(payload));controls+=1
        f.flush();os.fsync(f.fileno())
    if n!=232 or controls!=29:raise ValueError("exact 232 physical samples and 29 controls required")
    return dict(stats,physical_rows=n,controls=controls,unique_sources=len(unique),
        elapsed_cpu_replay_wall_s=time.perf_counter()-start,output_sha256=frozen.sha(output))

def compare(baseline,candidate,saved_live):
    old=events(baseline);new=events(candidate)
    live=(r for r in events(saved_live) if r["operation"] in ("physical","end"))
    mapping={};n=controls=0
    for a in old:
        b=next(new,None)
        if b is None or b["operation"]!=a["operation"]:raise ValueError("missing/misaligned replay operation")
        op=a["operation"];x=a["payload"];y=b["payload"]
        if op=="reset":
            y=deepcopy(y);contract=y.pop("measurement_origin_contract")
            if set(contract["fields"])!=set(FIELDS):raise ValueError("explicit origin contract")
            equal(x,y,"actual reset")
        elif op=="physical":
            original=next(live,None)
            if original is None or original["operation"]!=op:raise ValueError("native saved sequence differs")
            equal(x,original["payload"]["r0_row"],"baseline versus original saved physical R0")
            compare_row(x,y,mapping);n+=1
        elif op=="end":
            original=next(live,None)
            if original is None or original["operation"]!=op:raise ValueError("native saved sequence differs")
            equal(x,original["payload"],"baseline versus original saved control")
            x=deepcopy(x);y=deepcopy(y)
            for arow,brow in zip(x["physical_rows"],y["physical_rows"],strict=True):compare_row(arow,brow,mapping)
            x.pop("physical_rows");y.pop("physical_rows")
            x["control_receipt"]=reward_sources(x["control_receipt"],mapping)
            for key in ("reset_replay_audit","live_post_reset_initialization"):
                y[key].pop("measurement_origin_contract")
            equal(x,y,"all control settlement values and actions");controls+=1
    if next(new,None) is not None or next(live,None) is not None or n!=232 or controls!=29:
        raise ValueError("complete exact comparison required")
    return dict(physical_rows=n,control_windows=controls,
        original_native_R0_reproduced_exactly=True,all_values_validity_guards_and_actions_equal=True,
        raw_sample_mapping_count=len(mapping),stable_origin_count=len(set(mapping.values())))

def run(output):
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    if frozen.sha(frozen.RECEIPT)!=frozen.RECEIPT_SHA:raise ValueError("frozen source receipt differs")
    receipt=json.loads(frozen.RECEIPT.read_bytes());path=Path(receipt["output"])
    if receipt["native_qualification"] or receipt["duration_seconds"]<=180:
        raise ValueError("source must remain failed budget evidence")
    for name,expected in (("physical.jsonl",PHYSICAL_SHA),("live_r0.jsonl",frozen.LEDGER_SHA)):
        if frozen.sha(path/name)!=expected:raise ValueError("original native artifact differs: "+name)
    with codec_scope():
        a=replay(CompletePairLiveR0Observer,path,output/"baseline.jsonl")
        b=replay(OriginLiveR0Observer,path,output/"origin.jsonl")
        verified=compare(output/"baseline.jsonl",output/"origin.jsonl",path/"live_r0.jsonl")
    result=dict(version="dh116-r0-origin-equivalence-v1",source_receipt_sha256=frozen.RECEIPT_SHA,
        source_physical_sha256=PHYSICAL_SHA,source_r0_sha256=frozen.LEDGER_SHA,
        baseline=a,origin=b,comparison=verified,implementation_sha256=frozen.sha(__file__),
        new_physics_steps=0,ppo_updates=0,native_qualification=False,teacher_admission=False,
        scope="offline_same_failed_budget_native_ledger_replay_not_new_native",
        limitation="Sequential CPU replay timings are not an Isaac native wall-time qualification.")
    with (output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    print(json.dumps(run(p.parse_args().output),sort_keys=True))
if __name__=="__main__":main()
