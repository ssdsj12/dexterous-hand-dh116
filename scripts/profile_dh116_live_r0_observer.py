#!/usr/bin/env python3
"""Bounded CPU profile of immutable native Reset/R0 evidence, no physics."""
import argparse,cProfile,hashlib,json,os,pstats,signal,time
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-live-r0-failed-budget-cpu-profile-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def lock_receipt(path,expected):
    path=Path(path)
    if sha(path)!=expected:raise ValueError("receipt SHA differs")
    r=json.loads(path.read_bytes())
    if r.get("native_execution_performed") is not True:
        raise ValueError("actual native execution record required")
    if (r["qualification_scope"]!="actual_coupled_reset_then_live_original_R0_observer_234_steps_only"
        or r["outer_exit_code"]!=0 or r["debugger_inferior_signal"] is not None):
        raise ValueError("specific naturally-exited actual observer record required")
    audit=path.parent/"live-r0-post-exit-audit.json"
    if sha(audit)!=r["lifetime_post_exit_audit_sha256"]:
        raise ValueError("independent audit SHA differs")
    a=json.loads(audit.read_bytes())["evidence"]
    artifacts=a["artifact_sha256"] if r["native_qualification"] else a["available_artifact_sha256"]
    native=Path(r["output"])
    for name in ("physical.jsonl","director_reset.json","live_r0.jsonl","live_r0_reset.json","ownership.jsonl"):
        if sha(native/name)!=artifacts[name]:raise ValueError("native source bytes differ: "+name)
    for name,digest in r["lifetime_validator_source_sha256"].items():
        if sha(ROOT/name)!=digest:raise ValueError("source differs from original native run: "+name)
    return r

def compare_phases(prior,current):
    pairs=dict(reset_to_owned_release=("dual_reset_complete","owned hand and solver references dropped"),
               app_close=("before_app_close","after_app_close"))
    result={"causal_attribution_proven":False}
    def elapsed(rows,a,b):
        aa=[r["time_unix_s"] for r in rows if r["phase"]==a]
        bb=[r["time_unix_s"] for r in rows if r["phase"]==b]
        if len(aa)!=1 or len(bb)!=1 or bb[0]<aa[0]:raise ValueError("unique monotonic observed phases required")
        return bb[0]-aa[0]
    for name,(a,b) in pairs.items():
        old,new=elapsed(prior,a,b),elapsed(current,a,b)
        result[name]=dict(prior_s=old,current_s=new,delta_s=new-old)
    return result

def run(receipt,expected,output):
    from go2_pvcnn.control.dh116.tabletop_r0_live_observer import LiveR0Observer
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned,_digest
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS
    from scripts import replay_dh116_native_parameter_r0 as radius
    r=lock_receipt(receipt,expected);native=Path(r["output"])
    def rows(path):
        data=Path(path).read_bytes()
        if not data.endswith(b"\n"):raise ValueError("incomplete JSONL")
        return [decode_owned(json.loads(line)) for line in data.splitlines()]
    events=rows(native/"physical.jsonl")[1:]
    reset=next(e["payload"] for e in events if e["operation"]=="reset")
    parameter=decode_owned(json.loads((native/"director_reset.json").read_bytes()))["payload"]
    measured=radius.check_radius_packet(parameter,reset["reset"]["rod_reset"])
    task=load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT)
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2
    budget["physics"]["solver_substeps_per_physics_step"]=12
    obs=LiveR0Observer(task,"pinch",env_id=0,budget_config=budget,
        diagnostic_coefficients={k:float(i+1) for i,k in enumerate(R0_TERMS)},
        backend_artifact=radius.BACKEND,backend_sha256=radius.BACKEND_SHA,
        radius_sample=dict(value=measured["value"],source="native_parameter_snapshot:"+_digest(measured)))
    start=time.monotonic();profile=cProfile.Profile();counts=Counter();blocked=Counter()
    encoded_bytes=0
    encoded_stream=hashlib.sha256()
    def encode(value):
        nonlocal encoded_bytes
        payload=json.dumps(encode_owned(value),allow_nan=False).encode()
        encoded_bytes+=len(payload)
        encoded_stream.update(payload+b"\n")
    profile.enable()
    try:
        for e in events:
            op,p=e["operation"],e["payload"]
            if op=="reset":encode(obs.complete_physical_reset(p,reset_approach=None))
            elif op=="begin":obs.begin(**p);encode(dict(pre_action_sha256=_digest(p["pre_action_state"]),action_sha256=_digest(p["action"])))
            elif op=="physical":
                obs.record(p)
                row=obs._rows[-1];blocked.update(row["r0"]["blocked_terms"])
                encode(dict(identity=p["exchange"]["action_readback"]["post"]["identity"][0],
                    actual_event_sha256=_digest(p),r0_row=row))
            elif op=="end":encode(obs.end(p))
            else:continue
            counts[op]+=1
    finally:
        profile.disable();profile.dump_stats(str(output/"cpu.pstats"))
    elapsed=time.monotonic()-start
    if dict(counts)!=dict(reset=1,begin=29,physical=232,end=29):raise ValueError("incomplete actual owner sequence")
    stats=pstats.Stats(profile)
    top=[]
    for (path,line,name),(cc,nc,tt,ct,callers) in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:35]:
        top.append(dict(file=path,line=line,function=name,primitive_calls=cc,total_calls=nc,self_s=tt,cumulative_s=ct))
    # This is a profile replay, not a measurement of native per-function wall time.
    return dict(version=VERSION,status="complete",counts=dict(counts),elapsed_profiled_cpu_s=elapsed,
        blocked_terms=dict(blocked),encoded_bytes=encoded_bytes,
        encoded_stream_sha256=encoded_stream.hexdigest(),top_cumulative=top,
        source_receipt_sha256=expected,source_budget_passed=r["native_qualification"],
        native_duration_s=r["duration_seconds"],disk_io_profiled=False,
        profiler_overhead_present=True,causal_native_attribution_proven=False,
        new_physics_steps=0,ppo_updates=0,teacher_admission=False,native_qualification=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--receipt-sha256",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError("600s CPU limit")))
    signal.setitimer(signal.ITIMER_REAL,600)
    result=dict(version=VERSION,status="blocked",new_physics_steps=0,ppo_updates=0,teacher_admission=False,native_qualification=False)
    try:result=run(a.receipt,a.receipt_sha256,a.output)
    except Exception as e:result["reason"]=type(e).__name__+": "+str(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        with (a.output/"result.json").open("x") as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","elapsed_profiled_cpu_s","counts","blocked_terms")}))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
