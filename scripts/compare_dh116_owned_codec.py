#!/usr/bin/env python3
"""Same native ledger baseline/opt-in CPU comparison; no native admission."""
import argparse,json,os,re,signal
from pathlib import Path
from scripts import profile_dh116_live_r0_observer as profile
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope

VERSION="dh116-owned-codec-same-ledger-comparison-v1"

def compare_results(base,fast):
    for row in (base,fast):
        if (row.get("status")!="complete" or row.get("new_physics_steps")!=0
            or row.get("ppo_updates")!=0 or row.get("teacher_admission") is not False
            or row.get("native_qualification") is not False
            or not re.fullmatch("[0-9a-f]{64}",row.get("encoded_stream_sha256",""))):
            raise ValueError("complete pure CPU diagnostic and full encoded stream digest required")
    for key in ("encoded_stream_sha256","encoded_bytes","counts","blocked_terms","source_receipt_sha256","source_budget_passed"):
        if base[key]!=fast[key]:raise ValueError("same ledger output differs: "+key)
    if base["counts"]!=dict(reset=1,begin=29,physical=232,end=29):
        raise ValueError("full actual immutable event sequence required")
    return dict(version=VERSION,status="complete",byte_equivalent=True,
        encoded_stream_sha256=base["encoded_stream_sha256"],encoded_bytes=base["encoded_bytes"],
        counts=base["counts"],blocked_terms=base["blocked_terms"],
        source_native_budget_passed=base["source_budget_passed"],
        baseline_profiled_cpu_s=base["elapsed_profiled_cpu_s"],
        fast_profiled_cpu_s=fast["elapsed_profiled_cpu_s"],
        profiled_cpu_ratio=base["elapsed_profiled_cpu_s"]/fast["elapsed_profiled_cpu_s"],
        profiled_timing_is_not_native_wall_time=True,write_io_not_profiled=True,
        new_physics_steps=0,ppo_updates=0,native_qualification=False,teacher_admission=False)

def save(path,data):
    with Path(path).open("x") as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--receipt-sha256",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *x:(_ for _ in ()).throw(TimeoutError("600s CPU budget")))
    signal.setitimer(signal.ITIMER_REAL,600)
    result=dict(version=VERSION,status="blocked",new_physics_steps=0,ppo_updates=0,
                native_qualification=False,teacher_admission=False)
    try:
        base_path=a.output/"baseline";base_path.mkdir()
        base=profile.run(a.receipt,a.receipt_sha256,base_path);save(base_path/"result.json",base)
        fast_path=a.output/"finite_array";fast_path.mkdir()
        with codec_scope():fast=profile.run(a.receipt,a.receipt_sha256,fast_path)
        save(fast_path/"result.json",fast)
        result=compare_results(base,fast)
    except Exception as e:result["reason"]=type(e).__name__+": "+str(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);save(a.output/"result.json",result)
    print(json.dumps(result))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
