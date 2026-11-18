"""Frozen tracking-aware native contact episode; original-region evidence only."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from scripts import audit_dh116_effective_right_tracking as previous
from scripts import prepare_dh116_tracking_pinch as plan
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_deferred_physical_audit import validate_seal
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned

VERSION="dh116-tracking-pinch-native-contact-replay-v1"
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-tracking-pinch-native/4b900443ee3cf69ce0c387fa7489b554eaa948ce7143f4384e23e44c6263fc3e/attempt-zgggtx6_/receipt.json")
RECEIPT_SHA="83a8758210fd6aa35da239c32819b91b9b6967307d30f7e380c67f965381cf11"
SCOPE="tracking_aware_pinch_2_setup_232_original_guard_contact_diagnostic_only"
AUDIT_SHA="f37981e40594a10402b7f99fcb7cb6b8219aa91d40fdbed42ebd72d194c8397d"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_locked(path,expected):
    if not Path(path).is_file() or Path(path).is_symlink() or sha(path)!=expected:
        raise ValueError("complete frozen source bytes required")
    return Path(path).read_bytes()

def contiguous_windows(slots,dt):
    if not math.isfinite(dt) or dt<=0 or any(type(n) is not int or n<0 for n in slots):
        raise ValueError("finite actual solver interval and integer slots required")
    if list(slots)!=sorted(set(slots)):raise ValueError("unique measured ordered substeps required")
    runs=[]
    for n in slots:
        if runs and n==runs[-1][1]+1:runs[-1][1]=n
        else:runs.append([n,n])
    longest=max((b-a+1 for a,b in runs),default=0)
    return dict(count=len(slots),runs=runs,longest_samples=longest,longest_duration_s=longest*dt)

def run():
    r=json.loads(read_locked(RECEIPT,RECEIPT_SHA))
    if (r.get("native_qualification") is not True or r.get("native_execution_performed") is not True
        or r.get("qualification_scope")!=SCOPE or r.get("outer_exit_code")!=0
        or not 0<r.get("duration_seconds",float("inf"))<=180):
        raise ValueError("exact bounded diagnostic qualification required, never grasp success")
    audit=json.loads(read_locked(RECEIPT.parent/"tracking-pinch-post-exit-audit.json",AUDIT_SHA))
    a=audit["evidence"]
    if not a.get("full_post_exit_audit_passed") or not a.get("lifecycle_qualified") or a["qualification_scope"]!=SCOPE:
        raise ValueError("complete independent post-exit audit required")
    root=Path(r["output"]);raw=read_locked(root/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"])
    result=json.loads(read_locked(root/"result.json",r["artifact_sha256"]["result.json"]))
    validate_seal(root/"physical.jsonl",result["acquisition"]["journal"])
    if not raw.endswith(b"\n"):raise ValueError("incomplete native line")
    entries=[json.loads(line) for line in raw.splitlines()]
    rows=[decode_owned(e["payload"])["exchange"] for e in entries if e.get("operation")=="physical"]
    summary=previous.contact_summary(rows)
    required={s:{s+"."+n for n in ("finger14_link","finger23_link")} for s in ("left","right")}
    slots={s:[] for s in ("left","right","bilateral")};boundaries=[];forces=defaultdict(list);dt=None
    for x in rows:
        tick=x["physics_step"];cap=x["rod"]["rod_mesh_contact_pairs"]
        d=float(cap["substep_dt_s"])
        if not math.isfinite(d) or not np.isclose(d,1/(480*12),atol=1e-12,rtol=0) or (dt is not None and dt!=d):
            raise ValueError("actual 480Hz/12 solver time contract differs")
        dt=d;windows=defaultdict(set);counts=defaultdict(int)
        for p in x["closure_geometry"]["pad_region_diagnostic"]["pairs"]:
            sub=p["solver_substep"];name=p["mesh_name"]
            if type(sub) is not int or not 0<=sub<12:raise ValueError("actual solver substep identity required")
            if p["region"]["accepted_by_original_patch"]:
                force=float(p["normal_force_on_rod_n"])
                if not math.isfinite(force):raise ValueError("finite source normal-force magnitude required")
                windows[sub].add(name);counts[name]+=1;forces[name].append(force)
        for sub,names in sorted(windows.items()):
            index=(tick-3)*12+sub
            for side in ("left","right"):
                if required[side]<=names:slots[side].append(index)
            if required["left"]|required["right"]<=names:slots["bilateral"].append(index)
        boundaries.append(dict(physics_step=tick,policy_step=x["policy_step"],
            identity=x["closure_geometry"]["raw"]["identity"],accepted_region_pairs=dict(counts),
            guard_failure=x["guard_failure"]))
    geometry={}
    def minimum(side,key,pad=None):
        def value(x):
            side_data=x["closure_geometry"]["sides"][side]
            return side_data[key] if pad is None else side_data[key][pad]
        x=min(rows,key=value)
        return dict(value_mm=1000*value(x),physics_step=x["physics_step"],
            episode_uid=x["closure_geometry"]["raw"]["identity"]["episode_uid"])
    for side in ("left","right"):
        pads=[n for n in rows[0]["closure_geometry"]["sides"][side]["table_clearance_m"] if n.startswith("cooked:")]
        geometry[side]={key:minimum(side,key) for key in ("minimum_table_clearance_m","minimum_non_target_gap_lower_m")}
        geometry[side]["cooked_table_by_pad"]={n:minimum(side,"table_clearance_m",n) for n in pads}
    with bindings((previous,"plan",plan)):
        endpoint=previous.compare_endpoint(rows[-1])
    return dict(version=VERSION,source=dict(receipt=str(RECEIPT),receipt_sha256=RECEIPT_SHA,
        physical_sha256=r["artifact_sha256"]["physical.jsonl"],audit_sha256=AUDIT_SHA,
        outer_exit_code=r["outer_exit_code"],duration_seconds=r["duration_seconds"],
        qualification_scope=SCOPE),summary=summary,
        windows={s:contiguous_windows(v,dt) for s,v in slots.items()},
        accepted_pair_normal_force_n={n:dict(minimum=min(v),maximum=max(v),mean=float(np.mean(v)),samples=len(v),
            source="actual solver normal-force magnitude, simulation only, not hardware measurement") for n,v in forces.items()},
        measured_boundaries=boundaries,endpoint=endpoint,geometry_minima_mm=geometry,
        positive_effective_contact_qualified=False,fixed_material_slip_qualified=False,
        native_success_qualified=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0,
        limitations=["Region membership and force magnitude do not prove opposed bilateral support.",
            "Missing left contact is a measured failure, not substituted by right contact.",
            "This short native interval cannot establish 5s Hold or fixed-anchor Slip.",
            "Source-native diagnostic qualification is not new native execution by this CPU replay."])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=run()
    with a.output.open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output),
        accepted_by_pad=r["summary"]["accepted_by_pad"],windows=r["windows"],teacher_admission=False)))
if __name__=="__main__":main()
