"""Read-only original-region and projection statistics from one qualified interval."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,math
import numpy as np
from scripts.audit_dh116_tracking_pinch_contact import read_locked,contiguous_windows
from scripts.audit_dh116_effective_right_tracking import contact_summary
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
VERSION="dh116-projected-pinch-native-statistics-v1"
SCOPE="pinch_original_safety_projection_2_setup_232_diagnostic_only"
CODE="d2b175b7654b4b30c67028a20a6b0e1a50eda98b3804414554c6d8c9fd3f7d58"
RECIPE="129e0b0e3455e276733554d15c1ccbd431772dc04572e168943ed3d327109254"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def projection_summary(rows):
    if len(rows)!=232:raise ValueError("complete 232 measured rows")
    first=None;deltas=[];clips=[];final=None
    for tick,x in enumerate(rows,3):
        c=x["action_readback"]["post"]["identity"]
        if len(c)!=1:raise ValueError("single actual physical identity")
        c=c[0];key=tuple(c[n] for n in ("run_uuid","env_id","episode_uid","reset_id","rod_uid"))
        if (x["physics_step"]!=tick or c["physics_step"]!=tick
            or x["policy_step"]!=(tick-3)//8 or c["policy_step"]!=x["policy_step"]
            or not math.isfinite(c["timestamp_s"]) or abs(c["timestamp_s"]-tick/480)>1e-9
            or x["guard_failure"] is not None or first is not None and key!=first):
            raise ValueError("contiguous same physical episode without terminal")
        first=key;p=x["wrist_projection"]
        before=np.asarray(p["pre_projection_palm_offsets_m"]);after=np.asarray(p["palm_offsets_m"])
        flags=np.asarray(p["safe_z_clipped"])
        if (before.shape!=(1,2,3) or after.shape!=before.shape or flags.shape!=(1,2)
            or flags.dtype!=np.bool_ or not np.isfinite(before).all() or not np.isfinite(after).all()):
            raise ValueError("complete finite source projections")
        d=(after-before)[0]*1000;deltas.append(d)
        if flags.any():clips.append(dict(identity=c,sides=flags[0].tolist()))
        final=dict(identity=c,applied_offsets_mm=(after[0]*1000).tolist())
    return dict(clip_rows=len(clips),first_clip=clips[0] if clips else None,
        maximum_step_projection_mm=np.max(deltas,axis=0).tolist(),final=final,
        teacher_admission=False,native_success_qualified=False)

def summarize(rows):
    contact=contact_summary(rows);projection=projection_summary(rows)
    slots={s:[] for s in ("left","right","bilateral")}
    required={s:{s+"."+n for n in ("finger14_link","finger23_link")} for s in ("left","right")}
    minima={s:{} for s in ("left","right")}
    for x in rows:
        g=x["closure_geometry"];identity=g["raw"]["identity"];windows=defaultdict(set)
        cap=x["rod"]["rod_mesh_contact_pairs"]
        if not math.isclose(cap["substep_dt_s"],1/(480*12),rel_tol=0,abs_tol=1e-12):
            raise ValueError("original measured solver substep dt")
        for p in g["pad_region_diagnostic"]["pairs"]:
            if p["region"]["accepted_by_original_patch"]:windows[p["solver_substep"]].add(p["mesh_name"])
        for sub,names in sorted(windows.items()):
            if type(sub) is not int or not 0<=sub<12:raise ValueError("actual solver substep")
            slot=(x["physics_step"]-3)*12+sub
            for s in ("left","right"):
                if required[s]<=names:slots[s].append(slot)
            if required["left"]|required["right"]<=names:slots["bilateral"].append(slot)
        for s in minima:
            for name,v in g["sides"][s]["table_clearance_m"].items():
                if name.startswith("cooked:") and (name not in minima[s] or v*1000<minima[s][name]["mm"]):
                    minima[s][name]=dict(mm=v*1000,identity=identity)
    return dict(contact=contact,projection=projection,
        original_region_windows={s:contiguous_windows(v,1/(480*12)) for s,v in slots.items()},
        cooked_table_minima=minima,teacher_admission=False,native_success_qualified=False,
        effective_opposed_support_qualified=False,fixed_material_slip_qualified=False)

def run(receipt,receipt_sha):
    r=json.loads(read_locked(receipt,receipt_sha))
    if (r.get("qualification_scope")!=SCOPE or r.get("command_spec_sha256")!=RECIPE
        or r.get("code_config_sha256")!=CODE or r.get("native_qualification") is not True
        or r.get("native_execution_performed") is not True or r.get("outer_exit_code")!=0
        or r.get("debugger_inferior_exit_code")!=0 or r.get("debugger_inferior_signal") is not None
        or not 0<r.get("duration_seconds",float("inf"))<=180):
        raise ValueError("exact independently qualified bounded interval required")
    a=json.loads(read_locked(Path(receipt).parent/"projected-pinch-post-exit-audit.json",
        r["lifetime_post_exit_audit_sha256"]))["evidence"]
    if (not a.get("lifecycle_qualified") or not a.get("full_post_exit_audit_passed")
        or a.get("qualification_scope")!=SCOPE):raise ValueError("independent source/continuity/lifecycle audit")
    out=Path(r["output"]);raw=read_locked(out/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"])
    if not raw.endswith(b"\n"):raise ValueError("complete JSONL")
    rows=[decode_owned(e["payload"])["exchange"] for e in map(json.loads,raw.splitlines()) if e.get("operation")=="physical"]
    return dict(version=VERSION,source=dict(receipt=str(receipt),receipt_sha256=receipt_sha,
        physical_sha256=r["artifact_sha256"]["physical.jsonl"],
        post_exit_audit_sha256=r["lifetime_post_exit_audit_sha256"],outer_exit_code=0,
        native_duration_seconds=r["duration_seconds"],qualification_scope=SCOPE),
        **summarize(rows),new_physics_steps=0,ppo_updates=0,
        limitations=["Original patch membership is not opposed bilateral support.",
            "Solver contacts and force are simulation measurements, not real motor sensors.",
            "This short interval cannot establish 5s Hold, 30mm Lift or fixed-anchor Slip."])

def main():
    p=argparse.ArgumentParser();p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--receipt-sha256",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=run(a.receipt,a.receipt_sha256)
    with a.output.open("x") as f:json.dump(r,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output),contact=r["contact"],
        projection=r["projection"],windows=r["original_region_windows"])))
if __name__=="__main__":main()
