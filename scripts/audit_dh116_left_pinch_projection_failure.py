"""Audit one consumed failed native prefix; never promote it to native success."""
from collections import Counter
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np
from scripts.audit_dh116_tracking_pinch_contact import read_locked,contiguous_windows
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import SOURCE

VERSION="dh116-left-pinch-safe-projection-failure-audit-v1"
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-left-pinch-approach-native/1fb6e01d1b1d2e4b957a6997dbefdca98aef7cd88dd9aee8879afc792fe8165f/attempt-7o_2gq9g/receipt.json")
RECEIPT_SHA="25a75775ec43e03f6c722ae395df8584ac149e38c87e18ac77aba52f6d8d745d"
SCOPE="left_pinch_raised_approach_2_setup_232_original_guard_diagnostic_only"
LOCKS={
"physical.jsonl":"1896cdccf9fd8975219d24993661e9e3b3db3442bd0a1fd9639b6362328135f1",
"result.json":"d6630f4a0348c93ee893ee83fe2c1229ef054439d1d357510f49c050bfd913a0",
"ownership.jsonl":"04d551e0d8fb096e1ae7842ea8a9a2581fa92a59b8492cfb3ad83a31bdb87463",
"app_close_returned.json":"37b1ba15219bb2264687d2883ed6129cc5cd54fe64d0d7afbb1e5ca7d9308ae4",
"simulation_clear_receipt.json":"ef29ddc105acce0fa738d25cc5e4c4beb39adb8bd95d21f74a7e3da23c5ba572",
"sdk_clock.jsonl":"42cbcc17df572aa90f397f57e7d2c7e695d3f1016477743c8088304aa3ab76eb",
"gdb_finalize.bt.txt":"9033b55bda10c200c3489c8d32d9c665bddd31a4165db327233bdc8e9c8a8be8"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse_jsonl(raw):
    if not isinstance(raw,bytes) or not raw.endswith(b"\n"):raise ValueError("complete JSONL required")
    return [decode_owned(json.loads(line)) for line in raw.splitlines()]

def validate_prefix(entries):
    if (not entries or entries[0].get("physics_budget_end")!=234
        or entries[-1].get("operation")!="end"):
        raise ValueError("original bounded header and final end marker required")
    rows=[];ident=None;terminated=False;block=None;count=0;control=0
    for item in entries[1:]:
        op=item.get("operation")
        if terminated and op!="end":raise ValueError("terminal cannot be overwritten")
        if op=="begin":
            if block is not None:raise ValueError("nested policy boundary")
            block=item["payload"];count=0
            continue
        if op=="end":
            value=item["payload"]
            if (block is None or value["control_step"]!=control or value["physical_samples"]!=count
                or not rows or value["termination_reason"]!=rows[-1]["guard_failure"]
                or (not terminated and count!=8)):
                raise ValueError("matching actual policy end and physical count required")
            block=None;control+=1
            continue
        if op!="physical":
            if rows or block is not None or op not in ("setup","reset","initial_geometry"):
                raise ValueError("initial-only metadata or complete physical rows required")
            continue
        if block is None:raise ValueError("physical row requires owning policy begin")
        x=item["payload"]["exchange"];tick=3+len(rows)
        clock=x["action_readback"]["post"]["identity"]
        if len(clock)!=1:raise ValueError("one physical identity")
        clock=clock[0];pre=x["action_readback"]["pre"]["identity"][0]
        current=tuple(clock[k] for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid"))
        if (terminated or tick>234 or x["physics_step"]!=tick or clock["physics_step"]!=tick
            or pre["physics_step"]!=tick-1 or x["policy_step"]!=(tick-3)//8
            or clock["policy_step"]!=x["policy_step"] or clock["substep"]!=(tick-3)%8
            or abs(clock["timestamp_s"]-tick/480)>1e-9 or clock["clock_valid"] is not True
            or any(pre[k]!=clock[k] for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid"))
            or ident is not None and current!=ident):
            raise ValueError("contiguous same-episode actual 480Hz prefix required")
        g=x.get("closure_geometry")
        if g is not None and g["raw"]["identity"]!=clock:raise ValueError("same geometry clock")
        if not np.array_equal(block["action"],x["action_readback"]["request"]["action24"]):
            raise ValueError("same actual issued policy action required")
        ident=current;terminated=x["guard_failure"] is not None;rows.append(x);count+=1
    if block is not None or not rows or not terminated or entries[-1]["payload"]["termination_reason"]!=rows[-1]["guard_failure"]:
        raise ValueError("one retained final terminal and matching end marker required")
    return rows

def array(value,shape):
    a=np.asarray(value)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError("finite owned source shape")
    return a

def projection_decomposition(p,*,requested,submitted):
    before=array(p["pre_projection_palm_offsets_m"],(1,2,3))
    after=array(p["palm_offsets_m"],(1,2,3));safe=array(p["safe_z_offset_m"],(1,2))
    req=array(requested,(1,2,3))
    flags=array(p["safe_z_clipped"],(1,2));reasons=array(p["projection_reasons"],(1,2))
    roots=array(p["projected_root_poses"],(1,2,7))
    issued=array(submitted,(1,2,7))
    expected=before.copy();expected[:,:,2]=np.maximum(before[:,:,2],safe)
    clipped=abs(expected[:,:,2]-before[:,:,2])>1e-12
    if (p["projection_source"]!=SOURCE or not np.allclose(before,req,atol=1e-12,rtol=0)
        or not np.allclose(after,expected,atol=1e-12,rtol=0)
        or not np.array_equal(flags,clipped) or not np.array_equal(reasons,2*clipped.astype(int))
        or not np.array_equal(roots,issued)):
        raise ValueError("unmodified safe-z arithmetic, flags, reason and SDK submission required")
    return dict(classification="original_safe_z_projection" if clipped.any() else "no_projection",
        delta_mm=((after-before)[0]*1000).tolist(),
        requested_offsets_m=req[0].tolist(),safe_z_offsets_m=safe[0].tolist(),
        applied_offsets_m=after[0].tolist(),table_penetration_proven=False,
        independent_clearance_recomputed=False,
        scope="recorded safe-z equation plus submission parity; not independent pre-input reconstruction")

def lifecycle(receipt,app_closed,views_dead,context_cleared):
    zero=(receipt.get("outer_exit_code")==0 and receipt.get("debugger_inferior_exit_code")==0
        and receipt.get("debugger_inferior_signal") is None and receipt.get("debugger_status_verified") is True)
    return dict(app_close_returned=app_closed,native_views_dead_after_stop=views_dead,
        simulation_context_clear_returned=context_cleared,outer_exit_code=receipt.get("outer_exit_code"),
        inferior_exit_code=receipt.get("debugger_inferior_exit_code"),signal=receipt.get("debugger_inferior_signal"),
        natural_exit_zero=zero,native_qualified=False)

def region_windows(slots,dt):return contiguous_windows(slots,dt)

def load_frozen():
    receipt=json.loads(read_locked(RECEIPT,RECEIPT_SHA))
    if (receipt.get("native_execution_performed") is not True or receipt.get("native_qualification") is not False
        or receipt.get("qualification_scope")!=SCOPE or receipt.get("outer_exit_code")!=1
        or receipt.get("debugger_inferior_signal") is not None or not 0<receipt["duration_seconds"]<=180):
        raise ValueError("exact consumed failed-native receipt required")
    output=Path(receipt["output"]);raw={n:read_locked(output/n,s) for n,s in LOCKS.items()}
    return dict(receipt=receipt,physical_bytes=raw["physical.jsonl"],artifacts=raw)

def run():
    loaded=load_frozen();rows=validate_prefix(parse_jsonl(loaded["physical_bytes"]))
    result=json.loads(loaded["artifacts"]["result.json"]);last=rows[-1];prefix=rows[:-1]
    if result["physics_steps"]!=last["physics_step"] or result["ppo_updates"]!=0:raise ValueError("actual SDK budget")
    rawcounts=Counter();accepted=Counter();rejects=Counter();missing=[];boundaries=[];first=None
    slots={s:[] for s in ("left","right","bilateral")}
    required={s:{s+"."+n for n in ("finger14_link","finger23_link")} for s in ("left","right")}
    minima={s:{} for s in ("left","right")}
    for x in rows:
        identity=x["action_readback"]["post"]["identity"][0]
        p=x["wrist_projection"];a=x["action_readback"]
        check=projection_decomposition(p,requested=a["request"]["requested_palm_offsets_m"],
            submitted=a["submission"]["projected_root_poses"])
        sdkerror={}
        for side in ("left","right"):
            t=a["submission"]["expected_backend_targets"][side];q=a["post"]["hands"][side]
            actual=np.asarray(q["backend_targets"])[:,t["joint_indices"]]
            expected=np.asarray(t["values"])
            if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError("finite backend targets")
            sdkerror[side]=float(np.max(abs(actual-expected)))
        row=dict(identity=identity,safe_projection=check,backend_submission_error=sdkerror,
            post_clearance_m=x["post_step_clearance"]["clearance_m"],
            original_guard_failure=x["original_guard_failure"],guard_failure=x["guard_failure"],
            geometry_error=x.get("geometry_capture_error"))
        boundaries.append(row)
        if check["classification"]!="no_projection" and first is None:first=row
        cap=x["rod"]["rod_mesh_contact_pairs"];dt=cap["substep_dt_s"]
        if not np.isclose(dt,1/5760,atol=1e-12,rtol=0):raise ValueError("actual solver clock")
        g=x.get("closure_geometry")
        for pair in cap["pairs"]:
            if pair["mesh_name"]!="table":rawcounts[pair["mesh_name"]]+=1
        if g is None:
            missing.append(dict(identity=identity,reason=x.get("geometry_capture_error"),
                unclassified_hand_pairs=sum(p["mesh_name"]!="table" for p in cap["pairs"])))
            continue
        d=g["pad_region_diagnostic"]
        if d["identity"]!=identity:raise ValueError("actual region identity")
        windows={}
        for pair in d["pairs"]:
            name=pair["mesh_name"];region=pair["region"];sub=pair["solver_substep"]
            if type(sub) is not int or not 0<=sub<12:raise ValueError("actual solver substep")
            rejects[region["first_rejection"] or "accepted"]+=1
            if region["accepted_by_original_patch"]:
                accepted[name]+=1;windows.setdefault(sub,set()).add(name)
        for sub,names in sorted(windows.items()):
            slot=(x["physics_step"]-3)*12+sub
            for side in ("left","right"):
                if required[side]<=names:slots[side].append(slot)
            if required["left"]|required["right"]<=names:slots["bilateral"].append(slot)
        for side in ("left","right"):
            for name,value in g["sides"][side]["table_clearance_m"].items():
                if name not in minima[side] or value<minima[side][name]["clearance_m"]:
                    minima[side][name]=dict(clearance_m=value,identity=identity)
    owners=parse_jsonl(loaded["artifacts"]["ownership.jsonl"])
    stop=[r for r in owners if r["phase"]=="after_stop"]
    views=bool(stop) and all(v is False for k,v in stop[-1]["python_liveness"].items() if k!="simulation")
    close=json.loads(loaded["artifacts"]["app_close_returned.json"])["app_close_returned"]
    clear=json.loads(loaded["artifacts"]["simulation_clear_receipt.json"])["original_clear_instance_returned"]
    return dict(version=VERSION,source=dict(receipt=str(RECEIPT),receipt_sha256=RECEIPT_SHA,
        artifacts=LOCKS),physics_steps=last["physics_step"],measurement_rows=len(rows),
        completed_full_budget=False,last_geometry_step=prefix[-1]["physics_step"],
        terminal_geometry_valid=last["closure_geometry"] is not None,first_projection=first,
        failure_reason=last["guard_failure"],original_guard_failure=last["original_guard_failure"],
        geometry_failure=last["geometry_capture_error"],boundaries=boundaries,
        raw_pairs_all_measured_rows=dict(rawcounts),region_accepted_prefix_only=dict(accepted),
        region_rejections_prefix_only=dict(rejects),unknown_region_rows=missing,
        windows_prefix_only={s:region_windows(v,1/5760) for s,v in slots.items()},
        table_minima_m=minima,lifecycle=lifecycle(loaded["receipt"],close,views,clear),
        new_physics_steps=0,ppo_updates=0,teacher_admission=False,
        inference="Exact no-projection experimental contract failed when original safe-z activated. No table penetration or lifecycle crash is inferred.",
        next_requirement="Audit pre-action measured clearance and moving-wrist tracking before any new path.")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=run()
    with a.output.open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:r[k] for k in ("physics_steps","measurement_rows","first_projection",
        "region_accepted_prefix_only","unknown_region_rows","lifecycle")},default=lambda v:v.tolist()))
if __name__=="__main__":main()
