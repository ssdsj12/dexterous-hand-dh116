"""Replay one frozen native episode; separate tracking, rod motion and mapping."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scripts import prepare_dh116_effective_right_pinch as plan
from scripts import solve_dh116_pinch_effective_targets as target
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_pad_region import bind_geometry,classify_pair

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-effective-right-native-tracking-replay-v1"
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-effective-right-pinch-native/ad37bcb60d1b8a68b6c43b38202144abc0ce04cd80615648d85c21a379cf0205/attempt-ks37mrio/receipt.json")
RECEIPT_SHA="9525c46f257fb6d5d81f585f71b47c571dcff7a05019b583e3e93c7535ed0bd8"
SCOPE="effective_right_pinch_2_setup_232_guarded_region_diagnostic_only"

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def active_values(record):
    names=record.get("joint_names");q=np.asarray(record.get("q"),dtype=float)
    if (not isinstance(names,list) or len(set(names))!=len(names)
        or not set(ACTIVE)<=set(names) or q.shape!=(1,len(names))):
        raise ValueError("complete named active joint readback required")
    if not np.isfinite(q).all():raise ValueError("finite measured joints required")
    return np.array([q[0,names.index(n)] for n in ACTIVE])

def displacement_mm(actual,reference):
    a,b=np.asarray(actual,float),np.asarray(reference,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:raise ValueError("same explicit xyz shape required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("finite xyz required")
    return 1000*(a-b)

def load_source(*,expected_sha=RECEIPT_SHA):
    if expected_sha!=RECEIPT_SHA or digest(RECEIPT)!=expected_sha:raise ValueError("frozen receipt bytes required")
    r=json.loads(RECEIPT.read_text())
    if (r.get("native_qualification") is not True or r.get("qualification_scope")!=SCOPE
        or r.get("outer_exit_code")!=0 or not 0<r.get("duration_seconds",float("inf"))<=180):
        raise ValueError("exact bounded native diagnostic receipt, not Teacher qualification")
    path=Path(r["output"])/"physical.jsonl"
    if digest(path)!=r["artifact_sha256"]["physical.jsonl"]:raise ValueError("frozen native ledger required")
    rows=[]
    with path.open() as f:
        for line in f:
            if not line.endswith("\n"):raise ValueError("incomplete native JSONL")
            item=json.loads(line)
            if item.get("operation")=="physical":rows.append(decode_owned(item["payload"])["exchange"])
    contact_summary(rows)
    return dict(receipt=r,exchanges=rows,ledger_sha256=r["artifact_sha256"]["physical.jsonl"])

def contact_summary(rows):
    if len(rows)!=232:raise ValueError("complete bounded native episode required")
    raw_counts=Counter();accepted=Counter();reasons=Counter();ranges={};both=0;identity=None
    for step,x in enumerate(rows,3):
        g=x["closure_geometry"];clock=g["raw"]["identity"];d=g["pad_region_diagnostic"]
        keys=("run_uuid","env_id","episode_uid","reset_id","rod_uid")
        current={k:clock[k] for k in keys}
        if (x["physics_step"]!=step or clock["physics_step"]!=step or d["identity"]!=clock
            or identity is not None and current!=identity or x["policy_step"]!=(step-3)//8):
            raise ValueError("same episode and exact policy/physics identity required")
        if x["guard_failure"] is not None:raise ValueError("source contains a guard termination")
        identity=current;windows={}
        for p in d["pairs"]:
            name=p["mesh_name"];raw_counts[name]+=1;r=p["region"]
            ranges.setdefault(name,[]).append(r["ellipse_value"])
            reasons[r["first_rejection"] or "accepted"]+=1
            if r["accepted_by_original_patch"]:
                accepted[name]+=1;windows.setdefault(p["solver_substep"],set()).add(name)
        both+=sum({"right.finger14_link","right.finger23_link"}<=v for v in windows.values())
    return dict(identity=identity,physical_steps=list(range(3,235)),raw_pairs=dict(raw_counts),
        accepted_by_original_region=sum(accepted.values()),accepted_by_pad=dict(accepted),
        rejection_counts=dict(reasons),ellipse_ranges={k:[min(v),max(v)] for k,v in ranges.items()},
        both_right_region_windows=both,positive_effective_contact_qualified=False,
        fixed_material_slip_qualified=False,native_success_qualified=False,teacher_admission=False)

def compare_endpoint(x):
    raw=x["closure_geometry"]["raw"];clock=raw["identity"]
    if x["physics_step"]!=234 or clock["physics_step"]!=234:raise ValueError("complete actual terminal boundary required")
    h=raw["hands"]["right"];j=raw["joint_readback"]["hands"]["right"]
    g=target.surface.base.load_geometry(ROOT,side="right")
    grids,_=target.surface.base.grids_for(g,target.surface.base.checked_json(target.surface.base.SDF_DIR/"result.json",target.surface.base.SDF_SHA))
    atlas={n:target.surface._patch(g["source_colliders"][n]["vertices"],"right",n) for n in target.surface.PADS}
    expected=plan.expected_boundary(plan.load_pair(ROOT),232)
    commanded=target.surface.base.fk_poses(g,expected["root_poses_w"][1],expected["right_q_rad"])
    poses={n:target.surface.base.pose_matrix(np.r_[p[:3],p[6],p[3:6]])
           for n,p in zip(h["body_names"],h["link_poses_xyzw"][0])}
    root=h["link_poses_xyzw"][0,h["body_names"].index("base_link")]
    measured_fk=target.surface.base.fk_poses(g,np.r_[root[:3],root[6],root[3:6]],active_values(j))
    nominal=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)
    actual_nodes=np.asarray(raw["rod_nodes_m"],float)
    comparisons={}
    for n in target.surface.PADS:
        name="right."+n
        bound=bind_geometry(raw,clock,name,x["rod"]["mesh_bindings"][name])
        np.testing.assert_allclose(bound,poses[n],rtol=0,atol=1e-12)
        values={}
        for key,state,nodes in (("command_nominal",commanded,nominal),
            ("measured_fk_nominal",measured_fk,nominal),("measured_links_nominal",poses,nominal),
            ("measured_links_actual_rod",poses,actual_nodes)):
            values[key]=target.predict_region(grids[n],state[n],nodes,atlas[n])
        pairs=[p for p in x["rod"]["rod_mesh_contact_pairs"]["pairs"] if p["mesh_name"]==name]
        if not pairs:raise ValueError("no actual endpoint pair for required pad")
        latest=max(pairs,key=lambda p:p["substep"])
        predicted=values["measured_links_actual_rod"]
        if latest["node_index"]!=predicted["node_index"]:raise ValueError("solver/prediction node identity differs")
        values.update(solver_pair=latest,solver_region=classify_pair(latest,bound,atlas[n]),
            solver_point_prediction_error_mm=float(np.linalg.norm(displacement_mm(
                [latest["contact_point_world_m"]],[predicted["point_world_m"]])[0])),
            active_fk_vs_measured_link_position_error_mm=float(np.linalg.norm(displacement_mm(
                [measured_fk[n][:3,3]],[poses[n][:3,3]])[0])))
        comparisons[n]=values
    return dict(version=VERSION,identity=clock,pads=comparisons,
        right_root_translation_error_mm=displacement_mm([poses["base_link"][:3,3]],
            [commanded["base_link"][:3,3]])[0].tolist(),
        right_active_joint_names=list(ACTIVE),
        right_active_tracking_error_rad=(active_values(j)-expected["right_q_rad"]).tolist(),
        rod_displacement_max_abs_mm=np.max(abs(displacement_mm(actual_nodes,nominal)),axis=0).tolist(),
        inference_scope="same-boundary geometric decomposition; not a policy-learning causal experiment",
        actual_contact_source="frozen solver pairs, not synthetic predictor force",
        new_physics_steps=0,ppo_updates=0,teacher_admission=False)

def run():
    s=load_source()
    return dict(version=VERSION,receipt_sha256=RECEIPT_SHA,ledger_sha256=s["ledger_sha256"],
        native_duration_seconds=s["receipt"]["duration_seconds"],native_outer_exit_code=0,
        summary=contact_summary(s["exchanges"]),endpoint=compare_endpoint(s["exchanges"][-1]),
        this_command_is_cpu_replay=True,new_native_runs=0,new_physics_steps=0,new_ppo_updates=0,
        teacher_admission=False,rigid_proxy_used=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=run()
    with a.output.open("x") as f:json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"output":str(a.output),"sha256":digest(a.output),"raw_pairs":result["summary"]["raw_pairs"],
        "region_accepted":result["summary"]["accepted_by_original_region"],"teacher_admission":False}))
if __name__=="__main__":main()
