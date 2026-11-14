"""Read-only audit of locked native pairs using source-derived substep poses.

All classification remains diagnostic/unknown, including negative evidence.
Never launches an app, advances a physical step or modifies the source ledger.
"""
from pathlib import Path
import argparse, hashlib, json
from collections import Counter, defaultdict
import numpy as np
from scripts.audit_dh116_projected_pinch_tracking import RECEIPT, RECEIPT_SHA, validate_boundary
from scripts.audit_dh116_projected_pinch_contact import read_locked, projection_summary, SCOPE, CODE, RECIPE
from scripts import solve_dh116_pinch_effective_targets as geometry
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import HAND_NAMES, MESH_NAMES
from go2_pvcnn.control.dh116.tabletop_deformable_pad_region import classify_pair, bind_geometry
from go2_pvcnn.control.dh116.tabletop_deformx_substep_pose import (
    VERSION, BACKEND_SHA, verify_source_lock, target_pose, pose_at_substep,
    to_matrix, classification_status,
)
ROOT=Path(__file__).resolve().parents[1]
BUILD=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu")

def named_poses(hands,field):
    out={}
    for key in HAND_NAMES:
        side,name=key.split(".");h=hands[side];names=h["body_names"]
        a=np.asarray(h[field])
        if len(set(names))!=len(names) or a.shape!=(1,len(names),7):
            raise ValueError("one complete named articulation pose batch required")
        paths=h["link_paths"]
        if (len(paths)!=1 or len(paths[0])!=len(names) or name not in names
            or paths[0][names.index(name)].split("/")[-1]!=name
            or side+"Hand" not in paths[0][names.index(name)].split("/")):
            raise ValueError("own hand/body/path identity required")
        out[key]=target_pose(a[0,names.index(name)])
    return out

def summarize_candidates(records):
    old=Counter();new=Counter();changed=Counter();reasons=Counter()
    for r in records:
        name=r["mesh_name"]
        old[name]+=int(r["endpoint_region"]["accepted_by_original_patch"])
        new[name]+=int(r["substep_region"]["accepted_by_original_patch"])
        changed[name]+=int(r["endpoint_region"]["accepted_by_original_patch"]!=
                           r["substep_region"]["accepted_by_original_patch"])
        reasons[r["substep_region"]["first_rejection"] or "cpu_candidate"]+=1
    return dict(endpoint_candidates=dict(old),source_substep_candidates=dict(new),
        changed_classifications=dict(changed),substep_first_rejections=dict(reasons),
        positive_effective_contact_qualified=False,negative_effective_contact_qualified=False,
        reset_mesh_pose_qualified=False,teacher_admission=False)

def run():
    source=verify_source_lock(BUILD/"source")
    binary=BUILD/"build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so"
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=BACKEND_SHA:
        raise ValueError("actual compiled backend SHA differs")
    r=json.loads(read_locked(RECEIPT,RECEIPT_SHA))
    if (r["qualification_scope"]!=SCOPE or r["command_spec_sha256"]!=RECIPE
        or r["code_config_sha256"]!=CODE or r["outer_exit_code"]!=0
        or r["debugger_inferior_exit_code"]!=0 or r["debugger_inferior_signal"] is not None):
        raise ValueError("frozen completed bounded source receipt required")
    raw=read_locked(Path(r["output"])/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"])
    if not raw.endswith(b"\n"):raise ValueError("incomplete JSONL")
    rows=[];reset=None
    for e in map(json.loads,raw.splitlines()):
        if e.get("operation")=="reset":
            if reset is not None or rows:raise ValueError("exact single preceding reset")
            reset=decode_owned(e["payload"])["reset"]
        elif e.get("operation")=="physical":
            rows.append(decode_owned(e["payload"])["exchange"])
    del raw
    projection_summary(rows)
    if reset is None or reset["status"]!="complete":raise ValueError("recorded Reset missing")
    start=named_poses(reset["before_hands"],"body_poses_xyzw")
    after=named_poses(reset["hands"],"body_poses_xyzw")
    displacement={k:float(np.linalg.norm((after[k][:3]-start[k][:3]).astype(float))*1000) for k in HAND_NAMES}
    for k in HAND_NAMES:
        np.testing.assert_allclose(after[k],target_pose(np.concatenate(reset["pad_poses_xyzw"][k])),atol=1e-7,rtol=0)
    atlas={}
    for side in ("left","right"):
        g=geometry.surface.base.load_geometry(ROOT,side=side)
        for name in geometry.surface.PADS:
            atlas[side+"."+name]=geometry.surface._patch(g["source_colliders"][name]["vertices"],side,name)
    result=[];rawcounts=Counter()
    for x in rows:
        clock=validate_boundary(x)
        cap=x["rod"]["rod_mesh_contact_pairs"]
        if tuple(x["rod"]["mesh_names"])!=MESH_NAMES:raise ValueError("locked mesh slots differ")
        end=named_poses(x["closure_geometry"]["raw"]["hands"],"link_poses_xyzw")
        # Bind the original body/SDF identity as well as checking the pose names.
        for key in HAND_NAMES:
            bind_geometry(x["closure_geometry"]["raw"],clock,key,x["rod"]["mesh_bindings"][key])
        transforms={k:[to_matrix(pose_at_substep(start[k],end[k],s)) for s in range(12)] for k in HAND_NAMES}
        legacy={(p["mesh_name"],p["solver_substep"],p["node_index"]):p["region"]
            for p in x["closure_geometry"]["pad_region_diagnostic"]["pairs"]}
        seen=set()
        for pair in cap["pairs"]:
            name=pair["mesh_name"]
            if name=="table":continue
            sub=pair["substep"];key=(name,sub,pair["node_index"])
            if type(sub) is not int or not 0<=sub<12 or key in seen:
                raise ValueError("unique measured solver-substep pair identity")
            seen.add(key);rawcounts[name]+=1
            old=classify_pair(pair,to_matrix(end[name]),atlas[name])
            if old["accepted_by_original_patch"]!=legacy[key]["accepted_by_original_patch"]:
                raise ValueError("endpoint reconstruction differs from retained classification")
            new=classify_pair(pair,transforms[name][sub],atlas[name])
            result.append(dict(identity=clock,mesh_name=name,solver_substep=sub,
                substep_time_s=(clock["physics_step"]-1+(sub+1)/12)/480,
                node_index=pair["node_index"],endpoint_region=old,substep_region=new,
                source_derived_mesh_pose_xyzw=pose_at_substep(start[name],end[name],sub).tolist(),
                **classification_status(new["accepted_by_original_patch"])))
        if seen!=set(legacy):raise ValueError("retained pair set differs")
        start=end
    return dict(version=VERSION,source_receipt=str(RECEIPT),source_receipt_sha256=RECEIPT_SHA,
        physical_sha256=r["artifact_sha256"]["physical.jsonl"],backend_sha256=BACKEND_SHA,
        source_lock=source,physical_rows=len(rows),raw_pair_counts=dict(rawcounts),
        constructor_to_reset_translation_mm=displacement,summary=summarize_candidates(result),pairs=result,
        first_start_source="pre-reset actual body readback; inferred constructor commit from locked code, not native current getter",
        later_start_source="prior target committed by end_step in locked code",
        timing_formula="substep 0..11 alpha=(sub+1)/12; world left quaternion increment",
        uncertainty="CPU float32 reconstruction; CUDA FMA/trigonometric equivalence unverified; all effective contact remains NA",
        depth_source="kernel gap not retained; no depth reconstructed from endpoint nodes/force",
        new_native_runs=0,new_physics_steps=0,ppo_updates=0,teacher_admission=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=run()
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("pairs","source_lock")}))
    print("output_sha256="+hashlib.sha256(a.output.read_bytes()).hexdigest())
if __name__=="__main__":main()
