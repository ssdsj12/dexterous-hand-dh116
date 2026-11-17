#!/usr/bin/env python3
"""Prepare independent offline teacher acquisition proposals; never launches Isaac."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scripts.audit_dh116_pinch_initial_pose import (load_geometry,world_vertices,
    pose_matrix,box_relation,capsule_gap)
from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene,load_sdf_arrays
from go2_pvcnn.control.dh116.tabletop_teacher_acquisition_geometry import sample_nodes,required_reachable,freeze_decision,checked_geometry,translated_poses,source_radius,capsule_plane_bound,proposal_nodes

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError("preserve previous independent proposal")
    prepared=prepare_scene(a.root);grids=load_sdf_arrays(prepared)
    from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import supported_seed_plan
    original_plan,_=supported_seed_plan(a.root)
    auditpath=a.source_run/"initial_fk_parity_audit.json"
    audit=json.loads(auditpath.read_text());lo,hi=active_limits()
    report=dict(schema="teacher-acquisition-geometry-v1",status="candidateblocked",
        geometry_only=True,native_admission=False,native_runs=0,
        source_hashes={str(auditpath):digest(auditpath),
            str(a.source_run/"native_frames.jsonl"):digest(a.source_run/"native_frames.jsonl")},
        stage_source_sha256=prepared["local_sources"],locked_sdf_sha256=prepared["external_sources"],
        source_reference_fingerprint=prepared["fingerprint_sha256"],
        search_bounds=dict(root_translation_per_axis_m=.025,active_joint_delta_rad=.35),
        candidates={})
    scene_path=a.source_run/"supported_scene.json"
    supported=json.loads(scene_path.read_text())
    report["source_hashes"][str(scene_path)]=digest(scene_path)
    radius=source_radius(supported)
    report["actual_wire_radius_m"]=radius
    cfgpath=a.root/"configs/dh116_tasks/flexible_pinch_undamped_v1.json"
    cfgsha=digest(cfgpath)
    if cfgsha!="5843f88dc38aac8ed55706a9cfa03ff09302089be46fd1627b96e1c1ab7a1ca7":
        raise ValueError("actual source task SHA differs")
    if float(json.loads(cfgpath.read_text())["rod"]["radius_m"])!=radius:
        raise ValueError("actual scene and task radius differ")
    report["source_hashes"][str(cfgpath)]=cfgsha
    for rel in ("src/go2_pvcnn/control/dh116/tabletop_teacher_supported_runtime.py",
            "src/go2_pvcnn/control/dh116/tabletop_teacher_grasp_preview_runtime.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformx_end_supported_scene.py",
            "configs/dh116_tasks/flexible_teacher_end_supported_v1.json"):
        report["source_hashes"][str(a.root/rel)]=digest(a.root/rel)
    for style in ("pinch","tripod"):
        npz=a.source_run/(style+"_native.npz");report["source_hashes"][str(npz)]=digest(npz)
        with np.load(npz,allow_pickle=False) as data:nodes=data["f0_rod_nodes"].copy()
        # The native audit is SDK world; proposal is environment-local (3m lane spacing).
        envshift=np.array([0.,0. if style=="pinch" else 3.,0.])
        localnodes=nodes-envshift
        result=dict(selectedstyle=style,status="geometric_only",native_admission=False,
            sides={},before={},after={})
        roots=[];joints=[];required=[]
        for side in ("left","right"):
            source=audit[style][side];root=np.array(source["base_pose"]);q=np.array(source["q_seed_active"])
            geo=load_geometry(a.root,side=side)
            fk=HandPointFK(geo["usd"]["path"],side,root)
            names=["finger14_link","finger23_link"]+(["finger33_link"] if style=="tripod" else [])
            keys=[side+"."+n for n in names];required+=keys
            def poses(x):
                out=translated_poses(fk.poses(x[3:]),x[:3])
                return {k.rsplit("/",1)[-1]:v for k,v in out.items()}
            def rows(x):
                ts=poses(x)
                return {k:sample_nodes(g,nodes,ts[k.split(".")[1]],radius)
                    for k,g in grids.items() if k.startswith(side+".")}
            def objective(x):
                ts=poses(x);cost=0.
                for k in keys:
                    g=grids[k];t=ts[k.split(".")[1]]
                    p=(nodes-t[:3,3])@t[:3,:3]
                    c=(p-g["origin"])/g["spacing"]
                    outside=np.maximum(-c,0)+np.maximum(c-(np.array(g["values"].shape)-1),0)
                    rr=sample_nodes(g,nodes,t,radius)
                    gap=rr["nearest_sdf_gap_m"]
                    cost+=((gap-.0004)**2 if gap is not None else
                        .01**2+float(np.min(np.sum(outside**2,axis=1)))*g["spacing"]**2)
                    if gap is not None and gap<0:cost+=100*gap**2
                return cost*1e6+float(x[:3]@x[:3])*1e-2
            x0=np.r_[np.zeros(3),q];bounds=list(zip(np.full(3,-.025),np.full(3,.025)))+list(zip(np.maximum(lo,q-.35),np.minimum(hi,q+.35)))
            best=x0;history=[]
            for dz in (0.,-.008,.008):
                start=x0.copy();start[2]=dz
                opt=minimize(objective,start,bounds=bounds,method="Powell",
                    options=dict(maxiter=100,xtol=2e-6,ftol=1e-9,maxfev=12000))
                history.append(dict(success=bool(opt.success),evaluations=int(opt.nfev),objective=float(opt.fun)))
                if objective(opt.x)<objective(best):best=opt.x
                refined=minimize(objective,opt.x,bounds=bounds,method="SLSQP",
                    options=dict(maxiter=300,ftol=1e-10,eps=1e-6))
                history.append(dict(method="SLSQP",success=bool(refined.success),
                    evaluations=int(refined.nfev),objective=float(refined.fun)))
                if objective(refined.x)<objective(best):best=refined.x
            before=rows(x0);after=rows(best)
            result["before"].update(before);result["after"].update(after)
            newroot=root.copy();newroot[:3]+=best[:3];roots.append((newroot-np.r_[envshift,np.zeros(4)]).tolist());joints.append(expand(best[3:]).tolist())
            checks={}
            # SDK world table follows the environment lane shift.
            tablecenter=np.asarray(prepared["table"]["center_m"])+envshift
            for kind,cooked in (("source_convex",False),("locked_cooked_pad",True)):
                shapes=world_vertices(geo,newroot,best[3:],cooked=cooked)
                def rodcheck(v):
                    exact=checked_geometry(lambda:capsule_gap(v,nodes,radius))
                    if exact["intersects_capsule"] is None:
                        bound=capsule_plane_bound(v,nodes,radius)
                        bound["original_QP"]=exact
                        return bound
                    return exact
                checks[kind]={n:dict(table=box_relation(v,tablecenter,prepared["table"]["size_m"]),
                    rod=rodcheck(v),
                    supports={r["name"]:box_relation(v,np.asarray(r["center_m"])+envshift,r["size_m"]) for r in supported["supports"]}) for n,v in shapes.items()}
            sourcepass=all(not r["table"]["intersects"] and r["table"]["top_clearance_m"]>=.0019
                and all(not sr["intersects"] for sr in r["supports"].values())
                and (n in names or r["rod"]["intersects_capsule"] is False)
                for n,r in checks["source_convex"].items())
            result["sides"][side]=dict(root_translation_m=best[:3].tolist(),
                base_pose_before=root.tolist(),active_q_before=q.tolist(),active_q_after=best[3:].tolist(),search=history,
                geometry_checks=checks,source_table_rod_passed=sourcepass)
        reachable=required_reachable(result["after"],required)
        sourcepass=all(v["source_table_rod_passed"] for v in result["sides"].values())
        result.update(freeze_decision(reachable,sourcepass,False))
        result["blockers"].append("inter-hand and non-pad self collision clearance not certified")
        result["freeze_s0"]=False
        result["all_required_pads_reachable"]=reachable
        bound_nodes=proposal_nodes(original_plan["seeds_by_style"][style]["nodes"],localnodes)
        result["seeds"]=dict(nodes=bound_nodes["constructor_nodes"],joints=joints,roots=roots)
        result["observed_reference_nodes"]=bound_nodes["observed_reference_nodes"]
        result["constructor_nodes_sha256"]=hashlib.sha256(json.dumps(bound_nodes["constructor_nodes"],separators=(",",":")).encode()).hexdigest()
        result["geometry_scope"]="observed settled reference nodes with candidate FK; candidate reset requires fresh native settling"
        result["blockers"].append("candidate constructor straight nodes require fresh settling; observed curve is not equivalent reset")
        report["candidates"][style]=result
    report["implementation_sha256"]={str(p.relative_to(a.root)):digest(p) for p in (Path(__file__),a.root/"src/go2_pvcnn/control/dh116/tabletop_teacher_acquisition_geometry.py")}
    report["actual_reference_cohort_sha256"]=hashlib.sha256(json.dumps(report["source_hashes"],sort_keys=True).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(a.output),sha256=digest(a.output),
        candidates={s:dict(reachable=r["all_required_pads_reachable"],before={k:v["nearest_sdf_gap_m"] for k,v in r["before"].items()},
            after={k:v["nearest_sdf_gap_m"] for k,v in r["after"].items()},blockers=r["blockers"]) for s,r in report["candidates"].items()})))
if __name__=="__main__":main()
