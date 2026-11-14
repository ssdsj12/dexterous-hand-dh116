#!/usr/bin/env python3
"""Adaptive continuous fixed-root Tripod geometry proof; no native execution."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,load_geometry,world_vertices,path_displacement_bound)
from scripts.prepare_dh116_pinch_closure_margin import fk_poses,sample_contact
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
from scripts.solve_dh116_tripod_joint_root_direction import REQUIRED,envelope_support
from scripts.audit_dh116_power_closure_constraints import support_plane_bound
from scripts.audit_dh116_tripod_vertical_sdf import audit_paths
from scripts.screen_dh116_tripod_proximal import TRIPOD_SHA
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-tripod-fixed-root-path-reference-v1"
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-hard-constraints-20261114-run01/result.json")
PARENT_SHA="a484586dd4e08cbdfedd740fba3df53b4bcf9d636b16a7162e624422665fb1df"

def candidate(parent):
    if (parent["independent_style"]!="tripod" or parent["side"]!="left"
            or parent["independent_seed_sha256"]!=TRIPOD_SHA):
        raise ValueError("independent original left Tripod endpoint required")
    endpoint=np.array(parent["active_q_rad"],dtype=float);initial=endpoint.copy()
    initial[1:4]-=.03;expand(initial);expand(endpoint)
    delta=endpoint-initial
    return dict(side="left",independent_style="tripod",independent_seed_sha256=TRIPOD_SHA,
                base_pose_wxyz=list(parent["base_pose_wxyz"]),active_joint_names=list(ACTIVE),
                initial_q_rad=initial.tolist(),endpoint_q_rad=endpoint.tolist(),
                active_delta_rad=delta.tolist(),active_velocity_rad_s=(delta/(160/480)).tolist(),
                quiet_steps=48,closure_steps=160,root_motion="none",
                native_ready=False,reset_verified=False)

def certify_path(query,travel,thresholds,*,max_depth=12,max_samples=1025):
    if (set(travel)!=set(thresholds) or not travel or
            not all(np.isfinite(v) and v>=0 for v in travel.values())
            or not all(np.isfinite(v) for v in thresholds.values())
            or type(max_depth) is not int or not 0<=max_depth<=12
            or type(max_samples) is not int or not 3<=max_samples<=1025):
        raise ValueError("complete named finite global Lipschitz bounds and bounded subdivision required")
    result=dict(status="unresolved",proved=False,samples={},certified_intervals=[],
                unresolved_intervals=[],violations=[],minimum_clearance_margin_m=None,
                native_ready=False)
    def sample(t):
        if t not in result["samples"]:
            if len(result["samples"])>=max_samples:raise RuntimeError("sample_budget")
            values=query(t)
            if set(values)!=set(travel) or not all(np.isfinite(v) for v in values.values()):
                raise ValueError("shape identity or measured clearance changed")
            result["samples"][t]={k:float(v) for k,v in values.items()}
        return result["samples"][t]
    queue=[(0.,1.,0)]
    while queue:
        a,b,depth=queue.pop()
        try:rows=[sample(a),sample((a+b)/2),sample(b)]
        except RuntimeError as e:
            if str(e)!="sample_budget":raise
            result["unresolved_intervals"].extend([[a,b,depth],*queue]);break
        for t,row in zip((a,(a+b)/2,b),rows):
            for k,v in row.items():
                if v<thresholds[k]:
                    result["violations"].append(dict(fraction=t,shape=k,value_m=v,threshold_m=thresholds[k]))
        if result["violations"]:
            result["status"]="measured_violation";return result
        # Every point in [a,b] is at most (b-a)/4 from these three samples.
        lower={k:min(row[k] for row in rows)-travel[k]*(b-a)/4-thresholds[k] for k in travel}
        if min(lower.values())>=0:
            result["certified_intervals"].append(dict(start=a,end=b,depth=depth,margin_lower_m=lower))
        elif depth==max_depth:
            result["unresolved_intervals"].append([a,b,depth])
        else:
            middle=(a+b)/2
            queue.extend([(a,middle,depth+1),(middle,b,depth+1)])
    if not result["unresolved_intervals"]:
        result.update(status="proved_reference_path",proved=True,
            minimum_clearance_margin_m=min(min(r["margin_lower_m"].values()) for r in result["certified_intervals"]))
    return result

def shapes(g,pose,q):
    return {**{"source:"+k:v for k,v in world_vertices(g,pose,q).items()},
            **{"cooked:"+k:v for k,v in world_vertices(g,pose,q,cooked=True).items()}}

def run(root,result):
    prior=checked_json(PARENT,PARENT_SHA)
    if prior["status"]!="complete" or not prior["final_packet"]["measurement"]["endpoint_geometric_gate"]:
        raise ValueError("actual final near-contact endpoint with original geometric gates required")
    p=candidate(prior["final_packet"]["measurement"]["candidate"])
    g=load_geometry(root,side="left")
    if g["source_sha256"]!=prior["geometry_sources"]:raise ValueError("original side geometry changed")
    grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
    pose=p["base_pose_wxyz"];q0=np.array(p["initial_q_rad"]);delta=np.array(p["active_delta_rad"])
    layout=g["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
    registered=np.array(layout["wire_points"]);settled=registered.copy();settled[:,2]=top+.005
    initial=shapes(g,pose,q0);fk=fk_poses(g,pose,q0)
    result.update(parent_result_sha256=PARENT_SHA,candidate=p,geometry_sources=g["source_sha256"],
                  sdf_bindings=bindings)
    quiet=dict(table_m={k:float(v[:,2].min()-top) for k,v in initial.items()},
               envelope_lower_m={k:envelope_support(v,layout,.005)["surface_gap_lower_bound_m"] for k,v in initial.items()},
               node_sdf=audit_paths(grids,{k:fk[k] for k in grids},registered,settled,.005))
    quiet["reference_passed"]=bool(min(quiet["table_m"].values())>=CLEARANCE_M
        and min(quiet["envelope_lower_m"].values())>=0 and quiet["node_sdf"]["reference_no_contact"])
    result["quiet_start"]=quiet
    travel={**{"source:"+k:v for k,v in path_displacement_bound(g,q0,delta).items()},
            **{"cooked:"+k:v for k,v in path_displacement_bound(g,q0,delta,cooked=True).items()}}
    bound={**{"table:"+k:v for k,v in travel.items()},
           **{"rod:"+k:v for k,v in travel.items() if k.split(":")[1] not in REQUIRED}}
    thresholds={k:CLEARANCE_M if k.startswith("table:") else 0. for k in bound}
    def query(t):
        current=shapes(g,pose,q0+t*delta)
        table={"table:"+k:float(v[:,2].min()-top) for k,v in current.items()}
        rod={"rod:"+k:support_plane_bound(v,settled[[0,-1]],.005)["surface_gap_lower_bound_m"]-2e-6
             for k,v in current.items() if k.split(":")[1] not in REQUIRED}
        return {**table,**rod}
    result["fixed_root_path"]=certify_path(query,bound,thresholds)
    result["path_travel_bound_m"]=bound
    fke=fk_poses(g,pose,q0+delta)
    result["endpoint_float32_sdf"]={k:sample_contact(grids[k],fke[k],settled) for k in REQUIRED}
    result["reference_initial_and_path_passed"]=bool(quiet["reference_passed"] and result["fixed_root_path"]["proved"])
    result["source_sha256"]={n:digest(root/n) for n in (
        "scripts/audit_dh116_tripod_fixed_root_path.py","scripts/audit_dh116_pinch_initial_pose.py",
        "scripts/audit_dh116_tripod_vertical_sdf.py","scripts/prepare_dh116_pinch_closure_margin.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py")}
    result["status"]="complete"

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=55)
    a=p.parse_args(argv)
    if not np.isfinite(a.wall_seconds) or not 1<=a.wall_seconds<=55:p.error("CPU wall1..55s")
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",native_ready=False,teacher_admitted=False,
        reset_verified=False,new_physics_steps=0,ppo_updates=0,
        scope="fixed_root_linear_q_against_straight_settled_rod_reference_only")
    def expire(*unused):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,a.wall_seconds)
    try:run(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,KeyError) as e:result.update(status="error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","reference_initial_and_path_passed","error")}))
    return int(result["status"]=="error")

if __name__=="__main__":raise SystemExit(main())
