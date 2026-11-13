#!/usr/bin/env python3
"""Complete root+joint Tripod path bounds from the verified independent start."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_tripod_fixed_root_path import (
    PARENT,PARENT_SHA,certify_path,shapes)
from scripts.audit_dh116_pinch_initial_pose import checked_json,digest,load_geometry,path_displacement_bound
from scripts.prepare_dh116_pinch_closure_margin import fk_poses
from scripts.audit_dh116_tripod_vertical_sdf import audit_paths
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
from scripts.solve_dh116_tripod_joint_root_direction import REQUIRED,envelope_support
from scripts.audit_dh116_power_closure_constraints import support_plane_bound
from scripts.screen_dh116_right_pinch_closure import BudgetExpired
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.linear_coupling import expand

VERSION="dh116-tripod-complete-pose-path-reference-v1"

def rotations(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if (a.shape!=(7,) or b.shape!=(7,) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or abs(np.linalg.norm(a[3:])-1)>1e-10 or abs(np.linalg.norm(b[3:])-1)>1e-10):
        raise ValueError("finite unit-quaternion WXYZ poses required")
    r0=Rotation.from_quat(a[[4,5,6,3]]);r1=Rotation.from_quat(b[[4,5,6,3]])
    return a,b,r0,(r1*r0.inv()).as_rotvec()

def pose_at(a,b,fraction):
    if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError("path fraction0..1 required")
    a,b,r0,omega=rotations(a,b)
    r=Rotation.from_rotvec(omega*fraction)*r0
    return np.r_[a[:3]+fraction*(b[:3]-a[:3]),np.roll(r.as_quat(),1)]

def combine_bounds(joint_bounds,initial_world_vertices,origin,translation,angle):
    origin,translation=np.asarray(origin),np.asarray(translation)
    if (not joint_bounds or set(joint_bounds)!=set(initial_world_vertices)
            or origin.shape!=(3,) or translation.shape!=(3,)
            or not np.isfinite(origin).all() or not np.isfinite(translation).all()
            or not np.isfinite(angle) or angle<0):
        raise ValueError("complete finite point-motion sources required")
    out={}
    for k,j in joint_bounds.items():
        v=np.asarray(initial_world_vertices[k])
        if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all() or not np.isfinite(j) or j<0:
            raise ValueError("finite source vertices and nonnegative joint bound required")
        radius=float(np.max(np.linalg.norm(v-origin,axis=1)))
        out[k]=float(np.linalg.norm(translation)+angle*(radius+j)+j)
    return out

def run(root,result):
    prior=checked_json(PARENT,PARENT_SHA)
    if prior["status"]!="complete" or not prior["final_packet"]["measurement"]["endpoint_geometric_gate"]:
        raise ValueError("passing near-contact endpoint required")
    start=prior["parent_candidate"];end=prior["final_packet"]["measurement"]["candidate"]
    if (start["independent_style"]!="tripod" or start["side"]!="left"
            or end["independent_seed_sha256"]!=start["independent_seed_sha256"]):
        raise ValueError("same independent Tripod start and endpoint required")
    p0,p1=start["base_pose_wxyz"],end["base_pose_wxyz"]
    q0=np.array(start["initial_q_rad"]);q1=np.array(end["active_q_rad"]);delta=q1-q0
    expand(q0);expand(q1)
    g=load_geometry(root,side="left")
    if g["source_sha256"]!=prior["geometry_sources"]:raise ValueError("original side geometry changed")
    grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
    layout=g["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
    registered=np.array(layout["wire_points"]);settled=registered.copy();settled[:,2]=top+.005
    initial=shapes(g,p0,q0);fk=fk_poses(g,p0,q0)
    quiet=dict(table_m={k:float(v[:,2].min()-top) for k,v in initial.items()},
               envelope_lower_m={k:envelope_support(v,layout,.005)["surface_gap_lower_bound_m"] for k,v in initial.items()},
               node_sdf=audit_paths(grids,{k:fk[k] for k in grids},registered,settled,.005))
    quiet["reference_passed"]=bool(min(quiet["table_m"].values())>=CLEARANCE_M
        and min(quiet["envelope_lower_m"].values())>=0 and quiet["node_sdf"]["reference_no_contact"])
    joint={**{"source:"+k:v for k,v in path_displacement_bound(g,q0,delta).items()},
           **{"cooked:"+k:v for k,v in path_displacement_bound(g,q0,delta,cooked=True).items()}}
    a,b,_,omega=rotations(p0,p1)
    total=combine_bounds(joint,initial,a[:3],b[:3]-a[:3],float(np.linalg.norm(omega)))
    bound={**{"table:"+k:v for k,v in total.items()},
           **{"rod:"+k:v for k,v in total.items() if k.split(":")[1] not in REQUIRED}}
    thresholds={k:CLEARANCE_M if k.startswith("table:") else 0. for k in bound}
    result.update(parent_result_sha256=PARENT_SHA,quiet_start=quiet,start_candidate=start,
                  end_candidate=end,joint_delta_rad=delta.tolist(),root_translation_m=(b[:3]-a[:3]).tolist(),
                  world_rotation_vector_rad=omega.tolist(),joint_only_point_bound_m=joint,
                  complete_point_motion_bound_m=total,geometry_sources=g["source_sha256"],sdf_bindings=bindings)
    def query(t):
        current=shapes(g,pose_at(p0,p1,t),q0+t*delta)
        return {**{"table:"+k:float(v[:,2].min()-top) for k,v in current.items()},
                **{"rod:"+k:support_plane_bound(v,settled[[0,-1]],.005)["surface_gap_lower_bound_m"]-2e-6
                   for k,v in current.items() if k.split(":")[1] not in REQUIRED}}
    result["pose_path"]=certify_path(query,bound,thresholds)
    result["reference_initial_and_path_passed"]=bool(quiet["reference_passed"] and result["pose_path"]["proved"])
    result["source_sha256"]={n:digest(root/n) for n in (
        "scripts/audit_dh116_tripod_pose_path.py","scripts/audit_dh116_tripod_fixed_root_path.py",
        "scripts/audit_dh116_pinch_initial_pose.py","scripts/audit_dh116_tripod_vertical_sdf.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py")}
    result["status"]="complete"

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=55);a=p.parse_args(argv)
    if not np.isfinite(a.wall_seconds) or not 1<=a.wall_seconds<=55:p.error("CPU wall1..55seconds")
    a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",new_physics_steps=0,ppo_updates=0,
        native_ready=False,teacher_admitted=False,reset_verified=False,
        sdk_action_path_validated=False,scope="geometric_reference_not_actual_control_or_free_rod_motion")
    def expire(*unused):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,a.wall_seconds)
    try:run(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,KeyError) as e:result.update(status="error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    result["elapsed_s"]=time.monotonic()-begin
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","reference_initial_and_path_passed","error")}))
    return int(result["status"]=="error")

if __name__=="__main__":raise SystemExit(main())
