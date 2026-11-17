#!/usr/bin/env python3
"""One bounded nonlinear Tripod endpoint study, using original nodes and SDFs."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,load_geometry,world_vertices,convex_patch_closest)
from scripts.audit_dh116_power_closure_constraints import support_plane_bound
from scripts.prepare_dh116_pinch_closure_margin import sample_grid,fk_poses
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
from scripts.solve_dh116_tripod_joint_root_direction import REQUIRED
from scripts.screen_dh116_tripod_proximal import TRIPOD_SHA
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,active_limits,expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-tripod-node-aware-nonlinear-endpoint-v1"
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-vertical-sdf-20261114-run01/result.json")
PARENT_SHA="d0040fc6f282687b16bbc1336f7cbfdabc54bac52fc6b696facdf720e6ec4ba1"
SCALE=np.array([.005,.005,.005,.1,.1,.1,.1,.1,.1,.1])
MAX_EVALUATIONS=500
STRICT=.00005

class EvaluationLimit(RuntimeError):
    pass

class Budget:
    def __init__(self,query,maximum=MAX_EVALUATIONS):
        if type(maximum) is not int or not 1<=maximum<=MAX_EVALUATIONS:
            raise ValueError("bounded positive evaluation budget required")
        self.query,self.maximum,self.count,self.rows=query,maximum,0,[]
    def __call__(self,x):
        if self.count>=self.maximum:raise EvaluationLimit("geometry evaluation cap")
        row=self.query(x)
        with _commit_row_without_alarm():
            self.count+=1
            self.rows.append(dict(evaluation=self.count,x=np.asarray(x).tolist(),measurement=row))
        return row

def bounds(parent):
    q=finite_array(parent["initial_q_rad"],(6,));expand(q)
    low,high=active_limits()
    lo,hi=-np.ones(10),np.ones(10)
    lo[7:]=0
    lo[6:]=np.maximum(lo[6:],(low[:4]-q[:4])/.1)
    hi[6:]=np.minimum(hi[6:],(high[:4]-q[:4])/.1)
    return lo,hi

def decode(parent,x):
    x=finite_array(x,(10,))
    if (parent["independent_style"]!="tripod" or parent["side"]!="left"
            or parent["independent_seed_sha256"]!=TRIPOD_SHA):
        raise ValueError("verified independent left Tripod start required")
    lo,hi=bounds(parent)
    if np.any(x<lo-1e-12) or np.any(x>hi+1e-12):
        raise ValueError("original joint limits and bounded CPU root/closure range required")
    pose=finite_array(parent["base_pose_wxyz"],(7,)).copy()
    pose[:3]+=x[:3]*SCALE[:3]
    rotation=Rotation.from_rotvec(x[3:6]*SCALE[3:6])*Rotation.from_quat(pose[[4,5,6,3]])
    pose[3:]=np.roll(rotation.as_quat(),1)
    q=finite_array(parent["initial_q_rad"],(6,)).copy();q[:4]+=x[6:]*SCALE[6:]
    expand(q)
    return dict(independent_style="tripod",side="left",independent_seed_sha256=TRIPOD_SHA,
        base_pose_wxyz=pose.tolist(),active_q_rad=q.tolist(),active_joint_names=list(ACTIVE),
        normalized_delta=x.tolist(),native_ready=False,reset_verified=False)

def jacobian(evaluate,x,lo,hi,*,h=.001):
    x,lo,hi=map(np.asarray,(x,lo,hi))
    base=np.asarray(evaluate(x)["residual"]);cols=[]
    for i in range(len(x)):
        plus=x.copy();minus=x.copy()
        plus[i]=min(hi[i],x[i]+h);minus[i]=max(lo[i],x[i]-h)
        if plus[i]==minus[i]:
            cols.append(np.zeros_like(base));continue
        p=base if plus[i]==x[i] else np.asarray(evaluate(plus)["residual"])
        m=base if minus[i]==x[i] else np.asarray(evaluate(minus)["residual"])
        cols.append((p-m)/(plus[i]-minus[i]))
    return np.stack(cols,axis=1)

def bounded_fit(evaluate,lo,hi):
    # Dogbox retains an exact lower-bound start. TRF's 1e-10 interior nudge
    # produced a vanishing initial trust region on this all-zero closure.
    return least_squares(lambda x:np.asarray(evaluate(x)["residual"]),np.zeros(len(lo)),
        jac=lambda x:jacobian(evaluate,x,lo,hi),bounds=(lo,hi),method="dogbox",
        max_nfev=22,ftol=1e-9,xtol=1e-9,gtol=1e-9)

def endpoint_gate(row):
    g=row["node_sdf_gap_m"]
    return bool(all(v is not None and np.isfinite(v) and abs(v)<=STRICT for v in g)
        and all(v>0 for v in row["node_gradient_norm"])
        and row["minimum_table_gap_m"]>=CLEARANCE_M
        and row["minimum_nonrequired_gap_lower_m"]>=0
        and np.all(np.abs(row["required_hull_intervals_m"])<=STRICT)
        and all(v is not None and v<-.2 for v in row["geometric_opposition"])
        and all(v is not None and v<-.2 for v in row["sdf_opposition"]))

def measure(g,grids,parent,x):
    candidate=decode(parent,x);pose=candidate["base_pose_wxyz"];q=candidate["active_q_rad"]
    layout=g["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
    nodes=np.array(layout["wire_points"]);nodes[:,2]=top+.005
    shapes={**{"source:"+k:v for k,v in world_vertices(g,pose,q).items()},
            **{"cooked:"+k:v for k,v in world_vertices(g,pose,q,cooked=True).items()}}
    fk=fk_poses(g,pose,q)
    tables={k:float(v[:,2].min()-top) for k,v in shapes.items()}
    nonrequired={k:support_plane_bound(v,nodes[[0,-1]],.005)["surface_gap_lower_bound_m"]-2e-6
                 for k,v in shapes.items() if k.split(":")[1] not in REQUIRED}
    pads={};normals=[];directions=[];residual=[];labels=[]
    def add(label,value):labels.append(label);residual.append(float(value))
    for name in REQUIRED:
        local=(nodes-fk[name][:3,3])@fk[name][:3,:3]
        phi,grad,valid=sample_grid(grids[name],local,dtype=np.float64)
        ids=np.flatnonzero(valid)
        node=int(ids[np.argmin(phi[ids])]) if len(ids) else None
        gap=None if node is None else float(phi[node]-.005)
        norm=0. if node is None else float(np.linalg.norm(grad[node]))
        normal=None if norm<=0 else fk[name][:3,:3]@grad[node]/norm
        qp=convex_patch_closest(shapes["cooked:"+name],nodes[0],[nodes[-1]-nodes[0]])
        lower=support_plane_bound(shapes["cooked:"+name],nodes[[0,-1]],.005)["surface_gap_lower_bound_m"]-2e-6
        upper=qp["distance_m"]-.005+2e-6
        pads[name]=dict(node_id=node,node_sdf_gap_m=gap,gradient_norm=norm,
                       sdf_normal_world=None if normal is None else normal.tolist(),
                       hull_interval_m=[lower,upper],geometric_direction=qp["geometric_axis_to_hull_direction"])
        normals.append(normal);directions.append(qp["geometric_axis_to_hull_direction"])
        # Unsupported geometry is an explicit penalty, never a measured zero.
        add(name+":node_target_minus25um",10. if gap is None else (gap+.000025)/.001)
        add(name+":strict_hull_lower",max(0.,-STRICT-lower)/.001)
        add(name+":strict_hull_upper",max(0.,upper-STRICT)/.001)
    dots={}
    for kind,vecs in (("geometric",directions),("sdf",normals)):
        dots[kind]=[None if vecs[0] is None or n is None else float(np.dot(vecs[0],n)) for n in vecs[1:]]
        for i,d in enumerate(dots[kind]):
            add(kind+":opposition"+str(i),10. if d is None else max(0.,d+.200001))
    for name,gap in tables.items():add(name+":table",max(0.,CLEARANCE_M-gap)/.001)
    for name,gap in nonrequired.items():add(name+":nonrequired",max(0.,-gap)/.001)
    row=dict(candidate=candidate,required_pads=pads,table_gap_by_shape_m=tables,
        nonrequired_gap_lower_m=nonrequired,minimum_table_gap_m=min(tables.values()),
        minimum_nonrequired_gap_lower_m=min(nonrequired.values()),
        node_sdf_gap_m=[pads[n]["node_sdf_gap_m"] for n in REQUIRED],
        node_gradient_norm=[pads[n]["gradient_norm"] for n in REQUIRED],
        required_hull_intervals_m=[pads[n]["hull_interval_m"] for n in REQUIRED],
        geometric_opposition=dots["geometric"],sdf_opposition=dots["sdf"],
        residual=residual,residual_labels=labels,objective=float(np.dot(residual,residual)),
        continuous_path_verified=False,self_collision_verified=False,native_ready=False,teacher_admitted=False)
    row["endpoint_geometric_gate"]=endpoint_gate(row)
    return row

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=55.)
    args=p.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=55:p.error("CPU wall 1..55 seconds")
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();budget=None
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,ppo_updates=0,
                native_ready=False,teacher_admitted=False,reset_verified=False,
                maximum_geometry_evaluations=MAX_EVALUATIONS,least_squares_max_nfev=22)
    def expire(*unused):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    try:
        root=Path(__file__).resolve().parents[1];prior=checked_json(PARENT,PARENT_SHA)
        if prior["status"]!="complete" or not prior["continuous_reference"]["reference_no_contact"]:
            raise ValueError("completed independent Tripod quiet reference required")
        g=load_geometry(root,side="left")
        if g["source_sha256"]!=prior["geometry_sources"]:raise ValueError("Tripod geometry changed")
        grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
        parent=dict(independent_style="tripod",side="left",independent_seed_sha256=TRIPOD_SHA,
                    base_pose_wxyz=prior["base_pose_wxyz"],initial_q_rad=prior["initial_q_rad"])
        result.update(parent_sha256=PARENT_SHA,parent_candidate=parent,sdf_bindings=bindings,
                      geometry_sources=g["source_sha256"],source_sha256={
                      n:digest(root/n) for n in ("scripts/solve_dh116_tripod_node_closure.py",
                      "scripts/prepare_dh116_pinch_closure_margin.py",
                      "scripts/audit_dh116_pinch_initial_pose.py",
                      "scripts/audit_dh116_power_closure_constraints.py",
                      "src/go2_pvcnn/control/dh116/linear_coupling.py")})
        lo,hi=bounds(parent)
        budget=Budget(lambda x:measure(g,grids,parent,x))
        result["solver_method"]="dogbox"
        fit=bounded_fit(budget,lo,hi)
        result.update(status="complete",solver_success=bool(fit.success),solver_message=fit.message,
                      solver_nfev=fit.nfev,solver_njev=fit.njev,solver_x=fit.x.tolist())
    except BudgetExpired:result["status"]="partial_timeout"
    except EvaluationLimit:result["status"]="partial_evaluation_cap"
    except (ValueError,KeyError) as e:result.update(status="error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    if budget:
        result.update(evaluations=budget.count,completed_rows=budget.rows)
        if budget.rows:
            result["best"]=min(budget.rows,key=lambda r:r["measurement"]["objective"])
            result["baseline"]=budget.rows[0]
            result["endpoint_pass_count"]=sum(r["measurement"]["endpoint_geometric_gate"] for r in budget.rows)
    result["elapsed_s"]=time.monotonic()-started
    with (args.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","evaluations","endpoint_pass_count")}))
    return 1 if result["status"]=="error" else 0

if __name__=="__main__":raise SystemExit(main())
