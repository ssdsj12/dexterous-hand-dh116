#!/usr/bin/env python3
"""Legal opening and continuous-path audit for the frozen right static endpoint."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from itertools import product
from scripts import solve_dh116_right_pinch_surface_registration as surface

VERSION="dh116-right-pinch-owned-approach-v1"
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-joint-root-20261113-VJTnWwEu/run01/result.json")
PARENT_SHA="c53e0085a3e20aa0742460f3e0d51fbd08a5b14e4ee80554dfd4cf8f62d79ed6"


def minimum_norm_bounded(jacobian,desired,lower,upper):
    j=surface._array(jacobian,(2,3));b=surface._array(desired,(2,))
    lo=surface._array(lower,(3,));hi=surface._array(upper,(3,))
    if np.any(lo>hi):raise ValueError("bounded interval invalid")
    choices=[]
    # Exhaust all 27 active-bound patterns. Each free subproblem is the
    # minimum Euclidean norm exact solution, not an arbitrary TRF interior.
    for flags in product((-1,0,1),repeat=3):
        fixed=np.array(flags)!=0;free=~fixed
        d=np.zeros(3)
        for i,f in enumerate(flags):
            if f:d[i]=lo[i] if f<0 else hi[i]
        remaining=b-j[:,fixed]@d[fixed]
        if free.any():d[free]=np.linalg.lstsq(j[:,free],remaining,rcond=None)[0]
        if (np.max(abs(j@d-b))<=1e-9 and np.all(d>=lo-1e-12) and np.all(d<=hi+1e-12)):
            choices.append(d)
    if not choices:raise ValueError("no bounded original-joint opening proposal")
    return min(choices,key=lambda d:float(d@d))


def opening_proposal(measure,q,lower,upper,target):
    q=surface._array(q,(3,));lower=surface._array(lower,(3,));upper=surface._array(upper,(3,))
    if np.any(q-.0001<lower) or np.any(q+.0001>upper):
        raise ValueError("original bound cannot support central derivative neighborhood")
    if target not in (.00015,.00025,.0004,.0006,.0008,.001):raise ValueError("frozen precontact target required")
    initial=surface._array(measure(q),(2,))
    matrices=[]
    for h in (.0001,.00005):
        columns=[]
        for j in range(3):
            d=np.zeros(3);d[j]=h
            columns.append((surface._array(measure(q+d),(2,))-surface._array(measure(q-d),(2,)))/(2*h))
        matrices.append(np.column_stack(columns))
    relative=float(np.linalg.norm(matrices[0]-matrices[1])/max(np.linalg.norm(matrices[1]),1e-15))
    if relative>.05 or np.linalg.matrix_rank(matrices[1],tol=1e-8)<2:
        raise ValueError("nonlocal or singular absolute contact derivative")
    delta=minimum_norm_bounded(matrices[1],np.full(2,target)-initial,
                               np.maximum(lower-q,-.1),np.minimum(upper-q,.1))
    return dict(initial_gap_m=initial,jacobians=matrices,derivative_relative_error=relative,
                delta_rad=delta,target_gap_m=target,linear_residual_m=matrices[1]@delta-(target-initial))


def grid_lipschitz(grid):
    values=np.asarray(grid["values"],float);spacing=float(grid["spacing"])
    if values.ndim!=3 or min(values.shape)<2 or not np.isfinite(values).all() or not np.isfinite(spacing) or spacing<=0:
        raise ValueError("finite actual trilinear grid required")
    slopes=[np.max(abs(np.diff(values,axis=axis)))/spacing for axis in range(3)]
    return float(np.linalg.norm(slopes))


def run(out,emit,*,raised_start=False):
    base=surface.base
    parent=base.decode_owned(base.checked_json(PARENT,PARENT_SHA))
    candidates=[c["selected"] for c in parent["candidates"] if c["selected"]["geometry"]["endpoint_passed"]]
    if len(candidates)!=1:raise ValueError("one frozen independently passing endpoint required")
    selected=candidates[0];root=base.pose_matrix(selected["root_wxyz"]);end=np.asarray(selected["q_rad"])
    g=base.load_geometry(base.ROOT,side="right");nodes=np.asarray(parent["reset_nodes_m"])
    if g["source_sha256"]!=parent["geometry_sources"]:raise ValueError("original source geometry changed")
    grids,bindings=base.grids_for(g,base.checked_json(base.SDF_DIR/"result.json",base.SDF_SHA))
    if bindings!=parent["sdf_bindings"]:raise ValueError("locked SDF binding changed")
    low,high=surface.active_limits();budget=surface.EvaluationBudget(48)
    lift=.0005 if raised_start else 0.
    start_root=root.copy();start_root[2,3]+=lift
    if raised_start:low=low+.001

    out.update(parent_sha256=PARENT_SHA,geometry_sources=g["source_sha256"],sdf_bindings=bindings,
               root_wxyz=selected["root_wxyz"],end_q_rad=end,reset_nodes_m=nodes,candidates=[],
               raised_start=raised_start,start_root_lift_m=lift,extra_joint_margin_rad=.001 if raised_start else 0.)
    def measure(q3):
        count=budget.consume();q=end.copy();q[:3]=q3;surface.expand(q)
        t=base.fk_poses(g,surface._pose_vector(start_root),q)
        rows={n:base.sample_contact(grids[n],t[n],nodes) for n in surface.PADS}
        gaps=[r["minimum_gap_m"] for r in rows.values()]
        if any(v is None for v in gaps):raise ValueError("required opening node outside locked SDF")
        emit(dict(phase="direction",evaluation=count,q_rad=q,pads=rows));out["endpoint_evaluations"]=count
        return np.array(gaps)
    for target in ((.0006,.0008,.001) if raised_start else (.00015,.00025,.0004)):
        try:proposal=opening_proposal(measure,end[:3],low[:3],high[:3],target)
        except ValueError as exc:
            if "no bounded original-joint opening" not in str(exc):raise
            row=dict(target_gap_m=target,initial_clear=False,reason=str(exc))
            out["candidates"].append(row);emit(dict(phase="initial_candidate",**row));continue
        start=end.copy();start[:3]+=proposal["delta_rad"];surface.expand(start)
        measure(start[:3])
        audit=surface.geometry_record(g,grids,start_root,start,nodes)
        poses=base.fk_poses(g,surface._pose_vector(start_root),start)
        initial_sep={kind+":"+n:surface._separation(base.transform(v["vertices"],poses[n]),nodes)
                     for kind,key in (("source","source_colliders"),("cooked","cooked_pads")) for n,v in g[key].items()}
        passed=min(audit["table_m"].values())>=.0021 and min(initial_sep.values())>=.00005
        row=dict(target_gap_m=target,proposal=proposal,start_q_rad=start,start_root_wxyz=surface._pose_vector(start_root),geometry=audit,
                 all_shape_initial_rod_lower_m=initial_sep,initial_clear=bool(passed))
        out["candidates"].append(row);emit(dict(phase="initial_candidate",**row))
    ready=[c for c in out["candidates"] if c["initial_clear"]]
    out.update(initial_candidates_passed=len(ready),static_path_passed=False,native_ready=False,native_recipe_created=False,
               teacher_admission=False,self_collision_qualified=False)
    if ready:
        chosen=min(ready,key=lambda c:np.linalg.norm(c["proposal"]["delta_rad"]))
        start=chosen["start_q_rad"];delta=end-start
        travel={**{"source:"+n:v for n,v in base.path_displacement_bound(g,start,delta).items()},
                **{"cooked:"+n:v for n,v in base.path_displacement_bound(g,start,delta,cooked=True).items()}}
        travel={n:v+lift for n,v in travel.items()}
        lipschitz={n:grid_lipschitz(grids[n]) for n in surface.PADS}
        out.update(selected_start_q_rad=start,delta_rad=delta,physical_velocity_rad_s=delta/(160/480),root_velocity_world_m_s=[0.,0.,-lift/(160/480)],
                   path_travel_bounds_m=travel,sdf_lipschitz=lipschitz,path=[])
        for step,fraction in enumerate(np.linspace(0,1,161)):
            q=start+fraction*delta
            moving_root=root.copy();moving_root[2,3]+=lift*(1-fraction)
            audit=surface.geometry_record(g,grids,moving_root,q,nodes)
            table={n:v-travel[n]/320 for n,v in audit["table_m"].items()}
            other={n:v-travel[n]/320 for n,v in audit["non_target_rod_gap_m"].items()}
            depth={n:audit["pads"][n]["minimum_gap_m"]-lipschitz[n]*travel["cooked:"+n]/320 for n in surface.PADS}
            row=dict(index=step,fraction=float(fraction),table_lower_m=table,other_lower_m=other,
                     required_depth_lower_m=depth,pads=audit["pads"],normal_dot=audit.get("normal_dot"))
            out["path"].append(row);emit(dict(phase="path",**row))
        min_table=min(v for r in out["path"] for v in r["table_lower_m"].values())
        min_other=min(v for r in out["path"] for v in r["other_lower_m"].values())
        min_depth=min(v for r in out["path"] for v in r["required_depth_lower_m"].values())
        last=out["path"][-1];worst_end=max(r["minimum_gap_m"] for r in last["pads"].values())
        gate=base.gates(min_table,min_other,min_depth,worst_end,last["normal_dot"])
        out.update(path_gates=gate,static_path_passed=gate["static_path_passed"],
                   minimum_table_lower_m=min_table,minimum_other_lower_m=min_other,minimum_depth_lower_m=min_depth)
    out.update(status="complete",endpoint_evaluations=budget.count,
               path_samples=len(out.get("path",[])),limitations=[
                   "Static original Reset rod only; actual rod response and hand tracking remain native gates.",
                   "Original coupled bounds preserved; no implicit reopening beyond finger12 lower limit.",
                   "Self/interhand collision, runtime SDK root binding and actual Reset need separate qualification."])


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--raised-start",action="store_true");a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    out=dict(version=VERSION+("-minnorm-raised-v2" if a.raised_start else "-minnorm-v2"),status="partial",new_physics_steps=0,ppo_updates=0,native_qualified=False,teacher_admission=False,
             budget=dict(endpoint_evaluations=48,path_samples=161,wall_s=120))
    old=signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError("120s CPU budget exhausted")))
    signal.setitimer(signal.ITIMER_REAL,120)
    try:
        with (a.output/"progress.jsonl").open("x") as f:
            def emit(row):f.write(json.dumps(surface.base.encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(out,emit,raised_start=a.raised_start)
            finally:f.flush();os.fsync(f.fileno())
    except Exception as exc:out.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    out.update(elapsed_s=time.monotonic()-start,source_sha256=surface.base.digest(__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(surface.base.encode_owned(out),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:out.get(k) for k in ("status","reason","endpoint_evaluations","initial_candidates_passed","path_samples","static_path_passed","elapsed_s")}))
    return int(out["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
