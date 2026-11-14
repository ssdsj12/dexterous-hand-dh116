#!/usr/bin/env python3
"""One source-bound hard-constraint Tripod CPU study; no native recipe."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import minimize

from scripts import solve_dh116_tripod_node_closure as previous
from scripts.audit_dh116_pinch_initial_pose import checked_json,digest,load_geometry
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm

VERSION="dh116-tripod-hard-constraint-endpoint-v1"
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-node-closure-20261114-run02/result.json")
PARENT_SHA="f0e36fb782299ee48e41e9282f5c6943563c1aef51332e15201ef6981772e339"
EvaluationLimit=previous.EvaluationLimit
MAX_EVALUATIONS=500
CONTACT_PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-hard-constraints-20261114-run01/result.json")
CONTACT_PARENT_SHA="a484586dd4e08cbdfedd740fba3df53b4bcf9d636b16a7162e624422665fb1df"

def contract(row,*,require_contact=False):
    if (any(len(row[k])!=3 for k in ("node_sdf_gap_m","node_gradient_norm","required_hull_intervals_m"))
            or any(len(pair)!=2 for pair in row["required_hull_intervals_m"])
            or any(len(row[k])!=2 for k in ("geometric_opposition","sdf_opposition"))):
        raise ValueError("complete three pads and both opposition pairs required")
    values,labels=[],[]
    def add(label,value):
        labels.append(label);values.append(float(value))
    # These are tighter numerical search targets, never relaxed physical gates.
    for k,v in row["table_gap_by_shape_m"].items():
        add(k+":table",(v-(previous.CLEARANCE_M+.000005))/.001)
    for k,v in row["nonrequired_gap_lower_m"].items():
        add(k+":nonrequired",(v-.000005)/.001)
    upper_target=-.000005 if require_contact else .000045
    gaps=row["node_sdf_gap_m"]
    supported=all(g is not None and np.isfinite(g) for g in gaps) and all(
        np.isfinite(n) and n>0 for n in row["node_gradient_norm"])
    for i,(gap,interval) in enumerate(zip(gaps,row["required_hull_intervals_m"])):
        lo,hi=interval
        add(f"pad{i}:hull_lower",(lo+.000045)/.001)
        add(f"pad{i}:hull_upper",(.000045-hi)/.001)
        add(f"pad{i}:node_lower",-1000. if gap is None else (gap+.000045)/.001)
        add(f"pad{i}:node_upper",-1000. if gap is None else (upper_target-gap)/.001)
        add(f"pad{i}:gradient",1. if row["node_gradient_norm"][i]>0 else -1000.)
    for kind in ("geometric_opposition","sdf_opposition"):
        for i,v in enumerate(row[kind]):add(f"{kind}:{i}",-1000. if v is None else -.205-v)
    objective=1000. if not supported else float(np.linalg.norm((np.array(gaps)+.000025)/.001)**2)
    if not np.isfinite(values).all() or not np.isfinite(objective):
        raise ValueError("finite hard constraints and objective required")
    return dict(constraints=values,labels=labels,objective=objective,
                measurements_supported=bool(supported),native_ready=False,
                require_node_contact=bool(require_contact),node_upper_search_target_m=upper_target)

class Study:
    def __init__(self,query,maximum=MAX_EVALUATIONS):
        if type(maximum) is not int or not 1<=maximum<=MAX_EVALUATIONS:
            raise ValueError("one to500 actual geometry queries required")
        self.query,self.maximum=query,maximum
        self.count=0;self.rows=[];self.cache={};self.jac_cache={}
    def __call__(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=1 or not np.isfinite(x).all():raise ValueError("finite vector required")
        key=x.tobytes()
        if key in self.cache:return self.cache[key]
        if self.count>=self.maximum:raise EvaluationLimit("hard-constraint CPU geometry cap")
        value=self.query(x)
        if len(value["constraints"])!=len(value["labels"]):
            raise ValueError("ordered complete constraint identity required")
        with _commit_row_without_alarm():
            self.count+=1
            record=dict(evaluation=self.count,x=x.tolist(),packet=value)
            self.rows.append(record);self.cache[key]=value
        return value
    def derivatives(self,x,lo,hi,h=.0005):
        x=np.asarray(x,dtype=float);lo,hi=np.asarray(lo),np.asarray(hi)
        key=(x.tobytes(),lo.tobytes(),hi.tobytes(),h)
        if key in self.jac_cache:return self.jac_cache[key]
        base=self(x);labels=base["labels"]
        def vector(value):
            if value["labels"]!=labels:raise ValueError("constraint identity changed")
            return np.r_[value["objective"],value["constraints"]]
        center=vector(base);columns=[]
        for i in range(len(x)):
            a=x.copy();b=x.copy()
            a[i]=max(lo[i],x[i]-h);b[i]=min(hi[i],x[i]+h)
            if a[i]==b[i]:
                columns.append(np.zeros_like(center));continue
            av=center if a[i]==x[i] else vector(self(a))
            bv=center if b[i]==x[i] else vector(self(b))
            columns.append((bv-av)/(b[i]-a[i]))
        j=np.stack(columns,axis=1)
        result=dict(objective=j[0],constraints=j[1:])
        self.jac_cache[key]=result
        return result

def solve(study,x0,lo,hi,*,iterations=20,callback=None):
    if type(iterations) is not int or not 1<=iterations<=20:
        raise ValueError("bounded1..20 SQP iterations required")
    return minimize(lambda x:study(x)["objective"],x0,method="SLSQP",
        jac=lambda x:study.derivatives(x,lo,hi)["objective"],
        constraints=[dict(type="ineq",fun=lambda x:np.asarray(study(x)["constraints"]),
                          jac=lambda x:study.derivatives(x,lo,hi)["constraints"])],
        bounds=list(zip(lo,hi)),callback=callback,
        options=dict(maxiter=iterations,ftol=1e-10,disp=False))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=55)
    p.add_argument("--require-node-contact",action="store_true")
    a=p.parse_args(argv)
    if not np.isfinite(a.wall_seconds) or not 1<=a.wall_seconds<=55:p.error("CPU wall1..55seconds")
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();study=None
    result=dict(version=VERSION,status="partial_setup",new_physics_steps=0,ppo_updates=0,
                native_ready=False,teacher_admitted=False,reset_verified=False,
                max_geometry_queries=MAX_EVALUATIONS,max_sqp_iterations=20,solver_callback_iterates=[])
    def expire(*unused):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,a.wall_seconds)
    try:
        root=Path(__file__).resolve().parents[1]
        parent_path=CONTACT_PARENT if a.require_node_contact else PARENT
        parent_sha=CONTACT_PARENT_SHA if a.require_node_contact else PARENT_SHA
        prior=checked_json(parent_path,parent_sha)
        if prior["status"]!="complete":
            raise ValueError("exact complete prior Tripod candidate ledger required")
        if a.require_node_contact:
            if prior["version"]!=VERSION or not prior["final_packet"]["measurement"]["endpoint_geometric_gate"]:
                raise ValueError("independent passing original endpoint required")
            initial_x=prior["final_x"];initial_probe=None
        else:
            if prior["solver_method"]!="dogbox":raise ValueError("original dogbox study required")
            initial_x=prior["best"]["x"];initial_probe=prior["best"]["evaluation"]
        g=load_geometry(root,side="left")
        if g["source_sha256"]!=prior["geometry_sources"]:raise ValueError("original left geometry changed")
        grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
        parent=prior["parent_candidate"];x0=np.array(initial_x)
        previous.decode(parent,x0);lo,hi=previous.bounds(parent)
        def query(x):
            measured=previous.measure(g,grids,parent,x)
            return dict(contract(measured,require_contact=a.require_node_contact),measurement=measured)
        study=Study(query)
        result.update(parent_result_sha256=parent_sha,parent_result_path=str(parent_path),parent_candidate=parent,
                      require_node_contact=a.require_node_contact,
                      initial_probe_evaluation=initial_probe,sdf_bindings=bindings,
                      geometry_sources=g["source_sha256"],initial_x=x0.tolist(),
                      source_sha256={n:digest(root/n) for n in (
                          "scripts/solve_dh116_tripod_hard_constraints.py",
                          "scripts/solve_dh116_tripod_node_closure.py",
                          "scripts/audit_dh116_pinch_initial_pose.py",
                          "scripts/prepare_dh116_pinch_closure_margin.py",
                          "scripts/audit_dh116_power_closure_constraints.py")})
        def iteration(x):
            value=study(x)
            with _commit_row_without_alarm():
                result["solver_callback_iterates"].append(dict(x=np.asarray(x).tolist(),
                    objective=value["objective"],minimum_constraint=min(value["constraints"]),
                    original_endpoint_gate=value["measurement"]["endpoint_geometric_gate"]))
        fit=solve(study,x0,lo,hi,callback=iteration)
        final=study(fit.x)
        result.update(status="complete",solver_success=bool(fit.success),solver_status=int(fit.status),
                      solver_message=str(fit.message),solver_nit=int(fit.nit),final_x=fit.x.tolist(),
                      final_packet=final)
    except BudgetExpired:result["status"]="partial_timeout"
    except EvaluationLimit:result["status"]="partial_geometry_cap"
    except (ValueError,KeyError) as e:result.update(status="error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    if study:
        result.update(geometry_queries=study.count,query_records=study.rows)
        passed=[r for r in study.rows if r["packet"]["measurement"]["endpoint_geometric_gate"]]
        result["original_endpoint_pass_count"]=len(passed)
        result["hard_constraint_pass_count"]=sum(min(r["packet"]["constraints"])>=0 for r in passed)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in (
        "status","elapsed_s","geometry_queries","original_endpoint_pass_count","solver_message")}))
    return int(result["status"]=="error")

if __name__=="__main__":raise SystemExit(main())
