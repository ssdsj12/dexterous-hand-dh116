#!/usr/bin/env python3
"""Independent finite CPU max-clearance reset selection, not native approval."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scripts import audit_dh116_pinch_rod_axis_clearance as base

VERSION="dh116-pinch-rod-axis-max-margin-v1"
ITERATIONS=24

def extreme_indices(coefficients):
    """Keep actual constraints at 2D hull extremes for each identical RHS.

    Subsetting cannot invent an infeasibility certificate. Apparent feasibility
    MUST still be checked against every original row; hull numerics are not a
    geometry qualification.
    """
    a=base.finite_array(coefficients)
    if a.ndim!=2 or a.shape[1]!=3 or not len(a):raise ValueError("nonempty Nx3 required")
    selected=[]
    for rhs in np.unique(a[:,2]):
        ids=np.flatnonzero(a[:,2]==rhs)
        points,unique=np.unique(a[ids,:2],axis=0,return_index=True)
        if len(points)<=2:keep=np.arange(len(points))
        else:
            try:keep=ConvexHull(points).vertices
            except QhullError:
                # No approximation or rank tolerance: retain every distinct
                # original point if the hull cannot certify its own topology.
                keep=np.arange(len(points))
        selected.extend(ids[unique[keep]].tolist())
    return np.array(sorted(set(selected)),dtype=int)


def _midpoint(intervals):
    lo,hi=max(intervals,key=lambda pair:pair[1]-pair[0])
    return (lo+hi)/2


def select_margin(coefficients,intervals):
    full=base.finite_array(coefficients)
    ids=extreme_indices(full);subset=full[ids].copy()
    result=dict(angle_rad=None,lower_margin_m=None,upper_margin_m=None,
        iterations=0,full_constraint_count=len(full),retained_constraint_count=len(ids),
        full_constraint_rechecks=0,full_fallback_calls=0,native_ready=False,
        method="bounded_margin_bisection_24_with_original_constraint_rechecks")
    if not intervals:return result
    domain=[-base.ANGLE_LIMIT_RAD,base.ANGLE_LIMIT_RAD]
    # Independent caller may supply a tighter frozen domain, never widen it.
    domain=[max(domain[0],min(x[0] for x in intervals)),min(domain[1],max(x[1] for x in intervals))]
    if domain[0]>=domain[1]:raise ValueError("nonzero feasible static angle domain required")
    def margin(x):
        result["full_constraint_rechecks"]+=1
        return float(np.min(full[:,0]*np.cos(x)+full[:,1]*np.sin(x)-full[:,2]))
    original=base.choose_angle(intervals);middle=_midpoint(intervals)
    angle=max((original,middle),key=margin)
    lo=margin(angle)
    if lo<-1e-12:raise ValueError("supplied feasible domain failed complete original constraints")
    lo=max(0.,lo)
    # This unconstrained per-vertex maximum bounds the max-min from above.
    hi=max(lo,float(np.min(np.hypot(full[:,0],full[:,1])-full[:,2])))
    history=[]
    for n in range(ITERATIONS):
        trial=(lo+hi)/2
        shifted=subset.copy();shifted[:,2]+=trial
        arcs=base.feasible_intervals(shifted,domain)
        x=None if not arcs else _midpoint(arcs)
        actual=None if x is None else margin(x)
        if x is not None and actual<trial:
            result["full_fallback_calls"]+=1
            shifted=full.copy();shifted[:,2]+=trial
            arcs=base.feasible_intervals(shifted,domain)
            x=None if not arcs else _midpoint(arcs)
            actual=None if x is None else margin(x)
        feasible=x is not None and actual>=trial
        if feasible:lo,angle=trial,x
        else:hi=trial
        history.append(dict(iteration=n+1,trial_margin_m=trial,feasible=feasible,
                            candidate_angle_rad=x,actual_margin_m=actual))
    actual=margin(angle)
    if actual<lo-1e-12:raise ValueError("selected margin fails complete constraint audit")
    result.update(angle_rad=float(angle),lower_margin_m=float(lo),upper_margin_m=float(hi),
        selected_actual_margin_m=actual,iterations=ITERATIONS,history=history,
        bracket_scope="finite_float64_static_geometry_not_native_or_global_grasp_certificate")
    return result


def sampled_depth_gate(rows):
    if not rows:return False
    for row in rows:
        if set(row["pads"])!=set(base.PADS):return False
        for pad in row["pads"].values():
            value=pad["minimum_gap_m"]
            if value is None or not np.isfinite(value) or value<-.0003:return False
    return True


def run(project,result,progress):
    base.run(project,result,progress,angle_selector=select_margin,check_contact_path=True)
    result["implementation_sha256"]["scripts/audit_dh116_pinch_rod_axis_margin.py"]=base.digest(__file__)
    for side,row in result["sides"].items():
        path=[v["node_sdf"] for v in row["rotated_samples"]]
        row["required_depth_161_targets_passed"]=len(path)==base.SAMPLES and sampled_depth_gate(path)
        row["static_geometry_passed"] &= row["required_depth_161_targets_passed"]
        row["required_pad_sampled_min_gap_m"]={
            name:min((v["pads"][name]["minimum_gap_m"] for v in path
                      if v["pads"][name]["minimum_gap_m"] is not None),default=None)
            for name in base.PADS}
        row["native_ready"]=False
        row["limitations"]=[
            "Static full source17/cooked5 target trajectory, not measured dynamic tracking.",
            "All 161 required-pad target samples bounded; intermediate solver deformation is not predicted.",
            "Actual Reset, self collision, mounted wrist reach and execution remain unqualified.",
            "Margin optimization changes no joint angles, action amplitude, physical parameters or limits."]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False)
    result=dict(version=VERSION,status="partial",scope="new_CPU_max_clearance_reset_construction",
        budget=dict(wall_seconds=90,geometry_states=base.MAX_STATES,angle_candidates_per_side=1,
                    margin_iterations_per_side=ITERATIONS,native_physics_steps=0,ppo_updates=0))
    start=time.monotonic()
    old=signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(base.BudgetExpired()))
    signal.setitimer(signal.ITIMER_REAL,90)
    try:
        with (a.output/"progress.jsonl").open("x") as f:
            def progress(row):
                f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
            try:run(Path(__file__).resolve().parents[1],result,progress)
            finally:f.flush();os.fsync(f.fileno())
    except base.BudgetExpired:result["status"]="partial_budget"
    except (ValueError,KeyError,StopIteration) as exc:
        result.update(status="diagnostic_error",error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","geometry_states","elapsed_s","error")}))
    return int(result["status"]!="complete")

if __name__=="__main__":raise SystemExit(main())
