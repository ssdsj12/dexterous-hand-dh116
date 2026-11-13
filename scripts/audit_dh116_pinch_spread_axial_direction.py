#!/usr/bin/env python3
"""Bounded initial-clearance-aware axial/spread Pinch design; no native launch."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
from scipy.optimize import linprog
from scripts import audit_dh116_pinch_raised_reset_direction as previous
from scripts.audit_dh116_pinch_initial_pose import finite_array, digest, load_geometry, path_displacement_bound
from scripts.prepare_dh116_dual_pinch_readback_profile import load_pair_proposal, PAIR, PAIR_SHA, SEED_SHA
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for, SDF_DIR, SDF_SHA, checked_json
from scripts.screen_dh116_right_pinch_closure import BudgetExpired, _commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence

VERSION="dh116-pinch-spread-axial-direction-v1"
RUNTIME_FLOOR=previous.RUNTIME_FLOOR
SCALES=np.array([.002,.003,.06,.05,.05])
INITIAL_TABLE=.0025
INITIAL_ROD=.0001
ROOT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
RECEIPT=ROOT/"native-attempts/flex.f7-dual-pinch-guarded-closure-native/74f8378ee4648f30c8ecb2ba8f0aa5f2caf9ef396a8e49b9c532cbf44ff7fad6/attempt-1ouifpxg/receipt.json"
RECEIPT_SHA="dd64cce38312cb3fe044e210c8560f44fae33b218e5b8e2ba97435a3969c9bc0"
LEDGER_SHA="140584e4375b96343478ec973fdbdb94f62d8a2a687fc8dd3b1ba1a5db1a2ec9"


class BudgetLimit(RuntimeError):pass


class StateBudget:
    def __init__(self,query,maximum=160):
        if type(maximum) is not int or not 1<=maximum<=160:raise ValueError("1..160 states required")
        self.query,self.maximum,self.count=query,maximum,0
    def __call__(self,x):
        if self.count>=self.maximum:raise BudgetLimit("complete geometry-state cap")
        self.count+=1
        return self.query(x)


def decode(parent,x):
    x=finite_array(x,(5,))
    if parent["side"] not in ("left","right") or not 0<=x[0]<=1 or np.any(np.abs(x[1:3])>1):
        raise ValueError("independent side and bounded static reset required")
    q=finite_array(parent["active_q_rad"],(6,)).copy()
    delta=finite_array(parent["active_delta_rad"],(6,)).copy()
    pose=finite_array(parent["base_pose_wxyz"],(7,)).copy()
    pose[2]+=.002*x[0];pose[1]+=.003*x[1];q[0]+=.06*x[2]
    delta[1:3]+=.05*x[3:]
    if np.any(delta<0) or np.any(delta>.1+1e-12) or np.any(delta[[0,3,4,5]]):
        raise ValueError("original two-digit 0.3rad/s x 160/480s closure required")
    expand(q);expand(q+delta)
    return dict(side=parent["side"],base_pose_wxyz=pose.tolist(),active_q_rad=q.tolist(),
        active_delta_rad=delta.tolist(),static_reset_only=True,
        static_reset_delta=dict(world_y_m=.003*float(x[1]),world_z_m=.002*float(x[0]),
                               thumb_spread_rad=.06*float(x[2])),
        normalized_direction=x.tolist(),native_qualification=False,teacher_admission=False)


def bounds(parent):
    low,high=active_limits()
    q=np.asarray(parent["active_q_rad"]);d=np.asarray(parent["active_delta_rad"])
    return (np.r_[0,-1,max(-1,(low[0]-q[0])/.06),-d[1:3]/.05],
        np.r_[1,1,min(1,(high[0]-q[0])/.06),(np.minimum(.1,high[1:3]-q[1:3])-d[1:3])/.05])


def direction(query,lower,upper):
    base=query(np.zeros(5));labels=base["labels"]
    def value(row):
        if row["labels"]!=labels:raise ValueError("constraint ordering changed")
        return finite_array(row["values"],(len(labels),))
    c=value(base);jac=[]
    for h in (.01,.005):
        z=np.array([h,0.,0.,0.,0.])
        columns=[(-3*c+4*value(query(z))-value(query(2*z)))/(2*h)]
        columns += [(value(query(np.eye(5)[j]*h))-value(query(-np.eye(5)[j]*h)))/(2*h) for j in range(1,5)]
        jac.append(np.stack(columns,axis=1))
    relative=float(np.linalg.norm(jac[1]-jac[0])/max(np.linalg.norm(jac[1]),1e-12))
    result=dict(baseline=base,jacobians=[j.tolist() for j in jac],jacobian_relative_difference=relative,
        supported=relative<=.2,solver_calls=0,global_infeasibility_proven=False)
    if not result["supported"]:return result
    fit=linprog([.001,0,0,0,0,1.],A_ub=-np.column_stack((jac[1],np.ones(len(c)))),b_ub=c,
        bounds=[*zip(lower,upper),(0,None)],method="highs",
        options=dict(maxiter=100,time_limit=1.,threads=2))
    result.update(solver_calls=1,solver_status=int(fit.status),solver_message=fit.message)
    if fit.success:
        result.update(proposed_x=fit.x[:5].tolist(),predicted_common_slack_mm=float(fit.x[-1]),
            predicted_values=(c+jac[1]@fit.x[:5]).tolist())
    return result


def initial_constraints(row):
    labels=[];values=[]
    for k,v in row["table_m"].items():
        labels.append("initial:"+k+":table_design_2p5mm");values.append((v-INITIAL_TABLE)/.001)
    for k,v in row["all_shape_rod_lower_m"].items():
        labels.append("initial:"+k+":rod_clear_0p1mm");values.append((v-INITIAL_ROD)/.001)
    return labels,values


def path_lower(samples,travel):
    a=finite_array(samples,(33,))
    if not np.isfinite(travel) or travel<0:raise ValueError("finite travel bound required")
    return float(a.min()-travel/64)


def gate(table_lower,other_lower,initial_clear,contact):
    return dict(static_candidate_verified=bool(table_lower>=RUNTIME_FLOOR and other_lower>=0 and initial_clear and contact),
        native_qualification=False,teacher_admission=False,global_infeasibility_proven=False)


def native_nodes():
    receipt=checked_json(RECEIPT,RECEIPT_SHA)
    if (receipt["outer_exit_code"]!=0 or receipt["native_qualification"] is not True
        or receipt["qualification_scope"]!="dual_pinch_same_free_rod_one_guarded_finger_action_interval"):
        raise ValueError("completed exact parent native scope required")
    output=Path(receipt["output"]);path=output/"physical.jsonl"
    if digest(path)!=LEDGER_SHA or receipt["artifact_sha256"]["physical.jsonl"]!=LEDGER_SHA:
        raise ValueError("frozen complete parent ledger required")
    # The parent independent post-exit audit already qualified the exact bytes.
    rows=[decode_owned(r["payload"]) for l in path.open() if (r:=json.loads(l)).get("operation")=="physical"]
    if len(rows)!=16 or [r["exchange"]["physics_step"] for r in rows]!=list(range(3,19)):
        raise ValueError("complete 16-row parent required")
    last=rows[-1]["exchange"]
    return finite_array(last["rod"]["node_positions"],(1,41,3))[0],last["closure_geometry"]["raw"]["identity"]


def run(project,result):
    pair=load_pair_proposal();nodes,identity=native_nodes()
    result.update(pair_path=str(PAIR),pair_sha256=PAIR_SHA,reset_seed_sha256=SEED_SHA,
        parent_receipt_sha256=RECEIPT_SHA,parent_ledger_sha256=LEDGER_SHA,parent_identity=identity,
        actual_parent_nodes_m=nodes.tolist(),sources={},sides={},state_records=[],
        original_runtime_floor_m=RUNTIME_FLOOR,diagnostic_initial_table_target_m=INITIAL_TABLE,
        design_targets_are_not_changed_physics_thresholds=True)
    files=("scripts/audit_dh116_pinch_spread_axial_direction.py",
        "scripts/audit_dh116_pinch_raised_reset_direction.py",
        "scripts/prepare_dh116_dual_pinch_readback_profile.py",
        "scripts/prepare_dh116_pinch_closure_margin.py","scripts/audit_dh116_pinch_initial_pose.py",
        "src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py")
    result["implementation_sha256"]={str(project/f):digest(project/f) for f in files}
    holder=[None];budget=StateBudget(lambda args:holder[0](*args))
    try:
        for side in ("left","right"):
            g=load_geometry(project,side=side);parent=dict(pair[side]["candidate"],side=side)
            expected=parent["source_sha256"] if side=="left" else pair[side]["source_sha256"]
            if g["source_sha256"]!=expected:raise ValueError("independent side source differs")
            grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
            result["sources"][side]=dict(geometry_sha256=g["source_sha256"],sdf_bindings=bindings)
            start=budget.count
            def measure(x,fraction):
                selected=decode(parent,x)
                row=previous.measure(g,selected,grids,nodes,np.zeros(3),fraction)
                row["static_reset_delta"]=selected["static_reset_delta"]
                with _commit_row_without_alarm():result["state_records"].append(dict(side=side,measurement=row))
                return row
            holder[0]=measure
            def query(x):
                first=budget((x,0.));last=budget((x,1.))
                labels,values=initial_constraints(first)
                return dict(labels=labels+["endpoint:"+x for x in last["labels"]],
                    values=values+last["values"],initial=first,endpoint=last)
            fit=direction(query,*bounds(parent))
            side_result=dict(direction=fit,geometry_states=budget.count-start,static_candidate_verified=False)
            result["sides"][side]=side_result
            if "proposed_x" not in fit:continue
            x=fit["proposed_x"];path=[budget((x,float(f))) for f in np.linspace(0,1,33)]
            candidate=decode(parent,x);q=np.array(candidate["active_q_rad"]);delta=np.array(candidate["active_delta_rad"])
            travel={**{"source:"+k:v for k,v in path_displacement_bound(g,q,delta).items()},
                    **{"cooked:"+k:v for k,v in path_displacement_bound(g,q,delta,cooked=True).items()}}
            table={k:path_lower([r["table_m"][k] for r in path],b) for k,b in travel.items()}
            other={k:path_lower([r["all_shape_rod_lower_m"][k] for r in path],b)
                   for k,b in travel.items() if k.split(":")[1] not in previous.PADS}
            initial_clear=min(path[0]["all_shape_rod_lower_m"].values())>=0
            contact=all(path[-1]["node_sdf"][n]["minimum_gap_m"]<=-.0001
                        and bool(path[-1]["node_sdf"][n]["contact_predicate_nodes"]) for n in previous.PADS)
            side_result.update(candidate=candidate,path=path,point_travel_m=travel,
                continuous_table_lower_m=table,continuous_non_target_lower_m=other,
                initial_all_shape_rod_clear=bool(initial_clear),endpoint_target_sdf_contact=bool(contact),
                geometry_states=budget.count-start,
                **gate(min(table.values()),min(other.values()),initial_clear,contact))
        result["status"]="complete"
    finally:result["geometry_state_attempts"]=budget.count


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,training_updates=0,
        reset_proposal_created=False,native_qualification=False,teacher_admission=False)
    def timeout(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,50)
    try:run(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,BudgetLimit) as e:result.update(status="diagnostic_error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","geometry_state_attempts","error")}))
    return int(result["status"]=="diagnostic_error")


if __name__=="__main__":raise SystemExit(main())
