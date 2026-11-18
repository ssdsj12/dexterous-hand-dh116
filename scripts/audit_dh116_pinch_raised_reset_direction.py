#!/usr/bin/env python3
"""One bounded raised-root/two-digit Pinch direction per independent side."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
from scipy.optimize import linprog

from scripts.audit_dh116_pinch_initial_pose import (
    load_geometry,world_vertices,finite_array,digest,path_displacement_bound,convex_patch_closest)
from scripts.prepare_dh116_pinch_closure_margin import fk_poses,sample_contact
from scripts.prepare_dh116_dual_pinch_readback_profile import load_native,load_pair_proposal,PAIR,PAIR_SHA,SEED_SHA,LEDGER_SHA
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA,checked_json
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower
from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand,ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-pinch-raised-reset-direction-v1"
RUNTIME_FLOOR=CLEARANCE_M
CONTACT_TARGET=-.0001
SCALES=np.array([.002,.05,.05])
PADS=("finger14_link","finger23_link")
DESIGN_TABLE=.0024
DESIGN_NON_TARGET=.0002


class BudgetLimit(RuntimeError):pass


class StateBudget:
    def __init__(self,query,maximum=64):
        if type(maximum) is not int or not 1<=maximum<=64:raise ValueError("1..64 states required")
        self.query,self.maximum,self.count=query,maximum,0
    def __call__(self,x):
        if self.count>=self.maximum:raise BudgetLimit("complete geometry-state cap")
        self.count+=1
        return self.query(x)


def decode(parent,x):
    x=finite_array(x,(3,));q=finite_array(parent["active_q_rad"],(6,))
    old=finite_array(parent["active_delta_rad"],(6,))
    if parent["side"] not in ("left","right") or not 0<=x[0]<=1:
        raise ValueError("independent side and positive <=2mm static root change required")
    pose=finite_array(parent["base_pose_wxyz"],(7,)).copy();pose[2]+=SCALES[0]*x[0]
    delta=old.copy();delta[1:3]+=SCALES[1:]*x[1:]
    if np.any(delta<0) or np.any(delta>.1+1e-12) or np.any(delta[[0,3,4,5]]):
        raise ValueError("original two-digit 0.3rad/s x 160/480s closure required")
    expand(q);expand(q+delta)
    return dict(side=parent["side"],base_pose_wxyz=pose.tolist(),initial_q_rad=q.tolist(),
        endpoint_q_rad=(q+delta).tolist(),active_delta_rad=delta.tolist(),
        static_root_z_change_m=float(pose[2]-parent["base_pose_wxyz"][2]),
        active_velocity_rad_s=(delta/(160/480)).tolist(),active_joint_names=list(ACTIVE),
        normalized_direction=x.tolist(),native_qualification=False,reset_verified=False)


def bounds(parent):
    q=np.array(parent["active_q_rad"]);old=np.array(parent["active_delta_rad"]);high=active_limits()[1]
    return np.r_[0,-old[1:3]/.05],np.r_[1,(np.minimum(.1,high[1:3]-q[1:3])-old[1:3])/.05]


def direction(query,lower,upper):
    base=query(np.zeros(3));labels=base["labels"]
    def value(row):
        if row["labels"]!=labels:raise ValueError("constraint ordering changed")
        return finite_array(row["values"],(len(labels),))
    c=value(base);jac=[]
    for h in (.01,.005):
        # Positive one-sided root-Z derivative avoids inventing a lowered reset.
        z=np.array([h,0.,0.])
        columns=[(-3*c+4*value(query(z))-value(query(2*z)))/(2*h)]
        columns += [(value(query(np.eye(3)[j]*h))-value(query(-np.eye(3)[j]*h)))/(2*h) for j in (1,2)]
        jac.append(np.stack(columns,axis=1))
    relative=float(np.linalg.norm(jac[1]-jac[0])/max(np.linalg.norm(jac[1]),1e-12))
    result=dict(baseline=base,jacobians=[j.tolist() for j in jac],jacobian_relative_difference=relative,
        supported=relative<=.2,solver_calls=0,global_infeasibility_proven=False)
    if not result["supported"]:return result
    fit=linprog([.001,.0001,.0001,1.],A_ub=-np.column_stack((jac[1],np.ones(len(c)))),b_ub=c,
        bounds=[*zip(lower,upper),(0,None)],method="highs",
        options=dict(maxiter=100,time_limit=1.,threads=2))
    result.update(solver_calls=1,solver_status=int(fit.status),solver_message=fit.message)
    if fit.success:
        result.update(proposed_x=fit.x[:3].tolist(),predicted_common_slack_mm=float(fit.x[-1]),
            predicted_values=(c+jac[1]@fit.x[:3]).tolist())
    return result


def path_lower(samples,travel):
    a=finite_array(samples,(17,))
    if not np.isfinite(travel) or travel<0:raise ValueError("finite travel bound required")
    return float(a.min()-travel/32)


def measure(g,parent,grids,nodes,x,fraction=1.):
    c=decode(parent,x);q=np.array(c["initial_q_rad"])+fraction*np.array(c["active_delta_rad"])
    pose=c["base_pose_wxyz"];fk=fk_poses(g,pose,q)
    sets={**{"source:"+n:v for n,v in world_vertices(g,pose,q).items()},
          **{"cooked:"+n:v for n,v in world_vertices(g,pose,q,cooked=True).items()}}
    tables={k:float(v[:,2].min()-.3) for k,v in sets.items()}
    allgaps={k:rod_support_lower(v,nodes) for k,v in sets.items()}
    sdf={n:sample_contact(grids[n],fk[n],nodes) for n in PADS}
    if any(r["minimum_gap_m"] is None or r["gradient_norm_at_minimum"]<=0 for r in sdf.values()):
        raise ValueError("required side-specific SDF node coverage unavailable")
    labels=[];values=[]
    for k,v in tables.items():labels.append(k+":table_design_2p4mm");values.append((v-DESIGN_TABLE)/.001)
    for k,v in allgaps.items():
        if k.split(":")[1] not in PADS:labels.append(k+":non_target_design_0p2mm");values.append((v-DESIGN_NON_TARGET)/.001)
    for n,row in sdf.items():
        gap=row["minimum_gap_m"]
        labels += [n+":contact_minus0p1mm",n+":bounded_depth_minus0p3mm"]
        values += [(CONTACT_TARGET-gap)/.001,(gap+.0003)/.001]
    return dict(candidate=c,path_fraction=float(fraction),actual_q_rad=q.tolist(),
        labels=labels,values=values,table_m=tables,all_shape_rod_lower_m=allgaps,node_sdf=sdf,
        native_qualification=False,model_scope="source_FK_fixed_root_and_frozen_actual_parent_rod_nodes")


def run(project,result):
    pair=load_pair_proposal();native,_,audit=load_native()
    nodes=finite_array(native[-1]["exchange"]["rod"]["node_positions"],(1,41,3))[0]
    result.update(pair_path=str(PAIR),pair_sha256=PAIR_SHA,original_reset_seed_sha256=SEED_SHA,
        parent_native_ledger_sha256=LEDGER_SHA,native_parent_audit=audit,rod_seed=pair["rod_seed"],
        actual_parent_nodes_m=nodes.tolist(),sources={},sides={},state_records=[],
        original_runtime_floor_m=RUNTIME_FLOOR,diagnostic_design_targets=dict(
            table_m=DESIGN_TABLE,non_target_m=DESIGN_NON_TARGET,node_sdf_m=CONTACT_TARGET),
        design_targets_are_not_changed_physics_thresholds=True)
    files=("scripts/audit_dh116_pinch_raised_reset_direction.py",
        "scripts/prepare_dh116_dual_pinch_readback_profile.py","scripts/prepare_dh116_pinch_closure_margin.py",
        "scripts/audit_dh116_pinch_initial_pose.py",
        "src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py",
        "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")
    result["implementation_sha256"]={str(project/f):digest(project/f) for f in files}
    query_holder=[None]
    budget=StateBudget(lambda args:query_holder[0](*args))
    try:
        for side in ("left","right"):
            g=load_geometry(project,side=side);parent=dict(pair[side]["candidate"],side=side)
            expected=parent["source_sha256"] if side=="left" else pair[side]["source_sha256"]
            if g["source_sha256"]!=expected:raise ValueError("independent side source differs")
            grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
            result["sources"][side]=dict(geometry_sha256=g["source_sha256"],sdf_bindings=bindings)
            start_count=budget.count
            def query(x,fraction=1.):
                row=measure(g,parent,grids,nodes,x,fraction)
                with _commit_row_without_alarm():result["state_records"].append(dict(side=side,measurement=row))
                return row
            query_holder[0]=query
            fit=direction(lambda x:budget((x,)),*bounds(parent))
            side_result=dict(direction=fit,geometry_states=budget.count-start_count,static_candidate_verified=False)
            result["sides"][side]=side_result
            if "proposed_x" not in fit:continue
            x=fit["proposed_x"];path=[budget((x,float(f))) for f in np.linspace(0,1,17)]
            c=path[0]["candidate"];q0=np.array(c["initial_q_rad"]);delta=np.array(c["active_delta_rad"])
            travel={**{"source:"+k:v for k,v in path_displacement_bound(g,q0,delta).items()},
                    **{"cooked:"+k:v for k,v in path_displacement_bound(g,q0,delta,cooked=True).items()}}
            table={k:path_lower([r["table_m"][k] for r in path],b) for k,b in travel.items()}
            other={k:path_lower([r["all_shape_rod_lower_m"][k] for r in path],b)
                   for k,b in travel.items() if k.split(":")[1] not in PADS}
            initial_clear=min(path[0]["all_shape_rod_lower_m"].values())>=0
            endpoints={n:path[-1]["node_sdf"][n] for n in PADS}
            contact=all(r["minimum_gap_m"]<0 and bool(r["contact_predicate_nodes"]) for r in endpoints.values())
            side_result.update(candidate=c,path=path,point_travel_m=travel,continuous_table_lower_m=table,
                continuous_non_target_lower_m=other,initial_all_shape_rod_clear=bool(initial_clear),
                endpoint_both_pad_sdf_contact_predicate=bool(contact),
                static_candidate_verified=bool(initial_clear and min(table.values())>=RUNTIME_FLOOR and min(other.values())>=0 and contact),
                geometry_states=budget.count-start_count,endpoint_collision_witnesses={})
            # No new hand state: feasible QP upper witnesses only for failed endpoint bounds.
            endpoint_shapes={**{"source:"+n:v for n,v in world_vertices(g,c["base_pose_wxyz"],c["endpoint_q_rad"]).items()},
                **{"cooked:"+n:v for n,v in world_vertices(g,c["base_pose_wxyz"],c["endpoint_q_rad"],cooked=True).items()}}
            # The actual frozen native nodes are straight; refuse an invalid shortcut.
            chord=nodes[-1]-nodes[0];t=(nodes-nodes[0])@chord/(chord@chord)
            straight=np.max(np.linalg.norm(nodes-(nodes[0]+t[:,None]*chord),axis=1))<=1e-10
            if straight:
                for k in other:
                    if path[-1]["all_shape_rod_lower_m"][k]<0:
                        qp=convex_patch_closest(endpoint_shapes[k],nodes[0],[chord])
                        side_result["endpoint_collision_witnesses"][k]=dict(qp=qp,gap_upper_m=qp["distance_m"]-.005+2e-6)
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
