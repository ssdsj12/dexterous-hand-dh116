#!/usr/bin/env python3
"""Coupled bounded reset X/yaw + existing five-variable Pinch direction; CPU only."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scripts import audit_dh116_pinch_static_yaw as yaw
from scripts import solve_dh116_pinch_trust_region as trust
from scripts.audit_dh116_pinch_initial_pose import finite_array,digest,load_geometry,path_displacement_bound
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm

VERSION="dh116-pinch-coupled-static-lateral-yaw-v1"
DIM=7
DIRECTIONS=2
MAX_STATES=2*(DIRECTIONS*(29*2+2)+33)
PARENT=yaw.PARENT
PARENT_SHA=yaw.PARENT_SHA
WITNESS=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-yaw-witness-20261113-Nhixzu/run01/result.json")
WITNESS_SHA="61130374e62c82aa796b5b0bfbce95a92e015d0ae12ee3ad4435131bc37df715"
previous=trust.previous

def bounds(parent):
    low,high=previous.bounds(parent)
    return np.r_[low,-1.,-1.],np.r_[high,1.,1.]

def decode(parent,x):
    x=finite_array(x,(DIM,));low,high=bounds(parent)
    if np.any(x<low) or np.any(x>high):
        raise ValueError("absolute original joint/reset bounds and +/-3mm X, +/-0.03rad world yaw required")
    c=previous.decode(parent,x[:5])
    c["base_pose_wxyz"][0]+=.003*float(x[5])
    c=yaw.rotate_candidate(c,.03*float(x[6]))
    c.update(normalized_coupled_direction=x.tolist(),
        original_absolute_parent_basis=True,
        static_lateral_delta_m=.003*float(x[5]),
        static_reset_transform="world XYZ translation; world Z precomposition at the base origin",
        wrist_command_modified=False)
    return c

def model(query,center,lower,upper,radius):
    center,lower,upper=[finite_array(v,(DIM,)) for v in (center,lower,upper)]
    if np.any(center<lower) or np.any(center>upper) or np.any(lower>=upper) or not 0<radius<=.2:
        raise ValueError("bounded seven-variable center and trust region required")
    base=query(center.copy());labels=base["labels"];c=trust.values(base,labels)
    jacobians=[];stencils=[]
    for h in (radius/20,radius/40):
        columns=[];level=[]
        for j in range(DIM):
            s=trust.stencil(center,j,lower,upper,h)
            a,b=[trust.values(query(np.array(p)),labels) for p in s["points"]]
            if s["mode"]=="central":col=(a-b)/(2*s["step"])
            else:
                sign=1 if s["mode"]=="forward" else -1
                col=sign*(-3*c+4*a-b)/(2*s["step"])
            columns.append(col);level.append(s)
        jacobians.append(np.stack(columns,axis=1));stencils.append(level)
    relative=float(np.linalg.norm(jacobians[1]-jacobians[0])/max(np.linalg.norm(jacobians[1]),1e-12))
    return dict(baseline=base,jacobians=[j.tolist() for j in jacobians],stencils=stencils,
        jacobian_relative_difference=relative,supported=relative<=trust.MODEL_RELATIVE_LIMIT)

def refine(query,initial,lower,upper,trace):
    center=finite_array(initial,(DIM,)).copy()
    lower,upper=[finite_array(v,(DIM,)) for v in (lower,upper)]
    radius=.2;accepted=0
    for ordinal in range(DIRECTIONS):
        item=dict(direction=ordinal+1,center_x=center.tolist(),trust_radius_before=radius,
                  solver_calls=0,accepted=False,status="evaluating")
        with _commit_row_without_alarm():trace.append(item)
        m=model(query,center,lower,upper,radius);item["model"]=m
        if not m["supported"]:item["reason"]="two_scale_model_disagreement"
        else:
            labels=m["baseline"]["labels"];c=trust.values(m["baseline"],labels)
            soft=trust.contact_mask(labels)
            if not soft.any():raise ValueError("explicit target contact rows required")
            j=np.array(m["jacobians"][1])
            lo,hi=np.maximum(lower-center,-radius),np.minimum(upper-center,radius)
            fit=linprog([*([0.]*DIM),1.],
                A_ub=-np.column_stack((j,soft.astype(float))),b_ub=c,
                bounds=[*zip(lo,hi),(0,None)],method="highs",
                options=dict(maxiter=100,time_limit=1.,threads=2))
            item.update(solver_calls=1,solver_status=int(fit.status),solver_message=fit.message)
            if fit.success:
                proposed=center+fit.x[:DIM]
                if np.any(proposed<lower-1e-9) or np.any(proposed>upper+1e-9):
                    raise ValueError("proposed reset outside original absolute search bounds")
                proposed=np.clip(proposed,lower,upper);predicted=c+j@(proposed-center)
                item.update(proposed_x=proposed.tolist(),predicted_values=predicted.tolist(),
                            predicted_common_contact_slack_mm=float(fit.x[-1]))
                actual=query(proposed);item["actual"]=actual
                decision=trust.assess(labels,c,predicted,trust.values(actual,labels))
                item.update(assessment=decision,accepted=decision["accepted"],
                    reason="actual_predicted_improvement" if decision["accepted"] else "nonlinear_trial_rejected")
                if decision["accepted"]:center=proposed;accepted+=1
            else:item["reason"]="bounded_lp_no_solution"
        if not item["accepted"]:radius*=.5
        item.update(status="complete",selected_x=center.tolist(),trust_radius_after=radius)
    return dict(selected_x=center.tolist(),accepted_directions=accepted,
                global_infeasibility_proven=False,final_trust_radius=radius)

def run(project,result):
    old=previous.checked_json(PARENT,PARENT_SHA)
    w=previous.checked_json(WITNESS,WITNESS_SHA)
    if old["status"]!="complete" or old["geometry_state_attempts"]!=330 or w["status"]!="complete" or len(w["records"])!=16:
        raise ValueError("exact complete trust seed and archived yaw witness evidence required")
    for name,sha in old["implementation_sha256"].items():
        if digest(project/name)!=sha:raise ValueError("original geometry implementation changed: "+name)
    nodes=finite_array(old["actual_parent_nodes_m"],(41,3))
    pair=previous.load_pair_proposal()
    if old["pair_sha256"]!=previous.PAIR_SHA or old["reset_seed_sha256"]!=previous.SEED_SHA:
        raise ValueError("original absolute pair/reset identity differs")
    result.update(parent_result_path=str(PARENT),parent_result_sha256=PARENT_SHA,
        witness_path=str(WITNESS),witness_sha256=WITNESS_SHA,
        pair_sha256=previous.PAIR_SHA,reset_seed_sha256=previous.SEED_SHA,
        original_parent_identity=old["parent_identity"],actual_parent_nodes_m=nodes.tolist(),
        sources={},sides={},state_records=[],maximum_states=MAX_STATES,
        maximum_directions_per_side=DIRECTIONS,
        variable_order=["positive_static_z","static_y","initial_thumb_spread",
                        "thumb_closure_delta","index_closure_delta","static_x","static_world_yaw"],
        variable_scales=[.002,.003,.06,.05,.05,.003,.03],
        design_targets_are_not_changed_physics_thresholds=True,
        original_runtime_floor_m=previous.RUNTIME_FLOOR,
        implementation_sha256={f:digest(project/f) for f in (
            "scripts/solve_dh116_pinch_coupled_lateral.py",
            "scripts/audit_dh116_pinch_static_yaw.py",
            "scripts/audit_dh116_pinch_yaw_witness.py")})
    budget=trust.StateBudget(result["state_records"],maximum=MAX_STATES)
    try:
        for side in ("left","right"):
            g=load_geometry(project,side=side)
            parent=dict(pair[side]["candidate"],side=side)
            if g["source_sha256"]!=old["sources"][side]["geometry_sha256"]:
                raise ValueError("independent actual hand geometry differs")
            grids,binding=previous.grids_for(g,previous.checked_json(previous.SDF_DIR/"result.json",previous.SDF_SHA))
            if binding!=old["sources"][side]["sdf_bindings"]:raise ValueError("actual side SDF differs")
            result["sources"][side]=dict(geometry_sha256=g["source_sha256"],sdf_bindings=binding)
            seed=np.r_[old["sides"][side]["candidate"]["normalized_direction"],0.,0.]
            # The old candidate is a coordinate within the ORIGINAL parent bounds.
            # It must never be used as a new origin to silently expand the search.
            if previous.decode(parent,seed[:5])!=old["sides"][side]["candidate"]:
                raise ValueError("seed was rebased instead of original absolute parent")
            record=dict(directions=[],path=[],static_candidate_verified=False)
            result["sides"][side]=record;start=budget.count
            def measure(args):
                x,f=args;c=decode(parent,x)
                return previous.previous.measure(g,c,grids,nodes,np.zeros(3),f)
            def state(x,f,phase):
                return budget.call(measure,(x,f),side=side,phase=phase,
                    parameter_x=np.asarray(x).tolist(),path_fraction=float(f))
            def query(x):
                first=state(x,0.,"direction_initial");last=state(x,1.,"direction_endpoint")
                labels,values=previous.initial_constraints(first)
                return dict(labels=labels+["endpoint:"+n for n in last["labels"]],
                    values=values+last["values"],initial=first,endpoint=last)
            outcome=refine(query,seed,*bounds(parent),record["directions"])
            record.update(outcome);x=np.array(outcome["selected_x"]);c=decode(parent,x)
            for f in np.linspace(0,1,33):
                row=state(x,float(f),"selected_complete_path")
                with _commit_row_without_alarm():record["path"].append(row)
            path=record["path"];q=np.array(c["active_q_rad"]);d=np.array(c["active_delta_rad"])
            travel={**{"source:"+k:v for k,v in path_displacement_bound(g,q,d).items()},
                    **{"cooked:"+k:v for k,v in path_displacement_bound(g,q,d,cooked=True).items()}}
            table={k:previous.path_lower([r["table_m"][k] for r in path],v) for k,v in travel.items()}
            other={k:previous.path_lower([r["all_shape_rod_lower_m"][k] for r in path],v)
                   for k,v in travel.items() if k.split(":")[1] not in previous.previous.PADS}
            initial=min(path[0]["all_shape_rod_lower_m"].values())>=0
            contact=all(-.0003<=path[-1]["node_sdf"][n]["minimum_gap_m"]<=-.0001
                        and bool(path[-1]["node_sdf"][n]["contact_predicate_nodes"]) for n in previous.previous.PADS)
            record.update(candidate=c,geometry_states=budget.count-start,point_travel_m=travel,
                continuous_table_lower_m=table,continuous_non_target_lower_m=other,
                initial_all_shape_rod_clear=initial,endpoint_both_pad_target_contact=contact,
                **previous.gate(min(table.values()),min(other.values()),initial,contact))
        result["status"]="complete"
    finally:
        result["geometry_state_attempts"]=budget.count
        result["completed_geometry_states"]=sum(r["status"]=="complete" for r in result["state_records"])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,training_updates=0,
        reset_proposal_created=False,native_qualification=False,teacher_admission=False)
    def timeout(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,50)
    try:run(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except Exception as error:result.update(status="diagnostic_error",error=type(error).__name__+": "+str(error))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","geometry_state_attempts","error")}))
    return int(result["status"]=="diagnostic_error")

if __name__=="__main__":raise SystemExit(main())
