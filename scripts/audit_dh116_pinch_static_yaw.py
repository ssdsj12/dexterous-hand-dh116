#!/usr/bin/env python3
"""Bounded static reset yaw screen using actual side FK/hulls/SDF; CPU only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal,time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts import solve_dh116_pinch_trust_region as trust
from scripts.audit_dh116_pinch_initial_pose import finite_array,digest,load_geometry,path_displacement_bound
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import expand

VERSION="dh116-pinch-static-world-yaw-v1"
PARENT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-trust-region-20261113-1k09bsj6/run01/result.json")
PARENT_SHA="0b6ebea34f6ec22ee41c599343ac02aadd25b2560c600b969fe5e7e266819602"
ANGLES=tuple(float(x) for x in np.linspace(-.03,.03,9))
MAX_STATES=102
previous=trust.previous

def rotate_candidate(parent,yaw):
    if type(yaw) not in (int,float) or not np.isfinite(yaw) or abs(yaw)>.03:
        raise ValueError("finite static reset yaw within +/-0.03rad required")
    pose=finite_array(parent["base_pose_wxyz"],(7,)).copy()
    q=finite_array(parent["active_q_rad"],(6,))
    d=finite_array(parent["active_delta_rad"],(6,))
    if parent["side"] not in ("left","right") or not np.isclose(np.linalg.norm(pose[3:]),1,rtol=0,atol=1e-10):
        raise ValueError("actual side and unit original root quaternion required")
    if np.any(d<0) or np.any(d>.1+1e-12) or np.any(d[[0,3,4,5]]):
        raise ValueError("original two-finger bounded closure must be unchanged")
    expand(q);expand(q+d)
    old=pose[[4,5,6,3]]
    new=(Rotation.from_rotvec([0,0,yaw])*Rotation.from_quat(old)).as_quat()
    if np.dot(new,old)<0:new=-new
    pose[3:]=new[[3,0,1,2]]
    value=deepcopy(parent)
    value.update(base_pose_wxyz=pose.tolist(),static_reset_yaw_world_rad=float(yaw),
        static_reset_only=True,wrist_command_modified=False,
        native_qualification=False,teacher_admission=False)
    return value

def rank(hard_values,contact_gaps_mm):
    hard=np.asarray(hard_values,dtype=float);gaps=np.asarray(contact_gaps_mm,dtype=float)
    if hard.ndim!=1 or not hard.size or gaps.shape!=(2,) or not np.isfinite(hard).all() or not np.isfinite(gaps).all():
        raise ValueError("finite hard constraints and two actual pad gaps required")
    violation=max(0.,float(-hard.min()))
    contact=float(np.maximum(np.maximum(gaps+.1,-.3-gaps),0).max())
    return (violation>trust.HARD_NUMERIC_TOL_MM,violation,contact)

def evaluate(project,result):
    seed=previous.checked_json(PARENT,PARENT_SHA)
    if seed["status"]!="complete" or seed["geometry_state_attempts"]!=330:
        raise ValueError("exact completed previous trust-region evidence required")
    for name,sha in seed["implementation_sha256"].items():
        if digest(project/name)!=sha:raise ValueError("previous geometry implementation changed: "+name)
    nodes=np.asarray(seed["actual_parent_nodes_m"],dtype=float)
    pair=previous.load_pair_proposal()
    if seed["pair_sha256"]!=previous.PAIR_SHA or seed["reset_seed_sha256"]!=previous.SEED_SHA:
        raise ValueError("original pair/reset identity differs")
    result.update(parent_result_path=str(PARENT),parent_result_sha256=PARENT_SHA,
        original_parent_identity=seed["parent_identity"],actual_parent_nodes_m=nodes.tolist(),
        sides={},state_records=[],sources={},angles_rad=list(ANGLES),shared_state_cap=MAX_STATES,
        initial_pose_source="previous complete trust-region candidate; no old reset overwritten",
        root_rotation="world Z precomposition about each unchanged base origin, WXYZ",
        original_runtime_floor_m=previous.RUNTIME_FLOOR)
    budget=trust.StateBudget(result["state_records"],maximum=MAX_STATES)
    try:
        for side in ("left","right"):
            g=load_geometry(project,side=side)
            parent=seed["sides"][side]["candidate"]
            if parent["side"]!=side or g["source_sha256"]!=seed["sources"][side]["geometry_sha256"]:
                raise ValueError("independent actual side geometry required")
            grids,binding=previous.grids_for(g,previous.checked_json(previous.SDF_DIR/"result.json",previous.SDF_SHA))
            if binding!=seed["sources"][side]["sdf_bindings"]:
                raise ValueError("frozen independent side SDF bindings differ")
            result["sources"][side]=dict(geometry_sha256=g["source_sha256"],sdf_bindings=binding)
            record=dict(candidates=[],path=[],static_candidate_verified=False)
            result["sides"][side]=record;count=budget.count
            def measure(argument):
                c,fraction=argument
                return previous.previous.measure(g,c,grids,nodes,np.zeros(3),fraction)
            def state(candidate,fraction,phase):
                return budget.call(measure,(candidate,fraction),side=side,phase=phase,
                    static_yaw_world_rad=candidate["static_reset_yaw_world_rad"],path_fraction=float(fraction))
            for yaw in ANGLES:
                c=rotate_candidate(parent,yaw)
                first=state(c,0.,"grid_initial");last=state(c,1.,"grid_endpoint")
                labels,values=previous.initial_constraints(first)
                labels += ["endpoint:"+name for name in last["labels"]];values+=last["values"]
                hard=np.asarray(values)[~trust.contact_mask(labels)]
                gaps=[1000*last["node_sdf"][n]["minimum_gap_m"] for n in previous.previous.PADS]
                with _commit_row_without_alarm():
                    record["candidates"].append(dict(candidate=c,initial=first,endpoint=last,
                        selection_rank=list(rank(hard,gaps)),labels=labels,values=values))
            selected=min(record["candidates"],key=lambda r:(*r["selection_rank"],abs(r["candidate"]["static_reset_yaw_world_rad"])))
            c=selected["candidate"];record["selected_candidate"]=c
            for f in np.linspace(0,1,33):
                row=state(c,float(f),"selected_complete_path")
                with _commit_row_without_alarm():record["path"].append(row)
            path=record["path"];q=np.array(c["active_q_rad"]);d=np.array(c["active_delta_rad"])
            travel={**{"source:"+k:v for k,v in path_displacement_bound(g,q,d).items()},
                    **{"cooked:"+k:v for k,v in path_displacement_bound(g,q,d,cooked=True).items()}}
            table={k:previous.path_lower([r["table_m"][k] for r in path],v) for k,v in travel.items()}
            other={k:previous.path_lower([r["all_shape_rod_lower_m"][k] for r in path],v)
                for k,v in travel.items() if k.split(":")[1] not in previous.previous.PADS}
            contact=all(-.0003<=path[-1]["node_sdf"][n]["minimum_gap_m"]<=-.0001
                and bool(path[-1]["node_sdf"][n]["contact_predicate_nodes"]) for n in previous.previous.PADS)
            initial=min(path[0]["all_shape_rod_lower_m"].values())>=0
            hard=not selected["selection_rank"][0]
            zero=next(r for r in record["candidates"] if r["candidate"]["static_reset_yaw_world_rad"]==0)
            drift=max(abs(r["initial"]["table_m"][k]-zero["initial"]["table_m"][k])
                for r in record["candidates"] for k in zero["initial"]["table_m"])
            if drift>1e-10:raise ValueError("world-Z rotation unexpectedly changes static collider height")
            record.update(selected_rank=selected["selection_rank"],initial_all_shape_rod_clear=bool(initial),
                endpoint_both_pad_target_contact=bool(contact),hard_design_constraints_preserved=hard,
                continuous_table_lower_m=table,continuous_non_target_lower_m=other,
                point_travel_m=travel,initial_table_height_invariance_error_m=drift,
                geometry_states=budget.count-count,
                **previous.gate(min(table.values()),min(other.values()),initial and hard,contact))
        result["status"]="complete"
    finally:
        result["geometry_state_attempts"]=budget.count
        result["completed_geometry_states"]=sum(x["status"]=="complete" for x in result["state_records"])

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,training_updates=0,
        reset_proposal_created=False,native_qualification=False,teacher_admission=False,
        global_infeasibility_proven=False)
    def timeout(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
    try:evaluate(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,trust.BudgetLimit) as exc:result.update(status="diagnostic_error",error=str(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    with (args.output/"result.json").open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","geometry_state_attempts","error")}))
    return int(result["status"]=="diagnostic_error")

if __name__=="__main__":raise SystemExit(main())
