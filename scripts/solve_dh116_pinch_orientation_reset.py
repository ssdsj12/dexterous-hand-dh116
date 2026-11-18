#!/usr/bin/env python3
"""Independent bounded world-tilt Pinch reset experiment; no native execution."""
import argparse,hashlib,json,signal,subprocess,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts import solve_dh116_pinch_direct_sqp as sqp
from scripts.audit_dh116_pinch_initial_pose import finite_array,digest,load_geometry,path_displacement_bound
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm

coupled=sqp.coupled
previous=sqp.previous
VERSION="dh116-pinch-world-tilt-reset-v1"
PARENT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-direct-sqp-20261113-MS6xpU/run01/result.json")
PARENT_SHA="ef6e8d68f036f0a3a87389b6c3227cdaf403eae42f7e95d0b509378a8ae8d1c7"
PARENT_COMMIT="fd9e632f9a82dbb2a8ff2a1afea672bb172bff48"

def bounds(parent):
    lo,hi=coupled.bounds(parent)
    return np.r_[lo,-1.,-1.],np.r_[hi,1.,1.]

def decode(parent,x):
    x=finite_array(x,(9,));lo,hi=bounds(parent)
    if np.any(x<lo) or np.any(x>hi):raise ValueError("unchanged seven bounds and +/-0.03 rad world X/Y tilt required")
    c=coupled.decode(parent,x[:7])
    if np.any(x[7:]):
        pose=np.asarray(c["base_pose_wxyz"]).copy()
        old=pose[[4,5,6,3]]
        new=(Rotation.from_rotvec([.03*x[7],.03*x[8],0])*Rotation.from_quat(old)).as_quat()
        if new@old<0:new=-new
        pose[3:]=new[[3,0,1,2]];c["base_pose_wxyz"]=pose.tolist()
    c.update(normalized_orientation_direction=x.tolist(),static_world_tilt_rotvec_rad=[.03*float(x[7]),.03*float(x[8]),0.],
        orientation_contract="R=Exp(world_xy_tilt) * R_world_z_yaw * R_original; unchanged translated base origin",
        wrist_command_modified=False,native_qualification=False,teacher_admission=False)
    return c

def verify_historical_source(project,commit,name,expected):
    data=subprocess.check_output(["git","-c","safe.directory="+str(project),"-C",str(project),"show",commit+":"+name])
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError("immutable historical source does not match archived evidence")
    return True

def run(project,result):
    old=previous.checked_json(PARENT,PARENT_SHA)
    if old["status"]!="complete" or old["geometry_state_attempts"]!=808:
        raise ValueError("exact completed direct-SQP parent required")
    # Generic solver now accepts explicit 9D; old 7D source bytes are retained
    # in the immutable commit rather than falsely compared to its new version.
    for name,sha in old["implementation_sha256"].items():
        verify_historical_source(project,PARENT_COMMIT,name,sha)
    original=previous.checked_json(coupled.PARENT,coupled.PARENT_SHA)
    for name,sha in original["implementation_sha256"].items():
        if digest(project/name)!=sha:raise ValueError("original physical geometry implementation changed")
    pair=previous.load_pair_proposal();nodes=finite_array(old["actual_parent_nodes_m"],(41,3))
    result.update(parent_result_path=str(PARENT),parent_result_sha256=PARENT_SHA,parent_code_commit=PARENT_COMMIT,
        original_parent_identity=old["original_parent_identity"],actual_parent_nodes_m=nodes.tolist(),
        pair_sha256=previous.PAIR_SHA,reset_seed_sha256=previous.SEED_SHA,
        state_records=[],sides={},sources={},shared_state_cap=sqp.MAX_STATES,
        runtime_floor_m=previous.RUNTIME_FLOOR,maximum_queries_per_side=sqp.MAX_QUERIES,
        maximum_iterations_per_side=sqp.MAX_ITERATIONS,original_absolute_variable_box=True,
        implementation_sha256={f:digest(project/f) for f in (
            "scripts/solve_dh116_pinch_orientation_reset.py","scripts/solve_dh116_pinch_direct_sqp.py",
            "scripts/solve_dh116_pinch_coupled_lateral.py")})
    budget=sqp.StateBudget(result["state_records"])
    try:
        for side in ("left","right"):
            g=load_geometry(project,side=side)
            if g["source_sha256"]!=old["sources"][side]["geometry_sha256"]:raise ValueError("independent actual hand geometry changed")
            grids,binding=previous.grids_for(g,previous.checked_json(previous.SDF_DIR/"result.json",previous.SDF_SHA))
            if binding!=old["sources"][side]["sdf_bindings"]:raise ValueError("independent locked SDF changed")
            result["sources"][side]=dict(geometry_sha256=g["source_sha256"],sdf_bindings=binding)
            parent=dict(pair[side]["candidate"],side=side)
            seed=np.r_[old["sides"][side]["candidate"]["normalized_coupled_direction"],0.,0.]
            if coupled.decode(parent,seed[:7])!=old["sides"][side]["candidate"]:raise ValueError("seed not on original absolute bounds")
            record=dict(solver={},path=[],static_candidate_verified=False);result["sides"][side]=record;start=budget.count
            def measure(arg):
                x,f=arg
                return previous.previous.measure(g,decode(parent,x),grids,nodes,np.zeros(3),f)
            def state(x,f,phase):
                return budget.call(measure,(x,f),side=side,phase=phase,parameter_x=np.asarray(x).tolist(),path_fraction=float(f))
            def query(x):
                first=state(x,0.,"optimizer_initial");last=state(x,1.,"optimizer_endpoint")
                labels,values=previous.initial_constraints(first)
                return dict(labels=labels+["endpoint:"+n for n in last["labels"]],values=values+last["values"])
            outcome=sqp.solve(query,seed,*bounds(parent),record["solver"],dimension=9);record.update(outcome)
            x=np.array(outcome["selected_x"]);c=decode(parent,x)
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
        result["completed_geometry_states"]=sum(x["status"]=="complete" for x in result["state_records"])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,training_updates=0,
        reset_proposal_created=False,native_qualification=False,teacher_admission=False)
    def timeout(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,60)
    try:run(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except Exception as error:result.update(status="diagnostic_error",error=type(error).__name__+": "+str(error))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","geometry_state_attempts","error")}))
    return int(result["status"]=="diagnostic_error")

if __name__=="__main__":raise SystemExit(main())
