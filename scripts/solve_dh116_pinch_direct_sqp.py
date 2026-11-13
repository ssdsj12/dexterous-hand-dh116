#!/usr/bin/env python3
"""Finite direct nonlinear Pinch reset solve; unchanged FK/SDF and safety contracts."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scripts import solve_dh116_pinch_coupled_lateral as coupled
from scripts.audit_dh116_pinch_initial_pose import finite_array,digest,load_geometry,path_displacement_bound
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm

VERSION="dh116-pinch-direct-nonlinear-reset-v1"
PARENT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-coupled-lateral-20261113-MIOUYp/run01/result.json")
PARENT_SHA="d6032b457debc9f1962dfe556a1367e1fd38afc463d1f901292a0fa46cd9f157"
MAX_QUERIES=200
MAX_ITERATIONS=16
MAX_STATES=2*(2*MAX_QUERIES+33)
EPS=.001
previous=coupled.previous
trust=coupled.trust

class QueryBudget(RuntimeError):pass

class StateBudget:
    def __init__(self,records,maximum=MAX_STATES):
        if type(maximum) is not int or not 1<=maximum<=MAX_STATES:raise ValueError("finite complete-state cap required")
        self.records,self.maximum,self.count=records,maximum,0
    def call(self,fn,value,**metadata):
        if self.count>=self.maximum:raise QueryBudget("complete geometry-state cap")
        self.count+=1;row=dict(attempt=self.count,status="started",**metadata)
        with _commit_row_without_alarm():self.records.append(row)
        try:
            result=fn(value)
            with _commit_row_without_alarm():row.update(status="complete",measurement=result)
            return result
        except BaseException as error:
            with _commit_row_without_alarm():row.update(status="interrupted" if isinstance(error,BudgetExpired) else "error",error=type(error).__name__+": "+str(error))
            raise

class Queries:
    def __init__(self,query,records,maximum=MAX_QUERIES):
        if type(maximum) is not int or not 1<=maximum<=MAX_QUERIES:raise ValueError("1..200 unique full queries required")
        self.query,self.records,self.maximum=query,records,maximum
        self.cache={};self.labels=None
    def __call__(self,x):
        x=finite_array(x,(7,));key=x.tobytes()
        if key in self.cache:return self.cache[key]
        if len(self.records)>=self.maximum:raise QueryBudget("per-side full-query cap")
        row=dict(query=len(self.records)+1,x=x.tolist(),status="started")
        with _commit_row_without_alarm():self.records.append(row)
        try:
            actual=self.query(x.copy())
            if self.labels is None:self.labels=actual["labels"]
            values=trust.values(actual,self.labels)
            with _commit_row_without_alarm():row.update(status="complete",labels=self.labels,values=values.tolist())
            self.cache[key]=actual
            return actual
        except BaseException as error:
            with _commit_row_without_alarm():row.update(status="interrupted" if isinstance(error,BudgetExpired) else "error",error=type(error).__name__+": "+str(error))
            raise

def solve(query,initial,lower,upper,trace,*,maximum=MAX_QUERIES):
    seed,lower,upper=[finite_array(v,(7,)) for v in (initial,lower,upper)]
    if np.any(seed<lower) or np.any(seed>upper):raise ValueError("initial coordinate outside original absolute box")
    trace.update(queries=[],iterates=[],maximum_queries=maximum,maximum_iterations=MAX_ITERATIONS,finite_difference_step=EPS)
    q=Queries(query,trace["queries"],maximum=maximum)
    first=q(seed);labels=first["labels"];soft=trust.contact_mask(labels);hard=~soft
    if not soft.any() or not hard.any():raise ValueError("separate target-contact and hard physical-design constraints required")
    def data(x):
        x=finite_array(x,(7,))
        if np.any(x<lower-1e-12) or np.any(x>upper+1e-12):raise ValueError("optimizer queried outside original bounded reset")
        return trust.values(q(x),labels)
    def assess(x):
        c=data(x)
        return dict(x=np.asarray(x).tolist(),values=c.tolist(),
            feasible=bool(np.all(c[hard]>=-trust.HARD_NUMERIC_TOL_MM)),
            contact_deficit_mm=max(0.,float(-c[soft].min())),
            contact_squared_deficit=float(np.dot(np.minimum(c[soft],0),np.minimum(c[soft],0))))
    best=assess(seed)
    if not best["feasible"]:raise ValueError("independently verified seed is not hard-feasible")
    trace["selected"]=best
    def objective(x):
        c=data(x);d=np.minimum(c[soft],0)
        return float(d@d+1e-8*np.sum((x-seed)**2))
    def constraints(x):
        # Small stricter design reserve counters finite-candidate roundoff;
        # it does not change the runtime protection or final feasibility test.
        return data(x)[hard]-.002
    def retain(x):
        nonlocal best
        actual=assess(x)
        with _commit_row_without_alarm():
            trace["iterates"].append(actual)
            if actual["feasible"] and actual["contact_squared_deficit"]<best["contact_squared_deficit"]:
                best=actual;trace["selected"]=best
    try:
        result=minimize(objective,seed.copy(),method="SLSQP",bounds=list(zip(lower,upper)),
            constraints=[dict(type="ineq",fun=constraints)],callback=retain,
            options=dict(maxiter=MAX_ITERATIONS,eps=EPS,ftol=1e-10,disp=False))
        retain(result.x)
        trace.update(status="optimizer_returned",optimizer_success=bool(result.success),
            optimizer_status=int(result.status),optimizer_message=str(result.message),iterations=int(result.nit))
    except QueryBudget:
        trace.update(status="query_budget_exhausted",optimizer_success=False)
    return dict(status=trace["status"],selected_x=best["x"],selected_feasible=best["feasible"],
        selected_contact_deficit_mm=best["contact_deficit_mm"],unique_queries=len(trace["queries"]),
        global_infeasibility_proven=False,native_qualification=False)

def run(project,result):
    old=previous.checked_json(PARENT,PARENT_SHA)
    if old["status"]!="complete" or old["geometry_state_attempts"]!=306:
        raise ValueError("exact complete coupled parent required")
    for name,sha in old["implementation_sha256"].items():
        if digest(project/name)!=sha:raise ValueError("original coupled implementation changed")
    original=previous.checked_json(coupled.PARENT,coupled.PARENT_SHA)
    for name,sha in original["implementation_sha256"].items():
        if digest(project/name)!=sha:raise ValueError("original physical geometry implementation changed")
    pair=previous.load_pair_proposal();nodes=finite_array(old["actual_parent_nodes_m"],(41,3))
    result.update(parent_result_path=str(PARENT),parent_result_sha256=PARENT_SHA,
        original_parent_identity=old["original_parent_identity"],actual_parent_nodes_m=nodes.tolist(),
        pair_sha256=previous.PAIR_SHA,reset_seed_sha256=previous.SEED_SHA,
        state_records=[],sides={},sources={},shared_state_cap=MAX_STATES,
        runtime_floor_m=previous.RUNTIME_FLOOR,original_absolute_variable_box=True,
        implementation_sha256={"scripts/solve_dh116_pinch_direct_sqp.py":digest(project/"scripts/solve_dh116_pinch_direct_sqp.py")})
    budget=StateBudget(result["state_records"])
    try:
        for side in ("left","right"):
            g=load_geometry(project,side=side)
            if g["source_sha256"]!=old["sources"][side]["geometry_sha256"]:raise ValueError("independent side geometry differs")
            grids,binding=previous.grids_for(g,previous.checked_json(previous.SDF_DIR/"result.json",previous.SDF_SHA))
            if binding!=old["sources"][side]["sdf_bindings"]:raise ValueError("locked independent side SDF differs")
            result["sources"][side]=dict(geometry_sha256=g["source_sha256"],sdf_bindings=binding)
            parent=dict(pair[side]["candidate"],side=side)
            seed=np.array(old["sides"][side]["candidate"]["normalized_coupled_direction"])
            if coupled.decode(parent,seed)!=old["sides"][side]["candidate"]:raise ValueError("seed not original absolute coordinates")
            record=dict(solver={},path=[],static_candidate_verified=False);result["sides"][side]=record;start=budget.count
            def measure(arg):
                x,f=arg
                return previous.previous.measure(g,coupled.decode(parent,x),grids,nodes,np.zeros(3),f)
            def state(x,f,phase):
                return budget.call(measure,(x,f),side=side,phase=phase,parameter_x=np.asarray(x).tolist(),path_fraction=float(f))
            def query(x):
                first=state(x,0.,"optimizer_initial");last=state(x,1.,"optimizer_endpoint")
                labels,values=previous.initial_constraints(first)
                return dict(labels=labels+["endpoint:"+n for n in last["labels"]],values=values+last["values"])
            outcome=solve(query,seed,*coupled.bounds(parent),record["solver"]);record.update(outcome)
            x=np.array(outcome["selected_x"]);c=coupled.decode(parent,x)
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
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
