#!/usr/bin/env python3
"""Own-side LEFT Pinch endpoint proposals from the latest correlated native state.

The source held the left hand still. Reusing its errors during a proposed
closure is explicitly an empirical scenario check, never dynamic qualification.
"""
from copy import deepcopy
from pathlib import Path
import argparse,json,os,time
import numpy as np
from scripts import audit_dh116_tracking_pinch_contact as source
from scripts import prepare_dh116_tracking_pinch as plan
from scripts import solve_dh116_pinch_tracking_targets as observed
from scripts import solve_dh116_pinch_effective_targets as target
from scripts.solve_dh116_tripod_contact_closure import bounded_search,candidate_admission
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,active_limits,expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-left-pinch-correlated-endpoint-cpu-v1"
CONFIG=dict(version=VERSION,mode="cpu_only",side="left",maximum_evaluations=600,
    maximum_search_seconds=60.,search_physics_steps=[211,222,234],
    validation_physics_steps=list(range(211,235)),translation_scales_m=[.002]*3,
    rotation_scales_rad=[.03]*3,active_scales_rad=[.04]*3,
    native_execution=False,teacher_admission=False)

PRIOR=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/left-pinch-tracking-20261114/search-01/result.json")
NONLINEAR_CONFIG=dict(version="dh116-left-pinch-nonlinear-feasibility-cpu-v2",
    source_config=deepcopy(CONFIG),maximum_evaluations=600,maximum_search_seconds=60.,
    initial_candidate_source=str(PRIOR),initial_candidate_sha256="4c35c37916c7efd0a9fd470fd578195a43e589ac3119c59ade7bbb6c8508884a",
    optimizer="bounded_least_squares_constraint_violation",
    finite_difference_step=.002,native_execution=False,teacher_admission=False)

RESET_SOURCE=PRIOR.parent.parent/"linear-range-audit-01.json"
RESET_CONFIG={**CONFIG,"version":"dh116-left-pinch-independent-reset-cpu-v3",
    "translation_scales_m":[.004]*3,"rotation_scales_rad":[.06]*3,"active_scales_rad":[.08]*3,
    "initial_candidate_source":str(RESET_SOURCE),
    "initial_candidate_sha256":"f263d85299e95a8e554e3a021173491bea22d4c8b7e5ca27b16dec742b381803",
    "original_motion_unchanged":True,"scope":"reset_pose_proposals_only_not_action_amplitude"}

def nonlinear_search(query,start,*,maximum_evaluations,maximum_seconds):
    """Continuous residual search, independently checked using unmodified guards.

    Counts all finite-difference queries against one bound. Optimizer status
    never promotes a candidate. Historical minimax solver is unchanged.
    """
    from scipy.optimize import least_squares
    if type(maximum_evaluations) is not int or not 1<=maximum_evaluations<=600 or not 0<maximum_seconds<=60:
        raise ValueError("bounded CPU search")
    start=np.asarray(start,float)
    if start.ndim!=1 or not np.isfinite(start).all() or np.any(abs(start)>1):
        raise ValueError("finite bounded candidate")
    begun=time.monotonic();cache={};keys=None;best=None;count=0
    class Exhausted(Exception):pass
    def evaluate(x):
        nonlocal keys,best,count
        key=np.asarray(x,dtype="<f8").tobytes()
        if key in cache:return cache[key]
        if count>=maximum_evaluations or time.monotonic()-begun>=maximum_seconds:raise Exhausted()
        row=deepcopy(query(x));c=row["constraints"]
        if not c or not np.isfinite(list(c.values())).all():raise ValueError("finite complete guards")
        if keys is None:keys=tuple(c)
        elif keys!=tuple(c):raise ValueError("constraint identity changed")
        count+=1;row["admission"]=candidate_admission(c)
        vals=np.array(list(c.values()));viol=np.minimum(vals,0)
        row["minimum_margin_m"]=float(vals.min());row["violation_score"]=float(viol@viol)
        rank=(max(0.,-row["minimum_margin_m"]),row["violation_score"],float(x@x))
        if best is None or rank<best[0]:best=(rank,row)
        cache[key]=row;return row
    def residual(x):
        # 1 micrometre interior goal is only a planning reserve, never a new
        # physics criterion. All original guards are retained in evaluate().
        return np.minimum(np.array(list(evaluate(x)["constraints"].values()))-.000001,0.)/.001
    def jacobian(x):
        columns=[]
        for j in range(len(x)):
            a=x.copy();b=x.copy();a[j]=max(-1.,x[j]-.002);b[j]=min(1.,x[j]+.002)
            columns.append((residual(b)-residual(a))/(b[j]-a[j]))
        return np.stack(columns,axis=1)
    termination="budget_exhausted";solver={}
    try:
        evaluate(start)
        result=least_squares(residual,start,jac=jacobian,bounds=(-1.,1.),max_nfev=80,
            ftol=1e-10,xtol=1e-10,gtol=1e-10)
        evaluate(result.x);termination="nonlinear_solver_returned"
        solver=dict(status=int(result.status),message=str(result.message),nfev=result.nfev)
    except Exhausted:pass
    if best is None:raise ValueError("budget exhausted before candidate")
    return dict(best=best[1],evaluations=count,termination=termination,solver=solver,
        elapsed_s=time.monotonic()-begun,optimizer_status_is_not_qualification=True)

def load_packet():
    r=json.loads(source.read_locked(source.RECEIPT,source.RECEIPT_SHA))
    a=json.loads(source.read_locked(source.RECEIPT.parent/"tracking-pinch-post-exit-audit.json",source.AUDIT_SHA))["evidence"]
    if (r.get("native_qualification") is not True or r.get("qualification_scope")!=source.SCOPE
        or r.get("outer_exit_code")!=0 or not 0<r.get("duration_seconds",float("inf"))<=180
        or not a.get("full_post_exit_audit_passed") or a.get("qualification_scope")!=source.SCOPE):
        raise ValueError("frozen actual bounded scope and complete audit required")
    raw=source.read_locked(Path(r["output"])/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"])
    if not raw.endswith(b"\n"):raise ValueError("complete native lines required")
    rows=[decode_owned(e["payload"])["exchange"] for e in map(json.loads,raw.splitlines()) if e.get("operation")=="physical"]
    observed.tracking.contact_summary(rows)
    pair=plan.load_pair(ROOT);left=pair["left"]["candidate"]
    if np.any(plan.action_schedule().numpy()[:,:6]):raise ValueError("source must be held left, not claimed closure")
    executed=dict(root_pose_wxyz=pair["reset_seed"]["reset_root_pose_w"][0],
        q_rad=[pair["reset_seed"]["joint_positions"][0][JOINTS.index(n)] for n in ACTIVE])
    delta=np.asarray(left["active_delta_rad"],float)
    anchor=dict(root_pose_wxyz=executed["root_pose_wxyz"],q_rad=(np.asarray(executed["q_rad"])+delta).tolist())
    expand(anchor["q_rad"])
    scenarios=[]
    for x in rows[-24:]:
        g=x["closure_geometry"]["raw"];ident=g["identity"];j=g["joint_readback"];h=g["hands"]["left"];q=j["hands"]["left"]
        if j["identity"]!=[ident]:raise ValueError("same left hand/joint/rod identity")
        pose=h["link_poses_xyzw"][0,h["body_names"].index("base_link")]
        actual=np.r_[pose[:3],pose[6],pose[3:6]];active=observed.tracking.active_values(q)
        nodes=np.asarray(g["rod_nodes_m"],float)
        if nodes.shape!=(41,3) or not np.isfinite(nodes).all():raise ValueError("complete measured rod")
        full=np.array([q["q"][0,q["joint_names"].index(n)] for n in JOINTS])
        scenarios.append(dict(identity=deepcopy(ident),joint_identity=deepcopy(j["identity"][0]),
            rod_identity=deepcopy(ident),source_hand="left",empirical_only=True,
            root_translation_error_m=(actual[:3]-executed["root_pose_wxyz"][:3]).tolist(),
            world_rotation_error_xyzw=(observed.rotation(actual)*observed.rotation(executed["root_pose_wxyz"]).inv()).as_quat().tolist(),
            active_error_rad=(active-executed["q_rad"]).tolist(),rod_nodes_m=nodes.tolist(),
            measured_root_wxyz=actual.tolist(),measured_active_q_rad=active.tolist(),
            observed_coupling_residual_rad=(full-expand(active)).tolist()))
    if [s["identity"]["physics_step"] for s in scenarios]!=CONFIG["validation_physics_steps"]:
        raise ValueError("complete terminal correlated snapshots")
    return dict(side="left",executed_anchor=executed,anchor=anchor,scenarios=scenarios,
        original_joint_delta_rad=delta.tolist(),original_candidate=left,executed_left_actions_zero=True,
        source_sha256=dict(receipt=source.RECEIPT_SHA,audit=source.AUDIT_SHA,ledger=r["artifact_sha256"]["physical.jsonl"]),
        interpretation="observed held-left errors are unqualified extrapolations for any new left closure",
        native_ready=False,teacher_admission=False)

def start_for(packet,endpoint):
    return target.original_path(endpoint,packet["original_candidate"],0.)[0]

def decode_candidate(packet,x,*,config=None):
    config=CONFIG if config is None else config
    if config not in (CONFIG,RESET_CONFIG):raise ValueError("independent frozen reset proposal limits")
    x=np.asarray(x,float)
    if x.shape!=(9,) or not np.isfinite(x).all() or np.any(abs(x)>1):
        raise ValueError("bounded nine-variable candidate")
    anchor=packet["anchor"]
    if config==CONFIG:p,_=target.decode(anchor,np.r_[x[:6],np.zeros(3)])
    else:
        p=np.array(anchor["root_pose_wxyz"],float);p[:3]+=x[:3]*config["translation_scales_m"]
        rot=observed.Rotation.from_rotvec(x[3:6]*config["rotation_scales_rad"])*observed.rotation(p)
        p[3:]=np.roll(rot.as_quat(),1)
    q=np.asarray(anchor["q_rad"],float).copy();lo,hi=active_limits()
    errors=np.array([np.zeros(6),*[s["active_error_rad"] for s in packet["scenarios"]]])
    delta=np.asarray(packet["original_joint_delta_rad"])
    lower=np.maximum(lo-errors.min(axis=0),lo+delta)
    upper=np.minimum(hi-errors.max(axis=0),hi+delta)
    slack=np.minimum(q-lower,upper-q)
    if np.any(slack<0):raise ValueError("anchor outside own coupled scenario bounds")
    q[:3]+=x[6:]*np.minimum(config["active_scales_rad"],.99*slack[:3])
    expand(q);end=dict(root_pose_wxyz=p.tolist(),q_rad=q.tolist());start_for(packet,end)
    return end

class Geometry:
    def __init__(self,packet,*,config=None):
        if packet["side"]!="left":raise ValueError("own left side only")
        self.packet=packet;self.side="left";self.config=CONFIG if config is None else config
        self.g=target.surface.base.load_geometry(ROOT,side=self.side)
        self.grids,self.bindings=target.surface.base.grids_for(self.g,
            target.surface.base.checked_json(target.surface.base.SDF_DIR/"result.json",target.surface.base.SDF_SHA))
        self.atlas={n:target.surface._patch(self.g["source_colliders"][n]["vertices"],self.side,n) for n in target.surface.PADS}
        self.nominal_nodes=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)

    def evaluate(self,x,scenarios):
        end=decode_candidate(self.packet,x,config=self.config)
        nominal=target.evaluate(self.g,self.grids,self.atlas,end,np.zeros(9),self.nominal_nodes)
        nominal["strict_planner_constraints"]=nominal["constraints"]
        nominal["constraints"]=observed.physical_target_constraints(nominal)
        constraints={"nominal:"+k:v for k,v in nominal["constraints"].items()};out={}
        for s in scenarios:
            if s["source_hand"]!="left":raise ValueError("right residual cannot replace left measurement")
            p,q=observed.apply_scenario(end,s);key=str(s["identity"]["physics_step"])
            row=target.evaluate(self.g,self.grids,self.atlas,dict(root_pose_wxyz=p,q_rad=q),np.zeros(9),np.asarray(s["rod_nodes_m"]))
            row["strict_planner_constraints"]=row["constraints"];row["constraints"]=observed.physical_target_constraints(row)
            constraints.update({key+":"+k:v for k,v in row["constraints"].items()});out[key]=row
        return dict(x=np.asarray(x).tolist(),endpoint=end,nominal=nominal,scenarios=out,constraints=constraints,
            admission=candidate_admission(constraints),actual_contact_observed=False,native_ready=False,teacher_admission=False)

def run(output,config):
    if config not in (CONFIG,NONLINEAR_CONFIG,RESET_CONFIG):raise ValueError("frozen CPU-only bounded search")
    packet=load_packet();g=Geometry(packet,config=RESET_CONFIG if config==RESET_CONFIG else CONFIG)
    baseline=g.evaluate(np.zeros(9),packet["scenarios"])
    selected=[s for s in packet["scenarios"] if s["identity"]["physics_step"] in CONFIG["search_physics_steps"]]
    initial=np.zeros(9)
    if config==NONLINEAR_CONFIG:
        prior=decode_owned(json.loads(source.read_locked(PRIOR,config["initial_candidate_sha256"])))
        if prior["config"]!=CONFIG or prior["source"]["source_sha256"]!=packet["source_sha256"]:
            raise ValueError("same frozen native source and failed CPU proposal required")
        if prior["qualification"]["empirical_endpoint_passed"]:
            raise ValueError("do not repeat an already passed search")
        initial=np.array(prior["search"]["best"]["x"])
    if config==RESET_CONFIG:
        prior=decode_owned(json.loads(source.read_locked(RESET_SOURCE,config["initial_candidate_sha256"])))
        if prior[-1]["trust"]!=2. or len(prior[-1]["x"])!=9:
            raise ValueError("frozen full-range linear audit required")
        initial=np.asarray(prior[-1]["x"])/2
    with (output/"search.jsonl").open("x") as f:
        def query(x):
            row=g.evaluate(x,selected);f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");f.flush();return row
        search=(nonlinear_search if config==NONLINEAR_CONFIG else bounded_search)(query,initial,maximum_evaluations=config["maximum_evaluations"],maximum_seconds=config["maximum_search_seconds"])
        f.flush();os.fsync(f.fileno())
    validation=g.evaluate(np.asarray(search["best"]["x"]),packet["scenarios"])
    return dict(version=config["version"],status="complete",config=config,source=packet,baseline=baseline,search=search,
        all_observed_validation=validation,qualification=observed.admission(validation["nominal"],list(validation["scenarios"].values())),
        geometry_source_sha256=g.g["source_sha256"],sdf_bindings=g.bindings,
        native_ready=False,teacher_admission=False,continuous_path_proved=False,new_physics_steps=0,ppo_updates=0)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--config",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:r=run(a.output,json.loads(a.config.read_bytes()))
    except Exception as e:r=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),native_ready=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0)
    r.update(elapsed_s=time.monotonic()-start,script_sha256=source.sha(__file__),config_sha256=source.sha(a.config))
    with (a.output/"result.json").open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","qualification","native_ready")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
