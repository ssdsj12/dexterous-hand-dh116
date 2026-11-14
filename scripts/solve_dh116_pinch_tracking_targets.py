#!/usr/bin/env python3
"""CPU proposals checked against correlated native tracking/rod snapshots.

Observed discrepancies are scenario tests, not calibrated dynamics or native
contact evidence. Original pad regions, collision and actuator limits stay fixed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts import audit_dh116_effective_right_tracking as tracking
from scripts import prepare_dh116_effective_right_pinch as plan
from scripts import solve_dh116_pinch_effective_targets as target
from scripts.solve_dh116_tripod_contact_closure import bounded_search,candidate_admission
from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand,ACTIVE,JOINTS
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-pinch-correlated-tracking-targets-cpu-v1"
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-approach-r0-native/96fc30001e0d26e7421b683fc0e18ae9240054e3b6fcffb3e010aa6d6d0d70b3/attempt-wy27jh8g/receipt.json")
RECEIPT_SHA="9b0a8eb4f37ce20e27d5054a4d95043e21fb99febb00128a878fffa739608ee6"
SCOPE="actual_reset_and_same_clock_measured_approach_R0_234_steps_only"
CONFIG=dict(version=VERSION,mode="cpu_only",search_physics_steps=[211,222,234],
    validation_physics_steps=list(range(211,235)),maximum_evaluations=600,
    maximum_seconds=60.,translation_scales_m=[.002]*3,rotation_scales_rad=[.03]*3,
    active_scales_rad=[.04]*3,native_execution=False,teacher_admission=False)

PHYSICAL_CONFIG={**CONFIG,"version":"dh116-pinch-correlated-tracking-targets-cpu-v2",
    "planning_gate":"original_pad_acceptance_and_touch_feasibility",
    "contact_target_gap_m":0.,"ellipse_planning_max":1.}

REFINED_CONFIG={**PHYSICAL_CONFIG,"version":"dh116-pinch-correlated-tracking-targets-cpu-v3",
    "search_physics_steps":[211,212,214,222,234],
    "initial_candidate_source":"/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-tracking-target-20261114/search-02/result.json",
    "initial_candidate_sha256":"c6ddc92a3748572915f7a1da35b25d64b07805c545cdf59e6f5069ac698caa3a"}

def refinement_steps(prior):
    steps=set(CONFIG["search_physics_steps"])
    for key,row in prior["all_observed_validation"]["scenarios"].items():
        values=list(row["constraints"].values())
        if not values or not np.isfinite(values).all():raise ValueError("finite complete prior constraints")
        if min(values)<0:steps.add(int(key))
    return sorted(steps)

def physical_target_constraints(row):
    """Separate optional target depth/ellipse reserve from unchanged guards.

    This is only contact *feasibility*: no force/support/lift qualification.
    Table, non-target separation, depth floor, normal and plane gates remain.
    """
    out=dict(row["constraints"])
    for name,pad in row["pads"].items():
        out["sdf_target:"+name]=-pad["sdf_gap_m"]
        out["region:"+name]=(1.-pad["ellipse_value"])*.001
    return out

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rotation(p):
    p=np.asarray(p,float)
    if p.shape!=(7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1)>1e-5:
        raise ValueError("finite normalized wxyz root pose required")
    return Rotation.from_quat(p[[4,5,6,3]])

def load_observed_scenarios(*,expected_sha=RECEIPT_SHA):
    if expected_sha!=RECEIPT_SHA or digest(RECEIPT)!=expected_sha:
        raise ValueError("frozen native receipt SHA required")
    r=json.loads(RECEIPT.read_bytes())
    if (r.get("native_qualification") is not True or r.get("qualification_scope")!=SCOPE
        or r.get("outer_exit_code")!=0 or not 0<r.get("duration_seconds",float("inf"))<=180):
        raise ValueError("bounded natural-exit scope required")
    ledger=Path(r["output"])/"physical.jsonl"
    if digest(ledger)!=r["artifact_sha256"]["physical.jsonl"]:
        raise ValueError("native ledger SHA mismatch")
    rows=[]
    for line in ledger.open():
        if not line.endswith("\n"):raise ValueError("incomplete physical JSONL")
        item=json.loads(line)
        if item.get("operation")=="physical":
            rows.append(decode_owned(item["payload"])["exchange"])
    tracking.contact_summary(rows)
    original=plan.load_pair(ROOT);expected=plan.expected_boundary(original,232)
    anchor=dict(root_pose_wxyz=expected["root_poses_w"][1].tolist(),
        q_rad=expected["right_q_rad"].tolist())
    scenarios=[]
    for x in rows[-24:]:
        raw=x["closure_geometry"]["raw"];ident=raw["identity"]
        joint=raw["joint_readback"];h=raw["hands"]["right"];j=joint["hands"]["right"]
        if joint["identity"]!=[ident] or ident["physics_step"]!=x["physics_step"]:
            raise ValueError("actual root/joint/rod clock mismatch")
        pose=h["link_poses_xyzw"][0,h["body_names"].index("base_link")]
        actual=np.r_[pose[:3],pose[6],pose[3:6]]
        q=tracking.active_values(j);nodes=np.asarray(raw["rod_nodes_m"],float)
        if nodes.shape!=(41,3) or not np.isfinite(nodes).all():raise ValueError("complete actual rod nodes required")
        full=np.array([j["q"][0,j["joint_names"].index(n)] for n in JOINTS])
        expected_here=plan.expected_boundary(original,ident["physics_step"]-2)
        if not np.allclose(expected_here["right_q_rad"],anchor["q_rad"],atol=0,rtol=0):
            raise ValueError("only post-closure fixed command snapshots allowed")
        scenarios.append(dict(identity=deepcopy(ident),joint_identity=deepcopy(joint["identity"][0]),
            rod_identity=deepcopy(ident),empirical_only=True,
            root_translation_error_m=(actual[:3]-anchor["root_pose_wxyz"][:3]).tolist(),
            world_rotation_error_xyzw=(rotation(actual)*rotation(anchor["root_pose_wxyz"]).inv()).as_quat().tolist(),
            active_error_rad=(q-anchor["q_rad"]).tolist(),rod_nodes_m=nodes.tolist(),
            measured_root_wxyz=actual.tolist(),measured_active_q_rad=q.tolist(),
            observed_coupling_residual_rad=(full-expand(q)).tolist()))
    if [s["identity"]["physics_step"] for s in scenarios]!=CONFIG["validation_physics_steps"]:
        raise ValueError("complete 24 terminal hold snapshots required")
    return dict(anchor=anchor,scenarios=scenarios,original_joint_delta_rad=original["right"]["candidate"]["active_delta_rad"],
        source_sha256=dict(receipt=expected_sha,ledger=digest(ledger)),
        empirical_only=True,native_ready=False,teacher_admission=False)

def apply_scenario(endpoint,scenario):
    if (scenario["identity"]!=scenario["joint_identity"] or
        scenario["identity"]!=scenario["rod_identity"] or
        scenario["identity"].get("clock_valid") is not True):
        raise ValueError("same root/joint/rod clock required")
    root=np.array(endpoint["root_pose_wxyz"],float)
    error=np.asarray(scenario["root_translation_error_m"],float)
    dq=np.asarray(scenario["active_error_rad"],float)
    dr=np.asarray(scenario["world_rotation_error_xyzw"],float)
    if error.shape!=(3,) or dq.shape!=(6,) or dr.shape!=(4,) or not np.isfinite(np.r_[error,dq,dr]).all():
        raise ValueError("finite separate world and joint discrepancies required")
    if abs(np.linalg.norm(dr)-1)>1e-5:raise ValueError("unit world rotation discrepancy required")
    rot=Rotation.from_quat(dr)*rotation(root)
    root[:3]+=error;root[3:]=np.roll(rot.as_quat(),1)
    q=np.asarray(endpoint["q_rad"],float)+dq
    expand(q)
    return root,q

def original_start(endpoint):
    original=plan.load_pair(ROOT)["right"]["candidate"]
    return target.original_path(endpoint,original,0.)[0]

def decode_candidate(anchor,x,scenarios):
    x=np.asarray(x,float)
    if x.shape!=(9,) or not np.isfinite(x).all() or np.any(abs(x)>1):
        raise ValueError("bounded nine-variable independent reset proposal required")
    if not scenarios:raise ValueError("observed scenarios required")
    p,_=target.decode(anchor,np.r_[x[:6],np.zeros(3)])
    q=np.asarray(anchor["q_rad"],float).copy();lo,hi=active_limits()
    delta=np.asarray(plan.load_pair(ROOT)["right"]["candidate"]["active_delta_rad"])
    errors=np.array([np.zeros(6),*[s["active_error_rad"] for s in scenarios]])
    lower=np.maximum(lo-errors.min(axis=0),lo+delta)
    upper=np.minimum(hi-errors.max(axis=0),hi+delta)
    slack=np.minimum(q-lower,upper-q)
    if np.any(slack<0):raise ValueError("anchor outside original coupled scenario limits")
    q[:3]+=x[6:]*np.minimum(CONFIG["active_scales_rad"],.99*slack[:3])
    expand(q)
    out=dict(root_pose_wxyz=p.tolist(),q_rad=q.tolist())
    original_start(out)
    return out

def admission(nominal,observed):
    passed=bool(observed) and all(candidate_admission(r["constraints"])["static_constraints_passed"]
        for r in [nominal,*observed])
    return dict(empirical_endpoint_passed=passed,continuous_path_proved=False,
        unobserved_dynamics_qualified=False,native_ready=False,teacher_admission=False,
        actual_contact_observed=False)

class Geometry:
    def __init__(self,packet,*,physical_gate=False):
        self.physical_gate=physical_gate
        self.packet=packet;self.g=target.surface.base.load_geometry(ROOT,side="right")
        self.grids,self.bindings=target.surface.base.grids_for(self.g,
            target.surface.base.checked_json(target.surface.base.SDF_DIR/"result.json",target.surface.base.SDF_SHA))
        self.atlas={n:target.surface._patch(self.g["source_colliders"][n]["vertices"],"right",n) for n in target.surface.PADS}
        self.nominal_nodes=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)

    def evaluate(self,x,scenarios):
        end=decode_candidate(self.packet["anchor"],x,self.packet["scenarios"])
        nominal=target.evaluate(self.g,self.grids,self.atlas,end,np.zeros(9),self.nominal_nodes)
        if self.physical_gate:
            nominal["strict_planner_constraints"]=nominal["constraints"]
            nominal["constraints"]=physical_target_constraints(nominal)
        constraints={"nominal:"+k:v for k,v in nominal["constraints"].items()};out={}
        for s in scenarios:
            p,q=apply_scenario(end,s);step=str(s["identity"]["physics_step"])
            row=target.evaluate(self.g,self.grids,self.atlas,dict(root_pose_wxyz=p,q_rad=q),
                np.zeros(9),np.asarray(s["rod_nodes_m"]))
            if self.physical_gate:
                row["strict_planner_constraints"]=row["constraints"]
                row["constraints"]=physical_target_constraints(row)
            constraints.update({step+":"+k:v for k,v in row["constraints"].items()})
            out[step]=row
        return dict(x=np.asarray(x).tolist(),endpoint=end,nominal=nominal,scenarios=out,constraints=constraints,
            admission=candidate_admission(constraints),actual_contact_observed=False,
            native_ready=False,teacher_admission=False)

def run(output,config):
    if config not in (CONFIG,PHYSICAL_CONFIG,REFINED_CONFIG):raise ValueError("frozen CPU experiment bounds required")
    source=load_observed_scenarios();geometry=Geometry(source,physical_gate=config!=CONFIG)
    baseline=geometry.evaluate(np.zeros(9),source["scenarios"])
    initial=np.zeros(9)
    if config==REFINED_CONFIG:
        prior_path=Path(config["initial_candidate_source"])
        if digest(prior_path)!=config["initial_candidate_sha256"]:raise ValueError("prior CPU evidence SHA mismatch")
        prior=decode_owned(json.loads(prior_path.read_bytes()))
        if (prior["config"]!=PHYSICAL_CONFIG or prior["source"]["source_sha256"]!=source["source_sha256"]
            or refinement_steps(prior)!=config["search_physics_steps"]
            or set(prior["all_observed_validation"]["scenarios"])!={str(k) for k in range(211,235)}):
            raise ValueError("same observed complete prior and exact counterexamples required")
        initial=np.asarray(prior["search"]["best"]["x"])
    selected=[s for s in source["scenarios"] if s["identity"]["physics_step"] in config["search_physics_steps"]]
    with (output/"search.jsonl").open("x") as f:
        def query(x):
            row=geometry.evaluate(x,selected)
            f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");f.flush()
            return row
        search=bounded_search(query,initial,maximum_evaluations=config["maximum_evaluations"],
            maximum_seconds=config["maximum_seconds"])
        f.flush();os.fsync(f.fileno())
    validation=geometry.evaluate(np.asarray(search["best"]["x"]),source["scenarios"])
    verdict=admission(validation["nominal"],list(validation["scenarios"].values()))
    return dict(version=config["version"],status="complete",config=config,source=source,baseline=baseline,
        search=search,all_observed_validation=validation,qualification=verdict,
        original_geometry_sha256=geometry.g["source_sha256"],sdf_bindings=geometry.bindings,
        actual_contact_observed=False,native_ready=False,teacher_admission=False,
        new_physics_steps=0,ppo_updates=0,
        interpretation="correlated observed endpoint perturbations only; no robust-dynamics or path certificate")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,default=ROOT/"configs/dh116_tasks/flexible_pinch_tracking_targets_cpu_v3.json")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:result=run(a.output,json.loads(a.config.read_bytes()))
    except Exception as e:result=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),
        new_physics_steps=0,ppo_updates=0,native_ready=False,teacher_admission=False)
    result.update(elapsed_s=time.monotonic()-start,source_sha256=digest(__file__),config_sha256=digest(a.config))
    with (a.output/"result.json").open("x") as f:
        json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","elapsed_s","qualification","native_ready")}))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
