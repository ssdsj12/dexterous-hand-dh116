#!/usr/bin/env python3
"""Constrain left Reset and endpoint together; preserve original closing motion."""
import argparse,json,os,time
from pathlib import Path
from copy import deepcopy
import numpy as np
from scripts import audit_dh116_left_pinch_tracking_path as path
from scripts import solve_dh116_left_pinch_tracking as left
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,decode_owned

VERSION="dh116-left-pinch-start-conditioned-cpu-v1"
CONFIG=dict(version=VERSION,source_sha256=path.SOURCE_SHA,maximum_evaluations=600,
    maximum_seconds=60.,source_config=left.RESET_CONFIG,
    original_motion_unchanged=True,initial_separation_m=.00005,
    native_execution=False,teacher_admission=False)

def start_observation(packet):
    receipt=json.loads(left.source.read_locked(left.source.RECEIPT,left.source.RECEIPT_SHA))
    raw=left.source.read_locked(Path(receipt["output"])/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"])
    if not raw.endswith(b"\n"):raise ValueError("complete ledger")
    x=next(decode_owned(e["payload"])["exchange"] for e in map(json.loads,raw.splitlines()) if e.get("operation")=="physical")
    g=x["closure_geometry"]["raw"];ident=g["identity"];h=g["hands"]["left"]
    if ident["physics_step"]!=3 or x["physics_step"]!=3:raise ValueError("first measured actual clock")
    p=path.actual_pose(h);q=left.observed.tracking.active_values(g["joint_readback"]["hands"]["left"])
    a=packet["executed_anchor"]
    return dict(identity=ident,joint_identity=g["joint_readback"]["identity"][0],rod_identity=ident,
        root_translation_error_m=(p[:3]-a["root_pose_wxyz"][:3]).tolist(),
        world_rotation_error_xyzw=(left.observed.rotation(p)*left.observed.rotation(a["root_pose_wxyz"]).inv()).as_quat().tolist(),
        active_error_rad=(q-a["q_rad"]).tolist(),rod_nodes_m=np.asarray(g["rod_nodes_m"]).tolist(),
        source_left_was_held=True,closure_dynamics_qualified=False)

class Geometry(left.Geometry):
    def __init__(self,packet):
        super().__init__(packet,config=left.RESET_CONFIG);self.start_observed=start_observation(packet)
    def initial_guards(self,pose,nodes):
        base=left.target.surface.base
        row=left.target.evaluate(self.g,self.grids,self.atlas,pose,np.zeros(9),nodes)
        guards={k:v for k,v in row["constraints"].items() if k.startswith(("table:","rod:","sdf_depth:"))}
        transforms=base.fk_poses(self.g,pose["root_pose_wxyz"],pose["q_rad"])
        initial={kind+":"+n:left.target.surface._separation(base.transform(v["vertices"],transforms[n]),nodes)
            for kind,key in (("source","source_colliders"),("cooked","cooked_pads")) for n,v in self.g[key].items()}
        guards.update({"initial_rod:"+k:v-CONFIG["initial_separation_m"]-.00001 for k,v in initial.items()})
        return dict(constraints=guards,all_shape_separation_m=initial,table_m=row["table_m"])
    def evaluate(self,x,scenarios):
        row=super().evaluate(x,scenarios);start=left.start_for(self.packet,row["endpoint"])
        nominal=self.initial_guards(start,self.nominal_nodes)
        p,q=left.observed.apply_scenario(start,self.start_observed)
        observed=self.initial_guards(dict(root_pose_wxyz=p,q_rad=q),np.asarray(self.start_observed["rod_nodes_m"]))
        row["constraints"].update({"nominal_start:"+k:v for k,v in nominal["constraints"].items()})
        row["constraints"].update({"start:"+k:v for k,v in observed["constraints"].items()})
        row["start_state"]=dict(commanded=start,nominal=nominal,observed=observed,
            identity=self.start_observed["identity"],source_left_was_held=True)
        row["admission"]=left.candidate_admission(row["constraints"])
        return row

def load_inputs():
    r,_=path.load_candidate()
    return r,Geometry(r["source"])

def run(output,config):
    if config!=CONFIG:raise ValueError("frozen bounded own-left start conditioning")
    r,g=load_inputs();p=r["source"];initial=np.asarray(r["search"]["best"]["x"])
    ss=[s for s in p["scenarios"] if s["identity"]["physics_step"] in (211,222,234)]
    baseline=g.evaluate(initial,p["scenarios"])
    with (output/"search.jsonl").open("x") as f:
        def query(x):
            v=g.evaluate(x,ss);f.write(json.dumps(encode_owned(v),allow_nan=False)+"\n");f.flush();return v
        search=left.bounded_search(query,initial,maximum_evaluations=config["maximum_evaluations"],maximum_seconds=config["maximum_seconds"])
        f.flush();os.fsync(f.fileno())
    validation=g.evaluate(np.asarray(search["best"]["x"]),p["scenarios"])
    return dict(version=VERSION,status="complete",config=config,source=p,baseline=baseline,search=search,
        all_observed_validation=validation,qualification=dict(start_and_endpoint_passed=validation["admission"]["static_constraints_passed"],
            continuous_path_proved=False,native_ready=False,teacher_admission=False),
        geometry_source_sha256=g.g["source_sha256"],sdf_bindings=g.bindings,new_physics_steps=0,ppo_updates=0,native_ready=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--config",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:r=run(a.output,json.loads(a.config.read_bytes()))
    except Exception as e:r=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),new_physics_steps=0,ppo_updates=0,native_ready=False,teacher_admission=False)
    r.update(elapsed_s=time.monotonic()-started,script_sha256=left.source.sha(__file__),config_sha256=left.source.sha(a.config))
    with (a.output/"result.json").open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","qualification")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
