#!/usr/bin/env python3
"""Independent fixed Reset orientation relative to the unchanged rod."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts import solve_dh116_right_pinch_reset_direction as base

VERSION="dh116-right-pinch-relative-reset-orientation-v1"
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-reset-direction-v4-20261113-379rl4yl/run01/result.json")
PARENT_SHA="bd311cc064544f0b6c4ee295ac49d7b5e72f883dacd22c9f36ea6daa8787b865"
PIVOT=np.array([.43,-.2,.305])

def relative_rotation(pivot,angles):
    pivot=np.asarray(pivot,float);angles=np.asarray(angles,float)
    if pivot.shape!=(3,) or angles.shape!=(2,) or not np.isfinite(np.r_[pivot,angles]).all() or np.any(abs(angles)>1+1e-12):
        raise ValueError("finite pivot and bounded relative Reset angles required")
    x,z=.05*angles
    R=Rotation.from_rotvec([0,0,z]).as_matrix()@Rotation.from_rotvec([x,0,0]).as_matrix()
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=pivot-R@pivot
    return T

class FrozenRelativeAngles:
    def __init__(self,parent,angles):
        self.parent=parent;self.angles=np.asarray(angles,float).copy()
    def __getattr__(self,name):return getattr(self.parent,name)
    def poses(self,x,fraction):return self.parent.poses(np.r_[x,self.angles],fraction)

class RelativeGeometry(base.Geometry):
    def bounds(self):
        lo,hi=super().bounds()
        return np.r_[lo,-1.,-1.],np.r_[hi,1.,1.]
    def poses(self,x,fraction):
        x=np.asarray(x,float)
        if x.shape!=(9,):raise ValueError("nine independent bounded Reset coordinates required")
        old=super().poses(x[:7],fraction);T=relative_rotation(PIVOT,x[7:])
        return {name:T@pose for name,pose in old.items()}
    def path(self,x,emit):
        # The original full-path calculation receives unchanged seven joint/
        # translation coordinates; fixed extra angles are retained for EVERY
        # FK sample, never discarded at the path-validation boundary.
        return base.Geometry.path(FrozenRelativeAngles(self,x[7:]),x[:7],emit)

def run(result,emit,*,inactive_opposition_proposal=False):
    parent_path,parent_sha=PARENT,PARENT_SHA
    if inactive_opposition_proposal:
        parent_path=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-relative-reset-20261113-9shn7ze3/run01/result.json")
        parent_sha="532c6461b1ed30dd59b316b5e2718ef57a1ad9ba7ed3f47d17f50620fed86f86"
    if base.previous.digest(parent_path)!=parent_sha:raise ValueError("exact failed Reset source required")
    parent=base.previous.decode_owned(json.loads(parent_path.read_bytes()))
    if parent["static_path_passed"] is not False:raise ValueError("expected preserved failed candidate")
    geo=RelativeGeometry();lo,hi=geo.bounds();x=np.array(parent["best_x"]) if inactive_opposition_proposal else np.r_[parent["best_x"],0.,0.]
    records=[];passes=[]
    result.update(parent_result_sha256=parent_sha,inactive_opposition_proposal=inactive_opposition_proposal,source_ledger_sha256=geo.anchor["ledger_sha256"],
        source_identity=geo.anchor["identity"],source_geometry=geo.g["source_sha256"],sdf_bindings=geo.binding,
        initial_x=x.copy(),pivot_world_m=PIVOT,extra_angle_scales_rad=[.05,.05],
        extra_rotation_order="world_Rz_then_matrix_Rx__Rz_times_Rx_after_existing_fixed_Reset",
        measurements=records,local_passes=passes)
    def query(x):
        if len(records)>=246:raise ValueError("246 endpoint budget exhausted")
        row=geo.measure(x);records.append(row);emit(dict(phase="endpoint",index=len(records),**row))
        return np.array(row["constraint_values"])
    for _ in range(6):
        proposal_rows=[]
        if inactive_opposition_proposal:
            # The original measure order is frozen and independently checked.
            labels=records[-1]["constraint_labels"] if records else parent["selected"]["constraint_labels"]
            if labels[-1]!="opposition" or labels.count("opposition")!=1:
                raise ValueError("named sole opposition proposal row changed")
            proposal_rows=[len(labels)-1]
        fit=base.local_pass(query,x,lo,hi,trust_radius=.1,proposal_only_rows=proposal_rows)
        passes.append(fit);x=np.array(fit["best_x"])
        result.update(best_x=x.copy(),best_violation=fit["best_violation"])
        if fit["best_violation"]==0 or np.linalg.norm(x-np.array(fit["center"]))<1e-10:break
    selected=next(r for r in reversed(records) if np.array_equal(r["x"],x))
    result["selected"]=selected
    result["failed_constraints"]=[dict(name=n,value=v) for n,v in zip(selected["constraint_labels"],selected["constraint_values"]) if v<0]
    result["static_path_passed"]=False
    if selected["nonlinear_passed"]:
        result["path"]=geo.path(x,emit);result["static_path_passed"]=result["path"]["static_path_passed"]
    result.update(status="complete",candidate=dict(base.decode(geo.anchor["q"],geo.base_delta,x[:7]),
        relative_world_xz_angles_rad=(.05*x[7:]).tolist(),fixed_relative_reset_transform=relative_rotation(PIVOT,x[7:])),
        native_qualification=False,native_ready=False,teacher_admission=False,native_recipe_created=False,
        new_physics_steps=0,ppo_updates=0,
        limitations=["actual-node measured-frame CPU counterfactual only",
            "fresh authored root consistency, self-collision and actual Reset tracking require separate verification",
            "no actual rod response or contact measurement inferred from SDF"])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--inactive-opposition-proposal",action="store_true");a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    out=dict(version=VERSION+("-inactive-proposal-v2" if a.inactive_opposition_proposal else ""),status="partial",budget=dict(endpoint_evaluations=246,path_samples=161,wall_s=120),new_physics_steps=0,ppo_updates=0)
    handler=signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError("120s CPU batch exhausted")))
    signal.setitimer(signal.ITIMER_REAL,120)
    try:
        with (a.output/"progress.jsonl").open("x") as f:
            def emit(row):f.write(json.dumps(base.previous.encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(out,emit,inactive_opposition_proposal=a.inactive_opposition_proposal)
            finally:f.flush();os.fsync(f.fileno())
    except Exception as exc:out.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    out.update(elapsed_s=time.monotonic()-start,source_sha256=base.previous.digest(__file__),
        local_direction_source_sha256=base.previous.digest(base.__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(base.previous.encode_owned(out),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:out.get(k) for k in ("status","reason","elapsed_s","best_violation","static_path_passed")}))
    return int(out["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
