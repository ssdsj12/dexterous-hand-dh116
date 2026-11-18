#!/usr/bin/env python3
"""Bounded measured right Reset geometry, never a live policy/controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from scripts import audit_dh116_right_pinch_measured_direction as previous
from scripts.audit_dh116_pinch_rod_axis_clearance import axis_transform
from scripts.prepare_dh116_pinch_margin_contact import load_pair
from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand

VERSION="dh116-right-pinch-reset-direction-v4"
ROOT=Path(__file__).resolve().parents[1]
SCALES=np.array([.05,.025,.025,.1,.001,.001,.00625])
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-pinch-failed-path-20261113-0z8_jdc8/run01/result.json")
PARENT_SHA="2894befba70c78b506178c3a0b3bd95086d40eaf1d6fe951b77c4b5f38e48735"
PADS=previous.PADS

def decode(q,base_delta,x):
    q=np.asarray(q,float);d=np.asarray(base_delta,float);x=np.asarray(x,float)
    if (q.shape!=(6,) or d.shape!=(2,) or x.shape!=(7,) or not np.isfinite(q).all()
        or not np.isfinite(d).all() or not np.isfinite(x).all() or np.any(abs(x)>1+1e-12)):
        raise ValueError("finite bounded right Reset coordinates required")
    q0=q.copy();q0[0]+=SCALES[0]*x[0]
    delta=np.zeros(6);delta[1:3]=d+SCALES[1:3]*x[1:3]
    if np.any(delta<0) or np.any(delta>.1+1e-12):raise ValueError("original closure speed/time bound exceeded")
    expand(q0);expand(q0+delta)
    return dict(q0=q0,delta=delta,roll_rad=float(SCALES[3]*x[3]),
        reset_translation_world_m=np.array([SCALES[4]*x[4],SCALES[6]*x[6],SCALES[5]*x[5]]))

def reset_rotation(axis,angle):
    if not np.isfinite(angle) or abs(angle)>.1+1e-12:raise ValueError("frozen Reset angle domain exceeded")
    return axis_transform(axis,angle)

def reset_transform(axis,p):
    t=reset_rotation(axis,p["roll_rad"])
    shift=np.asarray(p["reset_translation_world_m"],float)
    if shift.shape!=(3,) or not np.isfinite(shift).all() or np.any(abs(shift)>np.array([.001,.00625,.001])+1e-12):
        raise ValueError("fixed Reset XYZ domain exceeded")
    t[:3,3]+=shift
    return t

def final_gate(table,other,depth,dot):
    values=np.r_[table,other,depth,dot]
    if not np.isfinite(values).all():raise ValueError("complete finite proof required")
    flags=dict(table_clear=bool(min(table)>=.0021),non_target_clear=bool(min(other)>=0),
        depth_safe=bool(min(depth)>=-.0003),contact_margin=bool(max(depth)<=-.0001),
        opposed=bool(dot<-.2))
    return dict(**flags,static_path_passed=all(flags.values()),native_ready=False,teacher_admission=False)

def local_pass(measure,center,lo,hi,trust_radius=.5,*,proposal_only_rows=()):
    if not np.isfinite(trust_radius) or not .005<=trust_radius<=.5:raise ValueError("frozen finite local trust radius required")
    center=np.asarray(center,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    n=len(center)
    if n not in (4,6,7,9) or any(x.shape!=(n,) or not np.isfinite(x).all() for x in (center,lo,hi)):
        raise ValueError("four, six, seven or nine finite domain coordinates required")
    if np.any(center-.002<lo-1e-12) or np.any(center+.002>hi+1e-12):raise ValueError("FD neighborhood leaves frozen box")
    count=0
    def query(x):
        nonlocal count
        if count>=4*n+5:raise ValueError("local measurement budget exhausted")
        if np.any(x<lo-1e-12) or np.any(x>hi+1e-12):raise ValueError("query outside original domain")
        count+=1;v=np.asarray(measure(x),float)
        if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError("finite constraints required")
        return v
    c=query(center);jac=[]
    if (not isinstance(proposal_only_rows,(list,tuple))
            or any(type(i) is not int or not 0<=i<len(c) for i in proposal_only_rows)
            or len(set(proposal_only_rows))!=len(proposal_only_rows)):
        raise ValueError("explicit distinct existing proposal row indices required")
    if any(c[i]<.25 for i in proposal_only_rows):
        raise ValueError("only safely inactive proposal rows with margin >=0.25 permitted")
    for h in (.002,.001):
        jac.append(np.column_stack([(query(center+np.eye(n)[j]*h)-query(center-np.eye(n)[j]*h))/(2*h) for j in range(n)]))
    relative=float(np.linalg.norm(jac[0]-jac[1])/max(np.linalg.norm(jac[1]),1e-12))
    violation=lambda v:float(max(0.,-v.min()))
    result=dict(center=center.tolist(),baseline=c.tolist(),jacobians=jac,
        jacobian_relative_error=relative,trust_radius=trust_radius,finite_difference_steps=[.002,.001],best_x=center.tolist(),best_violation=violation(c),
        trials=[],global_infeasibility_proven=False,linear_solver_calls=0)
    proposal_jac=[j.copy() for j in jac]
    for j in proposal_jac:j[list(proposal_only_rows)]=0.
    proposal_relative=float(np.linalg.norm(proposal_jac[0]-proposal_jac[1])/max(np.linalg.norm(proposal_jac[1]),1e-12))
    result.update(proposal_only_rows=list(proposal_only_rows),
        proposal_jacobian_relative_error=proposal_relative,
        full_nonlinear_constraints_always_verified=True)
    # Opt-in inactive rows affect the local proposal only. Full original
    # nonlinear vectors still determine EVERY trial loss and final gate.
    if proposal_relative>.2:
        result.update(measurements=count,reason="constraint_jacobian_nonsmooth");return result
    lower=np.maximum(lo+.003-center,-trust_radius);upper=np.minimum(hi-.003-center,trust_radius)
    # Minimize common violation first. Then prefer the smallest Reset adjustment.
    matrix=-np.column_stack((proposal_jac[-1],np.ones(len(c))))
    fit=linprog(np.r_[np.zeros(n),1.],A_ub=matrix,b_ub=c,
        bounds=[*zip(lower,upper),(0,None)],method="highs",options={"time_limit":1.,"maxiter":200})
    result.update(linear_solver_calls=1,lp_status=int(fit.status),lp_message=fit.message)
    if not fit.success:result.update(measurements=count,reason="local_LP_failed");return result
    t=float(fit.x[-1]);zeros=np.zeros((len(c),n))
    absrows=np.zeros((2*n,2*n+1))
    for i in range(n):
        absrows[2*i,i]=1;absrows[2*i,n+1+i]=-1
        absrows[2*i+1,i]=-1;absrows[2*i+1,n+1+i]=-1
    second=linprog(np.r_[np.zeros(n+1),np.ones(n)],A_ub=np.vstack([np.column_stack([matrix,zeros]),absrows]),
        b_ub=np.r_[c-(1e-6 if t==0 else 0),np.zeros(2*n)],bounds=[*zip(lower,upper),(0,t),*[(0,None)]*n],
        method="highs",options={"time_limit":1.,"maxiter":200})
    result["linear_solver_calls"]=2
    dx=second.x[:n] if second.success else fit.x[:n]
    result.update(linearized_slack=t,delta=dx.tolist(),smallest_adjustment_solver_status=int(second.status))
    for amount in (1.,.5,.25,.125):
        x=center+amount*dx;v=query(x);loss=violation(v)
        result["trials"].append(dict(fraction=amount,x=x.tolist(),constraints=v.tolist(),violation=loss))
        if loss<result["best_violation"]:result.update(best_x=x.tolist(),best_violation=loss)
    result["measurements"]=count
    return result

class Geometry:
    def __init__(self):
        self.anchor=previous.load_anchor()
        if previous.digest(PARENT)!=PARENT_SHA:raise ValueError("fixed measured failure required")
        parent=previous.decode_owned(json.loads(PARENT.read_bytes()))
        if parent["static_path_passed"] is not False:raise ValueError("expected failed direction")
        self.base_delta=np.array(parent["selected_delta_rad"])[1:3]
        self.g=previous.load_geometry(ROOT,side="right");self.pair=load_pair(ROOT)
        if self.g["source_sha256"]!=self.pair["right"]["source_sha256"]:raise ValueError("original right source differs")
        self.grids,self.binding=previous.grids_for(self.g,previous.checked_json(previous.SDF_DIR/"result.json",previous.SDF_SHA))
        self.old=previous.fk_poses(self.g,self.g["pose"],self.anchor["q"])
        self.axis=np.array([.43,0,.305])
        self.reset_nodes=np.array(self.pair["rod_seed"]["proposed_nodes_m"])
        if self.reset_nodes.shape!=(41,3) or not np.allclose(self.reset_nodes[:,[0,2]],[.43,.305],atol=1e-12):
            raise ValueError("frozen supported free-rod Reset required")
        self.names=set(self.g["source_colliders"])|set(self.g["cooked_pads"])
        self.labels=None
    def bounds(self):
        lo,hi=-np.ones(7),np.ones(7);ql,qh=active_limits();q=self.anchor["q"]
        lo[0]=max(lo[0],(ql[0]-q[0])/.05);hi[0]=min(hi[0],(qh[0]-q[0])/.05)
        lo[1:3]=np.maximum(lo[1:3],-self.base_delta/.025)
        hi[1:3]=np.minimum(hi[1:3],(np.minimum(.1,qh[1:3]-q[1:3])-self.base_delta)/.025)
        return lo,hi
    def poses(self,x,fraction):
        p=decode(self.anchor["q"],self.base_delta,x);q=p["q0"]+fraction*p["delta"]
        fk=previous.fk_poses(self.g,self.g["pose"],q)
        t=previous.predicted_poses(self.anchor["actual"],self.old,fk,self.names)
        roll=reset_transform(self.axis,p)
        return {name:roll@v for name,v in t.items()}
    def shapes(self,t):
        return {kind+":"+n:previous.transform(v["vertices"],t[n])
            for kind,key in (("source","source_colliders"),("cooked","cooked_pads"))
            for n,v in self.g[key].items()}
    def separation(self,v,nodes):
        lower=previous.rod_support_lower(v,nodes)
        if lower<0:
            certificate=previous.certify(v,nodes);previous.validate(certificate,v,nodes)
            return max(lower,certificate["lower_m"])
        return lower
    def pad_values(self,t):
        pads={n:previous.sample_contact(self.grids[n],t[n],self.anchor["nodes"]) for n in PADS}
        normals=[]
        for n,r in pads.items():
            if r["minimum_gap_m"] is None or r["gradient_norm_at_minimum"]<=0:raise ValueError("lost actual-node SDF domain")
            _,gradient,_=previous.sample_grid(self.grids[n],[r["local_node_at_minimum"]])
            normal=t[n][:3,:3]@gradient[0];normal/=np.linalg.norm(normal);normals.append(normal)
        return pads,float(np.dot(*normals))
    def measure(self,x):
        labels=[];values=[];packet=dict(x=np.asarray(x).tolist(),boundaries={})
        def add(label,value):labels.append(label);values.append(float(value))
        for f in (0.,1.):
            t=self.poses(x,f);shape=self.shapes(t);table={};others={};start={}
            for name,v in shape.items():
                table[name]=float(v[:,2].min()-.3);add(str(f)+":"+name+":table",(table[name]-.0022)/.001)
                if f==0:
                    start[name]=min(self.separation(v,self.anchor["nodes"]),self.separation(v,self.reset_nodes))
                    add(name+":reset_clear",(start[name]-.00005)/.001)
                elif name.split(":")[1] not in PADS:
                    others[name]=self.separation(v,self.anchor["nodes"])
                    add(name+":end_other",(others[name]-.0001)/.001)
            pads,dot=self.pad_values(t)
            if f==1:
                for name,r in pads.items():
                    gap=r["minimum_gap_m"];add(name+":depth",(gap+.00025)/.001);add(name+":margin",(-.0001-gap)/.001)
                add("opposition",-.200001-dot)
            packet["boundaries"][str(int(f))]=dict(tables=table,non_target=others,initial=start,pads=pads,dot=dot)
        if self.labels is not None and self.labels!=labels:raise ValueError("constraint order changed")
        self.labels=labels
        return dict(packet,constraint_labels=labels,constraint_values=values,nonlinear_passed=bool(min(values)>=0))
    def path(self,x,emit):
        p=decode(self.anchor["q"],self.base_delta,x)
        travel={**{"source:"+n:v for n,v in previous.path_displacement_bound(self.g,p["q0"],p["delta"]).items()},
                **{"cooked:"+n:v for n,v in previous.path_displacement_bound(self.g,p["q0"],p["delta"],cooked=True).items()}}
        rows=[];tables=[];others=[];depth=[]
        for index,f in enumerate(np.linspace(0,1,161)):
            t=self.poses(x,f);table={};other={}
            for name,v in self.shapes(t).items():
                remainder=travel[name]/320;table[name]=float(v[:,2].min()-.3-remainder);tables.append(table[name])
                if name.split(":")[1] not in PADS:
                    other[name]=self.separation(v,self.anchor["nodes"])-remainder;others.append(other[name])
            pads,dot=self.pad_values(t);depth.extend(r["minimum_gap_m"] for r in pads.values())
            row=dict(index=index,fraction=float(f),table_lower_m=table,other_lower_m=other,pads=pads,dot=dot)
            rows.append(row);emit(dict(phase="path",**row))
        end=[r["minimum_gap_m"] for r in rows[-1]["pads"].values()]
        gate=final_gate(tables,others,end,rows[-1]["dot"])
        gate["whole_path_depth_safe"]=bool(min(depth)>=-.0003);gate["static_path_passed"] &= gate["whole_path_depth_safe"]
        return dict(rows=rows,minimum_table_m=min(tables),minimum_other_m=min(others),
                    minimum_depth_m=min(depth),**gate)

def run(result,emit):
    geo=Geometry();lo,hi=geo.bounds();x=np.zeros(7);records=[];passes=[]
    prior_path=Path('/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-reset-direction-v3-20261113-2xfw3hqj/run01/result.json');prior_sha='8008acd225b32f09282dcf985dc93f0fe131d7c02d021df38e0916c9c7d4aaa4'
    if previous.digest(prior_path)!=prior_sha:raise ValueError("exact failed XZ Reset batch required")
    prior=previous.decode_owned(json.loads(prior_path.read_bytes()))
    if prior["static_path_passed"] is not False:raise ValueError("expected failed XZ candidate")
    x[:]=prior["best_x"]
    result.update(reset_parent_sha256=prior_sha,initial_solver_x=x.copy())
    result.update(source_identity=geo.anchor["identity"],source_ledger_sha256=geo.anchor["ledger_sha256"],
        prior_failure_sha256=PARENT_SHA,geometry_sources=geo.g["source_sha256"],sdf_bindings=geo.binding,
        initial_actual_q=geo.anchor["q"],base_delta_rad=geo.base_delta,coordinate_scales=SCALES,
        domain_lower=lo,domain_upper=hi,measurements=records,local_passes=passes)
    def query(x):
        if len(records)>=132:raise ValueError("132 endpoint-measurement batch exhausted")
        row=geo.measure(x);records.append(row);emit(dict(phase="endpoint",index=len(records),**row))
        return np.array(row["constraint_values"])
    for index in range(4):
        fit=local_pass(query,x,lo,hi,trust_radius=.05);passes.append(fit);x=np.array(fit["best_x"])
        result.update(best_x=x,best_violation=fit["best_violation"])
        if fit["best_violation"]==0:break
        if np.linalg.norm(x-np.array(fit["center"]))<1e-10:break
    selected=next((r for r in reversed(records) if np.array_equal(r["x"],x)),None)
    if selected is None:raise ValueError("actual selected nonlinear record absent")
    result["selected"]=selected
    result["failed_constraints"]=[dict(name=n,value=v) for n,v in zip(selected["constraint_labels"],selected["constraint_values"]) if v<0]
    if selected["nonlinear_passed"]:
        result["path"]=geo.path(x,emit)
        result["static_path_passed"]=result["path"]["static_path_passed"]
    else:result["static_path_passed"]=False
    result.update(status="complete",native_ready=False,teacher_admission=False,native_recipe_created=False,
        candidate=decode(geo.anchor["q"],geo.base_delta,x),
        limitations=["Measured-frame counterfactual only; no candidate dynamics/contact/Reset observation.",
            "No source/cooked shape, joint limit, wrist safety or material modification.",
            "Whole-path bounds use kinematics; do not establish tracking/model error bounds.",
            "Endpoint rod nodes remain the recorded SDK234 state; fresh Reset and same-rod dynamics require native validation."])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial",physics_steps=0,ppo_updates=0,
        budget=dict(wall_seconds=120,max_endpoint_measurements=132,max_full_path_samples=161))
    old=signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError("120s CPU batch exhausted")))
    signal.setitimer(signal.ITIMER_REAL,120)
    try:
        with (a.output/"progress.jsonl").open("x") as f:
            def emit(row):f.write(json.dumps(previous.encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(result,emit)
            finally:f.flush();os.fsync(f.fileno())
    except (ValueError,TimeoutError,KeyError,np.linalg.LinAlgError) as exc:
        result.update(status="blocked",reason=f"{type(exc).__name__}: {exc}")
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result.update(elapsed_s=time.monotonic()-start,source_sha256=previous.digest(__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(previous.encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","elapsed_s","best_x","best_violation","static_path_passed")},default=lambda x:x.tolist()))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
