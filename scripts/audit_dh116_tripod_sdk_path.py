#!/usr/bin/env python3
"""Compile and audit a CPU Tripod reference through original 24D kernels.

No SDK is started. Perfect target tracking is an explicit reference assumption;
float32 command reconstruction does not predict actual actuator tracking.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import time
import warnings
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_tripod_pose_path import pose_at,rotations,combine_bounds
from scripts.audit_dh116_tripod_fixed_root_path import PARENT,PARENT_SHA,certify_path,shapes
from scripts.audit_dh116_pinch_initial_pose import checked_json,digest,load_geometry,path_displacement_bound
from scripts.audit_dh116_power_closure_constraints import support_plane_bound
from scripts.solve_dh116_tripod_joint_root_direction import REQUIRED,STRICT
from scripts.screen_dh116_right_pinch_closure import BudgetExpired
from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST,compose_rotation
from go2_pvcnn.control.dh116.tabletop_command_kernels import (
    finger_pre_guard,surface_final_guard,wrist_requested_targets)
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets,CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_palm_mount import local_translation

VERSION="dh116-tripod-original-24d-target-path-audit-v1"
DT=1/480
RATES=dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],
           wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
           rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])

def relative_intrinsic(a,b):
    a,b,r0,_=rotations(a,b)
    relative=r0.inv()*Rotation.from_quat(b[[4,5,6,3]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:angles=relative.as_euler("XYZ")
        except UserWarning as e:raise ValueError("gimbal singularity") from e
    if abs(math.cos(angles[1]))<1e-6:raise ValueError("gimbal singularity")
    return angles

def compile_actions(p0,p1,q0,q1,*,side,physics_steps,mode):
    if side not in ("left","right") or mode not in ("intrinsic_linear","geodesic_knots"):
        raise ValueError("explicit side and known interpolation mode required")
    if type(physics_steps) is not int or not 8<=physics_steps<=232 or physics_steps%8:
        raise ValueError("whole 8-substep control intervals, bounded to232 required")
    p0,p1,_,_=rotations(p0,p1)
    q0,q1=np.asarray(q0,dtype=float),np.asarray(q1,dtype=float);expand(q0);expand(q1)
    n=physics_steps//8;t=np.linspace(0,1,n+1)
    end_angles=relative_intrinsic(p0,p1)
    angles=(t[:,None]*end_angles if mode=="intrinsic_linear" else
            np.array([relative_intrinsic(p0,pose_at(p0,p1,float(x))) for x in t]))
    offsets=t[:,None]*(p1[:3]-p0[:3]);qs=q0+t[:,None]*(q1-q0)
    if (np.any(offsets<WRIST["palm_lower_m"]) or np.any(offsets>WRIST["palm_upper_m"])
            or np.any(np.abs(angles)>math.pi/2)):
        raise ValueError("original position or intrinsic angle bounds")
    a=np.zeros((n,24));i=0 if side=="left" else 1
    a[:,6*i:6*i+6]=np.diff(qs,axis=0)/(8*DT*RATES["joint_rate_rad_s"])
    a[:,12+3*i:15+3*i]=np.diff(offsets,axis=0)/(8*DT*RATES["wrist_rate_m_s"])
    a[:,18+3*i:21+3*i]=np.diff(angles,axis=0)/(8*DT*RATES["rotation_rate_rad_s"])
    if (abs(a).max()>1+1e-12 or any(np.linalg.norm(a[:,j:j+3],axis=1).max()>1+1e-12
                                   for j in (12,15,18,21))):
        raise ValueError("original normalized rate and vector norm exceeded")
    return dict(version=VERSION,side=side,mode=mode,physics_steps=physics_steps,
        policy_steps=n,seed_pose_wxyz=p0.tolist(),end_pose_wxyz=p1.tolist(),
        q0_rad=q0.tolist(),q1_rad=q1.tolist(),intrinsic_knots_rad=angles.tolist(),
        world_offset_knots_m=offsets.tolist(),actions=a.tolist(),
        action_contract=ACTION_CONTRACT,native_ready=False)

def target_at(plan,fraction):
    if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError("fraction0..1")
    n=plan["policy_steps"];k=min(int(fraction*n),n-1);u=fraction*n-k
    knots=np.asarray(plan["intrinsic_knots_rad"])
    angles=(1-u)*knots[k]+u*knots[k+1]
    p0=np.array(plan["seed_pose_wxyz"]);p1=np.array(plan["end_pose_wxyz"])
    q0=np.array(plan["q0_rad"]);q1=np.array(plan["q1_rad"])
    quat=compose_rotation(torch.tensor(p0[None,3:]),torch.tensor(angles[None])).numpy()[0]
    return np.r_[p0[:3]+fraction*(p1[:3]-p0[:3]),quat],q0+fraction*(q1-q0)

def budget_audit(*,setup,quiet,motion,hold,maximum):
    values=(setup,quiet,motion,hold,maximum)
    if any(type(v) is not int or v<0 for v in values) or maximum<1:
        raise ValueError("nonnegative integer step accounting required")
    total=setup+quiet+motion+hold
    return dict(setup_steps=setup,quiet_steps=quiet,motion_steps=motion,hold_steps=hold,
        total_sdk_steps=total,maximum_steps=maximum,within_budget=total<=maximum,
        required_extra_steps=max(0,total-maximum),native_ready=False)

def runtime_reference(plan,g):
    """Original pure controller arithmetic with ideal tracking, no native claims.

    The unused hand slot is an arithmetic placeholder, not a right-hand model.
    Only the specified hand is used for geometry and all reported results.
    """
    i=0 if plan["side"]=="left" else 1
    seed=torch.tensor(plan["seed_pose_wxyz"],dtype=torch.float64).repeat(1,2,1)
    q0=np.array(plan["q0_rad"]);command=torch.tensor(np.tile(q0,2)[None]).float().double()
    seedq=command[0,i*6:i*6+6].numpy().copy()
    low,high=[torch.tensor(np.tile(v,2)) for v in active_limits()]
    palms=torch.zeros((1,2,3),dtype=torch.float64);angles=torch.zeros_like(palms)
    measured=seed.clone();top=g["reference"]["reference_layout"]["table_center"][2]+g["reference"]["reference_layout"]["table_size"][2]/2
    seed_r=Rotation.from_quat(seed[0,i,[4,5,6,3]].numpy())
    rows=[dict(physics_index=0,policy_index=None,pose_wxyz=seed[0,i].tolist(),q_rad=seedq.tolist(),
               intrinsic_rad=[0.,0.,0.],local_translation_m=[0.,0.,0.])]
    for step in range(plan["physics_steps"]):
        action=torch.tensor([plan["actions"][step//8]],dtype=torch.float64)
        qprev=command[0,i*6:i*6+6].numpy()
        current=shapes(g,measured[0,i].numpy(),qprev)
        clearance=min(float(v[:,2].min()-top) for v in current.values())
        measured_clearance=torch.full((1,2),1.,dtype=torch.float64)
        measured_clearance[0,i]=clearance
        finger=finger_pre_guard(command,action,command,torch.zeros((1,10),dtype=torch.float64),
            low,high,DT,RATES,policy_owned=True)
        _,finger=surface_final_guard(command,finger,low,high,DT)
        requested_p,requested_a=wrist_requested_targets(palms,angles,action,RATES,DT)
        projection=project_policy_wrist_targets(root_seed_w=seed,measured_base_w=measured,
            common_lift_offset_m=torch.zeros(1,dtype=torch.float64),measured_clearance_m=measured_clearance,
            palm_offsets_m=palms,requested_palm_offsets_m=requested_p,angles_rad=angles,
            requested_angles_rad=requested_a,allowed=torch.ones(1,dtype=torch.bool),
            live=torch.ones(1,dtype=torch.bool))
        local=local_translation(projection["projected_root_poses"][:,i],seed[:,i]).float().double()
        sdk_angles=projection["projected_angles"][:,i].float().double()
        measured[0,i,:3]=seed[0,i,:3]+torch.tensor(seed_r.apply(local.numpy()[0]))
        measured[0,i,3:]=compose_rotation(seed[:,i,3:],sdk_angles)[0]
        command=finger.float().double()
        palms,angles=projection["palm_offsets_m"],projection["projected_angles"]
        planned_pose,planned_q=target_at(plan,(step+1)/plan["physics_steps"])
        rows.append(dict(physics_index=step+1,policy_index=step//8,
            action24=action[0].tolist(),q_rad=command[0,i*6:i*6+6].tolist(),
            pose_wxyz=measured[0,i].tolist(),intrinsic_rad=sdk_angles[0].tolist(),
            local_translation_m=local[0].tolist(),reference_pre_clearance_m=clearance,
            projection_delta_m=(palms-requested_p)[0,i].tolist(),
            projection_reasons=int(projection["projection_reasons"][0,i]),
            requested_offset_m=requested_p[0,i].tolist(),
            sdk_target_position_error_m=float(np.linalg.norm(measured[0,i,:3].numpy()-planned_pose[:3])),
            sdk_target_joint_error_rad=float(abs(command[0,i*6:i*6+6].numpy()-planned_q).max())))
    return dict(rows=rows,inactive_slot="arithmetic_placeholder_not_other_hand_geometry",
                measured_tracking=False,ideal_tracking_assumption=True,new_physics_steps=0)

def path_constraints(total):
    rod={k:v for k,v in total.items() if k.startswith("cooked:") or k.split(":")[1] not in REQUIRED}
    bound={**{"table:"+k:v for k,v in total.items()},
           **{"rod:"+k:v for k,v in rod.items()}}
    thresholds={**{"table:"+k:CLEARANCE_M for k in total},
                **{"rod:"+k:(-STRICT if k.split(":")[1] in REQUIRED else 0.) for k in rod}}
    return bound,thresholds

def target_path_proof(plan,replay,g):
    rows=replay["rows"];n=len(rows)-1
    p=np.array([r["pose_wxyz"] for r in rows]);q=np.array([r["q_rad"] for r in rows])
    angle=np.array([r["intrinsic_rad"] for r in rows])
    derivative=np.max(abs(np.diff(q,axis=0)),axis=0)*n
    joint={}
    for cooked,prefix in ((False,"source:"),(True,"cooked:")):
        joint.update({prefix+k:v for k,v in path_displacement_bound(g,q[0],derivative,cooked=cooked).items()})
    translation=np.max(np.linalg.norm(np.diff(p[:,:3],axis=0),axis=1))*n
    omega=np.max(np.sum(abs(np.diff(angle,axis=0)),axis=1))*n
    initial=shapes(g,p[0],q[0])
    total=combine_bounds(joint,initial,p[0,:3],[translation,0,0],float(omega))
    bound,thresholds=path_constraints(total)
    layout=g["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
    nodes=np.array(layout["wire_points"]);nodes[:,2]=top+.005
    seed=torch.tensor([plan["seed_pose_wxyz"][3:]],dtype=torch.float64)
    def query(t):
        k=min(int(t*n),n-1);u=t*n-k
        angles=(1-u)*angle[k]+u*angle[k+1]
        quat=compose_rotation(seed,torch.tensor(angles[None])).numpy()[0]
        pose=np.r_[(1-u)*p[k,:3]+u*p[k+1,:3],quat]
        current=shapes(g,pose,(1-u)*q[k]+u*q[k+1])
        return {**{"table:"+k:float(v[:,2].min()-top) for k,v in current.items()},
                **{"rod:"+k:support_plane_bound(v,nodes[[0,-1]],.005)["surface_gap_lower_bound_m"]-2e-6
                   for k,v in current.items() if "rod:"+k in thresholds}}
    proof=certify_path(query,bound,thresholds)
    proof["required_rod_minimum_threshold_m"]=-STRICT
    proof["scope"]="piecewise_interpolated_float32_SDK_targets_assuming_perfect_tracking_not_native_motion"
    return proof

def summarize_replay(plan,replay,proof,budget):
    return dict(
        motion_steps=plan["physics_steps"],policy_steps=plan["policy_steps"],
        maximum_joint_target_rounding_error_rad=max(r.get("sdk_target_joint_error_rad",0) for r in replay["rows"]),
        maximum_world_target_rounding_error_m=max(r.get("sdk_target_position_error_m",0) for r in replay["rows"]),
        projection_changed_steps=int(sum(np.linalg.norm(r.get("projection_delta_m",[0,0,0]))>1e-12 for r in replay["rows"])),
        target_path_passed=proof["proved"],
        original_stage_budget_passed=budget["within_budget"])

def run(root,result,mode):
    prior=checked_json(PARENT,PARENT_SHA)
    start=prior["parent_candidate"];end=prior["final_packet"]["measurement"]["candidate"]
    if not prior["final_packet"]["measurement"]["endpoint_geometric_gate"] or start["side"]!="left":
        raise ValueError("same independent left Tripod near endpoint required")
    p0,p1=start["base_pose_wxyz"],end["base_pose_wxyz"]
    q0,q1=start["initial_q_rad"],end["active_q_rad"]
    plan=None;rejections=[]
    for steps in range(8,233,8):
        try:plan=compile_actions(p0,p1,q0,q1,side="left",physics_steps=steps,mode=mode)
        except ValueError as e:
            if "rate" not in str(e):raise
            rejections.append(dict(physics_steps=steps,reason=str(e)));continue
        break
    result.update(parent_result_sha256=PARENT_SHA,rate_rejected_plans=rejections,plan=plan)
    if plan is None:result["status"]="no_rate_feasible_plan";return
    result["unchanged_stage_budget"]=budget_audit(setup=2,quiet=48,motion=plan["physics_steps"],hold=24,maximum=240)
    g=load_geometry(root,side="left")
    if g["source_sha256"]!=prior["geometry_sources"]:raise ValueError("source geometry changed")
    result["geometry_sources"]=g["source_sha256"]
    replay=runtime_reference(plan,g);result["target_replay"]=replay
    result["target_path"]=target_path_proof(plan,replay,g)
    result["summary"]=summarize_replay(plan,replay,result["target_path"],result["unchanged_stage_budget"])
    result["source_sha256"]={n:digest(root/n) for n in (
        "scripts/audit_dh116_tripod_sdk_path.py",
        "src/go2_pvcnn/control/dh116/tabletop_command_kernels.py",
        "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py",
        "src/go2_pvcnn/control/dh116/tabletop_palm_mount.py")}
    result["status"]="complete"

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("intrinsic_linear","geodesic_knots"),required=True)
    p.add_argument("--wall-seconds",type=float,default=55);a=p.parse_args(argv)
    if not np.isfinite(a.wall_seconds) or not 1<=a.wall_seconds<=55:p.error("CPU wall1..55seconds")
    a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    result=dict(version=VERSION,mode=a.mode,status="partial_setup",new_physics_steps=0,ppo_updates=0,
        native_ready=False,teacher_admitted=False,reset_verified=False,actual_tracking_validated=False)
    def expire(*unused):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,a.wall_seconds)
    try:run(Path(__file__).resolve().parents[1],result,a.mode)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,KeyError) as e:result.update(status="error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    result["elapsed_s"]=time.monotonic()-begin
    encoded=json.dumps(result,indent=2,allow_nan=False)
    with (a.output/"result.json").open("x") as f:f.write(encoded+"\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","summary","error")}))
    return int(result["status"]=="error")

if __name__=="__main__":raise SystemExit(main())
