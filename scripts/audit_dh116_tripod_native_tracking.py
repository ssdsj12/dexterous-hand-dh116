#!/usr/bin/env python3
"""SHA-bound read-only analysis of the consumed Tripod terminal prefix.

Kinematic substitutions are NOT dynamic counterfactual experiments, contact
measurements, or permission to replay the consumed native recipe.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned

VERSION="dh116-tripod-native-tracking-audit-v1"
ROOT=Path(__file__).resolve().parents[1]
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-tripod-owned-contact-native/71f0fe738be3559c02c2a32ca5f1c2702f98cdb46ed403f3b2339fe732fa536a/attempt-r_cnsg84/receipt.json")
RECEIPT_SHA="9bb31008160da84e31c1e089eed925ba658d40c82a830173a958e21491ecf0eb"
LOCKS={
    "physical.jsonl":"fc30831476a2247df49c875e185f13106168ffa117b696491ba0b25d94748f33",
    "result.json":"8813114aa6c4f5963fc6c2301beb7ce67b67873922c6efd4da736918cf5fa8ff",
    "director_reset.json":"7ef7b21981c584b09e6e2a3860a64b229ffe44eec63938bccafc08cfce23c217",
    "sdk_clock.jsonl":"05d0c8da462f0c269cdf49414ff901c837eca2f5c93b6e588fb09c485d073c28"}
MOUNT=("wrist_x","wrist_y","wrist_z","wrist_roll","wrist_pitch","wrist_yaw")
IDENTITY=("run_uuid","env_id","episode_uid","reset_id","rod_uid")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def finite(a,shape):
    a=np.asarray(a,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError("finite complete array required")
    return a

def mount_pose(anchor,xyz,angles):
    anchor=finite(anchor,(7,));xyz=finite(xyz,(3,));angles=finite(angles,(3,))
    if abs(np.linalg.norm(anchor[3:])-1)>1e-5:raise ValueError("unit anchor quaternion")
    base=Rotation.from_quat(anchor[[4,5,6,3]])
    rotation=base*Rotation.from_euler("XYZ",angles)
    q=rotation.as_quat()
    return np.r_[anchor[:3]+base.apply(xyz),q[3],q[:3]]

def pose_matrix(pose):
    pose=finite(pose,(7,))
    if abs(np.linalg.norm(pose[3:])-1)>1e-5:raise ValueError("unit quaternion")
    t=np.eye(4);t[:3,3]=pose[:3];t[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
    return t

def fk_joint_positions(root,edges,base_pose,q_by_name):
    """Original USD frames with explicit measured passive angles, no expansion."""
    poses={root:pose_matrix(base_pose)}
    for name,parent,child,t0,t1,axis in edges:
        if parent not in poses:raise ValueError("ordered connected original joint tree required")
        motion=np.eye(4)
        if axis is not None:
            if name not in q_by_name or not math.isfinite(q_by_name[name]):
                raise ValueError("actual active/passive position required")
            motion[:3,:3]=Rotation.from_rotvec(axis*q_by_name[name]).as_matrix()
        poses[child]=poses[parent]@t0@motion@t1
    return {p.rsplit("/",1)[-1]:t for p,t in poses.items()}

class Sequence:
    def __init__(self,reset):
        self.identity={k:reset[k] for k in IDENTITY}
        if reset["physics_step"]!=2:raise ValueError("SDK2 actual reset required")
        self.rows=0;self.terminal=None
    def record(self,row,terminal=None):
        if self.terminal is not None:raise ValueError("physical row after terminal")
        n=self.rows+1
        if (any(row[k]!=self.identity[k] for k in IDENTITY)
            or row["physics_step"]!=n+2 or row["rod_tick"]!=12*n
            or row["policy_step"]!=(n-1)//8 or row["substep"]!=(n-1)%8
            or not math.isclose(row["timestamp_s"],(n+2)/480,abs_tol=1e-12,rel_tol=0)
            or n>232):raise ValueError("same episode and consecutive two-level clock required")
        self.rows=n;self.terminal=terminal
    def summary(self):
        return dict(physics_rows=self.rows,last_sdk_step=self.rows+2,terminal=self.terminal,
            complete_budget=self.rows==232 and self.terminal is None,new_physics_steps=0,teacher_admission=False)

def validate_control_end(sequence, row):
    if (sequence.rows<1 or row["control_step"]!=(sequence.rows-1)//8
        or row["physical_samples"]!=((sequence.rows-1)%8)+1
        or row["termination_reason"]!=sequence.terminal
        or (sequence.rows%8!=0 and sequence.terminal is None)):
        raise ValueError("control terminal sample accounting differs")

def classify_clearance(value):
    if not math.isfinite(value):raise ValueError("finite bound required")
    return dict(lower_bound_m=value,geometric_clearance_proven=value>=0,
        physical_contact_proven=False,penetration_depth_m=None)

def joint_tracking(hand):
    from scripts.audit_dh116_bilateral_native_evidence import measured_tracking
    tracked=measured_tracking(hand)
    names=hand["joint_names"];q=finite(hand["q"],(1,len(names)))[0]
    qd=finite(hand["qd"],(1,len(names)))[0];target=finite(hand["backend_targets"],(1,len(names)))[0]
    if len(names)!=len(set(names)) or not set(MOUNT+JOINTS)<=set(names):raise ValueError("17 distinct joint names")
    ii=[names.index(n) for n in MOUNT];err=q[ii]-target[ii]
    return dict(joint_names=names,actual_q=q,actual_qd=qd,sdk_target=target,
        maximum_active_error_rad=float(abs(tracked["active_error_rad"]).max()),
        passive_coupling_error_rad=tracked["passive_coupling_error_rad"],
        mount_error=err,maximum_mount_rotation_error_rad=float(abs(err[3:]).max()),
        maximum_mount_translation_error_m=float(abs(err[:3]).max()),
        passive_sdk_zero_targets_are_not_commands=True,
        torque_source="not_inferred_from_position_error",hardware_measurement=False)

def geometry_comparison(g,exchange):
    """Explicit same-row FK substitutions; preserve source geometry and actual rod."""
    from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import refined_lower
    raw=exchange["closure_geometry"]["raw"]
    hand=exchange["action_readback"]["post"]["hands"]["left"]
    t=joint_tracking(hand);names=t["joint_names"]
    q=dict(zip(names,t["actual_q"]));targets=dict(zip(names,t["sdk_target"]))
    anchor=np.asarray(hand["native_anchor_pose_w"])[0]
    actual=raw["hands"]["left"];poses=np.asarray(actual["link_poses_xyzw"])[0]
    measured={n:pose_matrix(np.r_[p[:3],p[6],p[3:6]]) for n,p in zip(actual["body_names"],poses)}
    xyz=np.array([q[n] for n in MOUNT[:3]]);angles=np.array([q[n] for n in MOUNT[3:]])
    command_xyz=np.array([targets[n] for n in MOUNT[:3]])
    command_angles=np.array([targets[n] for n in MOUNT[3:]])
    base=mount_pose(anchor,xyz,angles)
    all_actual=fk_joint_positions(g["fk"].root,g["fk"].edges,base,q)
    positions=max(np.linalg.norm(all_actual[n][:3,3]-measured[n][:3,3]) for n in all_actual)
    rotations=max(Rotation.from_matrix(all_actual[n][:3,:3].T@measured[n][:3,:3]).magnitude() for n in all_actual)
    # Do not explain an SDK-frame mismatch using an unverified FK.
    if positions>2e-6 or rotations>2e-5:raise ValueError("measured full17 FK does not reproduce actual links")
    ideal=dict(zip(JOINTS,expand([targets[n] for n in ACTIVE])))
    coupled_measured=dict(zip(JOINTS,expand([q[n] for n in ACTIVE])))
    scenarios={
        "measured_links":measured,
        "measured_all_joint_fk":all_actual,
        "only_ideal_passive_coupling":fk_joint_positions(g["fk"].root,g["fk"].edges,base,coupled_measured),
        "only_target_wrist_rotation":fk_joint_positions(g["fk"].root,g["fk"].edges,mount_pose(anchor,xyz,command_angles),q),
        "only_target_wrist_translation":fk_joint_positions(g["fk"].root,g["fk"].edges,mount_pose(anchor,command_xyz,angles),q),
        "target_wrist_measured_fingers":fk_joint_positions(g["fk"].root,g["fk"].edges,mount_pose(anchor,command_xyz,command_angles),q),
        "measured_wrist_target_fingers":fk_joint_positions(g["fk"].root,g["fk"].edges,base,ideal),
        "all_targets_ideal_coupling":fk_joint_positions(g["fk"].root,g["fk"].edges,mount_pose(anchor,command_xyz,command_angles),ideal)}
    nodes=np.asarray(raw["rod_nodes_m"]);shape=g["source_colliders"]["finger22_link"]["vertices"]
    results={}
    for name,transforms in scenarios.items():
        m=transforms["finger22_link"];world=shape@m[:3,:3].T+m[:3,3]
        results[name]=dict(**classify_clearance(float(refined_lower(world,nodes))),
            table_clearance_m=float(world[:,2].min()-.3))
    saved=exchange["closure_geometry"]["sides"]["left"]["non_target_surface_gap_lower_m"]["source:finger22_link"]
    if abs(results["measured_links"]["lower_bound_m"]-saved)>1e-12:
        raise ValueError("actual failed shape recomputation differs")
    return dict(identity=raw["identity"],fk_vs_measured_position_max_m=float(positions),
        fk_vs_measured_rotation_max_rad=float(rotations),scenarios=results,
        dynamic_causality_proven=False,contact_measurement=False,teacher_admission=False)

def audit(output):
    from scripts.replay_dh116_segment_contact_r0 import stream_events
    from scripts.audit_dh116_tripod_quiet_replay import inspect_quiet
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    if sha(RECEIPT)!=RECEIPT_SHA:raise ValueError("consumed receipt changed")
    receipt=json.loads(RECEIPT.read_bytes());native=Path(receipt["output"])
    if (receipt["native_qualification"] is not False or receipt["native_execution_performed"] is not True
        or receipt["outer_exit_code"]!=1 or receipt["debugger_inferior_exit_code"]!=1
        or receipt["debugger_inferior_signal"] is not None
        or receipt["command_spec_sha256"]!="9a226954121beff9c2d459c523dbafe24531ceb6913c730b5b7f761a7d833990"):
        raise ValueError("exact failed bounded native receipt required")
    for name,expected in LOCKS.items():
        if sha(native/name)!=expected:raise ValueError("native artifact changed "+name)
    quiet=inspect_quiet(native/"physical.jsonl",complete=False)
    counts=Counter();contacts=Counter();seq=None;pending=None;physical=[];last=None;ends=[]
    qpass=None
    with (output/"tracking.jsonl").open("x") as stream:
        for e in stream_events(native/"physical.jsonl",LOCKS["physical.jsonl"]):
            op=e.get("operation");counts[str(op)]+=1
            if op=="reset":seq=Sequence(e["payload"]["reset"]["after_clock"])
            elif op=="begin":
                if pending is not None:raise ValueError("unsettled control")
                pending=np.asarray(e["payload"]["action"])
            elif op=="physical":
                x=e["payload"]["exchange"];ids=x["action_readback"]["post"]["identity"]
                if seq is None or pending is None or len(ids)!=1:raise ValueError("missing real reset/action")
                if not np.array_equal(x["action_readback"]["request"]["action24"],pending):raise ValueError("action differs")
                seq.record(ids[0],x["guard_failure"])
                sides={side:joint_tracking(x["action_readback"]["post"]["hands"][side]) for side in ("left","right")}
                geometry={side:{k:v for k,v in data.items() if k not in ("required_pad_sdf",)}
                    for side,data in x["closure_geometry"]["sides"].items()}
                row=dict(identity=ids[0],action24=pending,tracking=sides,geometry=geometry,
                    guard=x["guard_failure"],reasons=x["closure_geometry"]["failure_reasons"],
                    required_pad_sdf_gaps_m={s:{n:v["minimum_gap_m"] for n,v in d["required_pad_sdf"].items()}
                        for s,d in x["closure_geometry"]["sides"].items()})
                stream.write(json.dumps(encode_owned(row),allow_nan=False)+"\n")
                physical.append(row)
                if x["tripod_quiet"].get("status")=="passed":qpass=x["tripod_quiet"]
                for pair in x["rod"]["rod_mesh_contact_pairs"]["pairs"]:contacts[pair["mesh_name"]]+=1
                last=x
            elif op=="end":
                end=e["payload"]
                if pending is None:raise ValueError("control end without action")
                validate_control_end(seq,end)
                ends.append(end);pending=None
        stream.flush();os.fsync(stream.fileno())
    if (seq is None or seq.rows!=197 or seq.terminal!="closure_geometry_guard"
        or pending is not None or len(ends)!=25 or counts["setup"]!=2 or counts["reset"]!=1):
        raise ValueError("expected consumed terminal prefix differs")
    result=json.loads((native/"result.json").read_text())
    if result["physics_steps"]!=seq.rows+2 or not result["failure"]:raise ValueError("outer result/ledger mismatch")
    g=load_geometry(ROOT,side="left")
    if g["source_sha256"]!=last["closure_geometry"]["sources"]["left"]["source_sha256"]:
        raise ValueError("source USD/cooked identities differ")
    comparison=geometry_comparison(g,last)
    maxima={}
    for side in ("left","right"):
        maxima[side]={k:max(float(r["tracking"][side][k]) for r in physical)
            for k in ("maximum_active_error_rad","passive_coupling_error_rad",
                "maximum_mount_rotation_error_rad","maximum_mount_translation_error_m")}
    return dict(version=VERSION,status="complete",analysis_scope="failed_native_prefix_not_qualified_execution",
        receipt_sha256=RECEIPT_SHA,artifact_sha256=LOCKS,sequence=seq.summary(),
        consumed_sdk_steps=199,consumed_measurement_steps=197,consumed_solver_substeps=2364,
        pending_budget_unused_steps=35,retry_permitted=False,control_windows=25,last_control_samples=5,
        operation_counts=dict(counts),quiet_replay=quiet,quiet_admission_before_motion=qpass,
        contact_mesh_pair_counts=dict(contacts),tracking_maxima=maxima,
        terminal=physical[-1],same_row_kinematic_substitutions=comparison,
        process=dict(outer_exit_code=receipt["outer_exit_code"],
            inferior_exit_code=receipt["debugger_inferior_exit_code"],
            signal=receipt["debugger_inferior_signal"],app_close_returned=json.loads((native/"app_close_returned.json").read_text())["app_close_returned"],
            duration_seconds=receipt["duration_seconds"],natural_exit_zero=False),
        new_physics_steps=0,ppo_updates=0,teacher_admission=False,grasp_success=False,
        limitations=["negative conservative separation is not a measured penetration",
            "target substitution is kinematic only; not a dynamic or causal native intervention",
            "passive SDK zero targets are not active commands",
            "source17 plus locked cooked5 does not qualify all live non-pad cooked hulls",
            "no hand contact; table-supported rod only; 5s Hold not executed",
            "app.close return is distinct from native view release and outer exit0"])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:result=audit(args.output)
    except Exception as e:result=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),
        new_physics_steps=0,ppo_updates=0,teacher_admission=False)
    result.update(elapsed_s=time.monotonic()-start,script_sha256=sha(__file__))
    with (args.output/"result.json").open("x") as f:
        json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","consumed_sdk_steps","elapsed_s")}))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
