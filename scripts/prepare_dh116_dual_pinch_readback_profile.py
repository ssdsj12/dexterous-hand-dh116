#!/usr/bin/env python3
"""Read-only native-ledger-bound dual Pinch 8-tick no-commanded-lift profile."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,load_geometry,transform,pose_matrix,path_displacement_bound)
from scripts.audit_dh116_power_closure_constraints import support_plane_bound
from scripts.prepare_dh116_pinch_closure_margin import fk_poses,sample_contact,relative_pose
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
from scripts.prepare_dh116_dual_pinch_interval import inspect_interval
from scripts.screen_dh116_right_pinch_closure import BudgetExpired
from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import PAIR,PAIR_SHA,SEED_SHA,load_pair_proposal
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_command_kernels import finger_pre_guard,surface_final_guard
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC,active_limits
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-dual-pinch-eight-step-readback-profile-v1"
ROOT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
NATIVE=ROOT/"approved-dual-pinch-zero-v4-20261113-utbAE0/native-runs/flex.f7-dual-pinch-zero-interval-v4-native/attempt-fz50kogx/native"
LEDGER_SHA="ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c"
RECEIPT=ROOT/"native-attempts/flex.f7-dual-pinch-zero-interval-v4-native/9e95d33e8b0565133c05d8fae5eb5c5eb68de269806ae561e4c5bd022bd9e202/attempt-fz50kogx/receipt.json"
RECEIPT_SHA="aa1a6877c29b460e8f12589674d4ba09381faf8aea200400a7cd0e71d1a5a1ec"
NUMERIC_ALLOWANCE=2e-6
PADS=("finger14_link","finger23_link")


def action():
    a=np.zeros(24);a[[1,2,7,8]]=[.1,.1,.05,.1]
    return a


def interval_count(steps):
    if type(steps) is not int or not 8<=steps<=240 or steps%8:
        raise ValueError("whole eight-tick intervals in 8..240 required")
    return steps//8


def targets(command,q):
    command=finite_array(command,(12,));q=finite_array(q,(12,))
    c=torch.from_numpy(command.copy())[None];measured=torch.from_numpy(q.copy())[None]
    low,high=(torch.from_numpy(np.tile(x,2)) for x in active_limits())
    a=torch.from_numpy(action())[None];rows=[command.tolist()]
    rates=dict(joint_rate_rad_s=.3,wrist_rate_m_s=.01,rotation_rate_rad_s=.5)
    # Keep measured q frozen: the largest requested lead here is only 0.0005rad.
    # This is exact target arithmetic under the stated q hypothesis, not dynamics.
    for unused in range(8):
        t=finger_pre_guard(c,a,measured,torch.zeros((1,10)),low,high,1/480,rates,policy_owned=True)
        _,c=surface_final_guard(c,t,low,high,1/480)
        # submit writes FP32, and the next begin reads that backend target.
        c=c.float().double()
        rows.append(c[0].numpy().copy().tolist())
    return np.asarray(rows)


def table_guard(clearances):
    if clearances is None:raise ValueError("same-clock measured two-hand clearance required")
    return bool(np.all(finite_array(clearances,(2,))>=CLEARANCE_M))


def profile(result_sha):
    if not isinstance(result_sha,str) or len(result_sha)!=64 or any(c not in "0123456789abcdef" for c in result_sha):
        raise ValueError("immutable offline result SHA required")
    return dict(version=VERSION,scope="one_future_guarded_readback_decision_not_contact_or_training_qualification",
        offline_result_sha256=result_sha,native_parent_ledger_path=str(NATIVE/"physical.jsonl"),
        native_parent_ledger_sha256=LEDGER_SHA,native_parent_receipt_path=str(RECEIPT),
        native_parent_receipt_sha256=RECEIPT_SHA,paired_reset_path=str(PAIR),
        paired_reset_sha256=PAIR_SHA,reset_seed_sha256=SEED_SHA,
        action_contract_sha256=hashlib.sha256(json.dumps(ACTION_CONTRACT,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        setup_sdk_steps=2,measurement_sdk_steps=16,total_sdk_steps=18,control_intervals=2,
        phases=[dict(name="repeat_original_zero_interval",sdk_steps=8,action24=[0.]*24),
                dict(name="one_nonzero_readback_interval",sdk_steps=8,action24=action().tolist())],
        expected_nonzero_indices=[1,2,7,8],expected_target_delta_rad=(action()[:12]*.3*8/480).tolist(),
        no_commanded_lift=True,no_commanded_rotation=True,
        original_limits=dict(joint_rate_rad_s=.3,joint_lead_rad=.1,joint_guard_margin_rad=.02,
            mimic_guard_rad=.02,per_pad_force_guard_n=30.,table_floor_m=CLEARANCE_M),
        required_fail_safe=[
            "whole interval reservation before control; no retry after partial SDK advancement",
            "same run/env/episode/reset/rod identity and consecutive SDK+1/rod+12 clocks",
            "original joint/mimic/nonfinite and >30N per-pad guard unchanged",
            "persist actual post-step named link transforms, q/qd/targets, pad poses and rod nodes",
            "same-clock measured source17 and locked5 clearance >=0.0021m for each hand",
            "stop on unexpected wrist safety lift or rotation; never suppress original safety projection",
            "stop on any negative non-target geometric gap or a required pad with no valid in-lattice SDF minimum; individual out-of-domain nodes remain unavailable",
            "one nonzero interval only; no automatic repetition, longer closure, lift or native qualification"],
        existing_missing_fields=["post-step pad/link poses absent from parent ledger",
            "static GPU hand-table contact report unavailable; never treat as zero"],
        native_recipe_created=False,native_qualification=False,teacher_admission=False,grasp_qualified=False)


def load_native(ledger=NATIVE/"physical.jsonl"):
    if digest(ledger)!=LEDGER_SHA:raise ValueError("exact passed native physical ledger required")
    receipt=checked_json(RECEIPT,RECEIPT_SHA)
    if receipt["native_qualification"] is not True or receipt["outer_exit_code"]!=0 or receipt["exit_code"]!=0:
        raise ValueError("actual passed scoped native receipt required")
    if receipt["artifact_sha256"]["physical.jsonl"]!=LEDGER_SHA:
        raise ValueError("receipt does not bind the original physical ledger")
    audit=inspect_interval(ledger)
    rows=[json.loads(line) for line in Path(ledger).read_text().splitlines()][1:]
    physical=[decode_owned(r["payload"]) for r in rows if r["operation"]=="physical"]
    initial={r["payload"]["side"]:decode_owned(r["payload"])["evidence"] for r in rows if r["operation"]=="initial_geometry"}
    return physical,initial,audit


def hand_state(hand):
    names=hand["joint_names"];q=finite_array(hand["q"],(1,17))[0]
    required=(*ACTIVE,"wrist_x","wrist_y","wrist_z","wrist_roll","wrist_pitch","wrist_yaw")
    if (len(names)!=17 or len(set(names))!=17 or not set(required)<=set(names)
            or hand["active_indices"]!=[names.index(n) for n in ACTIVE]):
        raise ValueError("actual named original mounted 17DOF required")
    anchor=finite_array(hand["native_anchor_pose_w"],(1,7))[0].copy()
    if abs(np.linalg.norm(anchor[3:])-1)>1e-5:raise ValueError("unit actual mount quaternion required")
    rot=Rotation.from_quat(anchor[[4,5,6,3]])
    anchor[:3]+=rot.apply([q[names.index(n)] for n in ("wrist_x","wrist_y","wrist_z")])
    angles=[q[names.index(n)] for n in ("wrist_roll","wrist_pitch","wrist_yaw")]
    anchor[3:]=np.roll((rot*Rotation.from_euler("XYZ",angles)).as_quat(),1)
    active=np.array([q[names.index(n)] for n in ACTIVE])
    mimic=[float(q[names.index(f)]-ratio*q[names.index(m)]-offset) for f,m,ratio,offset in MIMIC]
    return anchor,active,mimic


def measure(project,result):
    rows,initial,audit=load_native();pair=load_pair_proposal()
    last=rows[-1]["exchange"];first=rows[0]["exchange"]
    h=last["action_readback"]["post"]["hands"]
    q=np.concatenate([hand_state(h[s])[1] for s in ("left","right")])
    command=np.concatenate([h[s]["backend_targets"][0,h[s]["active_indices"]] for s in ("left","right")])
    target_rows=targets(command,q)
    result.update(native_parent_audit=audit,reset_seed_sha256=SEED_SHA,
        native_parent_receipt_sha256=RECEIPT_SHA,paired_result_sha256=PAIR_SHA,
        action24=action().tolist(),target_rows_rad=target_rows.tolist(),
        original_runtime_floor_m=CLEARANCE_M,numeric_allowance_m=NUMERIC_ALLOWANCE,
        allowance_scope="CPU numerical buffer not a bound on unknown future tracking error",
        sides={},source_sha256={},native_parent_ledger_sha256=LEDGER_SHA)
    files=("scripts/prepare_dh116_dual_pinch_readback_profile.py","scripts/prepare_dh116_pinch_closure_margin.py",
        "scripts/audit_dh116_pinch_initial_pose.py","scripts/audit_dh116_power_closure_constraints.py",
        "scripts/solve_dh116_tripod_sdf_constrained_direction.py","scripts/prepare_dh116_dual_pinch_interval.py",
        "src/go2_pvcnn/control/dh116/tabletop_dual_pinch_builder.py",
        "src/go2_pvcnn/control/dh116/tabletop_command_kernels.py",
        "src/go2_pvcnn/control/dh116/tabletop_palm_mount.py",
        "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py")
    result["source_sha256"]={str(project/f):digest(project/f) for f in files}
    for i,side in enumerate(("left","right")):
        g=load_geometry(project,side=side);init=initial[side]
        if g["source_sha256"]!=init["source_sha256"]:raise ValueError("actual initial geometry source differs")
        grids,binding=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
        pose0,q0,_=hand_state(first["action_readback"]["pre"]["hands"][side])
        pose,qpost,mimic=hand_state(h[side]);old=fk_poses(g,pose0,q0)
        xyzw=finite_array(init["actual_link_poses_xyzw"],(1,23,7))[0]
        actual={n:pose_matrix(np.r_[v[:3],v[6],v[3:6]]) for n,v in zip(init["actual_body_names"],xyzw)}
        residual={n:dict(position_m=float(np.linalg.norm(actual[n][:3,3]-old[n][:3,3])),
            rotation_rad=float(Rotation.from_matrix(actual[n][:3,:3]@old[n][:3,:3].T).magnitude()))
            for n in g["source_colliders"]}
        nodes=finite_array(last["rod"]["node_positions"],(1,41,3))[0]
        axis=nodes[-1]-nodes[0]
        f=np.clip((nodes-nodes[0])@axis/(axis@axis),0,1)
        deviation=float(np.linalg.norm(nodes-(nodes[0]+f[:,None]*axis),axis=1).max())
        records=[]
        # Include measured q followed by the eight intended target states;
        # the backend command-to-measurement difference is recorded separately.
        qs=np.vstack((qpost,target_rows[1:,i*6:i*6+6]))
        delta=qs[-1]-qs[0]
        travel={**{"source:"+k:v for k,v in path_displacement_bound(g,qs[0],delta).items()},
                **{"cooked:"+k:v for k,v in path_displacement_bound(g,qs[0],delta,cooked=True).items()}}
        # Per-segment bounds avoid assuming the tiny measured/command offset is uniform.
        segment_travel=[]
        for a,b in zip(qs[:-1],qs[1:]):
            segment_travel.append({**{"source:"+k:v for k,v in path_displacement_bound(g,a,b-a).items()},
                **{"cooked:"+k:v for k,v in path_displacement_bound(g,a,b-a,cooked=True).items()}})
        for j,qj in enumerate(qs):
            fk=fk_poses(g,pose,qj)
            pred={n:relative_pose(actual[n],old[n],fk[n]) for n in g["source_colliders"]}
            table={};gap={}
            for kind,shapes in (("source",g["source_colliders"]),("cooked",g["cooked_pads"])):
                for n,shape in shapes.items():
                    key=kind+":"+n;v=transform(shape["vertices"],pred[n])
                    table[key]=float(v[:,2].min()-.3)
                    if n not in PADS:
                        gap[key]=support_plane_bound(v,nodes[[0,-1]],.005)["surface_gap_lower_bound_m"]-deviation-NUMERIC_ALLOWANCE
            records.append(dict(sample=j,active_q_rad=qj.tolist(),table_m=table,nonrequired_gap_lower_m=gap,
                required_node_sdf={n:sample_contact(grids[n],pred[n],nodes) for n in PADS}))
        def continuous(field):
            keys=records[0][field]
            return {k:min(min(records[j][field][k],records[j+1][field][k])-segment_travel[j][k]/2
                          for j in range(8))-(NUMERIC_ALLOWANCE if field=="table_m" else 0) for k in keys}
        table_lower=continuous("table_m");gap_lower=continuous("nonrequired_gap_lower_m")
        oldend=np.array(pair[side]["candidate"]["active_q_rad"])+pair[side]["candidate"]["active_delta_rad"]
        fk=fk_poses(g,pose,oldend)
        paired_end={n:sample_contact(grids[n],relative_pose(actual[n],old[n],fk[n]),nodes) for n in PADS}
        result["sides"][side]=dict(source_sha256=g["source_sha256"],sdf_binding=binding,
            initial_native_to_fk_residual=residual,post_mimic_residual_rad=mimic,
            post_base_pose_fk_wxyz=pose.tolist(),post_active_q_rad=qpost.tolist(),
            actual_post_pad_pose_available=False,
            prediction_source="actual_initial_link_pose @ inverse(initial_measured_q_FK) @ target_FK_from_actual_post_q_and_anchor",
            actual_rod_nodes_m=nodes.tolist(),rod_endpoint_chord_deviation_m=deviation,
            command_minus_post_measured_rad=(command[i*6:i*6+6]-qpost).tolist(),
            samples=records,per_segment_point_travel_m=segment_travel,
            continuous_table_lower_m=table_lower,continuous_nonrequired_lower_m=gap_lower,
            conditional_table_passed=bool(min(table_lower.values())>=CLEARANCE_M),
            conditional_nonrequired_clear=bool(min(gap_lower.values())>=0),
            existing_paired_endpoint_node_sdf=paired_end,
            paired_full_path_not_newly_certified=True,actual_future_tracking_unknown=True,
            native_contact_observed=False,native_qualification=False)
    result["conditional_geometry_supported"]=all(
        r["conditional_table_passed"] and r["conditional_nonrequired_clear"] for r in result["sides"].values())
    result["status"]="complete"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,training_updates=0,
        native_recipe_created=False,native_qualification=False,teacher_admission=False)
    def expire(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,50)
    try:measure(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,KeyError) as e:result.update(status="diagnostic_error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    path=a.output/"result.json"
    with path.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    if result.get("conditional_geometry_supported") and result["status"]=="complete":
        with (a.output/"profile.json").open("x") as f:json.dump(profile(digest(path)),f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=result["status"],elapsed_s=result["elapsed_s"],
        conditional_geometry_supported=result.get("conditional_geometry_supported",False))))
    return int(result["status"]=="diagnostic_error")


if __name__=="__main__":raise SystemExit(main())
