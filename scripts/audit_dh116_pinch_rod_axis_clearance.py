#!/usr/bin/env python3
"""Analytic static rod-axis reset roll; no native launch or controller change.

The angle changes the fixed RESET transform about the rod, not a wrist action.
Joint path, USD, meshes, SDF, material, success thresholds and actuator limits
remain unchanged. A table interval is necessary, never contact qualification.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,pose_matrix,transform,load_geometry,
    world_vertices,path_displacement_bound)
from scripts.prepare_dh116_pinch_closure_margin import fk_poses,sample_contact,sample_grid
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
from scripts.solve_dh116_tripod_joint_root_direction import envelope_support
from scripts import replay_dh116_native_parameter_r0 as evidence
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,expand
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

VERSION="dh116-pinch-rod-axis-clearance-v1"
NUMERICAL_MARGIN_M=1e-9  # Extra margin, never relaxation of original 2.1mm.
SAMPLES=161
ANGLE_LIMIT_RAD=math.pi/12  # Independent static-reset diagnostic, +/-15 degrees.
MAX_STATES=4*SAMPLES
PADS=("finger14_link","finger23_link")
E=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
LEFT=E/"pinch-closure-margin-20261113-_dmnkcda/export/candidate.json"
LEFT_SHA="3d04ccca63263b4480e1831693b8d7c70f1336170f2006b5c6a3a1f414553fe0"
RIGHT=E/"right-pinch-closure-cpu-20261113-oIQiFC/run01/result.json"
RIGHT_SHA="1a271578961192096d3d4aa5ec03785807c3940908bd9444fd0c285be86dff1f"
RIGHT_ID="pinch-right-original_right-r0-p-10-b0-x0"


def feasible_intervals(coefficients,bounds):
    """Intersect EXACT sinusoid solution arcs: a*cos(t)+b*sin(t)>=c.

    This evaluates no new FK states and does not use a grid or local optimizer.
    Floating point interval edges are subsequently rechecked with extra margin.
    """
    rows=finite_array(coefficients)
    box=finite_array(bounds,(2,))
    if rows.ndim!=2 or rows.shape[1]!=3 or not len(rows) or not -math.pi<=box[0]<box[1]<=math.pi:
        raise ValueError("finite Nx3 constraints and ordered [-pi,pi] domain required")
    intervals=[tuple(box)]
    for a,b,c in rows:
        radius=math.hypot(a,b)
        if radius==0:
            if c>0:return []
            continue
        if c>radius:return []
        if c<=-radius:continue
        alpha=math.acos(c/radius);phase=math.atan2(b,a)
        allowed=[]
        for k in range(-2,3):
            low=phase-alpha+2*math.pi*k;high=phase+alpha+2*math.pi*k
            for lo,hi in intervals:
                x,y=max(lo,low),min(hi,high)
                if x<=y:allowed.append((x,y))
        intervals=sorted(set(allowed))
        if not intervals:break
    return [list(x) for x in intervals]


def choose_angle(intervals):
    if not intervals:return None
    if any(a<=0<=b for a,b in intervals):return 0.
    low,high=min(intervals,key=lambda x:min(abs(x[0]),abs(x[1])))
    eps=min(1e-5,(high-low)/2)
    return float(low+eps if low>0 else high-eps)


def axis_transform(axis,angle):
    axis=finite_array(axis,(3,))
    if not np.isfinite(angle) or abs(angle)>math.pi:raise ValueError("finite rotation required")
    rot=Rotation.from_rotvec([0.,float(angle),0.]).as_matrix()
    out=np.eye(4);out[:3,:3]=rot;out[:3,3]=axis-rot@axis
    return out


def rotate_about_axis(pose,axis,angle):
    old=pose_matrix(pose);out=axis_transform(axis,angle)@old
    return np.r_[out[:3,3],np.roll(Rotation.from_matrix(out[:3,:3]).as_quat(),1)].tolist()


def table_coefficients(vertices,axis,whole_path_travel,samples):
    vertices=finite_array(vertices);axis=finite_array(axis,(3,))
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices)
        or not np.isfinite(whole_path_travel) or whole_path_travel<0
        or type(samples) is not int or samples<2):
        raise ValueError("finite vertices and nonnegative continuous travel bound required")
    # z'=axis.z+(z-axis.z)*cos(t)-(x-axis.x)*sin(t)
    rhs=.3+CLEARANCE_M+whole_path_travel/(2*(samples-1))+NUMERICAL_MARGIN_M-axis[2]
    return np.column_stack((vertices[:,2]-axis[2],-(vertices[:,0]-axis[0]),
                            np.full(len(vertices),rhs)))


def validate_parent(parent):
    if parent["side"] not in ("left","right") or parent["active_joint_names"]!=list(ACTIVE):
        raise ValueError("independent side and original active joint order required")
    pose_matrix(parent["base_pose_wxyz"])
    q=finite_array(parent["active_q_rad"],(6,));d=finite_array(parent["active_delta_rad"],(6,))
    if np.any(d<0) or np.any(d>.1+1e-12) or np.any(d[[0,3,4,5]]):
        raise ValueError("original two-digit 160/480 second closure at <=0.3rad/s required")
    expand(q);expand(q+d)
    return q,d


def qualification(*,table,other,initial,gaps,opposed=True):
    finite=all(v is not None and np.isfinite(v) for v in gaps)
    checks=dict(table_clear=bool(table>=CLEARANCE_M),non_target_clear=bool(other>=0),
                initial_clear=bool(initial>=0),
                both_end_sdf_margin=bool(finite and all(-.0003<=v<=-.0001 for v in gaps)),
                opposed=bool(opposed))
    return dict(**checks,static_geometry_passed=all(checks.values()),
                native_ready=False,teacher_admitted=False)


def load_parents():
    left=checked_json(LEFT,LEFT_SHA);right=checked_json(RIGHT,RIGHT_SHA)
    if left["side"]!="left" or left["native_ready"] is not False or right["status"]!="complete":
        raise ValueError("exact historical unqualified source evidence required")
    r=next(x for x in right["rows"] if x["candidate_id"]==RIGHT_ID)
    parents={
        "left":dict(side="left",base_pose_wxyz=left["base_pose_world_wxyz"],
            active_joint_names=left["active_joint_names"],active_q_rad=left["candidate_initial_q_rad"],
            active_delta_rad=(np.array(left["target_end_q_rad"])-left["candidate_initial_q_rad"]).tolist(),
            source_sha256=left["original_geometry_source_sha256"]),
        "right":dict(**{k:r[k] for k in ("side","base_pose_wxyz","active_joint_names","active_q_rad",
                                        "active_delta_rad")},source_sha256=right["geometry_source_sha256"])}
    for parent in parents.values():validate_parent(parent)
    return parents


def archived_nodes():
    receipt=json.loads(evidence.read_pinned(evidence.RECEIPT,evidence.RECEIPT_SHA))
    if receipt["outer_exit_code"]!=0 or receipt["physics_clock"]["total_steps"]!=18:
        raise ValueError("exact completed parameter-task native receipt required")
    raw=evidence.read_pinned(Path(receipt["output"])/"physical.jsonl",
                             receipt["artifact_sha256"]["physical.jsonl"])
    rows=[decode_owned(json.loads(line)) for line in raw.splitlines()]
    physical=[r["payload"]["exchange"] for r in rows if r.get("operation")=="physical"]
    if len(physical)!=16 or [r["physics_step"] for r in physical]!=list(range(3,19)):
        raise ValueError("exact complete frozen 16-endpoint ledger required")
    row=physical[-1]
    nodes=finite_array(row["rod"]["node_positions"],(1,41,3))[0]
    return nodes,dict(receipt_sha256=evidence.RECEIPT_SHA,
        ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],
        identity=row["closure_geometry"]["raw"]["identity"])


def source_sets(g,pose,q):
    return {**{"source:"+n:v for n,v in world_vertices(g,pose,q).items()},
            **{"cooked:"+n:v for n,v in world_vertices(g,pose,q,cooked=True).items()}}


def endpoint_sdf(grids,poses,nodes):
    pads={};normals=[]
    for name in PADS:
        row=sample_contact(grids[name],poses[name],nodes)
        normal=None
        if row["local_node_at_minimum"] is not None:
            _,grad,_=sample_grid(grids[name],[row["local_node_at_minimum"]])
            if np.linalg.norm(grad[0])>0:
                normal=poses[name][:3,:3]@grad[0];normal/=np.linalg.norm(normal)
        row["predicted_gradient_normal_world"]=None if normal is None else normal.tolist()
        pads[name]=row;normals.append(normal)
    dot=None if any(n is None for n in normals) else float(np.dot(*normals))
    return dict(pads=pads,gradient_normal_dot=dot,opposed=dot is not None and dot<-.2,
                native_contact_observed=False)


def audit_side(g,parent,grids,nodes,result,progress,*,angle_selector=None,check_contact_path=False):
    q,d=validate_parent(parent)
    layout=g["reference"]["reference_layout"]
    top=layout["table_center"][2]+layout["table_size"][2]/2
    if abs(top-.3)>1e-12:raise ValueError("original table World Z changed")
    axis=np.array([layout["wire_points"][0][0],0.,top+.005])
    delta_to_axis=nodes[:,[0,2]]-axis[[0,2]]
    side_result=dict(parent=parent,axis_world_m=axis.tolist(),angle_domain_rad=[-ANGLE_LIMIT_RAD,ANGLE_LIMIT_RAD],
        actual_node_axis_residual_max_m=float(np.linalg.norm(delta_to_axis,axis=1).max()),
        selected_angle_rad=None,static_geometry_passed=False,native_ready=False,teacher_admitted=False,
        original_samples=[],rotated_samples=[])
    result["sides"][g["side"]]=side_result
    travel={**{"source:"+k:v for k,v in path_displacement_bound(g,q,d).items()},
            **{"cooked:"+k:v for k,v in path_displacement_bound(g,q,d,cooked=True).items()}}
    cached=[];coeff=[];old_min={}
    def count():
        if result["geometry_states"]>=MAX_STATES:raise ValueError("frozen geometry-state budget exhausted")
        result["geometry_states"]+=1
    for index,fraction in enumerate(np.linspace(0,1,SAMPLES)):
        count();state=q+fraction*d;shapes=source_sets(g,parent["base_pose_wxyz"],state)
        if len(shapes)!=22:raise ValueError("exact independent original17 plus cooked5 required")
        row=dict(path_index=index,fraction=float(fraction),table_lower_m={})
        for key,v in shapes.items():
            c=table_coefficients(v,axis,travel[key],SAMPLES);coeff.append(c)
            row["table_lower_m"][key]=float(v[:,2].min()-top-travel[key]/320)
            old_min[key]=min(old_min.get(key,float("inf")),row["table_lower_m"][key])
        cached.append((state,shapes))
        with _commit_row_without_alarm():
            side_result["original_samples"].append(row);progress(dict(side=g["side"],phase="original",**row))
    coefficients=np.concatenate(coeff)
    intervals=feasible_intervals(coefficients,[-ANGLE_LIMIT_RAD,ANGLE_LIMIT_RAD])
    selection=None if angle_selector is None else angle_selector(coefficients,intervals)
    angle=choose_angle(intervals) if selection is None else selection["angle_rad"]
    if selection is not None:side_result["angle_selection"]=selection
    side_result.update(analytic_constraint_count=len(coefficients),feasible_angle_intervals_rad=intervals,
                       original_continuous_table_lower_m=old_min,selected_angle_rad=angle)
    original_fk=fk_poses(g,parent["base_pose_wxyz"],q+d)
    side_result["original_endpoint_sdf"]=endpoint_sdf(grids,original_fk,nodes)
    if angle is None:
        side_result["blocked_reason"]="no_table_safe_static_rod_axis_angle_in_frozen_domain"
        return
    direct_margin=coefficients[:,0]*np.cos(angle)+coefficients[:,1]*np.sin(angle)-coefficients[:,2]
    if direct_margin.min()<-1e-12:raise ValueError("analytic angle failed direct constraint recheck")
    pose=rotate_about_axis(parent["base_pose_wxyz"],axis,angle);rotation=axis_transform(axis,angle)
    candidate={**parent,"base_pose_wxyz":pose,"static_reset_only":True,"native_ready":False}
    tables={};others={};initial={};path_depth=[]
    for index,(state,old) in enumerate(cached):
        count()
        shapes={k:transform(v,rotation) for k,v in old.items()}
        # At both endpoints independently recalculate source FK to catch composition/order errors.
        if index in (0,SAMPLES-1):
            independent=source_sets(g,pose,state)
            if any(not np.allclose(v,independent[k],rtol=0,atol=1e-12) for k,v in shapes.items()):
                raise ValueError("rigid rod-axis transform disagrees with recomputed original FK")
        row=dict(path_index=index,fraction=index/(SAMPLES-1),table_lower_m={},non_target_lower_m={})
        for key,v in shapes.items():
            remainder=travel[key]/320
            value=float(v[:,2].min()-top-remainder)
            tables[key]=min(tables.get(key,float("inf")),value);row["table_lower_m"][key]=value
            if key.split(":")[1] not in PADS:
                value=rod_support_lower(v,nodes)-remainder
                others[key]=min(others.get(key,float("inf")),value);row["non_target_lower_m"][key]=value
            if index==0:initial[key]=envelope_support(v,layout,.005)["surface_gap_lower_bound_m"]
        if check_contact_path or index in (0,SAMPLES-1):
            row["node_sdf"]=endpoint_sdf(grids,fk_poses(g,pose,state),nodes)
        with _commit_row_without_alarm():
            side_result["rotated_samples"].append(row);progress(dict(side=g["side"],phase="rotated",**row))
    end=side_result["rotated_samples"][-1]["node_sdf"]
    flags=qualification(table=min(tables.values()),other=min(others.values()),initial=min(initial.values()),
                        gaps=[end["pads"][n]["minimum_gap_m"] for n in PADS],opposed=end["opposed"])
    side_result.update(candidate=candidate,continuous_table_lower_m=tables,
        continuous_non_target_lower_m=others,initial_settling_envelope_lower_m=initial,
        recomputed_endpoint_sdf=end,
        base_position_delta_m=(np.array(pose[:3])-parent["base_pose_wxyz"][:3]).tolist(),**flags)
    side_result["limitations"]=[
        "Static source17/cooked5 path, not actual tracking or self-collision/arm reach qualification.",
        "Initial full vertical 305..308mm envelope is checked; actual settling/bounce is unmeasured.",
        "Actual frozen nodes are resampled after roll; ideal infinite-axis invariance alone is insufficient.",
        "Endpoint SDF depth is bounded; intervening required-pad penetration and dynamics remain unqualified.",
        "Table-safe roll does not repair an initially missing node/pad contact or non-target collision."]


def run(project,result,progress,*,angle_selector=None,check_contact_path=False):
    parents=load_parents();nodes,native=archived_nodes()
    result.update(parent_sources=dict(left=dict(path=str(LEFT),sha256=LEFT_SHA),
        right=dict(path=str(RIGHT),sha256=RIGHT_SHA,candidate_id=RIGHT_ID)),
        archived_native=native,actual_nodes_m=nodes.tolist(),sides={},geometry_states=0,
        physics_steps=0,ppo_updates=0,native_recipe_created=False,teacher_admitted=False)
    result["implementation_sha256"]={name:digest(project/name) for name in (
        "scripts/audit_dh116_pinch_rod_axis_clearance.py","scripts/audit_dh116_pinch_initial_pose.py",
        "scripts/prepare_dh116_pinch_closure_margin.py","scripts/solve_dh116_tripod_joint_root_direction.py",
        "src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py")}
    summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
    for side,parent in parents.items():
        g=load_geometry(project,side=side)
        if g["source_sha256"]!=parent["source_sha256"]:raise ValueError("independent side source SHA mismatch")
        grids,bindings=grids_for(g,summary)
        audit_side(g,parent,grids,nodes,result,progress,angle_selector=angle_selector,check_contact_path=check_contact_path)
        result["sides"][side]["sdf_bindings"]=bindings
    result["status"]="complete"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(version=VERSION,status="partial",scope="new_CPU_static_reset_construction",
        budget=dict(wall_seconds=90,geometry_states=MAX_STATES,per_side_reset_angles=1,
                    original_path_samples=SAMPLES,rotated_path_samples=SAMPLES,
                    native_physics_steps=0,ppo_updates=0))
    old=signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(BudgetExpired()))
    signal.setitimer(signal.ITIMER_REAL,90)
    try:
        with (args.output/"progress.jsonl").open("x") as f:
            def progress(row):
                f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
            try:run(Path(__file__).resolve().parents[1],result,progress)
            finally:f.flush();os.fsync(f.fileno())
    except BudgetExpired:result["status"]="partial_budget"
    except (ValueError,KeyError,StopIteration) as exc:
        result.update(status="diagnostic_error",error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-started
    with (args.output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","geometry_states","elapsed_s","error")}))
    return int(result["status"]!="complete")


if __name__=="__main__":raise SystemExit(main())
