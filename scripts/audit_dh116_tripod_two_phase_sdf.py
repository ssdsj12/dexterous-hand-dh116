#!/usr/bin/env python3
"""One independent Tripod quiet-start/settled-close hypothesis; CPU diagnostics."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,load_geometry,world_vertices,transform,
    convex_patch_closest,path_displacement_bound)
from scripts.audit_dh116_power_alternative_sdf_interval import contact_region
from scripts.audit_dh116_power_closure_constraints import support_plane_bound
from scripts.prepare_dh116_pinch_closure_margin import sample_contact,fk_poses
from scripts.solve_dh116_tripod_sdf_constrained_direction import (
    load_parent,grids_for,SDF_DIR,SDF_SHA,PARENT,PARENT_SHA)
from scripts.solve_dh116_tripod_joint_root_direction import REQUIRED,envelope_support
from scripts.screen_dh116_tripod_proximal import TRIPOD_SHA
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-tripod-two-phase-sdf-v1"
RUNTIME_FLOOR=CLEARANCE_M
INTERVALS=64
TARGET=.0001  # A diagnostic positive node gap, not a replacement contact threshold.


def command(parent,extra):
    extra=finite_array(extra,(3,))
    if (parent["side"]!="left" or parent["independent_style"]!="tripod"
            or parent["independent_seed_sha256"]!=TRIPOD_SHA):
        raise ValueError("independent original left Tripod source required")
    q=finite_array(parent["active_q_rad"],(6,))
    pose=finite_array(parent["base_pose_wxyz"],(7,))
    initial=q.copy();initial[1:4]-=.03
    endpoint=q.copy();endpoint[1:4]+=extra
    delta=endpoint-initial
    if np.any(extra<0) or np.max(delta)>.1+1e-12:
        raise ValueError("original positive <=0.3rad/s, 160/480s closure required")
    expand(initial);expand(endpoint)
    return dict(independent_style="tripod",independent_seed_sha256=TRIPOD_SHA,side="left",
        base_pose_wxyz=pose.tolist(),active_joint_names=list(ACTIVE),
        initial_q_rad=initial.tolist(),endpoint_q_rad=endpoint.tolist(),active_delta_rad=delta.tolist(),
        active_velocity_rad_s=(delta/(160/480)).tolist(),quiet_steps=48,closure_steps=160,
        reset_verified=False,native_ready=False,scope="offline_two_phase_hypothesis_not_reset_recipe")


def direction(query):
    base=finite_array(query(np.zeros(3)),(3,))
    jac=[]
    for h in (.001,.0005):
        jac.append(np.stack([(finite_array(query(np.eye(3)[j]*h),(3,))-
                              finite_array(query(-np.eye(3)[j]*h),(3,)))/(2*h)
                             for j in range(3)],axis=1))
    relative=float(np.linalg.norm(jac[1]-jac[0])/max(np.linalg.norm(jac[1]),1e-12))
    if relative>.2:raise ValueError("unstable two-scale node SDF derivative")
    if np.any(np.diag(jac[1])>=-1e-8) or np.max(abs(jac[1]-np.diag(np.diag(jac[1]))))>1e-7:
        raise ValueError("independent decreasing three-digit SDF derivative required")
    step=(TARGET-base)/np.diag(jac[1])
    if np.any(step<0) or np.any(step>.07):raise ValueError("single prediction exceeds original closure budget")
    return dict(baseline_node_sdf_gap_m=base.tolist(),jacobians=[j.tolist() for j in jac],
        jacobian_relative_difference=relative,proposed_extra_rad=step.tolist(),
        target_node_gap_m=TARGET,target_is_diagnostic_not_physics_threshold=True,
        newton_iterations=1,query_count=13)


def path_summary(tables,nonrequired,travel):
    if not tables or not nonrequired or set(tables)!=set(travel) or not set(nonrequired)<=set(travel):
        raise ValueError("complete named table/nonrequired/travel identities required")
    def lower(rows):
        out={}
        for k,v in rows.items():
            a=finite_array(v,(INTERVALS+1,))
            b=float(travel[k])
            if not np.isfinite(b) or b<0:raise ValueError("finite positive point travel required")
            out[k]=float(a.min()-b/(2*INTERVALS))
        return out
    t,n=lower(tables),lower(nonrequired)
    return dict(table_lower_by_shape_m=t,nonrequired_lower_by_shape_m=n,
        minimum_table_lower_m=min(t.values()),minimum_nonrequired_lower_m=min(n.values()),
        table_2p1mm_passed=bool(min(t.values())>=RUNTIME_FLOOR),
        nonrequired_settled_axis_clear=bool(min(n.values())>=0),
        intervals=INTERVALS,nearest_sample_fraction=1/(2*INTERVALS),
        point_travel_bound_m=travel,
        scope="fixed_root_linear_q_path_fixed_straight_settled_axis_only")


def shapes_at(g,pose,q):
    return {**{"source:"+k:v for k,v in world_vertices(g,pose,q).items()},
            **{"cooked:"+k:v for k,v in world_vertices(g,pose,q,cooked=True).items()}}


def node_samples(grids,g,pose,q,nodes):
    fk=fk_poses(g,pose,q)
    return {k:sample_contact(grids[k],fk[k],nodes) for k in REQUIRED}


def run(project,result):
    old=load_parent();summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
    files=("scripts/audit_dh116_tripod_two_phase_sdf.py",
        "scripts/audit_dh116_power_alternative_sdf_interval.py",
        "scripts/audit_dh116_power_closure_constraints.py",
        "scripts/prepare_dh116_pinch_closure_margin.py",
        "scripts/solve_dh116_tripod_sdf_constrained_direction.py",
        "scripts/solve_dh116_tripod_joint_root_direction.py",
        "scripts/audit_dh116_pinch_initial_pose.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py",
        "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")
    result.update(parent_path=str(PARENT),parent_sha256=PARENT_SHA,
        source_sha256={str(project/f):digest(project/f) for f in files},
        sdf_summary_sha256=SDF_SHA,geometry_sources={},sdf_bindings={},
        original_runtime_floor_m=RUNTIME_FLOOR,original_static_floor_m=.0019)
    for side in ("left","right"):
        g=load_geometry(project,side=side)
        if g["source_sha256"]!=old["geometry_source_sha256"][side]:raise ValueError("original side geometry changed")
        grids,binding=grids_for(g,summary);result["geometry_sources"][side]=g["source_sha256"]
        result["sdf_bindings"][side]=binding
        p=old["sides"][side]["best"]["candidate"];pose=p["base_pose_wxyz"];q=np.array(p["active_q_rad"])
        layout=g["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
        nodes=np.array(layout["wire_points"]);nodes[:,2]=top+.005
        if side=="right":
            v=shapes_at(g,pose,q)["source:finger13_link"]
            qp=convex_patch_closest(v,nodes[0],[nodes[-1]-nodes[0]])
            result["right_existing_endpoint"]=dict(candidate=p,
                source_finger13_settled_gap_upper_m=qp["distance_m"]-.005+2e-6,qp=qp,
                rejected=bool(qp["distance_m"]-.005+2e-6<0),
                scope="unchanged_endpoint_local_overlap_not_global_infeasibility")
            continue
        q0=q.copy();q0[1:4]-=.03;expand(q0)
        initial=shapes_at(g,pose,q0);fk0=fk_poses(g,pose,q0)
        quiet=dict(table_gap_by_shape_m={k:float(v[:,2].min()-top) for k,v in initial.items()},
            all_shape_rod_envelope_lower_m={k:envelope_support(v,layout,.005)["surface_gap_lower_bound_m"]
                                         for k,v in initial.items()},sdf_contact_region_envelope_lower_m={})
        # The region already contains radius dilation; never subtract R again.
        for k,grid in grids.items():
            region=contact_region(grid,.005)
            v=transform(np.array(region["vertices"]),fk0[k])
            quiet["sdf_contact_region_envelope_lower_m"][k]=envelope_support(v,layout,.005)["distance_lower_bound_m"]-2e-6
        quiet["static_conditions_passed"]=bool(min(quiet["table_gap_by_shape_m"].values())>=RUNTIME_FLOOR
            and min(quiet["all_shape_rod_envelope_lower_m"].values())>=0
            and min(quiet["sdf_contact_region_envelope_lower_m"].values())>0)
        quiet.update(initial_q_rad=q0.tolist(),base_pose_wxyz=pose,
            registered_rod_axis_z_m=float(np.array(layout["wire_points"])[0,2]),
            settled_hypothesis_axis_z_m=float(nodes[0,2]),reset_verified=False,
            actual_quiet_steps_observed=0,table_support_and_settling_not_simulated=True)
        result["left_quiet_start"]=quiet
        def query(extra):
            qs=q.copy();qs[1:4]+=extra;expand(qs)
            samples=node_samples(grids,g,pose,qs,nodes)
            values=[samples[k]["minimum_gap_m"] for k in REQUIRED]
            if any(v is None for v in values):raise ValueError("required node outside SDF domain")
            row=dict(extra_rad=np.asarray(extra).tolist(),samples=samples)
            with _commit_row_without_alarm():result["sdf_queries"].append(row)
            return values
        fit=direction(query);candidate=command(p,fit["proposed_extra_rad"])
        result.update(left_direction=fit,left_hypothesis=candidate)
        endq=np.array(candidate["endpoint_q_rad"]);delta=endq-q0
        endpoint=shapes_at(g,pose,endq);required={};normals=[]
        for k in REQUIRED:
            v=endpoint["cooked:"+k];qp=convex_patch_closest(v,nodes[0],[nodes[-1]-nodes[0]])
            lo=support_plane_bound(v,nodes[[0,-1]],.005)["surface_gap_lower_bound_m"]-2e-6
            hi=qp["distance_m"]-.005+2e-6
            required[k]=dict(gap_interval_m=[lo,hi],feasible_gap_m=qp["distance_m"]-.005,qp=qp)
            normals.append(qp["geometric_axis_to_hull_direction"])
        dots=[None if normals[0] is None or n is None else float(np.dot(normals[0],n)) for n in normals[1:]]
        gap=np.array([required[k]["gap_interval_m"] for k in REQUIRED])
        opposed=all(d is not None and d<-.2 for d in dots)
        result["left_endpoint"]=dict(required_pads=required,node_sdf=node_samples(grids,g,pose,endq,nodes),
            geometric_thumb_partner_dot=dots,original_coarse_0p5mm_passed=bool(np.all(abs(gap)<=.0005) and opposed),
            original_strict_50um_passed=bool(np.all(abs(gap)<=.00005) and opposed))
        tables={k:[] for k in initial};nonrequired={k:[] for k in initial if k.split(":")[1] not in REQUIRED}
        travel={**{"source:"+k:v for k,v in path_displacement_bound(g,q0,delta).items()},
                **{"cooked:"+k:v for k,v in path_displacement_bound(g,q0,delta,cooked=True).items()}}
        for i,fraction in enumerate(np.linspace(0,1,INTERVALS+1)):
            shapes=shapes_at(g,pose,q0+fraction*delta)
            t={k:float(v[:,2].min()-top) for k,v in shapes.items()}
            n={k:support_plane_bound(shapes[k],nodes[[0,-1]],.005)["surface_gap_lower_bound_m"]-2e-6 for k in nonrequired}
            with _commit_row_without_alarm():
                result["path_samples"].append(dict(index=i,fraction=float(fraction),table_m=t,nonrequired_settled_lower_m=n))
                for k,v in t.items():tables[k].append(v)
                for k,v in n.items():nonrequired[k].append(v)
        result["left_path"]=path_summary(tables,nonrequired,travel)
        # Negative lower bounds alone do not prove collision: preserve feasible QP witnesses.
        result["left_endpoint_nonrequired_witnesses"]={}
        for k in nonrequired:
            if result["path_samples"][-1]["nonrequired_settled_lower_m"][k]<0:
                qp=convex_patch_closest(endpoint[k],nodes[0],[nodes[-1]-nodes[0]])
                result["left_endpoint_nonrequired_witnesses"][k]=dict(qp=qp,gap_upper_m=qp["distance_m"]-.005+2e-6)
    result["status"]="complete"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=50)
    a=p.parse_args(argv)
    if not np.isfinite(a.wall_seconds) or not 1<=a.wall_seconds<=50:p.error("CPU soft wall limit 1..50s")
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",sdf_queries=[],path_samples=[],
        native_ready=False,reset_proposal_created=False,teacher_admitted=False,physics_steps=0,training_updates=0)
    def timeout(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_seconds)
    try:run(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,KeyError) as exc:result.update(status="diagnostic_error",error=str(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=result["status"],elapsed_s=result["elapsed_s"],
        sdf_queries=len(result["sdf_queries"]),path_samples=len(result["path_samples"]))))
    return int(result["status"]=="diagnostic_error")


if __name__=="__main__":raise SystemExit(main())
