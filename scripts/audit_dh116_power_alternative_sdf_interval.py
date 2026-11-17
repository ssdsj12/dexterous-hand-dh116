#!/usr/bin/env python3
"""Two unused original Power orientations: reset clearance and SDF closure bounds."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.spatial import ConvexHull

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,load_geometry,world_vertices,transform,path_displacement_bound)
from scripts.audit_dh116_power_closure_constraints import (
    PRIOR,PRIOR_SHA,POWER_SHA,SELECTION,support_plane_bound)
from scripts.prepare_dh116_flexible_style_poses import POWER_ARCHIVE,load_power_archive
from scripts.prepare_dh116_pinch_closure_margin import sample_contact,fk_poses
from scripts.solve_dh116_tripod_joint_root_direction import envelope_support
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,active_limits,expand
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-power-alternative-sdf-interval-v1"
SDF_DIR=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-sdf-register-rK7gw0/bake")
SDF_SHA="db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab"
NUMERIC_GUARD=2e-6
RESET_FLOOR=CLEARANCE_M+.0002


def select_unused(rows,side):
    used={index for s,index in SELECTION if s==side}
    valid=[r for r in rows if r["mode"]=="minimum_table_clearance_root"
        and r["archive_start_index"] not in used
        and min(v for group in r["initial_rod_settling_gaps_m"].values() for v in group.values())>=0]
    if not valid:raise ValueError("no unused nonoverlapping original Power orientation")
    return min(valid,key=lambda r:(max(r["initial_rod_settling_gaps_m"]["cooked"].values()),r["archive_start_index"]))


def contact_region(grid,radius):
    """Convex superset of every cell where trilinear phi could be < radius."""
    v=np.asarray(grid["values"]);o=finite_array(grid["origin"],(3,));h=float(grid["spacing"])
    if (v.ndim!=3 or min(v.shape)<2 or not np.isfinite(v).all()
            or not np.isfinite([h,radius]).all() or h<=0 or radius<=0):
        raise ValueError("finite SDF lattice and positive original radius required")
    cell_min=np.minimum.reduce([v[a:v.shape[0]-1+a,b:v.shape[1]-1+b,c:v.shape[2]-1+c]
                               for a in (0,1) for b in (0,1) for c in (0,1)])
    cells=np.argwhere(cell_min<radius+NUMERIC_GUARD)
    result=dict(active_cell_count=len(cells),threshold_radius_m=radius,
        scalar_rounding_guard_m=NUMERIC_GUARD,cell_shape=list(cell_min.shape),
        scope="convex_superset_of_possible_interpolant_contact_cells_not_a_new_collider")
    if not len(cells):return dict(result,vertices=[])
    corners=np.unique(np.concatenate([cells+np.array([a,b,c]) for a in (0,1) for b in (0,1) for c in (0,1)]),axis=0)
    points=o+h*corners
    vertices=points[ConvexHull(points).vertices]
    return dict(result,vertices=vertices.tolist(),lattice_corner_count=len(corners))


def interval_certificate(amounts,lower_bounds,travel):
    a=finite_array(amounts);d=finite_array(lower_bounds)
    if (a.shape!=(9,) or d.shape!=(9,) or not 0<a[-1]<=.1+1e-12
            or not np.allclose(a,np.linspace(0,a[-1],9),rtol=0,atol=1e-12)
            or not np.isfinite(travel) or travel<0):
        raise ValueError("nine uniform samples over an original <=0.1rad closure required")
    lower=float(d.min()-travel/16-NUMERIC_GUARD)
    return dict(sample_amount_rad=a.tolist(),sample_axis_to_contact_region_lower_m=d.tolist(),
        uniform_point_travel_m=float(travel),half_sample_fraction=1/16,numeric_guard_m=NUMERIC_GUARD,
        continuous_axis_to_contact_region_lower_m=lower,
        fixed_root_contact_excluded=bool(lower>0),radius_subtracted_again=False,
        global_infeasible=False,native_ready=False)


def load_grids(geometry,summary):
    grids={};bindings={}
    for name,cooked in geometry["cooked_pads"].items():
        key=geometry["side"]+"."+name;record=summary["pads"][key];p=SDF_DIR/(key+".npz")
        if (digest(p)!=record["sdf_file_sha256"] or record["gpu_row_sha256"]!=cooked["capture_sha256"]
                or record["spacing_m"]!=.001 or record["rod_radius_m"]!=.005
                or record["frame"]!="original_pad_coincident_local_basis_m"):
            raise ValueError("original per-side SDF/cooked source changed")
        with np.load(p,allow_pickle=False) as d:grids[name]={k:d[k].copy() for k in d.files}
        bindings[name]=dict(path=str(p),sha256=record["sdf_file_sha256"],capture_sha256=cooked["capture_sha256"])
    return grids,bindings


def original_alternative(prior,archive,geometry):
    side=geometry["side"];record=prior["power"]["sides"][side]
    if record["source_sha256"]!=geometry["source_sha256"]:raise ValueError("original side source geometry changed")
    selected=deepcopy(select_unused(record["records"],side))
    old=[r for r in archive[side] if r["start_index"]==selected["archive_start_index"]]
    if (len(old)!=1 or selected["active_q_rad"]!=old[0]["active_angles"]
            or selected["base_pose_wxyz"][:2]!=old[0]["wrist_pose_w"][:2]
            or selected["base_pose_wxyz"][3:]!=old[0]["wrist_pose_w"][3:]):
        raise ValueError("actual original Power orientation and q required")
    shift=RESET_FLOOR-selected["initial_all_shape_table_clearance_m"]
    selected["base_pose_wxyz"][2]+=shift
    return dict(selected,side=side,independent_style="power",independent_seed_sha256=POWER_SHA,
        additional_vertical_shift_m=float(shift),candidate_scope="unused_original_orientation_reset_only")


def audit_side(geometry,parent,grids,progress):
    pose=parent["base_pose_wxyz"];q=finite_array(parent["active_q_rad"],(6,));expand(q)
    layout=geometry["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
    if set(grids)!=set(DISTAL_PAD_LINKS):raise ValueError("all original five named pad grids required")
    regions={n:contact_region(grids[n],.005) for n in DISTAL_PAD_LINKS}
    transforms=fk_poses(geometry,pose,q)
    table={};rod={}
    for kind,cooked in (("source",False),("cooked",True)):
        for name,v in world_vertices(geometry,pose,q,cooked=cooked).items():
            key=kind+":"+name;table[key]=float(v[:,2].min()-top)
            rod[key]=envelope_support(v,layout,.005)["surface_gap_lower_bound_m"]
    region_quiet={}
    for n,r in regions.items():
        if not r["vertices"]:raise ValueError("original pad SDF has no potential contact cells")
        world=transform(r["vertices"],transforms[n])
        # The region already contains contactRadius dilation. Only use the
        # distance field of the support witness; do not subtract radius twice.
        w=envelope_support(world,layout,.005)
        region_quiet[n]=w["distance_lower_bound_m"]-NUMERIC_GUARD
    if abs(min(table.values())-RESET_FLOOR)>1e-9:raise ValueError("all-shape reset floor correction differs")
    start=dict(table_gap_m=table,all_shape_rod_envelope_lower_m=rod,
        sdf_contact_region_quiet_lower_m=region_quiet,
        all_shape_runtime_floor_passed=min(table.values())>=CLEARANCE_M,
        all_shape_rod_clear=min(rod.values())>=0,all_sdf_quiet_proven=min(region_quiet.values())>0)
    result=dict(candidate=parent,initial=start,contact_regions=regions,curves={},closure_contact_excluded=False,
        native_ready=False,reset_verified=False,teacher_admitted=False,reset_proposal_created=False)
    if not all(start[k] for k in ("all_shape_runtime_floor_passed","all_shape_rod_clear","all_sdf_quiet_proven")):
        return dict(result,blocker="alternative_initial_geometry_unproven")
    nodes=np.array(layout["wire_points"]);nodes[:,2]=top+.005
    upper=active_limits()[1]
    for channel,name in enumerate(DISTAL_PAD_LINKS,1):
        maximum=min(.1,float(upper[channel]-q[channel]))
        delta=np.zeros(6);delta[channel]=maximum
        local=deepcopy(geometry["cooked_pads"]);local[name]["vertices"]=np.array(regions[name]["vertices"])
        surrogate=dict(geometry,cooked_pads=local)
        travel=path_displacement_bound(surrogate,q,delta,cooked=True)[name]
        rows=[]
        for amount in np.linspace(0,maximum,9):
            moved=q.copy();moved[channel]+=amount;expand(moved)
            t=fk_poses(geometry,pose,moved)[name]
            region_world=transform(regions[name]["vertices"],t)
            witness=support_plane_bound(region_world,nodes[[0,-1]],.005)
            sampled=sample_contact(grids[name],t,nodes)
            row=dict(side=geometry["side"],pad=name,active_channel=channel,
                active_delta_rad=float(amount),support_witness=witness,
                axis_to_contact_region_lower_m=witness["distance_lower_bound_m"],node_sdf=sampled)
            with _commit_row_without_alarm():rows.append(row);progress(row)
        cert=interval_certificate([r["active_delta_rad"] for r in rows],
            [r["axis_to_contact_region_lower_m"] for r in rows],travel)
        with _commit_row_without_alarm():result["curves"][name]=dict(samples=rows,certificate=cert)
    result["excluded_pads"]=[n for n,r in result["curves"].items() if r["certificate"]["fixed_root_contact_excluded"]]
    result["closure_contact_excluded"]=bool(result["excluded_pads"])
    result["blocker"]="at_least_one_pad_cannot_reach_original_SDF_contact_in_bounded_fixed_root_interval" if result["excluded_pads"] else "no_exclusion_certificate_not_contact_or_path_success"
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=50.);args=p.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50:p.error("CPU soft budget 1..50 seconds")
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",sides={},completed_pad_samples=[],
        new_orientations=0,native_ready=False,reset_proposal_created=False,physics_steps=0,training_updates=0)
    def expire(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    try:
        project=Path(__file__).resolve().parents[1];prior=checked_json(PRIOR,PRIOR_SHA)
        for n,k in (("scripts/prepare_dh116_flexible_style_poses.py","preparation_source_sha256"),
                    ("scripts/audit_dh116_pinch_initial_pose.py","geometry_audit_sha256")):
            if digest(project/n)!=prior[k]:raise ValueError("frozen original geometry implementation differs")
        archive=load_power_archive(POWER_ARCHIVE,POWER_SHA);summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
        result.update(prior_sha256=PRIOR_SHA,power_archive_sha256=POWER_SHA,sdf_summary_sha256=SDF_SHA,
            source_sha256={str(project/n):digest(project/n) for n in (
                "scripts/audit_dh116_power_alternative_sdf_interval.py","scripts/audit_dh116_power_closure_constraints.py",
                "scripts/prepare_dh116_pinch_closure_margin.py","scripts/solve_dh116_tripod_joint_root_direction.py",
                "scripts/audit_dh116_pinch_initial_pose.py","scripts/prepare_dh116_flexible_style_poses.py",
                "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")},
            geometry_sources={},sdf_bindings={})
        for side in ("left","right"):
            geometry=load_geometry(project,side=side);parent=original_alternative(prior,archive,geometry)
            grids,bindings=load_grids(geometry,summary)
            result["geometry_sources"][side]=geometry["source_sha256"];result["sdf_bindings"][side]=bindings
            row=audit_side(geometry,parent,grids,result["completed_pad_samples"].append)
            with _commit_row_without_alarm():result["sides"][side]=row;result["new_orientations"]+=1
        result["status"]="complete"
    except BudgetExpired:result["status"]="partial_timeout"
    except ValueError as exc:result.update(status="diagnostic_error",error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-started
    with (args.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=result["status"],elapsed_s=result["elapsed_s"],pad_samples=len(result["completed_pad_samples"]))))
    return 1 if result["status"]=="diagnostic_error" else 0


if __name__=="__main__":raise SystemExit(main())
