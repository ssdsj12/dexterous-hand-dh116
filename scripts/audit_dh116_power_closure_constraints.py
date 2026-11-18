#!/usr/bin/env python3
"""Five original Power roots: continuous fixed-root closure bounds, no native grant."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import time

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,load_geometry,path_displacement_bound,settling_gap,world_vertices)
from scripts.prepare_dh116_flexible_style_poses import (
    ACTIVE,POWER_ARCHIVE,POWER_SHA,MARGIN,MAX_ACTIVE_DELTA,DURATION_S,REACH_TOLERANCE,
    load_power_archive,measure_closure_grid)
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm,SOURCES
from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-power-continuous-closure-constraints-v2"
EVIDENCE=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
PRIOR=EVIDENCE/"tripod-power-preparation-20261113-7zhi9kgm/final-preparation.json"
PRIOR_SHA="e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6"
SELECTION=(("left",15),("right",1),("right",12),("right",13),("right",15))
ADDITIONAL={
    "teacher-wrap-geometry-20261112-joWzRM/multifinger_pregrasp_v1.json":
        "234eb791df392d0b78f3902d86b156b2255d22b74cfbc0a39eb8c5ba74ef29a7",
    "teacher-wrap-local-fd-20261112-AGNuAX/fixed_patch_fd.json":
        "183a70c0aa7db7270004d6c8b7f6a12a90de5889698704a65a6aef5cec97734e",
}


def certify_curve(samples,*,maximum_delta,travel_bound,support_lower_bounds):
    if (not np.isfinite([maximum_delta,travel_bound]).all()
            or not 0<maximum_delta<=MAX_ACTIVE_DELTA+1e-12 or travel_bound<0):
        raise ValueError("original positive closure interval and finite displacement bound required")
    if len(samples)!=9:
        raise ValueError("nine samples over the entire closure interval required")
    amounts=finite_array([r["active_delta_rad"] for r in samples],(9,))
    gaps=finite_array([r["capsule_surface_gap_m"] for r in samples],(9,))
    if not np.allclose(amounts,np.linspace(0,maximum_delta,9),rtol=0,atol=1e-12):
        raise ValueError("curve does not cover the exact original interval")
    support=finite_array(support_lower_bounds,(9,))
    if np.any(support>gaps+2e-6):
        raise ValueError("analytic lower bound exceeds stored feasible distance estimate")
    # Every point in the interval is within half a sample spacing of a sample.
    # The original angle-times-lever bound is uniform and linear in path length;
    # convex-solid distance is 1-Lipschitz under the resulting point motion.
    radius_fraction=float(np.max(np.diff(amounts))/(2*maximum_delta))
    numerical_margin=2e-6
    lower=float(support.min()-travel_bound*radius_fraction-numerical_margin)
    return dict(sample_delta_rad=amounts.tolist(),sample_qp_distance_estimate_m=gaps.tolist(),
        sample_support_surface_gap_lower_bound_m=support.tolist(),
        bound_source="explicit_support_plane_separation_not_QP_optimality",
        maximum_positive_delta_rad=maximum_delta,uniform_point_travel_bound_m=travel_bound,
        nearest_sample_path_fraction=radius_fraction,numeric_margin_m=numerical_margin,
        continuous_surface_gap_lower_bound_m=lower,required_endpoint_tolerance_m=REACH_TOLERANCE,
        fixed_root_contact_excluded=bool(lower>REACH_TOLERANCE),
        necessary_root_translation_m=max(0.,lower-REACH_TOLERANCE),
        necessary_translation_scope="lower_bound_only_same_orientation_and_closure_interval",
        global_infeasible=False,native_ready=False)


def support_plane_bound(vertices,axis_endpoints,radius):
    vertices=finite_array(vertices);axis=finite_array(axis_endpoints,(2,3))
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(radius) or radius<=0:
        raise ValueError("finite convex hull vertices, axis endpoints and positive radius required")
    angles=np.arange(64)*2*np.pi/64
    normals=np.stack((np.cos(angles),np.zeros(64),np.sin(angles)),axis=1)
    # Any norm<=1 direction is dual feasible; shrinking also avoids roundoff >1.
    normals/=np.linalg.norm(normals,axis=1)[:,None]+1e-12
    hull=(vertices@normals.T).min(axis=0);patch=(axis@normals.T).max(axis=0)
    index=int(np.argmax(hull-patch));separation=float(hull[index]-patch[index])
    distance=max(0.,separation)
    return dict(direction_world=normals[index].tolist(),direction_count=64,
        hull_projection_min_m=float(hull[index]),axis_projection_max_m=float(patch[index]),
        distance_lower_bound_m=distance,surface_gap_lower_bound_m=distance-radius,
        method="convex_hull_min_minus_finite_axis_max_support_projection")


def curve_certificates(geometry,row,curve):
    q=np.asarray(row["active_q_rad"]);budget=joint_budget(q)
    layout=geometry["reference"]["reference_layout"]
    axis=np.array(layout["wire_points"])[[0,-1]].copy()
    radius=geometry["old_candidate"]["rod"]["radius_m"]
    axis[:,2]=layout["table_center"][2]+layout["table_size"][2]/2+radius
    certificates={}
    for channel,name in enumerate(curve["required_pads"],1):
        delta=np.zeros(6);delta[channel]=budget["allowed_positive_delta_rad"][channel]
        travel=path_displacement_bound(geometry,q,delta,cooked=True)[name]
        witnesses=[]
        for sample in curve["closure_grid"][name]:
            moved=q.copy();moved[channel]+=sample["active_delta_rad"]
            vertices=world_vertices(geometry,row["base_pose_wxyz"],moved,cooked=True)[name]
            witnesses.append(support_plane_bound(vertices,axis,radius))
        certificate=certify_curve(curve["closure_grid"][name],
            maximum_delta=float(delta[channel]),travel_bound=travel,
            support_lower_bounds=[w["surface_gap_lower_bound_m"] for w in witnesses])
        certificates[name]=dict(certificate,support_witnesses=witnesses)
    return certificates


def joint_budget(q):
    q=finite_array(q,(6,));expand(q)
    lower,upper=active_limits();headroom=upper-q
    allowed=np.minimum(MAX_ACTIVE_DELTA,headroom)
    return dict(active_joint_names=list(ACTIVE),active_q_rad=q.tolist(),
        lower_limits_rad=lower.tolist(),upper_limits_rad=upper.tolist(),
        upper_headroom_rad=headroom.tolist(),allowed_positive_delta_rad=allowed.tolist(),
        limited_by_control_duration=(headroom>MAX_ACTIVE_DELTA).tolist(),
        maximum_velocity_rad_s=.3,duration_s=DURATION_S,thumb_spread_fixed_during_closure=True)


def load_selection(path,project,geometries):
    prior=checked_json(path,PRIOR_SHA)
    if (digest(Path(project)/"scripts/prepare_dh116_flexible_style_poses.py")!=prior["preparation_source_sha256"]
            or digest(Path(project)/"scripts/audit_dh116_pinch_initial_pose.py")!=prior["geometry_audit_sha256"]
            or prior["power"]["archive_sha256"]!=POWER_SHA):
        raise ValueError("immutable Power geometry helper or archive differs")
    archive=load_power_archive(POWER_ARCHIVE,POWER_SHA)
    selected=[]
    for side,index in SELECTION:
        record=prior["power"]["sides"][side]
        if record["source_sha256"]!=geometries[side]["source_sha256"]:
            raise ValueError("actual per-side Power source geometry changed")
        rows=[r for r in record["records"] if r["archive_start_index"]==index
              and r["mode"]=="minimum_table_clearance_root"]
        originals=[r for r in archive[side] if r["start_index"]==index]
        if len(rows)!=1 or len(originals)!=1:
            raise ValueError("unique original negative Power record required")
        row=rows[0];old=originals[0]
        if (row["archived_wrap_qualified"] is not False or row["active_joint_names"]!=list(ACTIVE)
                or row["active_q_rad"]!=old["active_angles"]
                or row["base_pose_wxyz"][:2]!=old["wrist_pose_w"][:2]
                or row["base_pose_wxyz"][3:]!=old["wrist_pose_w"][3:]):
            raise ValueError("independent original Power q/root lineage differs")
        selected.append(dict(row,side=side,candidate_id=f"power-constraint-{side}-{index}",
            independent_style="power",independent_seed_sha256=POWER_SHA))
    return selected,prior["power"]


def initial_geometry(geometry,row):
    q=row["active_q_rad"];pose=row["base_pose_wxyz"]
    layout=geometry["reference"]["reference_layout"];radius=geometry["old_candidate"]["rod"]["radius_m"]
    top=layout["table_center"][2]+layout["table_size"][2]/2
    sets={"source":world_vertices(geometry,pose,q),"cooked":world_vertices(geometry,pose,q,cooked=True)}
    table={kind:{n:float(v[:,2].min()-top) for n,v in shapes.items()} for kind,shapes in sets.items()}
    gaps={kind:{n:settling_gap(v,layout,radius)["capsule_surface_gap_m"] for n,v in shapes.items()}
          for kind,shapes in sets.items()}
    minimum=min(v for group in table.values() for v in group.values())
    original=row["initial_rod_settling_gaps_m"]
    residual=max(abs(gaps[k][n]-original[k][n]) for k in gaps for n in gaps[k])
    if residual>1e-8 or abs(minimum-row["initial_all_shape_table_clearance_m"])>1e-10:
        raise ValueError("fresh source geometry differs from immutable original record")
    return dict(table_gap_by_shape_m=table,rod_envelope_gap_by_shape_m=gaps,
        prior_gap_max_abs_residual_m=residual,minimum_table_gap_m=minimum,
        limiting_table_shapes=[f"{k}/{n}" for k,group in table.items() for n,v in group.items()
                               if abs(v-minimum)<1e-9],
        static_floor_m=MARGIN,initial_static_floor_margin_m=minimum-MARGIN,
        runtime_floor_m=CLEARANCE_M,initial_runtime_floor_margin_m=minimum-CLEARANCE_M,
        maximum_downward_translation_before_static_floor_m=max(0.,minimum-MARGIN),
        source_rod_overlaps={n:v for n,v in gaps["source"].items() if v<0},
        cooked_rod_overlaps={n:v for n,v in gaps["cooked"].items() if v<0})


def obtain_curves(geometry,row,prior):
    if row["side"]=="right":
        matches=[r["candidate"] for r in prior["sides"]["right"]["closure_grid_probes"]
                 if r["archive_start_index"]==row["archive_start_index"] and r["grid_performed"]]
        if len(matches)!=1:
            raise ValueError("unique already measured right closure curve required")
        return deepcopy(matches[0]),"immutable_prior_nine_angle_measurements"
    return measure_closure_grid(geometry,row,(1,2,3,4,5)),"new_left_five_digit_nine_angle_measurements"


def audit_one(geometry,row,prior):
    initial=initial_geometry(geometry,row)
    result=dict(candidate_id=row.get("candidate_id"),side=row["side"],
        archive_start_index=row["archive_start_index"],initial=initial,
        closure_evaluated=False,native_ready=False,teacher_admitted=False,
        reset_proposal_created=False,global_infeasible=False)
    if initial["source_rod_overlaps"] or initial["cooked_rod_overlaps"]:
        return dict(result,blockers=["initial_source_or_cooked_rod_overlap"])
    q=np.asarray(row["active_q_rad"]);budget=joint_budget(q)
    curve,origin=obtain_curves(geometry,row,prior)
    certificates=curve_certificates(geometry,row,curve)
    excluded=[n for n,c in certificates.items() if c["fixed_root_contact_excluded"]]
    return dict(result,closure_evaluated=True,curve_source=origin,joint_budget=budget,
        per_pad_continuous_certificates=certificates,
        independently_excluded_pads=excluded,fixed_root_five_pad_contact_excluded=bool(excluded),
        original_grid_endpoint=dict(active_delta_rad=curve["active_delta_rad"],
            max_required_endpoint_abs_gap_m=curve["max_required_endpoint_abs_gap_m"],
            geometric_thumb_partner_dot=curve["geometric_thumb_partner_dot"],
            blockers=curve["coarse_blockers"]),
        blockers=["fixed_root_positive_closure_cannot_reach_all_five_pads"] if excluded
            else ["necessary_reach_only_no_full_path_or_contact_qualification"])


FROZEN_V1_RESULT_SHA="763032412514c70d34910eaa7b0bd5f4a4d3b53d751ddd41e65c26b01c0ca529"
FROZEN_V1_SOURCE_SHA="8e53ada5a36703150b306e8312cd868100672db1e0907ff6ff214ec1b03009e7"


def read_frozen_report(path,project,geometries):
    report=checked_json(path,FROZEN_V1_RESULT_SHA)
    script="scripts/audit_dh116_power_closure_constraints.py"
    if report["source_sha256"][script]!=FROZEN_V1_SOURCE_SHA:
        raise ValueError("original numerical-estimate producer differs")
    for name,expected in report["source_sha256"].items():
        if name!=script and digest(Path(project)/name)!=expected:
            raise ValueError("original report numerical geometry source changed")
    if report["geometry_source_sha256"]!={s:g["source_sha256"] for s,g in geometries.items()}:
        raise ValueError("original per-side actual geometry source changed")
    if report["status"]!="complete" or len(report["rows"])!=5:
        raise ValueError("complete frozen original five-root report required")
    return report


def recheck_row(geometry,row,old):
    if not old["closure_evaluated"]:
        return dict(deepcopy(old),provenance="frozen_initial_geometry_no_new_queries")
    curve=dict(required_pads=list(old["per_pad_continuous_certificates"]),closure_grid={})
    for name,cert in old["per_pad_continuous_certificates"].items():
        curve["closure_grid"][name]=[dict(active_delta_rad=a,capsule_surface_gap_m=g)
            for a,g in zip(cert["sample_delta_rad"],cert["sample_surface_gap_m"])]
    certificates=curve_certificates(geometry,row,curve)
    excluded=[n for n,c in certificates.items() if c["fixed_root_contact_excluded"]]
    return dict(deepcopy(old),per_pad_continuous_certificates=certificates,
        original_curve_source=old["curve_source"],curve_source="frozen_report_samples_no_new_QP",
        independently_excluded_pads=excluded,fixed_root_five_pad_contact_excluded=bool(excluded),
        support_FK_samples=45,initial_geometry_repeated=False,
        blockers=["fixed_root_positive_closure_cannot_reach_all_five_pads"] if excluded
            else ["necessary_reach_only_no_full_path_or_contact_qualification"])


def source_inventory():
    out=[]
    for relative,expected in ADDITIONAL.items():
        row=checked_json(EVIDENCE/relative,expected)
        fields=("scope","geometry_candidates","archived_scene_matches_reference",
                "native_contact_verified","native_grasp_verified","teacher_training_ready")
        out.append(dict(path=str(EVIDENCE/relative),sha256=expected,
            observed={k:row[k] for k in fields if k in row},
            admission_granted=False,used_as_full_collider_geometry=False))
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,default=50.)
    parser.add_argument("--recheck-report",type=Path)
    args=parser.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50:
        parser.error("CPU deadline must be 1..50 seconds")
    project=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();finished=set()
    result=dict(version=VERSION,status="partial_setup",rows=[],physics_steps=0,optimizer_updates=0,
        native_ready=False,teacher_admitted=False,global_infeasible=False,
        reset_proposal_created=False,total_original_roots=5,new_pose_candidates=0)
    def expire(signum,frame):raise BudgetExpired()
    prior_signal_handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    with (args.output/"progress.jsonl").open("x") as ledger:
        try:
            geometries={s:load_geometry(project,side=s) for s in ("left","right")}
            selected,prior=load_selection(PRIOR,project,geometries)
            result.update(parent_result_sha256=PRIOR_SHA,archive_sha256=POWER_SHA,
                original_selected_records=selected,additional_source_inventory=source_inventory(),
                geometry_source_sha256={s:g["source_sha256"] for s,g in geometries.items()})
            frozen=read_frozen_report(args.recheck_report,project,geometries) if args.recheck_report else None
            if frozen:
                result.update(mode="support_plane_recheck_no_new_QP",
                    derived_from_report_sha256=FROZEN_V1_RESULT_SHA,
                    prior_estimate_not_a_certified_lower_bound=True,
                    new_closure_curve_queries=0,new_initial_geometry_queries=0)
            for row in selected:
                if frozen:
                    old=next(x for x in frozen["rows"] if x["side"]==row["side"]
                             and x["archive_start_index"]==row["archive_start_index"])
                    record=recheck_row(geometries[row["side"]],row,old)
                else:
                    record=audit_one(geometries[row["side"]],row,prior)
                with _commit_row_without_alarm():
                    result["rows"].append(record);finished.add((row["side"],row["archive_start_index"]))
                    ledger.write(json.dumps(record,allow_nan=False)+"\n");ledger.flush()
            result["status"]="complete"
        except BudgetExpired:
            result["status"]="partial_timeout"
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior_signal_handler)
    result.update(elapsed_s=time.monotonic()-started,
        remaining_original_roots=[list(x) for x in SELECTION if x not in finished],
        new_left_closure_curves=sum(r.get("curve_source","").startswith("new_left") for r in result["rows"]),
        reused_right_closure_curves=sum(r.get("curve_source","").startswith("immutable") for r in result["rows"]),
        source_sha256={name:digest(project/name) for name in (*SOURCES,
            "scripts/audit_dh116_power_closure_constraints.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")},
        progress_sha256=digest(args.output/"progress.jsonl"),soft_wall_seconds=args.wall_seconds)
    with (args.output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","elapsed_s","remaining_original_roots",
        "new_left_closure_curves","reused_right_closure_curves","native_ready")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
