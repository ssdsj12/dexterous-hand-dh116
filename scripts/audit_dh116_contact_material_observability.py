#!/usr/bin/env python3
"""Read-only contact/material observability audit; never an R0 slip provider."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from scripts.dh116_pinch_evidence_v2 import validate_phase_pair_record,validate_action_targets
from scripts.validate_dh116_pinch_v4 import summarize_contacts,validate_simulation_clear_receipt
from go2_pvcnn.control.dh116.tabletop_deformable_pinch_force_precision import validate_ledger
from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import material_station_resolution
from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract

VERSION="dh116-contact-material-observability-cpu-v1"
SOURCE_HASHES={
 "third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh":"8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb",
 "third_party/yarnball/KittenEngine/YarnBall/sim/iteration.cu":"84dab66f671bc0052640b55a89056acbc319071a42f4d9197ff8199acad8753c",
 "third_party/yarnball/KittenEngine/YarnBall/YarnBall.h":"152dfc26742e03867f0a4e7ce4791ce5f718414169582443bb1d3b4fd72576cc",
 "csrc/sdf.cu":"e5b3b1b00881aa1aff1697214f36b4e12310f8d62655455be7ce6b6df0369c12",
 "csrc/bindings.cpp":"89b3928faf49ec26e2d53614c0e30c5d6242892e54c8345c45f4d5702077d8d5"}
PADS=("finger14_link","finger23_link","finger33_link","finger43_link","finger53_link")
CONFIG="configs/dh116_tasks/flexible_pinch_pose_candidate_v4.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_contract(root):
    result={}
    for name,expected in SOURCE_HASHES.items():
        path=(Path(root)/name).resolve(strict=True)
        if sha(path)!=expected:raise ValueError("compiled-tree source changed: "+name)
        result[name]=dict(path=str(path),sha256=expected)
    return result


def friction_ratio(speed,*,mu_s,mu_k,v_slip):
    """Exact real-arithmetic scalar version of contact.cuh:27-45, not CUDA."""
    if (any(type(x) not in (float,int) or not math.isfinite(x) for x in (speed,mu_s,mu_k,v_slip))
            or speed<0 or not mu_s>mu_k>0 or v_slip<=0):
        raise ValueError("finite speed and explicit positive weakening coefficients required")
    if speed<=v_slip:return mu_s*speed/v_slip
    blend=1.-min(1.,speed/v_slip-1.)
    return blend*mu_s+(1.-blend)*mu_k


def force_inversion_counterexamples(*,mu_s,mu_k,v_slip,substep_dt):
    if type(substep_dt) not in (float,int) or not math.isfinite(substep_dt) or substep_dt<=0:
        raise ValueError("positive substep duration required")
    ratio=(mu_s+mu_k)/2
    speeds=[ratio/mu_s*v_slip,1.5*v_slip]
    forces=[friction_ratio(v,mu_s=mu_s,mu_k=mu_k,v_slip=v_slip) for v in speeds]
    plateau=[2*v_slip,max(3*v_slip,.001/substep_dt)]
    return dict(scope="real_arithmetic_noninjectivity_of_exact_source_law",
        transition_speeds_m_s=speeds,transition_same_force_ratios=forces,
        transition_displacements_m=[v*substep_dt for v in speeds],
        plateau_speeds_m_s=plateau,plateau_same_force_ratios=[
            friction_ratio(v,mu_s=mu_s,mu_k=mu_k,v_slip=v_slip) for v in plateau],
        plateau_same_force_displacements_m=[v*substep_dt for v in plateau],
        unique_displacement_inversion=False,physical_slip_upper_bound_m=None,
        caveat="counterexample_is_not_a_claim_of_these_speeds_in_the_native_run")


def rolling_counterexample(*,radius,angle_rad):
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (radius,angle_rad)) or radius<=0:
        raise ValueError("explicit finite radius/angle required")
    travel=radius*abs(angle_rad)
    return dict(scope="unobserved_cross_section_material_rotation_counterexample",
        radius_m=radius,angle_rad=angle_rad,surface_material_travel_m=travel,
        identical_node_sphere_geometry=True,exceeds_r0_limit=travel>.0005,
        native_run_rotation_measured=False)


def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:
        raise ValueError("actual finite WXYZ unit pad quaternion required")
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def final_substep_endpoint(pair,row,*,radius):
    """Use only the final solver substep, simultaneous with the stored SDK pose.

    This is the position of a fixed centerline material node in a pad frame.
    It is not the location of a fixed surface material marker or contact s.
    """
    if (type(radius) not in (float,int) or not math.isfinite(radius) or radius<=0
            or pair["substep"]!=11 or type(pair["mesh_index"]) is not int
            or not 0<=pair["mesh_index"]<5):
        raise ValueError("explicit radius and final-substep actual hand pair required")
    nodes=np.asarray(row["rod_nodes_w"],dtype=float)
    n,p=np.asarray(pair["outward_normal_world"]),np.asarray(pair["contact_point_world_m"])
    i,m=pair["node_index"],pair["mesh_index"]
    if (type(i) is not int or not 0<=i<41 or nodes.shape!=(41,3)
            or not np.isfinite(nodes).all() or n.shape!=(3,) or p.shape!=(3,)
            or not np.isfinite(n).all() or not np.isfinite(p).all()
            or not np.isclose(np.linalg.norm(n),1,atol=2e-5,rtol=0)):
        raise ValueError("complete actual node/contact vectors required")
    error=float(np.linalg.norm(p+radius*n-nodes[i]))
    if error>2e-6:raise ValueError("pair surface point and final same-node readback differ")
    R=rotation(row["post_state"]["link_quat"][m])
    origin=np.asarray(row["post_state"]["link_pos"][m],dtype=float)
    if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError("finite actual pad origin required")
    return dict(phase_step=row["phase_step"],physics_step=row["physics_step"],rod_tick=row["rod_tick"],
        solver_substep=11,node_index=i,pad=PADS[m],
        pad_local_node_m=((nodes[i]-origin)@R).tolist(),pad_local_normal=(n@R).tolist(),
        pair_reconstructed_center_error_m=error,
        consistency_tolerance_m=2e-6,consistency_tolerance_is_not_slip_error_bound=True,
        continuous_material_s_m=None,physical_slip_m=None)


def audit(native_output,*,project,source_root):
    d,project=Path(native_output),Path(project)
    sources=source_contract(source_root)
    def read(name):return json.loads((d/name).read_text())
    def lines(name):
        text=(d/name).read_text()
        if not text.endswith("\n"):raise ValueError("incomplete ledger")
        return [json.loads(x) for x in text.splitlines()]
    plan=read("plan.json");cfg=json.loads((project/CONFIG).read_text())
    if plan["sources"]["local_sources"][CONFIG]!=sha(project/CONFIG):
        raise ValueError("successor candidate source differs from captured plan")
    rows,pairs=lines("physics.jsonl"),lines("pairs.jsonl")
    force=validate_ledger(rows);validate_action_targets(rows,plan,cfg)
    contact=summarize_contacts(rows,pairs,expected_bindings=plan["actual_pad_bindings"])
    if rows[-1]["state_backend_sha256"]!="92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0":
        raise ValueError("pair backend identity differs")
    radius=cfg["rod"]["radius_m"];tracks={};ratios=[]
    for row,record in zip(rows,pairs):
        parsed=validate_phase_pair_record(record,row)
        for pair in parsed["pairs"]:
            if pair["mesh_index"]>=5:continue
            N=pair["normal_force_on_rod_n"]
            if N>0:ratios.append(float(np.linalg.norm(pair["tangential_force_on_rod_world_n"])/N))
            if pair["substep"]!=11 or N<=0:continue
            end=final_substep_endpoint(pair,row,radius=radius)
            key=end["pad"]+":"+str(end["node_index"])
            track=tracks.setdefault(key,dict(reference=end,samples=[]))
            ref=track["reference"]
            displacement=np.asarray(end["pad_local_node_m"])-ref["pad_local_node_m"]
            normal=np.asarray(ref["pad_local_normal"])
            tangent=displacement-normal*np.dot(displacement,normal)
            end["fixed_centerline_tangent_displacement_m"]=float(np.linalg.norm(tangent))
            track["samples"].append(end)
    for track in tracks.values():
        samples=track["samples"]
        track.update(sample_count=len(samples),last=samples[-1],
            contiguous_endpoint_contact=all(b["phase_step"]==a["phase_step"]+1 for a,b in zip(samples,samples[1:])),
            max_centerline_tangent_displacement_m=max(v["fixed_centerline_tangent_displacement_m"] for v in samples),
            scope="fixed_node_centerline_endpoint_diagnostic_not_physical_surface_slip")
    hand=hand_material_contract(cfg["hand"]["material_profile"])["hand_contact_material"]
    mu_s=max(cfg["rod"]["mu_s"],hand["static_friction"])
    mu_k=max(cfg["rod"]["mu_k"],hand["dynamic_friction"])
    rest=plan["reset_readback"]["rest_lengths_m"]
    if np.asarray(rest).ndim==2:rest=rest[0]
    owners={phase:read(phase+".simulation-owners.json") for phase in
            ("resources_acquired","before_stop","after_stop","before_app_close","after_app_close")}
    source_files=("plan.json","physics.jsonl","pairs.jsonl","simulation_clear_receipt.json",
                  *[p+".simulation-owners.json" for p in owners])
    return dict(version=VERSION,scope="source_and_completed_native_data_observability_audit_only",
        native_output=str(d.resolve()),artifact_sha256={n:sha(d/n) for n in source_files},
        source_contract=sources,force_audit=force,contact_observation=contact,
        contact_source_fields=["substep","node","mesh","point","normal","forceOnRod","torqueOnMesh"],
        contact_source_line_ranges=[[27,45],[94,109],[165,206]],
        substep_pose_source="iteration.cu:17-32_fraction_(sub+1)/nSub",
        final_node_clock_source="iteration.cu:114-120_readout_then_position_update",
        material_resolution=material_station_resolution(rest,source="actual_reset_readback_rest_lengths_m"),
        fixed_centerline_endpoint_tracks=tracks,
        force_ratio_range=None if not ratios else [min(ratios),max(ratios)],
        force_inversion=force_inversion_counterexamples(mu_s=mu_s,mu_k=mu_k,
            v_slip=cfg["rod"]["v_slip_m_s"],substep_dt=1/5760),
        friction_parameter_source="frozen_constructor_setter_values_not_native_contact_getters",
        radius_source="frozen_constructor_contact_radius_not_native_contact_getter",
        consecutive_contact_node_center_reconstructible=True,
        reconstruction_limit="contact_only_centers_plus_source_mesh_interpolation_do_not_label_surface_material",
        rolling_counterexample=rolling_counterexample(radius=radius,angle_rad=.2),
        simulation_clear_receipt=validate_simulation_clear_receipt(d/"simulation_clear_receipt.json"),
        simulation_owners=owners,
        missing=["continuous_segment_contact_material_coordinate_and_error_bound",
            "fixed_surface_material_marker_identity_and_cross_section_director_history",
            "direct_surface_material_relative_displacement_including_contact_gaps",
            "contact_material_correspondence_across_no_contact_and_node_switches",
            "validated_combined_anchor_and_tangential_uncertainty_at_0.5mm"],
        proposed_future_source="validated_continuous_contact_correspondence_and_fixed_surface_marker_transport_with_substep_directors",
        proposal_is_currently_available=False,r0_material_s_m=None,r0_tangential_slip_m=None,
        r0_slip_acceptance=None,physical_steps_executed=0,native_execution_performed=False,
        native_qualification=False,teacher_admission=False,training_reward=None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-output",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists() or a.output.is_symlink():raise FileExistsError("preserve existing evidence")
    result=audit(a.native_output,project=Path(__file__).resolve().parents[1],source_root=a.source_root)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output),native_execution_performed=False)))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
