#!/usr/bin/env python3
"""Actual endpoint contact/material correspondence; no new native execution."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json,os,time,math
from pathlib import Path
import numpy as np
from scripts import replay_dh116_bilateral_contact_r0 as frozen
from scripts import replay_dh116_segment_contact_r0 as stream
from scripts import replay_dh116_native_parameter_r0 as parameters
from scripts import audit_dh116_bilateral_native_evidence as pinned
from go2_pvcnn.control.dh116 import tabletop_contact_material_correspondence as model
from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as native_source
from go2_pvcnn.control.dh116.tabletop_cosserat_material_marker_reference import _rotation
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_contact_diagnostics import MIN_NORMAL_FORCE_N

VERSION="dh116-actual-bilateral-material-endpoint-audit-v1"
FROZEN_CONFIG=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/integration")/native_source.CONFIG

def archived_source(expected):
    """Only the exact immutable original config path may differ by worktree."""
    actual=deepcopy(expected)
    current=str(native_source.ROOT/native_source.CONFIG)
    if (actual["sources"].pop(current)!=native_source.CONFIG_SHA
            or FROZEN_CONFIG.is_symlink() or pinned.sha(FROZEN_CONFIG)!=native_source.CONFIG_SHA):
        raise ValueError("archived original configuration bytes changed")
    actual["sources"][str(FROZEN_CONFIG)]=native_source.CONFIG_SHA
    return actual


def pose_matrix(xyzw):
    a=np.asarray(xyzw,dtype=float)
    if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError("finite actual XYZ/XYZW pose required")
    t=np.eye(4);t[:3,3]=a[:3];t[:3,:3]=_rotation(a[3:])
    return t

def same_endpoint_sequence(prior,now):
    pinned.require_episode(prior,now)
    if (type(now["physics_step"]) is not int or type(now["rod_tick"]) is not int
            or now["physics_step"]!=prior["physics_step"]+1 or now["rod_tick"]!=prior["rod_tick"]+12
            or not math.isclose(now["timestamp_s"],now["physics_step"]/480,rel_tol=0,abs_tol=1e-9)):
        raise ValueError("consecutive actual SDK/12 solver endpoints required")

def partition_pairs(pairs):
    result=[];reasons=Counter()
    for pair in pairs:
        sub=pair["substep"]
        if type(sub) is not int or not 0<=sub<12:raise ValueError("actual 12-substep pair index required")
        if pair["mesh_name"]=="table":reasons["table_outside_hand_material_scope"]+=1
        elif sub!=11:reasons["no_same_substep_director"]+=1
        else:result.append(pair)
    return result,dict(reasons)

def qualification():
    return dict(scope="actual_endpoint_sources_with_unqualified_cross_section_model",
        new_physics_steps=0,ppo_updates=0,teacher_admission=False,native_qualification=False,
        contact_marker_correspondence_verified=False,collision_surface_model_equivalence_verified=False,
        substep_director_history_available=False,substep_path_bound_available=False,
        r0_tangential_slip_m=None,physical_error_bound_m=None,training_reward=None)

def data(packet,rod,source):
    if packet["sources"]!=source:raise ValueError("compiled endpoint source differs")
    radius=parameters.check_radius_packet(packet,rod)
    nodes=np.asarray(packet["node_positions_m"])[0]
    q=np.asarray(packet["segment_quaternions_xyzw"])[0]
    rest=np.asarray(packet["raw_native_parameters"]["rest_lengths_m"])[0]
    model.real_directors(rest,nodes,q)
    return nodes,q,rest,radius["value"]

def run(output):
    folder,receipt=frozen.load_source()
    compiled_source=archived_source(native_source.source_basis())
    p0=decode_owned(json.loads((folder/"director_reset.json").read_bytes())["payload"])
    counts=Counter();rejections=Counter();pair_counts=Counter();tracks={}
    mappings=[];last=None;reference_uid=None;last_rod=None;initial_rest=None
    capture_rows=0;row_count=0
    def emit(stream,row):stream.write(json.dumps(encode_owned(row),allow_nan=False)+"\n")
    with (output/"contact_mapping.jsonl").open("x") as pairs_file,(output/"fixed_marker_tracks.jsonl").open("x") as tracks_file:
        for event in stream.stream_events(folder/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"]):
            op=event.get("operation");counts[str(op)]+=1
            if op=="reset":
                if last is not None:raise ValueError("one actual reset per frozen native source")
                p=event["payload"]["reset"];last=p["after_clock"]
                nodes,q,initial_rest,radius=data(p0,p["rod_reset"],compiled_source)
                reference_uid=p0["material_reference_uid"];last_rod=p["rod_reset"]
                if not reference_uid:raise ValueError("actual material reference UID required")
            elif op=="physical":
                if last is None:raise ValueError("actual reset before material endpoint required")
                x=event["payload"]["exchange"];packet=x["material_director"]
                identity=dict(packet["identity"],policy_step=x["policy_step"],substep=x["substep"])
                actual_action_clock=x["action_readback"]["post"]["identity"][0]
                if any(actual_action_clock.get(k)!=v for k,v in identity.items()):
                    raise ValueError("material endpoint and actual executed action identity differ")
                same_endpoint_sequence(last,identity)
                nodes,q,rest,radius=data(packet,x["rod"],compiled_source)
                if reference_uid!=packet["material_reference_uid"] or not np.array_equal(rest,initial_rest):
                    raise ValueError("material identity or native rest lengths changed")
                raw=x["closure_geometry"]["raw"]
                for key in ("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s"):
                    if raw["identity"][key]!=identity[key]:raise ValueError("pad/rod/director endpoint clock differs")
                poses={}
                for side in ("left","right"):
                    h=raw["hands"][side];names=h["body_names"];transforms=np.asarray(h["link_poses_xyzw"])[0]
                    if len(set(names))!=len(names) or transforms.shape!=(len(names),7):raise ValueError("complete unique hand body poses required")
                    for i,name in enumerate(names):poses[side+"."+name]=pose_matrix(transforms[i])
                final,rejected=partition_pairs(x["rod"]["rod_mesh_contact_pairs"]["pairs"])
                rejections.update(rejected);seen=set();loaded=set()
                for pair in final:
                    name=pair["mesh_name"];node=pair["node_index"];key=name+":"+str(node)
                    if key in seen:raise ValueError("duplicate node/mesh at same final solver substep")
                    seen.add(key);pair_counts[name]+=1
                    if name not in poses:raise ValueError("actual pad link missing")
                    candidate=model.contact_candidate(pair,rest,nodes,q,radius_m=radius,
                        material_reference_uid=reference_uid)
                    row=dict(identity=identity,solver_substep=11,pad=name,node_index=node,
                        source_pair=pair,mapping=candidate["mapping"],candidate_marker=asdict(candidate["marker"]),
                        candidate_qualified=False)
                    mappings.append(row);emit(pairs_file,row)
                    if pair["normal_force_on_rod_n"]>MIN_NORMAL_FORCE_N:
                        loaded.add(key)
                        if key not in tracks:
                            tracks[key]=dict(reference=model.bind_candidate(candidate,rest,nodes,q,poses[name]),
                                pad=name,node=node,first_identity=identity,initial_mapping=candidate["mapping"],
                                first_contact_pair=pair,samples=0,loaded_endpoints=0,missing_contact_endpoints=0,
                                maximum=None,contact_losses=0,last_contact=True)
                for key,track in tracks.items():
                    contact=key in loaded
                    measured=model.transport(track["reference"],rest,nodes,q,poses[track["pad"]],endpoint_contact_present=contact)
                    sample=dict(identity=identity,solver_substep=11,pad=track["pad"],node_index=track["node"],
                        first_reference_identity=track["first_identity"],**measured)
                    emit(tracks_file,sample);capture_rows+=1;track["samples"]+=1
                    track["loaded_endpoints"]+=int(contact);track["missing_contact_endpoints"]+=int(not contact)
                    if track["last_contact"] and not contact:track["contact_losses"]+=1
                    track["last_contact"]=contact
                    if track["maximum"] is None or measured["marker_tangent_displacement_m"]>track["maximum"]["value_m"]:
                        track["maximum"]=dict(value_m=measured["marker_tangent_displacement_m"],identity=identity,
                            pad=track["pad"],node_index=track["node"],endpoint_contact_present=contact)
                last=identity;last_rod=x["rod"];row_count+=1
        for file in (pairs_file,tracks_file):file.flush();os.fsync(file.fileno())
    expected=Counter({"None":1,"setup":2,"initial_geometry":2,"reset":1,"begin":29,"physical":232,"end":29})
    if counts!=expected or row_count!=232 or last["physics_step"]!=234:raise ValueError("complete frozen native prefix required")
    by_pad={}
    for name in pair_counts:
        rows=[r for r in mappings if r["pad"]==name]
        by_pad[name]=dict(endpoint_pair_count=len(rows),
            sphere_center_residual_m=pinned.stat([r["mapping"]["sphere_center_residual_m"] for r in rows]),
            axial_cross_section_residual_m=pinned.stat([r["mapping"]["axial_cross_section_residual_m"] for r in rows]),
            radial_surface_residual_m=pinned.stat([r["mapping"]["radial_surface_residual_m"] for r in rows]),
            candidate_model_residual_m=pinned.stat([r["mapping"]["surface_model_candidate_residual_m"] for r in rows]),
            above_original_0p5mm_model_residual=sum(r["mapping"]["exceeds_original_slip_threshold_as_model_residual"] for r in rows))
    summaries={key:{k:v for k,v in t.items() if k not in ("reference","last_contact")} for key,t in tracks.items()}
    if not mappings:raise ValueError("actual hand endpoint contact required, no synthetic replacement")
    worst=max(mappings,key=lambda r:r["mapping"]["surface_model_candidate_residual_m"])
    return dict(version=VERSION,status="complete",source_receipt_sha256=pinned.ACTUAL_RECEIPT_SHA,
        ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],source_native_steps=234,
        source_contract=compiled_source,operation_counts=dict(counts),endpoint_rows=row_count,
        final_substep_pair_count=len(mappings),pair_partition_reasons=dict(rejections),by_pad=by_pad,
        worst_model_residual=worst,track_count=len(tracks),track_rows=capture_rows,tracks=summaries,
        no_candidate_dropped_for_large_model_residual=True,reference_reselected_on_contact_loss=False,
        candidate_bound_to_first_loaded_node_and_pad=True,
        limitation="480Hz endpoint observations cannot bound intervening5760Hz path; node-sphere surface need not equal Cosserat cross-section material",
        **qualification())

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:result=run(a.output)
    except Exception as exc:result=dict(version=VERSION,status="blocked",reason=type(exc).__name__+": "+str(exc),**qualification())
    result.update(elapsed_s=time.monotonic()-start,script_sha256=pinned.sha(__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","elapsed_s","final_substep_pair_count","track_count","track_rows","by_pad")}))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
