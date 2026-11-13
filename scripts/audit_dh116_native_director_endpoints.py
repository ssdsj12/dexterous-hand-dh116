#!/usr/bin/env python3
"""Read immutable native endpoint archives; never execute physics or infer slip."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts import audit_dh116_dual_pinch_closure_readback as previous
from scripts.validate_dh116_dual_pinch_director_interval import inspect_directors
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

VERSION="dh116-native-director-endpoint-summary-v1"
RECEIPT=previous.EVIDENCE/("native-attempts/flex.f7-dual-pinch-director-endpoints-native/"
    "abfae81f2cc9bfad7cb1698ad88afd5f9aeb8a89f87861fbd436c6defc16cf74/attempt-qr3qhoda/receipt.json")
RECEIPT_SHA="11731b7d0a95edfd9573c53ccd08af238f9db47831eb37bf7b3ea0c27b2d9dd6"
SCOPE="exact_2_setup_plus_16_material_director_endpoints_only"
AUDIT="dual-pinch-director-post-exit-audit.json"
require=previous.require


def _array(value,shape,name):
    a=np.asarray(value,dtype=float)
    require(a.shape==shape and np.isfinite(a).all(),"finite "+name+" shape required")
    return a


def _quaternions(value):
    q=_array(value,(41,4),"directors including padding")[:40]
    norms=np.linalg.norm(q,axis=-1)
    require(np.allclose(norms,1,atol=1e-4,rtol=0),"unit real segment directors required")
    return q/norms[:,None],norms


def _stats(value):
    a=np.asarray(value,dtype=float)
    require(a.size>0 and np.isfinite(a).all(),"finite nonempty statistics required")
    return dict(p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),
        p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()))


def geometry_metrics(nodes,quaternions,rest_lengths,reset_nodes,reset_quaternions):
    """Endpoint geometric changes; rotations are SO(3), never tangential slip."""
    p=_array(nodes,(41,3),"nodes");p0=_array(reset_nodes,(41,3),"reset nodes")
    rest=_array(rest_lengths,(40,),"constructor rest")
    require(np.all(rest>0),"positive material rest lengths required")
    q,norms=_quaternions(quaternions);q0,_=_quaternions(reset_quaternions)
    delta=np.diff(p,axis=0);length=np.linalg.norm(delta,axis=-1)
    require(np.all(length>0),"nondegenerate current centerline segments required")
    x,y,z,w=q.T
    material_tangent=np.stack((1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)),axis=-1)
    cosine=np.sum(material_tangent*delta/length[:,None],axis=-1)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    rotation=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q*q0,axis=-1)),0,1)))
    return dict(real_segments=40,padding_index_excluded=40,
        node_displacement_m=_stats(np.linalg.norm(p-p0,axis=-1)),
        director_norm_error_max=float(np.max(np.abs(norms-1))),
        director_centerline_angle_deg=_stats(angle),
        director_rotation_from_reset_deg=_stats(rotation),
        relative_segment_length_change=_stats(length/rest-1),
        r0_tangential_slip_m=None,continuous_contact_s_m=None,
        physical_error_bound_m=None,contact_qualified=False)


def _load(receipt,expected):
    receipt=Path(receipt)
    r=previous.strict_json(previous.read_pinned(receipt,expected))
    require(r["native_qualification"] is True and r["native_execution_performed"] is True
        and r["qualification_scope"]==SCOPE and type(r["outer_exit_code"]) is int
        and r["outer_exit_code"]==0 and r["debugger_inferior_signal"] is None,
        "qualified same-clock native endpoint archive required")
    p=Path(r["output"])
    data={}
    for name in ("physical.jsonl","result.json","sdk_clock.jsonl","director_reset.json"):
        data[name]=previous.read_pinned(p/name,r["artifact_sha256"][name])
    a=previous.strict_json(previous.read_pinned(receipt.parent/AUDIT,
        r["lifetime_post_exit_audit_sha256"]))["evidence"]
    require(a["qualification_scope"]==SCOPE and a["lifecycle_qualified"] is True
        and a["endpoint_director_readback_qualified"] is True
        and a["physics_steps"]==18 and a["ppo_updates"]==0,
        "archived independent post-exit endpoint scope differs")
    for name in data:
        require(a["available_artifact_sha256"][name]==r["artifact_sha256"][name],
                "independently recorded artifact SHA differs")
    inspected=inspect_directors(p/"physical.jsonl")
    require(inspected==a["material_director_audit"],"actual director packet replay differs")
    result=previous.strict_json(data["result.json"])
    require(result["failure"] is None and result["physics_steps"]==18
        and result["measurement_steps"]==16 and result["ppo_updates"]==0,
        "complete nonterminal archived result required")
    rows=[decode_owned(previous.strict_json(line)) for line in data["physical.jsonl"].splitlines()]
    reset=decode_owned(previous.strict_json(data["director_reset.json"]))["payload"]
    return r,a,result,rows,reset


def compare_parent(physical,result):
    """Compare matching step indices from two distinct actual native episodes."""
    old_rows,ctx=previous.load_archive()
    old=previous.validate_records(old_rows,ctx)
    require(result["acquisition"]["action_schedule"]==ctx["schedule"],
            "original action schedule changed")
    differences={"rod_nodes_m":0.0}
    for side in previous.SIDES:
        for field,unit in (("q","rad"),("qd","rad_s"),("backend_targets","rad")):
            differences[side+"_"+field+"_"+unit]=0.0
    for fresh,prior in zip(physical,old,strict=True):
        left,right=[e["exchange"] for e in (fresh,prior)]
        for key in ("physics_step","policy_step","substep"):
            require(left[key]==right[key],"paired original physical/control indices differ")
        for key in ("physics_step","rod_tick","timestamp_s","solver_time_s"):
            require(left["rod"][key]==right["rod"][key],"paired endpoint clocks differ")
        require(np.array_equal(left["action_readback"]["request"]["action24"],
                               right["action_readback"]["request"]["action24"]),
                "actually requested canonical actions differ")
        differences["rod_nodes_m"]=max(differences["rod_nodes_m"],
            float(np.max(np.abs(left["rod"]["node_positions"]-right["rod"]["node_positions"]))))
        for side in previous.SIDES:
            for field,unit in (("q","rad"),("qd","rad_s"),("backend_targets","rad")):
                values=[e["action_readback"]["post"]["hands"][side][field] for e in (left,right)]
                key=side+"_"+field+"_"+unit
                differences[key]=max(differences[key],float(np.max(np.abs(values[0]-values[1]))))
    return dict(parent_receipt_sha256=previous.RECEIPT_SHA,clocks_and_action_schedule_equal=True,
        distinct_native_episodes=True,maximum_abs_readback_difference=differences,
        repeatability_tolerance_qualified=False,causal_effect_proven=False)


def summarize(*,receipt=RECEIPT,receipt_sha=RECEIPT_SHA):
    r,a,result,rows,reset=_load(receipt,receipt_sha)
    physical=[e["payload"] for e in rows if e.get("operation")=="physical"]
    packets=[reset]+[v["exchange"]["material_director"] for v in physical]
    p0=reset["node_positions_m"][0];q0=reset["quaternions_xyzw_with_padding"][0]
    endpoints=[]
    for packet in packets:
        endpoints.append(dict(identity=packet["identity"],material_reference_uid=packet["material_reference_uid"],
            **geometry_metrics(packet["node_positions_m"][0],packet["quaternions_xyzw_with_padding"][0],
               packet["constructor_rest_lengths_m"],p0,q0)))
    pairs=Counter(x["mesh_name"] for row in physical
        for x in row["exchange"]["rod"]["rod_mesh_contact_pairs"]["pairs"])
    return dict(version=VERSION,scope="read_only_archived_endpoint_geometry_not_contact_slip",
        source_native_receipt_sha256=receipt_sha,source_native_tested_head=r["tested_head"],
        source_native_physics_steps=18,source_artifact_sha256=r["artifact_sha256"],
        source_post_exit_sha256=r["lifetime_post_exit_audit_sha256"],
        native=dict(outer_exit_code=r["outer_exit_code"],inferior_exit_code=r["debugger_inferior_exit_code"],
            signal=r["debugger_inferior_signal"],wall_seconds=r["duration_seconds"],
            material_packets=len(packets),rod_mesh_pair_samples=dict(pairs),
            table_pair_samples=pairs["table"],hand_rod_pair_samples=sum(v for k,v in pairs.items() if k!="table"),
            director_reader_liveness=a["director_reader_liveness"]),
        comparison=compare_parent(physical,result),endpoints=endpoints,
        this_audit_new_physics_steps=0,ppo_updates=0,native_qualification=False,
        teacher_admission=False,grasp_qualified=False,r0_tangential_slip_m=None,
        continuous_contact_s_m=None,physical_error_bound_m=None,
        hand_table_measurement_available=False,contact_pair_substep_alignment_verified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    result=summarize()
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,allow_nan=False,indent=2);stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("endpoints","source_artifact_sha256")},indent=2))


if __name__=="__main__":
    main()
