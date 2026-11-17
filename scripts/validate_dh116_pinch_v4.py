#!/usr/bin/env python3
"""V4 post-exit audit: explicit successor actions, force precision and raw contacts.

The unchanged v3/v2 full audit remains the lifecycle gate. Simulation callback
API returns are not native release; same-substep pair loads are not a grasp,
effective opposition, material slip, or Teacher admission.
"""
import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
import math
from pathlib import Path

from scripts import validate_dh116_pinch_v3 as prior
from scripts import validate_dh116_pinch_v2 as full
from scripts.dh116_pinch_evidence_v2 import validate_phase_pair_record
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_deformable_grasp_runtime as runtime
from go2_pvcnn.control.dh116.tabletop_deformable_pinch_force_precision import validate_ledger
from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import VERSION as CLEAR_VERSION
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS

VERSION="dh116-single-left-pinch-v4"
CONFIG="configs/dh116_tasks/flexible_pinch_pose_candidate_v4.json"
MINIMUM_TABLE_GAP_M=.0019
CONTACT_SCOPE="same_solver_substep_thumb_index_positive_normal_load_observation_only"


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def _cfg(value):
    """Trust only the explicit caller/project config, never an output-side claim."""
    cfg=deepcopy(value)
    canonical(cfg)  # Reject nonfinite/opaque values without normalizing their types.
    expected=dict(side="left",physics_hz=480,setup_steps=2,measured_steps=238)
    full.require(all(cfg.get(k)==v for k,v in expected.items()),"explicit v4 config scope differs")
    action=cfg["action"]
    full.require(action.get("zero_steps")==48 and action.get("move_steps")==160,
                 "explicit v4 quiet/move budget differs")
    for key in ("thumb_velocity_normalized","index_velocity_normalized"):
        v=action[key]
        full.require(type(v) in (int,float) and math.isfinite(v) and -1<=v<=1,
                     "explicit v4 action velocity is invalid")
    pose=cfg["initial_pose"]
    for key,count in (("active_q_rad",6),("base_pose_wxyz",7)):
        values=pose[key]
        full.require(type(values) is list and len(values)==count and
            all(type(v) in (int,float) and math.isfinite(v) for v in values),
            "explicit v4 reset pose is invalid")
    return cfg


def validate_simulation_clear_receipt(path):
    receipt=full.read(path)
    expected=dict(version=CLEAR_VERSION,subscription_status="explicitly_unsubscribed",
        unsubscribe_returned=True,subscription_field_cleared=True,
        original_clear_instance_returned=True,extra_gc_calls=0,native_release_verified=False,
        simulation_liveness_after_return=None,constructor_frame_owner_resolved=False,
        physics_steps_executed=0)
    full.require(canonical(receipt)==canonical(expected),
                 "simulation clear receipt lacks the exact known subscription/API sequence")
    return dict(api_sequence_verified=True,receipt_sha256=full.sha(path),
        scope="known_ready_subscription_and_original_clear_API_returns_only",
        native_release_verified=False,simulation_released=None,
        constructor_frame_owner_resolved=False)


def validate_initial_clearance(record):
    """Add the fixed-base 1.9 mm gate to metrics recomputed by the full audit."""
    result={}
    for key,count in (("source_colliders",17),("locked_pads",5)):
        rows=record[key]
        full.require(type(rows) is dict and len(rows)==count,"complete initial geometry set required")
        if key=="locked_pads":
            full.require(set(rows)==set(DISTAL_PAD_LINKS),"exact locked five pad identities required")
        for row in rows.values():
            gap=row["table"]["top_clearance_m"]
            rod_gap=row["rod"]["capsule_surface_gap_m"]
            full.require(type(gap) in (int,float) and math.isfinite(gap)
                and gap>=MINIMUM_TABLE_GAP_M and row["table"]["intersects"] is False,
                "v4 initial source/locked table clearance below 1.9 mm")
            full.require(type(rod_gap) in (int,float) and math.isfinite(rod_gap) and rod_gap>=0,
                         "v4 initial source/locked rod gap is negative or unknown")
        result[key]=dict(shape_count=count,
            minimum_table_gap_m=min(r["table"]["top_clearance_m"] for r in rows.values()),
            minimum_rod_gap_m=min(r["rod"]["capsule_surface_gap_m"] for r in rows.values()))
    return dict(required_table_gap_m=MINIMUM_TABLE_GAP_M,groups=result,
        scope="source17_and_locked5_at_actual_reset_pose_not_live_shape_attestation")


def _lines(path):
    raw=Path(path).read_text()
    full.require(raw.endswith("\n"),"incomplete final ledger line")
    return [json.loads(line) for line in raw.splitlines()]


def _stamp(row,substep):
    # timestamp_s is the enclosing SDK boundary, not a fabricated solver
    # substep timestamp; the actual solver substep is carried separately.
    return {**{k:row[k] for k in ("phase_step","physics_step","sdk_step","rod_tick",
        "episode_uid","reset_id","env_id","rod_uid","timestamp_s")},
        "solver_substep":substep}


def summarize_contacts(rows,records,*,expected_bindings):
    """Reparse atomic raw samples; never infer contact from net mesh force."""
    full.require(len(rows)==len(records)==238,"exact 238 physical/pair records required")
    full.require(set(expected_bindings)==set(DISTAL_PAD_LINKS),"complete measured pad bindings required")
    pads={name:dict(mesh_index=i,body_index=expected_bindings[name]["body_index"],
        body_path=expected_bindings[name]["body_path"],pair_samples=0,positive_normal_load_samples=0,
        measured_rows=0,first_contact=None,first_positive_normal_load=None,
        maximum_normal_load_n=None,normal_load_integral_ns=0.)
        for i,name in enumerate(DISTAL_PAD_LINKS)}
    first=None;table_samples=0;synchronous=[]
    for row,record in zip(rows,records):
        full.require(row["actual_pad_bindings"]==expected_bindings,
                     "pair row actual body/mesh bindings differ from acquisition")
        parsed=validate_phase_pair_record(record,row)
        loads={};seen=set()
        for sample in parsed["pairs"]:
            mesh=sample["mesh_index"]
            if mesh==5:
                table_samples+=1
                continue
            name=DISTAL_PAD_LINKS[mesh];pad=pads[name]
            stamp=_stamp(row,sample["substep"])
            load=sample["normal_force_on_rod_n"]
            if first is None:first=stamp
            if pad["first_contact"] is None:pad["first_contact"]=stamp
            pad["pair_samples"]+=1;seen.add(name)
            pad["maximum_normal_load_n"]=load if pad["maximum_normal_load_n"] is None else max(
                pad["maximum_normal_load_n"],load)
            pad["normal_load_integral_ns"]+=load*parsed["substep_dt_s"]
            if load>0:
                pad["positive_normal_load_samples"]+=1
                if pad["first_positive_normal_load"] is None:pad["first_positive_normal_load"]=stamp
            entry=loads.setdefault(sample["substep"],{})
            entry[name]=entry.get(name,0.)+load
        for name in seen:pads[name]["measured_rows"]+=1
        for substep,values in sorted(loads.items()):
            if all(values.get(name,0.)>0 for name in DISTAL_PAD_LINKS[:2]):
                synchronous.append({**_stamp(row,substep),"normal_load_n":values})
    return dict(scope=CONTACT_SCOPE,raw_pairs_revalidated=True,pair_rows=238,
        hand_pair_samples=sum(p["pair_samples"] for p in pads.values()),
        table_pair_samples=table_samples,pads=pads,first_hand_contact=first,
        thumb_index_synchronous_positive_load_substeps=len(synchronous),
        first_thumb_index_synchronous_positive_load=synchronous[0] if synchronous else None,
        synchronous_load_samples=synchronous,effective_opposed_contact=None,
        material_anchor_slip_m_s=None,force_threshold_n=0.,
        threshold_scope="strict_positive_raw_normal_load_not_grasp_threshold",
        native_release_verified=False,grasp_qualified=False,teacher_admission=False)


def audit_outcome(output,outer_exit,cleanup,*,cfg):
    """Pure post-exit disk audit; wrapper must bind its own source-checked cfg."""
    output=Path(output)
    errors=[];force={}
    result=dict(lifecycle_qualified=False,case=cleanup)
    try:
        candidate=_cfg(cfg)
        action_validator=full.validate_action_targets
        def actions(rows,plan,_old_config):
            return action_validator(rows,plan,candidate)
        def precision(rows):
            value=validate_ledger(rows);force.update(value);return value
        # All changed globals are restored, including failure paths. V3 still
        # checks its original clean USD import boundary and every v2 artifact.
        with bindings((prior,"VERSION",VERSION),(full,"validate_action_targets",actions),
                      (runtime,"validate_pinch_ledger",precision)):
            result=prior.audit_outcome(output,outer_exit,cleanup)
        result["successor_cfg_sha256"]=hashlib.sha256(canonical(candidate).encode()).hexdigest()
        if force:result["force_precision_audit"]=force
    except (OSError,ValueError,KeyError,TypeError,OverflowError,IndexError) as error:
        errors.append(type(error).__name__+": "+str(error))
    # Preserve useful disk diagnostics even if the original process/ledger gate
    # fails. Their raw summaries cannot override that failure.
    for name,operation in (
        ("simulation_clear_receipt_audit",partial(validate_simulation_clear_receipt,
                                                   output/"simulation_clear_receipt.json")),
        ("initial_clearance_audit",lambda:validate_initial_clearance(full.read(output/"initial_geometry.json"))),
        ("contact_evidence",lambda:summarize_contacts(_lines(output/"physics.jsonl"),_lines(output/"pairs.jsonl"),
            expected_bindings=full.read(output/"plan.json")["actual_pad_bindings"]))):
        try:result[name]=operation()
        except (OSError,ValueError,KeyError,TypeError,OverflowError,IndexError) as error:
            errors.append(name+": "+type(error).__name__+": "+str(error))
    if errors:
        result["lifecycle_qualified"]=False
        result["reason"]="; ".join(filter(None,(result.get("reason"),*errors)))
    result.update(version=VERSION,contact_qualification_scope=CONTACT_SCOPE,
        contact_observation_qualified=(result["lifecycle_qualified"] and
            result.get("contact_evidence",{}).get("thumb_index_synchronous_positive_load_substeps",0)>0),
        effective_opposed_contact=None,material_anchor_slip_m_s=None,
        simulation_released=None,native_release_verified=False,
        grasp_qualified=False,teacher_admission=False)
    result["available_artifact_sha256"]={p.name:full.sha(p) for p in sorted(output.glob("*"))
        if p.is_file() and not p.is_symlink()}
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--outer-exit",type=Path,required=True)
    parser.add_argument("--cleanup",choices=("explicit",),default="explicit")
    args=parser.parse_args(argv)
    cfg=full.read(Path(__file__).resolve().parents[1]/CONFIG)
    result=audit_outcome(args.output,args.outer_exit,args.cleanup,cfg=cfg)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result["lifecycle_qualified"] else 1


if __name__=="__main__":
    raise SystemExit(main())
