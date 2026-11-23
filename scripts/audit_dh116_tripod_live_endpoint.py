"""Audit the sealed live Tripod 902 endpoint and screen an offline next window."""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import IDENTITY_FIELDS, measure_supported_distal_contact
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from scripts.audit_dh116_tripod_early_boundary import SIDES, WINDOW, _controller_endpoint, _hand_geometry, _initial_boundary, _window_proof
from scripts.audit_dh116_tripod_index_contact_loss import _contact_row, _validated_pose
from scripts.audit_dh116_tripod_triad_continuation import RIGHT_CHANNELS, _gaps, _geometry, _relative_frames, triad_direction
from scripts.audit_dh116_tripod_transverse_approach import _source_grid
from scripts.audit_dh116_tripod_wrist_relief_command import _read_bound_wrist, _rotation
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery

def sealed_inventory(run, receipt_sha256):
    outer = json.loads(checked_file(run/"outer.json",receipt_sha256))
    if (outer["exit_code"]!=0 or outer["debugger_inferior_exit_code"]!=0
            or outer["debugger_status_verified"] is not True
            or outer["debugger_inferior_signal"] is not None or outer["forced_exit"] or outer["exceeded_budget"]):
        raise ValueError("sealed natural exit required")
    entries = outer["artifact_sha256"]
    for name, digest in entries.items():
        if Path(name).name != name:
            raise ValueError("invalid receipt artifact filename")
        checked_file(run/"native"/name,digest)
    return entries


def _require(condition, error):
    if not condition:
        raise ValueError(error)


def replay(run, *, receipt_sha256, prior_probe, prior_probe_sha256, project):
    entries = sealed_inventory(run, receipt_sha256)

    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name,entries[name])))

    close = read("app_close_returned.json")
    _require(close["close_mode"] == "normal" and close["app_close_returned"] is True,
             "sealed normal native app close required")
    seed_record=read("tripod_reset_source_1.json")["reset"]["seed"]
    seed=np.asarray(seed_record["reset_root_pose_w"])
    spec=read("tripod_reset_source_2.json")["evidence"]
    table=spec["table_center_m"][2]+.5*spec["table_size_m"][2]
    radius=spec["rod_radius_m"]
    boxes=[(np.asarray(box["center_m"])-.5*np.asarray(box["size_m"]),
            np.asarray(box["center_m"])+.5*np.asarray(box["size_m"]))
           for box in read("tripod_support_scene.json")["supports"]]
    plan=read("frozen_plan.json")
    gates=[read(f"tripod_902_interlock_{n}.json") for n in (240,552,568)]
    _require(gates[-1]["candidate_precontact_complete"]
             and all(r["physics_step"] == s for r, s in zip(gates, (240, 552, 568), strict=True)),
             "the three original Tripod interlocks must be complete")
    old = json.loads(checked_file(prior_probe,prior_probe_sha256))
    _require(old["archived_anchor_sdk"] == 552 and old["virtual_suffix_anchor_sdk"] == 568,
             "R6 endpoint probe must use the original compatible anchor and suffix")
    _require(seed_record["sha256"] == old["archive_reset_seed_sha256"] == plan["tripod_seed_sha256"],
             "actual native and prior conditional Reset seeds differ")
    rows=[_contact_row(read,n) for n in range(552,569)]
    for previous, current in zip(rows[:-1],rows[1:]):
        _require(all(previous["rod"][k] == current["rod"][k] for k in IDENTITY_FIELDS[:-2])
                 and np.array_equal(previous["rod"]["rest_lengths_m"],current["rod"]["rest_lengths_m"]),
                 "physical rows changed rod, material, episode or Reset")
        check_reaction_delivery(previous,current)
    loads=[measure_supported_distal_contact(r["rod"],grasp="tripod") for r in rows]
    _require(all(not x["wrong_side_contacts"] and not x["unexpected_hand_contacts"] for x in loads),
             "unaligned hand contact at the original precontact anchor")
    no_hand=all(sub["total_normal_load_n"]==0. for x in loads for h in x["per_hand"].values() for sub in h["per_substep"])
    support={k:min(min(x["external_support_force_norm_n"][k]) for x in loads)
             for k in ("support_0","support_1")}
    _require(no_hand and min(support.values()) > 0., "the recorded precontact handoff must retain both supports")
    commanded=np.asarray(plan["tripod_approach"])[39:41]
    _require(all(np.array_equal(r["action_readback"]["request"]["action24"][0],commanded[i//8])
                 for i,r in enumerate(rows[1:])), "recorded targets differ from the two frozen command windows")
    t0,_,m0,_=_read_bound_wrist(rows[0])
    t1,_,m1,errors=_read_bound_wrist(rows[-1])
    palms={}
    for i,s in enumerate(SIDES):
        motion=np.asarray([_validated_pose(r["sdk_hand_geometry"]["poses"][f"{s}.base_link"])[0] for r in rows])
        mr0,mr1=_rotation(m0[i,3:]),_rotation(m1[i,3:])
        palms[s]=dict(target_translation_m=(t1[i,:3]-t0[i,:3]).tolist(),
            measured_translation_m=(m1[i,:3]-m0[i,:3]).tolist(),
            measured_max_translation_m=float(np.linalg.norm(motion-motion[0],axis=1).max()),
            target_rotation_vector_rad=(_rotation(t1[i,3:])*_rotation(t0[i,3:]).inv()).as_rotvec().tolist(),
            measured_rotation_vector_rad=(mr1*mr0.inv()).as_rotvec().tolist(), endpoint_tracking=errors[s])
        names=rows[-1]["action_readback"]["post"]["hands"][s]["joint_names"]
        palms[s]["finger_q_delta_rad"]={
            k:float(rows[-1]["action_readback"]["post"]["hands"][s]["q"][0,j]
                    -rows[0]["action_readback"]["post"]["hands"][s]["q"][0,j])
            for j,k in enumerate(names) if k.startswith("finger")}
    rods=np.stack([r["rod"]["node_positions"][0] for r in rows]).astype(float)
    d=np.linalg.norm(rods-rods[0],axis=-1)
    near=np.linalg.norm(rods[-9:]-rods[-1],axis=-1)
    past8, scaled8=float(near.max()),float(WINDOW*near[-2].max())
    rod_budget=max(past8,scaled8)
    models=[load_geometry(project,side=s) for s in SIDES]
    previous,last=rows[-2:]
    source,state=_initial_boundary(last,previous,models,radius=radius,boxes=boxes,table_top=table)
    prev_t,_,prev_m,_=_read_bound_wrist(previous)
    prev_shapes,_,_=_hand_geometry(previous,models,prev_t,prev_m)
    target, angle, measured, shapes, q, backend=state
    nodes=rods[-1]
    grids,_=_source_grid(models,last)
    corrections=_relative_frames(shapes)
    neutral={}
    for mode,pose,qs in (("archived_offset_held",measured,q),("absolute_backend",target,backend)):
        geom,_=_geometry(models,shapes,corrections,pose,qs,table_top=table)
        neutral[mode]=geom
    witnesses=old["conditional_triad"]["witness_nodes"]

    def _trial(action):
        result=_controller_endpoint(action,last,seed,state)
        if result is None:
            return None
        t=np.asarray(result["target_poses_wxyz"])
        j=np.asarray(result["finger_targets_rad"])
        y={}
        for mode in neutral:
            poses=t.copy()
            qs=j.copy()
            if mode=="archived_offset_held":
                residual=_rotation(target[:,3:]).inv()*_rotation(measured[:,3:])
                poses[:,3:]=np.roll((_rotation(t[:,3:])*residual).as_quat(),1,axis=1)
                poses[:,:3]+=measured[:,:3]-target[:,:3]
                qs+=q-backend
            shape,_=_geometry(models,shapes,corrections,poses,qs,table_top=table)
            y[mode]=_gaps(shape,nodes,radius,grids,witnesses)
            if y[mode] is None:
                return None
        return y

    def flatten(result):
        return np.concatenate([result[k]["gaps_m"] for k in neutral])
    actual={k:_gaps(v,nodes,radius,grids,witnesses) for k,v in neutral.items()}
    hold=_trial(np.zeros(24))
    _require(hold is not None and np.allclose(flatten(hold),flatten(actual),atol=1e-12,rtol=0.),
             "frozen hold did not preserve the measured and backend endpoint geometries")
    prior=old["conditional_triad"]
    signed_difference={k:(np.asarray(actual[k]["gaps_m"])-prior["hold_endpoint"][k]["gaps_m"]).tolist()
                       for k in actual}
    derivative=np.full((6,len(RIGHT_CHANNELS)),np.nan)
    for j,i in enumerate(RIGHT_CHANNELS):
        basis=np.zeros(24)
        basis[i]=.02
        a,b=_trial(basis),_trial(-basis)
        if a is not None and b is not None:
            derivative[:,j]=(flatten(a)-flatten(b))/.04
    proposal=triad_direction(derivative)
    trials={}
    options={"hold":np.zeros(24),"old_original_forward":np.asarray(prior["original_forward_action24"]),
             "old_triad_direction":np.asarray(prior["direction_candidate"]["action24"]),
             "old_original_forward_0.5":.5*np.asarray(prior["original_forward_action24"])}
    if proposal is not None:
        options.update({f"live_direction_{scale}":scale*proposal for scale in (1.,.5,.25,.125,.0625,.05,.03125)})
    for name,action in options.items():
        end=_trial(action)
        proof=_window_proof(action,row=last,seed=seed,state=state,measured_previous=prev_shapes,
                            models=models,grids=grids,radius=radius,
                            depth=plan["right_tripod_distal_contact"]["max_pad_penetration_m"],
                            boxes=boxes,table_top=table,observed_rod_m=rod_budget/WINDOW)
        failed=[{k:r[k] for k in ("side","group","link","mode","hull")}
                for r in proof.get("geometry",[]) if not r["hull"]["passed"] or not r.get("pad",{}).get("passed",True)]
        margins=[r["hull"]["rod"]["lower_m"]-r["hull"]["rod"]["threshold_m"]-r["hull"]["rod"].get("reserve_m",0.)
                  for r in proof.get("geometry",[])]
        trials[name]=dict(action24=action.tolist(),frozen_endpoint=end,
            minimum_gap_reduction_m=None if end is None else float((flatten(actual)-flatten(end)).min()),
            extrapolated_continuous_proof_passed=proof["passed"], minimum_rod_margin_m=min(margins) if margins else None,
            geometry_checked=len(proof.get("geometry",[])), first_failure=failed[0] if failed else None,
            reason=proof.get("reason"))
    return dict(receipt_sha256=receipt_sha256, prior_probe_sha256=prior_probe_sha256, seed_sha256=seed_record["sha256"],
        verified_ledger_count=len(entries), natural_exit=True, native_endpoint_sdk=568,
        checked_motion_rows=list(range(552,569)), downstream_command_unexecuted=True,
        rod_measured=dict(max_all_nodes_m=float(d.max()), max_right_material_nodes_1_to_5_m=float(d[:,1:6].max()),
            terminal_max_node_m=float(d[-1].max()),max_adjacent_node_m=float(np.linalg.norm(np.diff(rods,axis=0),axis=-1).max()),
            prior_eight_sdk_max_m=past8, scaled_last_sdk_m=scaled8, extrapolated_rod_reserve_m=rod_budget),
        palms=palms,no_hand_load=no_hand,support_minimums_n=support,static_start_clear=source["zero_reserve_scene_passed"],
        witnesses=witnesses,live_measured_and_backend_model=actual,prior_conditional_hold=prior["hold_endpoint"],
        live_minus_prior_conditional_m=signed_difference,trials=trials)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "prior-probe", "project", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("receipt-sha256", "prior-probe-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = replay(args.run, receipt_sha256=args.receipt_sha256, prior_probe=args.prior_probe,
                    prior_probe_sha256=args.prior_probe_sha256, project=args.project)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write(chr(10))


if __name__ == "__main__":
    main()
