"""Screen a conditional Tripod tri-pad continuation and its full-history start guard."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import IDENTITY_FIELDS
from go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance import WireClearance
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from scripts.audit_dh116_tripod_early_boundary import (
    PADS, SIDES, WINDOW, _controller_endpoint, _initial_boundary, _hand_geometry, _matrix,
    _shape_proof, require_boundary,
)
from scripts.audit_dh116_tripod_index_contact_loss import _contact_row, _validated_pose
from scripts.audit_dh116_tripod_transverse_approach import (
    _certify_sequence, _path, _source_grid, compile_two_windows, prior_rod_reserve,
)
from scripts.audit_dh116_tripod_wrist_relief import _opposed
from scripts.audit_dh116_tripod_wrist_relief_command import _read_bound_wrist, _rotation
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery


RIGHT_CHANNELS = np.r_[6:10, 15:18, 21:24]
POSE_MODELS = ("archived_offset_held", "absolute_backend")
SUBSEQUENT_WINDOW = 3


def triad_direction(derivative):
    """Maximize the worst linearized same-node gap reduction in the two pose models."""
    derivative = np.asarray(derivative, dtype=float)
    if derivative.shape != (6, len(RIGHT_CHANNELS)):
        raise ValueError("three named pads in both pose models are required")
    eligible = np.isfinite(derivative).all(axis=0)
    if not eligible.any():
        return None
    # Box constraints are inside the original per-hand wrist L2 unit balls.
    limits = np.r_[np.ones(4), np.full(6, 1. / np.sqrt(3.))]
    result = linprog(np.r_[np.zeros(int(eligible.sum())), -1.],
                     A_ub=np.column_stack((1e6*derivative[:, eligible], np.ones(6))),
                     b_ub=np.zeros(6), bounds=[(-l, l) for l in limits[eligible]] + [(0., None)],
                     method="highs")
    if not result.success or result.x[-1] <= 1e-8:
        return None
    action = np.zeros(24)
    action[RIGHT_CHANNELS[eligible]] = result.x[:-1]
    for group in (slice(15,18),slice(21,24)):
        action[group] /= max(1., float(np.linalg.norm(action[group])))
    return action


def exact_start_deficit(rod):
    """Identify a finite-segment upper witness below the motion-reserved floor."""
    segment = rod.get("failed_segment")
    if segment is None:
        return None
    required = rod["threshold_m"] + rod.get("reserve_m", 0.)
    deficit = required - segment["upper_m"]
    return float(deficit) if deficit > 0. else None


def _relative_frames(shapes):
    return {key: np.linalg.solve(s["fk_at_measured"], _matrix(s["position"], s["orientation"]))
            for key, s in shapes.items()}


def _geometry(models, shapes, frames, poses, q, *, table_top):
    scene, table_clearance = {}, []
    for i, side in enumerate(SIDES):
        fk = fk_poses(models[i], poses[i], q[6*i:6*i+6])
        clearance = []
        for (owner, group, link), shape in shapes.items():
            if owner != side:
                continue
            key = owner, group, link
            transform = fk[link] @ frames[key]
            vertices = shape["vertices"] @ transform[:3, :3].T + transform[:3, 3]
            scene[key] = dict(transform=transform, vertices=vertices)
            clearance.append(float(vertices[:,2].min() - table_top))
        table_clearance.append(min(clearance))
    return scene, np.asarray(table_clearance)


def _modeled_boundary(prefix, seed, state):
    target, angles, measured, _, q, backend = state
    boundaries = {}
    for mode in POSE_MODELS:
        rows = [r for r in _path(prefix,target,angles,backend,seed,measured,q,
                                  carry_offset=mode == POSE_MODELS[0]) if r[0] == 1]
        if len(rows) != len(SIDES):
            raise ValueError("the complete two-window modeled boundary is required")
        boundaries[mode] = dict(poses=np.stack([r[2][-1] for r in rows]),
                                joints=np.concatenate([r[3][-1] for r in rows]))
    return boundaries


def _witnesses(scene, nodes, grids, stations):
    witnesses = {}
    for pad in PADS:
        name = f"right.{pad}"
        tf = scene["right","cooked_pads",pad]["transform"]
        ids = stations[name]
        local = (nodes[ids] - tf[:3,3]) @ tf[:3,:3]
        phi, _, valid = sample_grid(grids[name],local,dtype=np.float64)
        if not valid.any():
            raise ValueError("original named right-pad material station outside locked SDF")
        witnesses[name] = ids[int(np.nanargmin(phi))]
    return witnesses


def _gaps(scene, nodes, radius, grids, witnesses):
    distances, normals = [], []
    for pad in PADS:
        name = f"right.{pad}"
        tf = scene["right","cooked_pads",pad]["transform"]
        local = (nodes[[witnesses[name]]] - tf[:3,3]) @ tf[:3,:3]
        phi, grad, valid = sample_grid(grids[name],local,dtype=np.float64)
        if not valid[0]:
            return None
        distances.append(float(phi[0] - radius))
        normals.append(tf[:3,:3] @ grad[0])
    opposed = bool(np.all(np.linalg.norm(normals,axis=1)>0.) and _opposed(np.asarray(normals)))
    return dict(gaps_m=distances, opposed_normals=opposed)


def _necessary_start_guard(scene, shapes, previous, nodes, *, rod_budget, horizon, radius, boxes, table_top):
    wire = WireClearance(nodes[None], radius_m=radius)
    rows = []
    for key, part in scene.items():
        side, group, link = key
        history = float(horizon * np.linalg.norm(shapes[key]["measured"]-previous[key]["measured"],axis=1).max())
        proof = _shape_proof(wire,part["vertices"],selected=link in PADS,boxes=boxes,table_top=table_top,
                             hand_motion=history,rod_motion=rod_budget)
        rows.append(dict(side=side, group=group, link=link, historical_hand_allowance_m=history,
                         exact_rod_upper_deficit_m=exact_start_deficit(proof["rod"]), **proof))
    return rows


def _triad_probe(*, prefix, seed, row_state, models, nodes, radius, table_top, grids, stations, forward):
    _, _, _, shapes, _, _ = row_state
    frames = _relative_frames(shapes)
    boundary = _modeled_boundary(prefix,seed,row_state)
    starting, clear = {}, {}
    for mode, model in boundary.items():
        starting[mode], clear[mode] = _geometry(models,shapes,frames,model["poses"],model["joints"],
                                                table_top=table_top)
    witness = _witnesses(starting[POSE_MODELS[0]],nodes,grids,stations)
    terminal = prefix[-1]
    t, a, q = (np.asarray(terminal[k]) for k in ("target_poses_wxyz","intrinsic_angles_rad","finger_targets_rad"))

    def query(action):
        observations, target_values = {}, []
        for mode, model in boundary.items():
            source = {"sdk_hand_geometry":{"table_clearance_m":clear[mode][None]}}
            projected = _controller_endpoint(action,source,seed,(t,a,model["poses"],shapes,model["joints"],q))
            if projected is None:
                return None
            goal = np.asarray(projected["target_poses_wxyz"])
            fingers = np.asarray(projected["finger_targets_rad"])
            target_values.append(np.r_[goal.ravel(), fingers])
            if mode == POSE_MODELS[0]:
                residual = _rotation(t[:,3:]).inv()*_rotation(model["poses"][:,3:])
                goal[:,3:] = np.roll((_rotation(goal[:,3:])*residual).as_quat(),1,axis=1)
                goal[:,:3] += model["poses"][:,:3] - t[:,:3]
                fingers += model["joints"] - q
            geom, _ = _geometry(models,shapes,frames,goal,fingers,table_top=table_top)
            gaps = _gaps(geom,nodes,radius,grids,witness)
            if gaps is None:
                return None
            observations[mode] = gaps
        if not np.allclose(target_values[0],target_values[1],atol=1e-12,rtol=0.):
            return None
        return observations

    def flatten(result):
        return np.concatenate([result[mode]["gaps_m"] for mode in POSE_MODELS])

    hold = query(np.zeros(24))
    if hold is None:
        raise ValueError("conditional endpoint cannot retain all original targets")
    fwd = query(forward)
    initial = flatten(hold)
    derivative = np.full((6,len(RIGHT_CHANNELS)),np.nan)
    perturbation = .02
    for j, slot in enumerate(RIGHT_CHANNELS):
        basis = np.zeros(24)
        basis[slot] = perturbation
        positive, negative = query(basis), query(-basis)
        if positive is not None and negative is not None:
            derivative[:,j] = (flatten(positive)-flatten(negative))/(2*perturbation)
    move = triad_direction(derivative)
    choice = None
    if move is not None:
        for scale in (1.,.5,.25):
            actual = scale*move
            endpoint = query(actual)
            if endpoint is None:
                continue
            improvement = initial-flatten(endpoint)
            if (improvement > 0.).all() and all(x["opposed_normals"] for x in endpoint.values()):
                choice = dict(action24=actual.tolist(), endpoint=endpoint,
                              minimum_same_node_gap_reduction_m=float(improvement.min()))
                break
    return dict(witness_nodes=witness, hold_endpoint=hold, original_forward_endpoint=fwd,
                original_forward_action24=forward.tolist(),
                direction_candidate=choice, scope="conditional_endpoint_no_swept_suffix_or_measured_force")


def replay(run, *, receipt_sha256, project, cutoff_proof, cutoff_proof_sha256, cut_sdk, control_start_sdk):
    outer = json.loads(checked_file(run/"outer.json",receipt_sha256))
    proof = json.loads(checked_file(cutoff_proof,cutoff_proof_sha256))
    selected = proof["selected"]
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or not outer["debugger_status_verified"] or outer["debugger_inferior_signal"] is not None
            or outer["forced_exit"] or outer["exceeded_budget"] or proof["receipt_sha256"] != receipt_sha256
            or selected["sdk"] != cut_sdk or max(map(int,proof["source_rows_sha256"])) != cut_sdk):
        raise ValueError("original natural-exit archive and exact R5 causal-cutoff proof required")
    inventory = outer["artifact_sha256"]

    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name,inventory[name])))

    close = read("app_close_returned.json")
    if close["close_mode"] != "normal" or close["app_close_returned"] is not True:
        raise ValueError("sealed normal app.close required")
    hist = [_contact_row(read,i) for i in range(cut_sdk-SUBSEQUENT_WINDOW*WINDOW,cut_sdk+1)]
    row, last = hist[-1],hist[-2]
    horizon_nodes = np.stack([item["rod"]["node_positions"][0] for item in hist]).astype(float)
    for older, newer in zip(hist[:-1],hist[1:],strict=True):
        if (any(older["rod"][k] != newer["rod"][k] for k in IDENTITY_FIELDS[:-2])
                or not np.array_equal(older["rod"]["rest_lengths_m"],newer["rod"]["rest_lengths_m"])):
            raise ValueError("causal history must keep one rod, material, run and Reset")
        check_reaction_delivery(older,newer)
    reset = read("tripod_reset_source_2.json")["evidence"]
    old_seed = read("tripod_reset_source_1.json")["reset"]["seed"]
    if old_seed["sha256"] != proof["archived_reset_seed_sha256"]:
        raise ValueError("historical Reset seed must match R5; no fresh-seed reuse")
    seed = np.asarray(old_seed["reset_root_pose_w"])
    radius = reset["rod_radius_m"]
    table_top = reset["table_center_m"][2] + .5*reset["table_size_m"][2]
    boxes = [(np.asarray(z["center_m"]) - .5*np.asarray(z["size_m"]),
              np.asarray(z["center_m"]) + .5*np.asarray(z["size_m"]))
             for z in read("tripod_support_scene.json")["supports"]]
    models = [load_geometry(project,side=side) for side in SIDES]
    frozen = read("frozen_plan.json")
    spec = frozen["right_tripod_distal_contact"]
    stations = spec["stations"]
    idx = require_boundary(row,sdk=cut_sdk,start_sdk=control_start_sdk)
    if idx != selected["next_policy"]:
        raise ValueError("recorded R5 phase identity must agree")
    incoming, state = _initial_boundary(row,last,models,radius=radius,boxes=boxes,table_top=table_top)
    if not incoming["zero_reserve_scene_passed"] or not incoming["no_hand_load"]:
        raise ValueError("original clear precontact anchor required")
    target, _, measured, shapes, q, cmd = state
    last_t, _, last_m, _ = _read_bound_wrist(last)
    previous, _, _ = _hand_geometry(last,models,last_t,last_m)
    grids, lipschitz = _source_grid(models,row)
    old_actions = np.asarray(selected["alternative"]["actions24"])
    prefix = compile_two_windows(old_actions,row=row,seed=seed,state=state)
    if prefix is None or any(projected[key] != archived[key] for projected, archived in
                             zip(prefix,selected["alternative"]["compiled_targets"],strict=True)
                             for key in archived):
        raise ValueError("recompiled R5 targets do not reproduce the sealed proof")
    reserve16 = prior_rod_reserve(horizon_nodes[-17:])
    thumb = stations["right.finger14_link"]
    prefix_cert = _certify_sequence(prefix,seed,state,previous,models,grids,lipschitz,horizon_nodes[-1],
        radius=radius,depth=spec["max_pad_penetration_m"],rod_budget=reserve16["allowance_m"],boxes=boxes,
        table=table_top,index_node=selected["index_witness_node"],thumb_segments=range(min(thumb)-1,max(thumb)+1))
    if (not prefix_cert["passed"] or not np.isclose(prefix_cert["minimum_rod_margin_m"],
            selected["alternative"]["minimum_rod_margin_m"], rtol=0., atol=1e-12)):
        raise ValueError("two-window positive-control proof does not reproduce the sealed margin")

    horizon = SUBSEQUENT_WINDOW * WINDOW
    drift = np.linalg.norm(horizon_nodes-horizon_nodes[-1],axis=-1)
    allowance = max(float(drift.max()),float(horizon*drift[-2].max()))
    matrices = _relative_frames(shapes)
    start_guards = {}
    for mode, poses, nq in ((POSE_MODELS[0],measured,q),(POSE_MODELS[1],target,cmd)):
        scene,_ = _geometry(models,shapes,matrices,poses,nq,table_top=table_top)
        start_guards[mode] = _necessary_start_guard(scene,shapes,previous,horizon_nodes[-1],
                rod_budget=allowance,horizon=horizon,radius=radius,boxes=boxes,table_top=table_top)
    proved_start_obstacle = any(s["exact_rod_upper_deficit_m"] is not None
                                for mode in start_guards.values() for s in mode)
    thumb = shapes["right","source_colliders","finger14_link"]
    prior_thumb_motion = []
    for sample in hist:
        p, r = _validated_pose(sample["sdk_hand_geometry"]["poses"]["right.finger14_link"])
        prior_thumb_motion.append(float(np.linalg.norm(p + r.apply(thumb["vertices"]) - thumb["measured"],axis=1).max()))
    probe = _triad_probe(prefix=prefix,seed=seed,row_state=state,models=models,nodes=horizon_nodes[-1],radius=radius,
        table_top=table_top,grids=grids,stations=stations,forward=np.asarray(frozen["tripod_approach"][idx+2]))
    return dict(version="dh116-tripod-triad-continuation-v1",receipt_sha256=receipt_sha256,
                archive_reset_seed_sha256=old_seed["sha256"],cutoff_proof_sha256=cutoff_proof_sha256,
                control_start_sdk=control_start_sdk, archived_anchor_sdk=cut_sdk, virtual_suffix_anchor_sdk=cut_sdk+2*WINDOW,
                virtual_suffix_policy=idx+2, virtual_suffix_sdk_interval=[cut_sdk+2*WINDOW+1,cut_sdk+3*WINDOW],
                last_read_physical_sdk=cut_sdk,
                source_rows_sha256={str(i):inventory[f"tripod_control_{i}.json"]
                                    for i in range(cut_sdk-horizon,cut_sdk+1)},
                prefix_two_window_control=dict(passed=True,minimum_rod_margin_m=prefix_cert["minimum_rod_margin_m"]),
                three_window_horizon=dict(covered_sdk_steps=horizon,measured_prior_drift_m=float(drift.max()),
                    scaled_last_sdk_drift_m=float(horizon*drift[-2].max()),rod_allowance_m=allowance,
                    actual_prior_right_thumb_max_point_displacement_m=max(prior_thumb_motion),
                    exact_reserved_start_obstacle=proved_start_obstacle,start_guards=start_guards),
                conditional_triad=probe,native_steps_executed=0,teacher_admission=False,grasp_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("run","project","output","cutoff-proof"):
        parser.add_argument(f"--{key}",required=True,type=Path)
    for key in ("receipt-sha256","cutoff-proof-sha256"):
        parser.add_argument(f"--{key}",required=True)
    for key in ("cut-sdk","control-start-sdk"):
        parser.add_argument(f"--{key}",required=True,type=int)
    args=parser.parse_args()
    result=replay(args.run,receipt_sha256=args.receipt_sha256,project=args.project,
                  cutoff_proof=args.cutoff_proof,cutoff_proof_sha256=args.cutoff_proof_sha256,
                  cut_sdk=args.cut_sdk,control_start_sdk=args.control_start_sdk)
    with args.output.open("x",encoding="utf-8") as obj:
        json.dump(result,obj,allow_nan=False,indent=2)
        obj.write(chr(10))


if __name__=="__main__":
    main()
