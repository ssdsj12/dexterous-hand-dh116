"""Attribute live Tripod pad gaps and screen geometry-limited contact directions."""

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import IDENTITY_FIELDS, measure_supported_distal_contact
from scripts.audit_dh116_certified_segment_distance import segment_hull_bounds
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from scripts.audit_dh116_tripod_early_boundary import (
    PADS, WINDOW, _controller_endpoint, _hand_geometry, _initial_boundary, require_boundary,
)
from scripts.audit_dh116_tripod_index_contact_loss import _contact_row, _validated_pose
from scripts.audit_dh116_tripod_live_endpoint import sealed_inventory
from scripts.audit_dh116_tripod_transverse_approach import _certify_sequence, _source_grid
from scripts.audit_dh116_tripod_triad_continuation import _gaps, _geometry, _relative_frames
from scripts.audit_dh116_tripod_wrist_relief_command import _read_bound_wrist, _rotation
from scripts.prepare_dh116_pinch_closure_margin import sample_grid
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery


RIGHT_FINGERS = np.r_[6:10]
RIGHT_WRIST = np.r_[15:18, 21:24]
RIGHT_CHANNELS = np.r_[RIGHT_FINGERS, RIGHT_WRIST]
MODES = ("measured_offset", "absolute_backend")
SOURCE_HULLS = {"thumb": ("right", "source_colliders", "finger14_link"),
                "middle": ("right", "source_colliders", "finger33_link")}


def factor_effects(values):
    """Average every factor ordering in a measured-endpoint factorial comparison."""
    values = np.asarray(values, dtype=float)
    if (values.ndim not in (2, 3) or values.shape != (2,) * values.ndim or not np.isfinite(values).all()):
        raise ValueError("two or three finite binary geometry factors required")
    effects = np.zeros(values.ndim)
    orders = tuple(permutations(range(values.ndim)))
    for order in orders:
        state = [0] * values.ndim
        for factor in order:
            before = values[tuple(state)]
            state[factor] = 1
            effects[factor] += values[tuple(state)] - before
    return effects / len(orders)


def contact_direction(derivative, channels, *, hull_guard=None):
    """Maximize the worst linearized pad improvement across both pose models."""
    channels = np.asarray(channels)
    derivative = np.asarray(derivative, dtype=float)
    if derivative.shape != (6, len(channels)) or not set(channels).issubset(set(RIGHT_CHANNELS)):
        raise ValueError("six signed pad effects and original right-only action channels required")
    eligible = np.isfinite(derivative).all(axis=0)
    if hull_guard is not None:
        hull_guard = np.asarray(hull_guard, dtype=float)
        if hull_guard.shape not in ((2, len(channels)), (4, len(channels))):
            raise ValueError("one finite-segment derivative per guarded hull and pose model required")
        eligible &= np.isfinite(hull_guard).all(axis=0)
    if not eligible.any():
        return None

    # Translation and intrinsic rotation component limits inscribe each native L2 action ball.
    use = channels[eligible]
    limits = np.where(np.isin(use, RIGHT_WRIST), 1. / np.sqrt(3.), 1.)
    a = np.column_stack((derivative[:, eligible]*1e6, np.ones(6)))
    if hull_guard is not None:
        clearance_constraints = np.column_stack((-hull_guard[:, eligible] * 1e6, np.zeros(len(hull_guard))))
        a = np.vstack((a, clearance_constraints))
    result = linprog(np.r_[np.zeros(len(use)), -1.], A_ub=a, b_ub=np.zeros(len(a)),
                     bounds=[(-limit, limit) for limit in limits]+[(0., None)], method="highs")
    if not result.success or result.x[-1] <= 1e-8:
        return None
    action = np.zeros(24)
    action[use] = result.x[:-1]
    for group in (slice(15, 18), slice(21, 24)):
        action[group] /= max(1., float(np.linalg.norm(action[group])))
    return action


def _gap(grid, node, position, orientation, radius):
    local = orientation.inv().apply(node-position)[None]
    phi, _, valid = sample_grid(grid, local, dtype=np.float64)
    if not valid[0]:
        raise ValueError("fixed material witness left the original locked pad lattice")
    return float(phi[0] - radius)


def _measured_motion(first, last, *, grids, witnesses, radius):
    endpoints = (first, last)
    palms = [_validated_pose(x["sdk_hand_geometry"]["poses"]["right.base_link"]) for x in endpoints]
    records = {}
    for link, index in witnesses.items():
        relative = []
        for row, (p, r) in zip(endpoints, palms, strict=True):
            lp, lr = _validated_pose(row["sdk_hand_geometry"]["poses"][link])
            relative.append((r.inv().apply(lp-p), r.inv()*lr))
        values = np.empty((2, 2, 2))
        for a, b, c in np.ndindex(values.shape):
            p, r = palms[a]
            lp, lr = relative[b]
            node = endpoints[c]["rod"]["node_positions"][0, index]
            values[a,b,c] = _gap(grids[link], node, p+r.apply(lp), r*lr, radius)
        effects = factor_effects(values)
        records[link] = dict(initial_gap_m=float(values[0,0,0]), final_gap_m=float(values[1,1,1]),
                             wrist_effect_m=float(effects[0]), finger_pose_effect_m=float(effects[1]),
                             rod_effect_m=float(effects[2]), crossed_endpoint_gaps_m=values.tolist())
    return records


def _tracking_matrix(state, models, frames, *, table_top, grids, witnesses, nodes, radius):
    target, _, measured, shapes, q, cmd = state
    gaps = np.empty((2,2,3))
    for i, j in np.ndindex((2,2)):
        scene, _ = _geometry(models,shapes,frames,(measured,target)[i],(q,cmd)[j],table_top=table_top)
        effect = _gaps(scene,nodes,radius,grids,witnesses)
        if effect is None:
            raise ValueError("backend substitution loses the original required station")
        gaps[i,j] = effect["gaps_m"]
    results = {}
    for k, name in enumerate(f"right.{pad}" for pad in PADS):
        wrist, fingers = factor_effects(gaps[:,:,k])
        results[name] = dict(measured_gap_m=float(gaps[0,0,k]), wrist_only_gap_m=float(gaps[1,0,k]),
                             fingers_only_gap_m=float(gaps[0,1,k]), backend_gap_m=float(gaps[1,1,k]),
                             wrist_tracking_effect_m=float(wrist), finger_tracking_effect_m=float(fingers))
    return results


def _query(action, *, row, seed, state, models, frames, table_top, grids, witnesses, nodes, radius, source_segments):
    target, _, measured, shapes, q, cmd = state
    compiled = _controller_endpoint(action,row,seed,state)
    if compiled is None:
        return None
    result = {}
    goal = np.asarray(compiled["target_poses_wxyz"])
    joints = np.asarray(compiled["finger_targets_rad"])
    for mode in MODES:
        pose = goal.copy()
        finger = joints.copy()
        if mode == MODES[0]:
            residual = _rotation(target[:,3:]).inv() * _rotation(measured[:,3:])
            pose[:,3:] = np.roll((_rotation(goal[:,3:])*residual).as_quat(),1,axis=1)
            pose[:,:3] += measured[:,:3] - target[:,:3]
            finger += q - cmd
        geometry, _ = _geometry(models,shapes,frames,pose,finger,table_top=table_top)
        gaps = _gaps(geometry,nodes,radius,grids,witnesses)
        if gaps is None:
            return None
        for pad, key in SOURCE_HULLS.items():
            i = source_segments[pad]
            hull = segment_hull_bounds(geometry[key]["vertices"],nodes[i],nodes[i+1])
            gaps[f"{pad}_source_gap_m"] = (hull["lower_m"]+hull["upper_m"])/2.-radius
        result[mode] = gaps
    return result, compiled


def _flat_gaps(modes):
    return np.concatenate([modes[mode]["gaps_m"] for mode in MODES])


def _flat_hull(modes, pad):
    return np.array([modes[m][f"{pad}_source_gap_m"] for m in MODES])


def _reaction_summary(rows):
    samples = [measure_supported_distal_contact(row["rod"], grasp="tripod") for row in rows]
    if any(x["wrong_side_contacts"] or x["unexpected_hand_contacts"] for x in samples):
        raise ValueError("actual solver packet has unexpected or wrong-side hand contact")
    per_pad = {}
    substeps = 0
    for side in ("left", "right"):
        hands = [x["per_hand"][side] for x in samples]
        times = [step for hand in hands for step in hand["per_substep"]]
        substeps = len(times)
        per_pad[side] = {pad: [x["per_pad"][pad]["normal_load_n"] for x in times]
                         for pad in hands[0]["required_pads"]}
    return dict(
        solver_substeps=substeps,
        maximum_pad_load_n={side: {pad: float(max(values)) for pad, values in pads.items()}
                            for side, pads in per_pad.items()},
        positive_pad_substeps={side: {pad: int(sum(v > 0. for v in values)) for pad, values in pads.items()}
                               for side, pads in per_pad.items()},
        minimum_support_force_n={support: float(min(min(x["external_support_force_norm_n"][support]) for x in samples))
                                 for support in ("support_0", "support_1")},
    )


def replay(run, *, receipt_sha256, project):
    entries = sealed_inventory(run,receipt_sha256)

    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name,entries[name])))

    close = read("app_close_returned.json")
    if close["close_mode"] != "normal" or close["app_close_returned"] is not True:
        raise ValueError("sealed normal end of the actual 902 run required")
    rows = [_contact_row(read,sdk) for sdk in range(552,569)]
    first, prior, last = rows[0],rows[-2],rows[-1]
    reaction_steps = 0
    for older, newer in zip(rows[:-1], rows[1:], strict=True):
        if any(older["rod"][k] != newer["rod"][k] for k in IDENTITY_FIELDS[:-2]):
            raise ValueError("one rod and original Reset required across the live phase")
        reaction_steps += int(check_reaction_delivery(older, newer))
    reactions = _reaction_summary(rows)
    reactions["nonzero_delivered_hand_wrench_steps"] = reaction_steps
    next_policy = require_boundary(last,sdk=568,start_sdk=240)
    interlock = read("tripod_902_interlock_568.json")
    if interlock["physics_step"] != 568 or interlock["candidate_precontact_complete"] is not True:
        raise ValueError("the actual completed two-window 902 interlock is required")
    frozen = read("frozen_plan.json")
    spec = frozen["right_tripod_distal_contact"]
    reset = read("tripod_reset_source_2.json")["evidence"]
    seed = read("tripod_reset_source_1.json")["reset"]["seed"]
    if seed["sha256"] != frozen["tripod_seed_sha256"]:
        raise ValueError("live 902 Reset identity differs from its plan")
    root_seed = np.asarray(seed["reset_root_pose_w"])
    radius = reset["rod_radius_m"]
    table = reset["table_center_m"][2] + .5*reset["table_size_m"][2]
    boxes = [(np.asarray(z["center_m"])-.5*np.asarray(z["size_m"]),
              np.asarray(z["center_m"])+.5*np.asarray(z["size_m"]))
             for z in read("tripod_support_scene.json")["supports"]]
    models = [load_geometry(project,side=side) for side in ("left","right")]
    boundary,state = _initial_boundary(last,prior,models,radius=radius,boxes=boxes,table_top=table)
    if not boundary["no_hand_load"] or min(boundary["support_minimums_n"].values()) <= 0.:
        raise ValueError("the loaded supports and clean right precontact endpoint are required")
    prior_t, _, prior_m, _ = _read_bound_wrist(prior)
    prior_shapes, _, _ = _hand_geometry(prior,models,prior_t,prior_m)
    _, _, measured, shapes, q, _ = state
    nodes = last["rod"]["node_positions"][0]
    grids,lipschitz = _source_grid(models,last)
    frames = _relative_frames(shapes)
    measured_scene, _ = _geometry(models,shapes,frames,measured,q,table_top=table)
    witnesses = {}
    for name, ids in spec["stations"].items():
        local = measured_scene["right","cooked_pads",name.split(".")[1]]["transform"]
        phi, _, valid = sample_grid(grids[name],(nodes[ids]-local[:3,3])@local[:3,:3],dtype=np.float64)
        if not valid.any():
            raise ValueError("required right station missing in measured source")
        witnesses[name] = ids[int(np.nanargmin(phi))]
    source_segments = {}
    for pad, key in SOURCE_HULLS.items():
        stations = spec["stations"][f"right.{key[-1]}"]
        vertices = measured_scene[key]["vertices"]
        source_segments[pad] = min(range(min(stations)-1,max(stations)+1),
            key=lambda i: segment_hull_bounds(vertices,nodes[i],nodes[i+1])["upper_m"])
    thumb_stations = spec["stations"]["right.finger14_link"]
    thumb_range = range(min(thumb_stations)-1,max(thumb_stations)+1)

    def query(action):
        return _query(action,row=last,seed=root_seed,state=state,models=models,frames=frames,table_top=table,
                      grids=grids,witnesses=witnesses,nodes=nodes,radius=radius,source_segments=source_segments)

    hold = query(np.zeros(24))
    if hold is None:
        raise ValueError("actual SDK568 must hold the original target state")
    derivative = np.full((6,len(RIGHT_CHANNELS)),np.nan)
    hull_derivatives = {pad:np.full((2,len(RIGHT_CHANNELS)),np.nan) for pad in SOURCE_HULLS}
    for j, channel in enumerate(RIGHT_CHANNELS):
        unit = np.zeros(24)
        unit[channel] = .02
        plus, minus = query(unit), query(-unit)
        if plus is not None and minus is not None:
            derivative[:,j] = (_flat_gaps(plus[0])-_flat_gaps(minus[0]))/.04
            for pad in SOURCE_HULLS:
                hull_derivatives[pad][:,j] = (_flat_hull(plus[0],pad)-_flat_hull(minus[0],pad))/.04

    arms = {}
    for label, channels, guards in (
        ("wrist_only",RIGHT_WRIST,()), ("fingers_only",RIGHT_FINGERS,()), ("coordinated",RIGHT_CHANNELS,()),
        ("thumb_guarded_coordinated",RIGHT_CHANNELS,("thumb",)),
        ("thumb_middle_guarded_coordinated",RIGHT_CHANNELS,("thumb","middle")),
    ):
        selected = np.isin(RIGHT_CHANNELS,channels)
        guarded = np.concatenate([hull_derivatives[pad][:,selected] for pad in guards]) if guards else None
        move = contact_direction(derivative[:,selected],channels,hull_guard=guarded)
        result = None if move is None else query(move)
        if result is None:
            arms[label] = dict(status="no_unclipped_direction_in_model")
            continue
        effect = _flat_gaps(hold[0]) - _flat_gaps(result[0])
        arms[label] = dict(status="endpoint_only", action24=move.tolist(), endpoint=result[0],
                           minimum_signed_gap_reduction_m=float(effect.min()),
                           maximum_remaining_gap_m=float(_flat_gaps(result[0]).max()),
                           source_hull_gap_change_m={
                               pad:(_flat_hull(result[0],pad)-_flat_hull(hold[0],pad)).tolist() for pad in SOURCE_HULLS})

    history = np.stack([row["rod"]["node_positions"][0] for row in rows[-9:]]).astype(float)
    motion = np.linalg.norm(history-history[-1],axis=-1)
    allowance = max(float(motion.max()),float(WINDOW*motion[-2].max()))
    proof_rows = {}
    for kind, scales in (("coordinated",(.0625,)), ("thumb_guarded_coordinated",(1.,.75,.5,.25,.125,.0625)),
                         ("thumb_middle_guarded_coordinated",(1.,.75,.5,.25,.125,.0625))):
        arm = arms[kind]
        if "action24" not in arm:
            continue
        proof_rows[kind] = []
        for scale in scales:
            action = np.asarray(arm["action24"])*scale
            probe = query(action)
            if probe is None:
                continue
            endpoints, compiled = probe
            swept = _certify_sequence([compiled],root_seed,state,prior_shapes,models,grids,lipschitz,nodes,
                radius=radius,depth=spec["max_pad_penetration_m"],rod_budget=allowance,boxes=boxes,
                table=table,index_node=witnesses["right.finger23_link"],thumb_segments=thumb_range)
            gain = _flat_gaps(hold[0]) - _flat_gaps(endpoints)
            proof_rows[kind].append(dict(scale=scale,action24=action.tolist(),minimum_signed_gap_reduction_m=float(gain.min()),
                                         endpoint=endpoints,proof=swept))

    return dict(version="dh116-tripod-contact-readiness-v1",receipt_sha256=receipt_sha256,
                seed_sha256=seed["sha256"],inventory_size=len(entries),source_sdk=568,next_policy=next_policy,
                measured_precontact=boundary,no_native_step=True,selected_witnesses=witnesses,
                actual_552_through_568_owned_reactions=reactions,
                measured_552_to_568=_measured_motion(first,last,grids=grids,witnesses=witnesses,radius=radius),
                static_tracking_substitutions=_tracking_matrix(state,models,frames,table_top=table,grids=grids,
                                                               witnesses=witnesses,nodes=nodes,radius=radius),
                limiting_source_segments=source_segments, hold_endpoint=hold[0], authority_arms=arms,
                one_window_rod_allowance_m=allowance, swept_proof_comparison=proof_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run","project","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--receipt-sha256",required=True)
    args = parser.parse_args()
    answer = replay(args.run,receipt_sha256=args.receipt_sha256,project=args.project)
    with args.output.open("x",encoding="utf-8") as output:
        json.dump(answer,output,indent=2,sort_keys=True,allow_nan=False)
        output.write(chr(10))


if __name__ == "__main__":
    main()
