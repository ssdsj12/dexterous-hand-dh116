"""Certify a transverse right-wrist relief and authored Tripod advance on one frozen anchor."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import IDENTITY_FIELDS
from go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance import WireClearance
from scripts.audit_dh116_certified_segment_distance import segment_hull_bounds
from scripts.audit_dh116_pinch_initial_pose import load_geometry, path_displacement_bound
from scripts.audit_dh116_tripod_early_boundary import (
    GROUPS, PADS, SIDES, WINDOW, _controller_endpoint, _hand_geometry, _initial_boundary, _matrix,
    _shape_proof, require_boundary,
)
from scripts.audit_dh116_tripod_index_contact_loss import _contact_row
from scripts.audit_dh116_tripod_wrist_relief_command import _read_bound_wrist, _rotation
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.solve_dh116_right_pinch_owned_approach import grid_lipschitz
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery


THUMB = ("right", "source_colliders", "finger14_link")
INDEX = ("right", "cooked_pads", "finger23_link")
SUBDIVISIONS = 8


def relief_action(away, tangent, index_gradient, *, amplitude):
    """Project the source thumb separation off rod axial and index normal motion."""
    matrix = np.asarray([away, tangent, index_gradient], dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.isfinite(amplitude) or not 0. < amplitude <= 1.):
        raise ValueError("finite contact directions and one existing wrist-window rate required")
    lengths = np.linalg.norm(matrix, axis=1)
    if np.any(lengths <= 0.):
        raise ValueError("nonzero contact directions required")
    unit = matrix / lengths[:, None]
    thumb, constraints = unit[0], unit[1:]
    tangent_relief = thumb - constraints.T @ np.linalg.lstsq(constraints.T, thumb, rcond=None)[0]
    length = np.linalg.norm(tangent_relief)
    if length <= 64 * np.finfo(float).eps:
        return None
    vector = amplitude * tangent_relief / length
    # Retain the actuator's exact closed unit norm bound after rounding.
    vector /= max(1., float(np.linalg.norm(vector)))
    action = np.zeros(24)
    action[15:18] = vector
    return action


def compile_two_windows(actions, *, row, seed, state):
    """Reproduce original target writes while keeping the archive measurement fixed."""
    actions = np.asarray(actions, dtype=float)
    if actions.shape != (2, 24):
        raise ValueError("two complete canonical authored action windows required")
    target, angles, measured, shapes, q, backend = state
    compiled = []
    for action in actions:
        segment = _controller_endpoint(action, row, seed, (target, angles, measured, shapes, q, backend))
        if segment is None:
            return None
        compiled.append(segment)
        target = np.asarray(segment["target_poses_wxyz"])
        angles = np.asarray(segment["intrinsic_angles_rad"])
        backend = np.asarray(segment["finger_targets_rad"])
    return compiled


def _thumb_distance(vertices, nodes, segments):
    pairs = [segment_hull_bounds(vertices, nodes[i], nodes[i+1]) for i in segments]
    lower = min(bounds["lower_m"] for bounds in pairs)
    chosen = min(pairs, key=lambda bounds: bounds["upper_m"])
    return lower, chosen


def _point_sdf(grid, node, p, r):
    values, _, valid = sample_grid(grid, r.inv().apply(node - p).reshape(1, 3), dtype=np.float64)
    return float(values[0]) if valid[0] else None


def _index_witness(nodes, grid, body, ids):
    local = body["orientation"].inv().apply(nodes[ids] - body["position"])
    values, gradient, valid = sample_grid(grid, local, dtype=np.float64)
    if not valid.any():
        return None
    index = int(np.nanargmin(values))
    return int(ids[index]), body["orientation"].apply(gradient[index])


def _path(compiled, start_target, start_angles, start_q, seed, measured, actual_q, *, carry_offset):
    """Lift unclipped original target interpolation into one of the two frozen geometry models."""
    weights = np.linspace(0., 1., SUBDIVISIONS+1)
    for first, final in enumerate(compiled):
        end_target = np.asarray(final["target_poses_wxyz"])
        end_angles = np.asarray(final["intrinsic_angles_rad"])
        end_q = np.asarray(final["finger_targets_rad"])
        shift = end_target[:, :3] - start_target[:, :3]
        turn = end_angles - start_angles
        bend = end_q - start_q
        positions = start_target[None, :, :3] + weights[:, None, None]*shift[None]
        angles = start_angles[None] + weights[:, None, None]*turn[None]
        joints = start_q[None] + weights[:, None]*bend[None]
        # The offset is tied to the original archive, even after the first phase.
        if first == 0:
            initial_target = start_target
            initial_q = start_q
        for side in range(2):
            rot = _rotation(seed[side, 3:]) * Rotation.from_euler("XYZ", angles[:, side])
            selected = slice(6 * side, 6 * side + 6)
            q = joints[:, selected]
            p = positions[:, side]
            if carry_offset:
                rot = rot * (_rotation(initial_target[side, 3:]).inv() * _rotation(measured[side, 3:]))
                p = p + measured[side, :3] - initial_target[side, :3]
                q = q + actual_q[selected] - initial_q[selected]
            pose = np.column_stack((p, np.roll(rot.as_quat(), 1, axis=1)))
            yield first, side, pose, q, float(np.linalg.norm(shift[side])), float(np.abs(turn[side]).sum())
        start_target, start_angles, start_q = end_target, end_angles, end_q


def prior_rod_reserve(nodes):
    """Compare measured whole-horizon drift to the scaled last-SDK increment."""
    nodes = np.asarray(nodes, dtype=float)
    horizon = 2 * WINDOW
    if nodes.shape != (horizon + 1, 41, 3) or not np.isfinite(nodes).all():
        raise ValueError("consecutive same-rod prior horizon required")
    distance = np.linalg.norm(nodes - nodes[-1], axis=-1)
    measured = float(distance.max())
    scaled = float(horizon * distance[-2].max())
    return dict(measured_horizon_m=measured, last_step_scaled_m=scaled, allowance_m=max(measured, scaled))


def _bound_pad_sdf(grid, local, reserve, radius, depth, lipschitz):
    phi, _, inside = sample_grid(grid, local.reshape(-1, 3), dtype=np.float64)
    inside = inside.reshape(local.shape[:-1])
    phi = phi.reshape(local.shape[:-1])
    low = np.asarray(grid["origin"])
    high = low + (np.asarray(grid["values"].shape) - 1) * grid["spacing"]
    distance = np.linalg.norm(np.maximum(np.maximum(low-local, local-high), 0.), axis=-1)
    if np.any(~inside & (distance <= reserve)):
        return dict(passed=False, lower_gap_m=None, reason="unresolved_lattice_entry")
    if not inside.any():
        return dict(passed=True, lower_gap_m=None)
    lower = float(np.min((phi - radius - lipschitz*reserve)[inside]))
    return dict(passed=lower >= -depth, lower_gap_m=lower)


def _certify_sequence(compiled, seed, state, older_shapes, models, grids, lipschitz, nodes, *,
                      radius, depth, rod_budget, boxes, table, index_node, thumb_segments):
    target, angles, measured, shapes, actual_q, backend_q = state
    wire = WireClearance(nodes[None], radius_m=radius)
    horizon = 2 * WINDOW
    records, endpoints = [], {}
    for carry in (True, False):
        mode = "archived_offset_held" if carry else "absolute_backend"
        path = _path(compiled, target, angles, backend_q, seed, measured, actual_q, carry_offset=carry)
        for phase, si, poses, q, linear, angular in path:
            side, model = SIDES[si], models[si]
            corrected = dict(model)
            finger_rotation = float(np.abs(expand(q[-1]) - expand(q[0])).sum())
            matrices = [fk_poses(model, pose, joint) for pose, joint in zip(poses, q, strict=True)]
            for group in GROUPS:
                frames = {}
                corrected[group] = {}
                for link, shape in model[group].items():
                    old = shapes[side, group, link]
                    fix = np.linalg.solve(old["fk_at_measured"], _matrix(old["position"], old["orientation"]))
                    frames[link] = fix
                    local = np.vstack((old["vertices"], np.zeros(3)))
                    corrected[group][link] = dict(shape, vertices=local @ fix[:3, :3].T + fix[:3, 3])
                finger = path_displacement_bound(corrected, q[0], q[-1] - q[0], cooked=group == "cooked_pads")
                for link in model[group]:
                    key = side, group, link
                    shape = shapes[key]
                    world = np.stack([mat[link] @ frames[link] for mat in matrices])
                    vertices = np.einsum("tij,nj->tni", world[:, :3, :3], shape["vertices"]) + world[:, None, :3, 3]
                    relative = np.vstack((vertices[0], world[0, :3, 3])) - poses[0, :3]
                    lever = float(np.linalg.norm(relative, axis=1).max()) + finger[link]
                    interval_motion = (linear + angular*lever + finger[link]) / (2 * SUBDIVISIONS)
                    history = horizon * np.linalg.norm(shape["measured"] - older_shapes[key]["measured"], axis=1).max()
                    hand_budget = float(interval_motion + history)
                    checks = [_shape_proof(wire, v, selected=link in PADS, boxes=boxes, table_top=table,
                                          hand_motion=hand_budget, rod_motion=rod_budget) for v in vertices]
                    rod_margin = min(c["rod"]["lower_m"]-c["rod"]["threshold_m"]-hand_budget-rod_budget for c in checks)
                    record = dict(mode=mode, phase=phase+1, side=side, group=group, link=link,
                                  passed=all(c["passed"] for c in checks), rod_margin_m=float(rod_margin),
                                  table_margin_m=min(c["table_margin_m"] for c in checks),
                                  support_margin_m=min(c["support_margin_m"] for c in checks),
                                  history_hand_reserve_m=float(history), interpolation_reserve_m=float(interval_motion))
                    if group == "cooked_pads" and link in PADS:
                        name = f"{side}.{link}"
                        displacement = nodes[None] - world[:, None, :3, 3]
                        local = np.einsum("tni,tij->tnj", displacement, world[:, :3, :3])
                        node_lever = np.linalg.norm(displacement, axis=-1)
                        old = older_shapes[key]
                        origin_history = horizon * np.linalg.norm(shape["position"] - old["position"])
                        orientation_history = horizon * (shape["orientation"] * old["orientation"].inv()).magnitude()
                        origin_reserve = interval_motion + origin_history + rod_budget
                        rotation_reserve = (angular + finger_rotation) / (2 * SUBDIVISIONS) + orientation_history
                        reserve = origin_reserve + rotation_reserve * (node_lever + origin_reserve)
                        record["pad"] = _bound_pad_sdf(grids[name], local, reserve, radius, depth, lipschitz[name])
                    records.append(record)
                    if key in (THUMB, INDEX):
                        for endpoint, label in ((0, "start"), (-1, "end")):
                            if phase == 1 and endpoint == 0:
                                continue
                            dest = endpoints.setdefault((mode, phase+1, label), {})
                            if key == THUMB:
                                dist, near = _thumb_distance(vertices[endpoint], nodes, thumb_segments)
                                dest["thumb_hull_lower_gap_m"] = dist - radius
                                dest["thumb_hull_upper_gap_m"] = near["upper_m"] - radius
                            else:
                                pose = world[endpoint]
                                phi = _point_sdf(grids["right.finger23_link"], nodes[index_node],
                                                 pose[:3, 3], Rotation.from_matrix(pose[:3, :3]))
                                dest["index_same_node_sdf_gap_m"] = None if phi is None else phi - radius

    return dict(passed=all(r["passed"] and r.get("pad", {}).get("passed", True) for r in records),
                records=records, rod_motion_allowance_m=float(rod_budget),
                endpoints={f"{m}_window_{i}_{where}":x for (m,i,where),x in endpoints.items()},
                minimum_rod_margin_m=min(r["rod_margin_m"] for r in records))


def _counterfactual(candidate, hold_forward, original):
    """Require right thumb separation gain and progress of the fixed index witness in both models."""
    result = {}
    for mode in ("archived_offset_held", "absolute_backend"):
        before = candidate["endpoints"][f"{mode}_window_1_start"]
        after = candidate["endpoints"][f"{mode}_window_2_end"]
        other = hold_forward["endpoints"][f"{mode}_window_2_end"]
        g0, g1 = before["index_same_node_sdf_gap_m"], after["index_same_node_sdf_gap_m"]
        result[mode] = dict(
            thumb_gain_lower_m=after["thumb_hull_lower_gap_m"]-other["thumb_hull_upper_gap_m"],
            forward_index_reduction_m=None if g0 is None or g1 is None else g0-g1)
    admitted = (candidate["passed"] and candidate["minimum_rod_margin_m"] > original["minimum_rod_margin_m"]
                and all(v["thumb_gain_lower_m"] > 0. and v["forward_index_reduction_m"] is not None
                        and v["forward_index_reduction_m"] > 0. for v in result.values()))
    return admitted, result


def _source_grid(models, row):
    grids, lipschitz = {}, {}
    for i, side in enumerate(SIDES):
        for link in PADS:
            name = f"{side}.{link}"
            desc = row["rod"]["mesh_bindings"][name]
            if (desc["frame"] != "PhysX_link_origin" or desc["unit"] != "m"
                    or desc["capture_sha256"] != models[i]["cooked_pads"][link]["capture_sha256"]):
                raise ValueError("locked original mode-consistent convex and SDF sources required")
            raw = checked_file(Path(desc["sdf_path"]), desc["sdf_sha256"])
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                grids[name] = {k:data[k] for k in ("values", "origin", "spacing")}
            lipschitz[name] = grid_lipschitz(grids[name])
    return grids, lipschitz


def replay(run, *, receipt_sha256, project, first_sdk, last_sdk, control_start_sdk):
    outer = json.loads(checked_file(run/"outer.json", receipt_sha256))
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or not outer["debugger_status_verified"] or outer["debugger_inferior_signal"] is not None
            or outer["forced_exit"] or outer["exceeded_budget"]):
        raise ValueError("original successful process receipt required")
    inventory = outer["artifact_sha256"]

    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name, inventory[name])))

    close = read("app_close_returned.json")
    if close["app_close_returned"] is not True or close["close_mode"] != "normal":
        raise ValueError("sealed normal app.close required")
    reset = read("tripod_reset_source_2.json")["evidence"]
    center = reset["table_center_m"]
    table = center[2] + .5*reset["table_size_m"][2]
    boxes = [(np.asarray(z["center_m"]) - .5*np.asarray(z["size_m"]),
              np.asarray(z["center_m"]) + .5*np.asarray(z["size_m"]))
             for z in read("tripod_support_scene.json")["supports"]]
    reset_seed = read("tripod_reset_source_1.json")["reset"]["seed"]
    seed = np.asarray(reset_seed["reset_root_pose_w"])
    radius = reset["rod_radius_m"]
    frozen = read("frozen_plan.json")
    schedule = np.asarray(frozen["tripod_approach"])
    bounds = frozen["right_tripod_distal_contact"]
    stations = bounds["stations"]
    thumb_segments = range(min(stations["right.finger14_link"])-1, max(stations["right.finger14_link"])+1)
    models = [load_geometry(project, side=side) for side in SIDES]
    if first_sdk > last_sdk or (last_sdk-first_sdk) % WINDOW:
        raise ValueError("ordered complete-window candidate range required")
    tried, best, receipts = [], None, {}
    for sdk in range(last_sdk,first_sdk-1,-WINDOW):
        row, prior = (_contact_row(read, tick) for tick in (sdk,sdk-1))
        idx = require_boundary(row, sdk=sdk, start_sdk=control_start_sdk)
        if any(row["rod"][key] != prior["rod"][key] for key in IDENTITY_FIELDS[:-2]):
            raise ValueError("the preceding measured readback must come from the same rod and Reset")
        check_reaction_delivery(prior, row)
        receipts[str(sdk)] = inventory[f"tripod_control_{sdk}.json"]
        receipts[str(sdk-1)] = inventory[f"tripod_control_{sdk-1}.json"]
        boundary, state = _initial_boundary(row, prior, models, radius=radius, boxes=boxes, table_top=table)
        if (not boundary["zero_reserve_scene_passed"] or not boundary["no_hand_load"]
                or min(boundary["support_minimums_n"].values()) <= 0.):
            tried.append(dict(sdk=sdk, cause="precontact_start_gate", boundary=boundary))
            continue
        old_t, _, old_m, _ = _read_bound_wrist(prior)
        old_shapes, _, _ = _hand_geometry(prior, models, old_t, old_m)
        grids, lipschitz = _source_grid(models, row)
        nodes = row["rod"]["node_positions"][0]
        _, nearest = _thumb_distance(state[3][THUMB]["measured"], nodes, thumb_segments)
        away = np.asarray(nearest["hull_point_m"]) - nearest["segment_point_m"]
        witness = _index_witness(nodes, grids["right.finger23_link"], state[3][INDEX], stations["right.finger23_link"])
        if witness is None:
            tried.append(dict(sdk=sdk, cause="no_original_index_witness"))
            continue
        node, gradient = witness
        tangent = nodes[node+1]-nodes[node-1]
        history = []
        for tick in range(sdk-2*WINDOW, sdk+1):
            archived = _contact_row(read,tick)["rod"]
            if (any(row["rod"][key] != archived[key] for key in IDENTITY_FIELDS[:-2])
                    or not np.array_equal(row["rod"]["rest_lengths_m"], archived["rest_lengths_m"])):
                raise ValueError("causal displacement history must use the same rod and Reset")
            receipts[str(tick)] = inventory[f"tripod_control_{tick}.json"]
            history.append(archived["node_positions"][0])
        rod_allowance = prior_rod_reserve(history)

        def evaluate(actions):
            compiled = compile_two_windows(actions, row=row, seed=seed, state=state)
            if compiled is None:
                return None
            proof = _certify_sequence(compiled,seed,state,old_shapes,models,grids,lipschitz,nodes,
                radius=radius,depth=bounds["max_pad_penetration_m"],rod_budget=rod_allowance["allowance_m"],
                boxes=boxes,table=table,index_node=node,thumb_segments=thumb_segments)
            return dict(proof, actions24=np.asarray(actions).tolist(),
                        compiled_targets=[{key:value for key,value in w.items()
                                           if key in ("target_poses_wxyz","intrinsic_angles_rad","finger_targets_rad")}
                                          for w in compiled])

        forward = schedule[idx+1]
        reference = evaluate(np.stack((np.zeros(24), forward)))
        if reference is None:
            tried.append(dict(sdk=sdk, cause="reference_projection_rejected"))
            continue
        anchor = dict(sdk=sdk, next_policy=idx, index_witness=node, candidates=[], rod_allowance=rod_allowance,
                      hold_then_forward_margin_m=reference["minimum_rod_margin_m"])
        hold_start = [r for r in reference["records"] if r["phase"] == 1]
        if any(not r["passed"] or not r.get("pad", {}).get("passed", True) for r in hold_start):
            tried.append(dict(anchor, cause="start_fails_history_reserve_even_at_hold"))
            continue
        original = evaluate(schedule[idx:idx+2])
        if original is None:
            tried.append(dict(anchor, cause="original_two_window_projection_rejected"))
            continue
        anchor["original_authored_margin_m"] = original["minimum_rod_margin_m"]
        if original["passed"]:
            tried.append(dict(anchor, cause="original_authored_pair_already_certified"))
            continue
        for amplitude in (.25, .5, .75, 1.):
            relief = relief_action(away, tangent, gradient, amplitude=amplitude)
            alternative = None if relief is None else evaluate(np.stack((relief,forward)))
            if alternative is None:
                anchor["candidates"].append(dict(amplitude=amplitude, cause="relief_projection_or_direction_rejected"))
                continue
            passed, comparison = _counterfactual(alternative,reference,original)
            anchor["candidates"].append(dict(amplitude=amplitude,passed=passed,comparison=comparison,
                                             minimum_rod_margin_m=alternative["minimum_rod_margin_m"]))
            if passed and (best is None or alternative["minimum_rod_margin_m"] > best["alternative"]["minimum_rod_margin_m"]):
                best = dict(sdk=sdk,next_policy=idx,index_witness_node=node,amplitude=amplitude,
                            actual_support_minimums_n=boundary["support_minimums_n"],
                            target_vs_measured=boundary["target_vs_measured"],
                            alternative=alternative,hold_then_forward=reference,comparison=comparison,
                            authored_forward_policy_index=idx+1, original_authored_two_windows=original,
                            rod_allowance=rod_allowance)
        tried.append(anchor)
    return dict(version="dh116-tripod-transverse-approach-v1",receipt_sha256=receipt_sha256,
                source_head=outer["head"], archived_reset_seed_sha256=reset_seed["sha256"],
                tried=tried, selected=best, source_rows_sha256=receipts,
                native_steps_executed=0,teacher_admission=False,grasp_qualified=False)


def _matrix(position, rotation):
    value = np.eye(4)
    value[:3,:3] = rotation.as_matrix()
    value[:3,3] = position
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("run","project","output"):
        parser.add_argument(f"--{key}",required=True,type=Path)
    parser.add_argument("--receipt-sha256",required=True)
    parser.add_argument("--first-sdk",required=True,type=int)
    parser.add_argument("--last-sdk",required=True,type=int)
    parser.add_argument("--control-start-sdk",required=True,type=int)
    args=parser.parse_args()
    result=replay(args.run,receipt_sha256=args.receipt_sha256,project=args.project,
                  first_sdk=args.first_sdk,last_sdk=args.last_sdk,control_start_sdk=args.control_start_sdk)
    with args.output.open("x",encoding="utf-8") as obj:
        json.dump(result,obj,allow_nan=False,indent=2)
        obj.write(chr(10))


if __name__=="__main__":
    main()
