#!/usr/bin/env python3
"""CPU audit of locked F4 cooked hulls, failed Pinch poses and a separate pose candidate.

No Isaac, CUDA, live cooking or simulation is imported or launched. Source USD
convex geometry for the other twelve links remains explicitly an estimate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from scripts.bake_dh116_deformx_pad_sdfs import checked_json, input_packets
from scripts.verify_dh116_cooked_convex_hulls_native import _sources
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand

EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
CAPTURE = EVIDENCE/"flex-f4-parity-prep-Vu9H0f"
VERSION = "dh116-pinch-initial-pose-audit-v1"
FAILED_HASHES = {
    "plan.json": "8f803f05c04d946e87cf28416ce478bb8f7b740293a76e244c83cf9bf4e98efd",
    "physics.jsonl": "0c8062301e82b3127b0c49aadc5fdfd0096310f27d00a1f9c9a2f552216a20d1",
    "result.json": "d6a12c3dc7fed06ebe0cedf086b22a74ab495c4019d5da76d4af975b0579502b",
}


def default_failed_output():
    return EVIDENCE/("approved-c5-pinch-20261112-v9l5ajv3/native-runs/"
        "flex.f4-free-pinch-left-candidate-native/attempt-crp1pahm/native")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_array(value, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError("finite array with required shape expected")
    return a


def pose_matrix(pose):
    pose = finite_array(pose, (7,))
    if abs(np.linalg.norm(pose[3:])-1) > 2e-6:
        raise ValueError("unit wxyz quaternion required")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
    t[:3, 3] = pose[:3]
    return t


def transform(vertices, matrix):
    return finite_array(vertices) @ matrix[:3, :3].T + matrix[:3, 3]


def box_relation(vertices, center, size):
    """Exact convex-polyhedron / finite AABB SAT, not a vertex-inside heuristic."""
    v, c, s = finite_array(vertices), finite_array(center, (3,)), finite_array(size, (3,))
    if np.any(s <= 0):
        raise ValueError("positive box size required")
    hull = ConvexHull(v)
    edges = np.vstack([v[hull.simplices[:, (i+1)%3]]-v[hull.simplices[:, i]]
                       for i in range(3)])
    axes = np.vstack([np.eye(3), hull.equations[:, :3],
        *[np.cross(edges, unit) for unit in np.eye(3)]])
    axes = axes[np.linalg.norm(axes, axis=1) > 1e-12]
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    projections = (v-c) @ axes.T
    radii = np.abs(axes) @ (s/2)
    separation = np.maximum(projections.min(0)-radii, -radii-projections.max(0))
    return {"intersects": bool(np.max(separation) <= 1e-10),
            "max_separating_axis_gap_m": float(np.max(separation)),
            "top_clearance_m": float(v[:, 2].min()-(c[2]+s[2]/2)),
            "bounds_world_m": [v.min(0).tolist(), v.max(0).tolist()]}


def convex_patch_closest(vertices, start, directions):
    """Convex QP to a finite segment (1 axis) or rectangle (2 axes), in metres.

    The rectangle is the geometric envelope of a straight rod axis settling
    vertically. This computes geometry only; no material normal/contact is read.
    """
    v, a = finite_array(vertices), finite_array(start, (3,))
    axes = finite_array(directions)
    if axes.ndim != 2 or axes.shape[1] != 3 or len(axes) not in (1, 2):
        raise ValueError("one or two finite patch directions required")
    origin = v.mean(0)
    scale = max(float(np.ptp(v, axis=0).max()), float(np.linalg.norm(axes, axis=1).max()), .001)
    v, a, axes = (v-origin)/scale, (a-origin)/scale, axes/scale
    planes = ConvexHull(v).equations
    def objective(x):
        d = x[:3]-(a+x[3:]@axes)
        return float(d@d)
    def jacobian(x):
        d = x[:3]-(a+x[3:]@axes)
        return np.r_[2*d, -2*axes@d]
    matrix = np.column_stack([planes[:, :3], np.zeros((len(planes), len(axes)))])
    constraint = {"type": "ineq", "fun": lambda x: -matrix@x-planes[:, 3],
                  "jac": lambda x: -matrix}
    result = minimize(objective, np.r_[v.mean(0), np.full(len(axes), .5)], jac=jacobian,
        bounds=[(None, None)]*3+[(0, 1)]*len(axes), constraints=constraint,
        method="SLSQP", options={"ftol": 1e-14, "maxiter": 200})
    if not result.success or np.max(matrix@result.x+planes[:, 3]) > 1e-8:
        raise ValueError("continuous patch/convex-hull distance did not converge")
    point = origin+scale*result.x[:3]
    patch_point = origin+scale*(a+result.x[3:]@axes)
    difference = point-patch_point
    distance = float(np.linalg.norm(difference))
    return {"distance_m": distance, "hull_point_w": point.tolist(),
            "axis_point_w": patch_point.tolist(),
            "geometric_axis_to_hull_direction": (difference/distance).tolist() if distance > 1e-10 else None}


def segment_hull_distance(vertices, start, end):
    """Continuous finite segment to convex solid distance; never samples nodes."""
    return convex_patch_closest(vertices, start,
        [finite_array(end, (3,))-finite_array(start, (3,))])["distance_m"]


def capsule_gap(vertices, nodes, radius):
    nodes = finite_array(nodes)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2 or radius <= 0:
        raise ValueError("continuous rod nodes and positive radius required")
    # Collapse only truly straight, ordered geometry. Otherwise test every edge.
    direction = nodes[-1]-nodes[0]
    length2 = direction@direction
    fractions = ((nodes-nodes[0])@direction)/max(length2, 1e-30)
    straight = (length2 > 0 and np.max(np.linalg.norm(
        nodes-(nodes[0]+fractions[:, None]*direction), axis=1)) < 1e-10
        and np.all(np.diff(fractions) >= 0))
    segments = [(nodes[0], nodes[-1])] if straight else zip(nodes[:-1], nodes[1:])
    distance = min(segment_hull_distance(vertices, a, b) for a, b in segments)
    return {"axis_to_solid_distance_m": distance, "capsule_surface_gap_m": distance-radius,
            "intersects_capsule": bool(distance < radius-1e-9),
            "method": "continuous_segment_convex_solid_QP",
            "penetration_depth_is_not_computed": True}


def load_geometry(project, capture_dir=CAPTURE, *, side='left'):
    from pxr import Usd, UsdPhysics
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    from go2_pvcnn.control.dh116.usd_surface import read_surface

    project = Path(project).resolve()
    refpath = project/"configs/dh116_training/pose_reference.json"
    oldpath = project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json"
    old = json.loads(oldpath.read_text())
    for name in ("configs/dh116_training/pose_reference.json",
                 "docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json"):
        checked_json(project/name, old["sources"][name])
    reference = json.loads(refpath.read_text())
    lockpath = project/"docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json"
    packets = input_packets(capture_dir, json.loads(lockpath.read_text()))
    if side not in ("left", "right"):
        raise ValueError("left or right hand required")
    usd = _sources(project)[side]
    stage = Usd.Stage.Open(usd["path"])
    source_colliders, cooked = {}, {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        record = read_surface(stage, str(prim.GetPath()))
        if record["collision_approximation"] != "convexHull":
            raise ValueError("unexpected source collision approximation")
        vertices = np.asarray(record["points_body"])
        vertices = vertices[ConvexHull(vertices).vertices]
        source_colliders[prim.GetName()] = {"vertices": vertices, "source": record,
            "kind": "source_USD_convex_envelope_not_live_cooked"}
    if len(source_colliders) != 17:
        raise ValueError("expected original seventeen enabled colliders")
    for name in DISTAL_PAD_LINKS:
        cap, hull, basis_delta, sha = packets[side+"."+name]
        if cap["source"]["geometry_sha256"] != source_colliders[name]["source"]["geometry_sha256"]:
            raise ValueError("captured cooked hull is not this USD geometry")
        cooked[name] = {"vertices": np.asarray(hull["vertices"]),
            "faces": np.asarray(hull["faces"]), "capture_sha256": sha,
            "geometry_sha256": cap["source"]["geometry_sha256"],
            "coincident_mesh_body_basis_max_delta": basis_delta,
            "kind": "locked_F4_live_GPU_cooked_hull"}
    pose = reference["resolved_poses"][side]["dh116_base_pose_w"]
    q = reference["pose_contract"][side]["hand_active_q"]
    return {"project": project, "side": side, "reference": reference, "old_candidate": old,
            "usd": usd, "source_colliders": source_colliders, "cooked_pads": cooked,
            "all_17_live_cooked_available": False, "pose": pose, "active_q": q,
            "fk": HandPointFK(usd["path"], side, pose),
            "source_sha256": {"pose_reference": digest(refpath), "usd": usd["sha256"],
                "pad_evidence_lock": digest(lockpath), "old_candidate": digest(oldpath),
                **{name: digest(project/name) for name in (
                    "src/go2_pvcnn/control/dh116/hand_point_kinematics.py",
                    "src/go2_pvcnn/control/dh116/usd_surface.py",
                    "src/go2_pvcnn/control/dh116/usd_contract.py",
                    "src/go2_pvcnn/control/dh116/linear_coupling.py")},
                **{str(capture_dir/f"gpu/{side}_{n}.json"): r["capture_sha256"]
                   for n, r in cooked.items()}}}


def world_vertices(geometry, pose, active_q, *, cooked=False):
    # Reuse immutable source joint frames; only the root transform is replaced.
    expand(active_q)
    delta = pose_matrix(pose) @ np.linalg.inv(pose_matrix(geometry["pose"]))
    transforms = {k.rsplit("/", 1)[-1]: delta@v
                  for k, v in geometry["fk"].poses(active_q).items()}
    shapes = geometry["cooked_pads"] if cooked else geometry["source_colliders"]
    return {name: transform(row["vertices"], transforms[name]) for name, row in shapes.items()}


def shape_metrics(shapes, layout, radius):
    return {name: {"table": box_relation(v, layout["table_center"], layout["table_size"]),
                   "rod": capsule_gap(v, layout["wire_points"], radius)}
            for name, v in shapes.items()}


def audit_initial_pose(geometry, native_output):
    if geometry["side"] != "left":
        raise ValueError("failed Pinch evidence is left only")
    native_output = Path(native_output)
    for name, sha in FAILED_HASHES.items():
        if digest(native_output/name) != sha:
            raise ValueError("failed native evidence SHA256 differs: "+name)
    plan = json.loads((native_output/"plan.json").read_text())
    rows = [json.loads(x) for x in (native_output/"physics.jsonl").read_text().splitlines()]
    if len(rows) != 1 or rows[0]["phase_step"] != 1 or rows[0]["sdk_step"] != 3:
        raise ValueError("expected the recorded single failed measurement at SDK tick 3")
    row = rows[0]
    if row["failure"] != "mimic_error" or plan["source_usd"]["sha256"] != geometry["usd"]["sha256"]:
        raise ValueError("failed run source/result identity differs")
    # SDK q order was not serialized. Do not fabricate it from a different run.
    before = row["pre_state"]
    transforms = {name: pose_matrix(np.r_[before["link_pos"][i], before["link_quat"][i]])
                  for i, name in enumerate(DISTAL_PAD_LINKS)}
    position_deltas, rotation_deltas = [], []
    expected = {p.rsplit("/", 1)[-1]: t for p, t in geometry["fk"].poses(geometry["active_q"]).items()}
    for i, name in enumerate(DISTAL_PAD_LINKS):
        if row["actual_pad_bindings"][name]["mesh_index"] != i:
            raise ValueError("pad order differs")
        if plan["pad_bindings"][name]["capture_sha256"] != geometry["cooked_pads"][name]["capture_sha256"]:
            raise ValueError("failed native row uses another cooked capture")
        position_deltas.append(float(np.linalg.norm(expected[name][:3, 3]-transforms[name][:3, 3])))
        rotation_deltas.append(float(np.max(np.abs(expected[name][:3, :3]-transforms[name][:3, :3]))))
    if max(position_deltas) > 1e-6 or max(rotation_deltas) > 2e-6:
        raise ValueError("source FK does not reproduce recorded actual link poses")
    actual = {name: transform(shape["vertices"], transforms[name])
              for name, shape in geometry["cooked_pads"].items()}
    layout = geometry["reference"]["reference_layout"]
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    source = world_vertices(geometry, geometry["pose"], geometry["active_q"])
    metrics = shape_metrics(actual, layout, radius)
    return {"version": VERSION, "native_execution_performed": False,
        "failed_run_artifacts_sha256": {n: digest(native_output/n)
            for n in ("plan.json", "physics.jsonl", "result.json")},
        "clock": {k: row[k] for k in ("env_id", "episode_uid", "reset_id", "rod_uid",
            "sdk_step", "physics_step", "timestamp_s", "policy_step", "substep", "rod_tick")},
        "actual_pre_state": {"cooked_pads": metrics,
            "pose_source": "failed_physics_row_pre_state_at_SDK_tick_2",
            "raw_link_pos_w": before["link_pos"], "raw_link_quat_wxyz": before["link_quat"],
            "T_world_link_column_vector": {n: t.tolist() for n, t in transforms.items()}},
        "joint_readback": {
            "joint_names": None, "order_status": "not_serialized_do_not_infer",
            "pre_q_rad": before["q"], "pre_qd_rad_s": before["qd"],
            "post_q_rad": row["post_state"]["q"], "post_qd_rad_s": row["post_state"]["qd"],
            "max_abs_delta_q_rad": float(np.max(np.abs(
                np.asarray(row["post_state"]["q"])-before["q"]))),
            "max_abs_post_qd_rad_s": float(np.max(np.abs(row["post_state"]["qd"])))},
        "source_transform_chain": {
            "world_base_pose_wxyz": geometry["pose"],
            "world_base_pose_status": "authored_reset_not_native_root_readback",
            "active_joint_names": list(ACTIVE), "active_q_rad": geometry["active_q"],
            "expanded_joint_names": list(JOINTS),
            "expanded_q_rad": expand(geometry["active_q"]).tolist(),
            "joint_frames": [{"joint": n, "parent": p, "child": c,
                "T_parent_joint_column_vector": t0.tolist(),
                "T_joint_child_column_vector": t1.tolist(),
                "axis_joint_local": axis.tolist() if axis is not None else None}
                for n, p, c, t0, t1, axis in geometry["fk"].edges],
            "T_world_link_source_FK_column_vector": {n: t.tolist() for n, t in expected.items()},
            "collision_shapes": {n: {k: r["source"][k] for k in (
                "body_path", "mesh_path", "collision_owner", "mesh_to_body",
                "geometry_sha256", "collision_approximation")}
                for n, r in geometry["source_colliders"].items()},
            "cooked_basis": "locked_F4_mesh_local_coincident_with_PhysX_link_origin"},
        "source_fk_vs_actual": {"max_position_delta_m": max(position_deltas),
            "max_rotation_matrix_delta": max(rotation_deltas)},
        "source_all_17": {name: box_relation(v, layout["table_center"], layout["table_size"])
                          for name, v in source.items()},
        "initial_geometry_passed": not any(r["table"]["intersects"] for r in metrics.values()),
        "joint_name_order_recorded": False,
        "contact_impulse_or_causality_proven": False,
        "all_17_live_cooked_available": False, "source_sha256": geometry["source_sha256"]}


def path_displacement_bound(geometry, active_q, delta, *, cooked=False):
    """Uniform point travel bound for the entire joint-space path, per link.

    Each revolute contribution is |angle| times a triangle-inequality upper
    bound on its downstream lever arm. Fixed source joint frames are included.
    This bounds unsampled points between adjacent commanded FK states.
    """
    full_delta = dict(zip(JOINTS, expand(np.asarray(active_q)+delta)-expand(active_q)))
    by_child = {e[2]: e for e in geometry["fk"].edges}
    out = {}
    shapes = geometry["cooked_pads"] if cooked else geometry["source_colliders"]
    for name, shape in shapes.items():
        path = geometry["source_colliders"][name]["source"]["body_path"]
        downstream = float(np.linalg.norm(shape["vertices"], axis=1).max())
        distance = 0.
        while path in by_child:
            joint, parent, _, t0, t1, axis = by_child[path]
            radius = downstream+float(np.linalg.norm(t1[:3, 3]))
            if axis is not None:
                distance += abs(full_delta[joint])*radius
            downstream = radius+float(np.linalg.norm(t0[:3, 3]))
            path = parent
        out[name] = distance
    return out


def settling_gap(vertices, layout, radius):
    nodes = np.asarray(layout["wire_points"], float)
    if (np.max(np.abs(nodes[:, [0, 2]]-nodes[0, [0, 2]])) > 1e-10
            or np.any(np.diff(nodes[:, 1]) <= 0)):
        raise ValueError("settling envelope requires the unchanged straight y-axis rod")
    top = layout["table_center"][2]+layout["table_size"][2]/2
    floor = top+radius
    if nodes[0, 2] < floor:
        raise ValueError("initial rod penetrates the table")
    start = nodes[0].copy()
    start[2] = floor
    result = convex_patch_closest(vertices, start, [
        nodes[-1]-nodes[0], [0, 0, nodes[0, 2]-floor]])
    return {**result, "capsule_surface_gap_m": result["distance_m"]-radius,
            "axis_height_range_m": [floor, float(nodes[0, 2])],
            "scope": "vertical_geometric_envelope_not_dynamic_settling_proof"}


def evaluate_candidate(geometry, candidate):
    if candidate["version"] != "dh116-pinch-pose-candidate-v2" or geometry["side"] != "left":
        raise ValueError("distinct left Pinch pose candidate version required")
    if candidate["scope"] != "CPU_geometry_candidate_only":
        raise ValueError("pose candidate cannot enable a native runtime")
    if candidate["source_sha256"] != geometry["source_sha256"]:
        raise ValueError("candidate source SHA256 differs")
    if candidate["failed_native_artifacts_sha256"] != FAILED_HASHES:
        raise ValueError("candidate failed artifact SHA256 differs")
    if any(candidate[k] is not False for k in (
            "native_recipe_ready", "grasp_qualified", "all_17_live_cooked_available")):
        raise ValueError("CPU pose candidate cannot claim native qualification")
    pose, q = candidate["base_pose_wxyz"], np.asarray(candidate["active_q_rad"], float)
    if candidate["active_joint_names"] != list(ACTIVE):
        raise ValueError("candidate named active joint order differs")
    expand(q)
    approach = candidate["approach"]
    delta = finite_array(approach["active_delta_rad"], (6,))
    if (not np.array_equal(np.flatnonzero(delta), [1, 2])
            or np.any(delta < 0) or np.max(delta) > .025+1e-12
            or approach["move_steps"] != 160 or approach["physics_hz"] != 480
            or not 0 < approach["velocity_rad_s"] <= .3):
        raise ValueError("candidate approach exceeds original two-channel bounded limits")
    if not np.allclose(delta[[1, 2]], approach["velocity_rad_s"]*160/480, atol=1e-12):
        raise ValueError("candidate approach velocity/duration differs")
    layout = geometry["reference"]["reference_layout"]
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    top = layout["table_center"][2]+layout["table_size"][2]/2
    cooked_world = world_vertices(geometry, pose, q, cooked=True)
    cooked = shape_metrics(cooked_world, layout, radius)
    source = world_vertices(geometry, pose, q)
    tables = {n: box_relation(v, layout["table_center"], layout["table_size"]) for n, v in source.items()}
    initial_settling = {n: settling_gap(v, layout, radius) for n, v in source.items()}
    initial_cooked_settling = {n: settling_gap(v, layout, radius) for n, v in cooked_world.items()}
    minimum, per_link_min, proximal_min = float("inf"), {}, {}
    bounds = path_displacement_bound(geometry, q, delta)
    cooked_bounds = path_displacement_bound(geometry, q, delta, cooked=True)
    cooked_min = {}
    intervals = 160
    for fraction in np.linspace(0, 1, intervals+1):
        active = q+fraction*delta
        shapes = world_vertices(geometry, pose, active)
        for name, v in world_vertices(geometry, pose, active, cooked=True).items():
            clearance = float(v[:, 2].min()-top)
            cooked_min[name] = min(cooked_min.get(name, float("inf")), clearance)
        for name, v in shapes.items():
            clearance = float(v[:, 2].min()-top)
            per_link_min[name] = min(per_link_min.get(name, float("inf")), clearance)
            minimum = min(minimum, clearance)
            if name not in DISTAL_PAD_LINKS:
                gap = settling_gap(v, layout, radius)["capsule_surface_gap_m"]
                proximal_min[name] = min(proximal_min.get(name, float("inf")), gap)
    continuous_table = {n: d-bounds[n]/(2*intervals) for n, d in per_link_min.items()}
    continuous_proximal = {n: d-bounds[n]/(2*intervals)-2e-6 for n, d in proximal_min.items()}
    continuous_cooked_table = {n: d-cooked_bounds[n]/(2*intervals)
                              for n, d in cooked_min.items()}
    endpoint_world = world_vertices(geometry, pose, q+delta, cooked=True)
    end_metrics = shape_metrics(endpoint_world, layout, radius)
    settled_nodes = np.asarray(layout["wire_points"], float).copy()
    settled_nodes[:, 2] = top+radius
    endpoint_settled = {}
    for name, v in endpoint_world.items():
        closest = convex_patch_closest(v, settled_nodes[0], [settled_nodes[-1]-settled_nodes[0]])
        endpoint_settled[name] = {**closest, "capsule_surface_gap_m": closest["distance_m"]-radius}
    directions = [np.asarray(endpoint_settled[n]["geometric_axis_to_hull_direction"])
                  for n in DISTAL_PAD_LINKS[:2]]
    dot = float(directions[0]@directions[1])
    margin = float(candidate["minimum_source_table_clearance_m"])
    if not np.isfinite(margin) or margin < .0019:
        raise ValueError("candidate cannot weaken the 1.9 mm source clearance gate")
    initial_pass = all(r["top_clearance_m"] >= margin for r in tables.values())
    cooked_initial_pass = all(r["table"]["top_clearance_m"] >= margin for r in cooked.values())
    cooked_approach_pass = all(v >= margin for v in continuous_cooked_table.values())
    approach_pass = all(v >= margin for v in continuous_table.values())
    proximal_pass = all(v >= 0 for v in continuous_proximal.values())
    initial_rod_clear = all(r["capsule_surface_gap_m"] >= 0 for r in
        [*initial_settling.values(), *initial_cooked_settling.values()])
    reach = all(abs(endpoint_settled[n]["capsule_surface_gap_m"]) < .00005 for n in DISTAL_PAD_LINKS[:2])
    return {"version": VERSION, "candidate_version": candidate["version"],
        "native_execution_performed": False, "grasp_qualified": False, "native_recipe_ready": False,
        "source_geometry_initial_clearance_passed": initial_pass,
        "cooked_geometry_initial_clearance_passed": cooked_initial_pass,
        "geometry_candidate_passed": bool(initial_pass and cooked_initial_pass
            and cooked_approach_pass and approach_pass and proximal_pass
            and initial_rod_clear and reach and dot < -.2),
        "initial_cooked_pads": cooked, "initial_source_all_17": tables,
        "initial_source_rod_settling_envelope": initial_settling,
        "initial_cooked_rod_settling_envelope": initial_cooked_settling,
        "approach": {"sample_count": intervals+1, "max_active_delta_rad": float(delta.max()),
            "duration_s": 160/480, "velocity_rad_s": approach["velocity_rad_s"],
            "all_source_colliders_above_table": approach_pass,
            "all_cooked_pads_above_table": cooked_approach_pass,
            "continuous_cooked_table_clearance_lower_bound_m": continuous_cooked_table,
            "minimum_source_table_clearance_m": minimum,
            "per_link_minimum_table_clearance_m": per_link_min,
            "continuous_table_clearance_lower_bound_m": continuous_table,
            "continuous_proximal_rod_gap_lower_bound_m": continuous_proximal,
            "sampling_bound_method": "per_joint_angle_times_downstream_lever_bound_half_interval",
            "endpoint_cooked_pads": end_metrics,
            "endpoint_cooked_pads_settled_rod": endpoint_settled,
            "required_pair_geometric_direction_dot": dot,
            "effective_opposed_contact": None,
            "is_dynamics_or_tracking_proof": False},
        "remaining_gates": candidate["remaining_gates"],
        "all_17_live_cooked_available": False}


def audit_style_seed(project, style, seed_path=None):
    """Independent style input; missing seed is a hard blocked result.

    A seed supplies canonical eleven-joint order, separate roots for both hands
    and matching original USD hashes. No Pinch fallback is permitted.
    """
    if style not in ("pinch", "tripod", "power"):
        raise ValueError("unknown independent grasp style")
    result = {"style": style, "native_recipe_ready": False,
              "grasp_qualified": False, "native_execution_performed": False}
    if seed_path is None:
        return {**result, "geometry_seed_passed": False,
                "blocked_reasons": ["missing_independent_"+style+"_seed"]}
    seed_path = Path(seed_path)
    seed = json.loads(seed_path.read_text())
    if seed["diagnostics"]["joint_order"] != list(JOINTS):
        raise ValueError("style seed requires explicit canonical joint order")
    source = seed["source_contract"]
    hands = {}
    required = {"pinch": 2, "tripod": 3, "power": 5}[style]
    reasons = []
    for index, side in enumerate(("left", "right")):
        geometry = load_geometry(project, side=side)
        if (source["source_sha256"][side] != geometry["usd"]["sha256"]
                or source["pose_sha256"] != geometry["source_sha256"]["pose_reference"]):
            raise ValueError("independent style source SHA256 differs")
        full = finite_array(seed["joint_positions"][index], (11,))
        q = np.asarray([full[JOINTS.index(n)] for n in ACTIVE])
        if not np.allclose(full, expand(q), atol=1e-10, rtol=0):
            raise ValueError("independent style seed violates canonical mimic mapping")
        pose = seed["roots"][index]
        layout = geometry["reference"]["reference_layout"]
        if not np.allclose(seed["wire_pose"], [.43, 0, .308, 1, 0, 0, 0], atol=1e-12, rtol=0):
            raise ValueError("independent style rod pose differs from this audit layout")
        radius = geometry["old_candidate"]["rod"]["radius_m"]
        shapes = world_vertices(geometry, pose, q)
        metrics = shape_metrics(shapes, layout, radius)
        cooked = shape_metrics(world_vertices(geometry, pose, q, cooked=True), layout, radius)
        clearance = min(m["table"]["top_clearance_m"] for m in metrics.values())
        if clearance < .0019:
            reasons.append(side+"_source_table_clearance_below_1p9mm")
        if any(r["rod"]["capsule_surface_gap_m"] < 0 for r in metrics.values()):
            reasons.append(side+"_source_rod_initial_overlap")
        hands[side] = {"source_all_17": metrics, "cooked_pads": cooked,
            "required_pad_names": list(DISTAL_PAD_LINKS[:required]),
            "minimum_source_table_clearance_m": clearance,
            "joint_names": list(ACTIVE), "active_q_rad": q.tolist(),
            "base_pose_wxyz": pose, "source_sha256": geometry["source_sha256"],
            "live_pose_readback_available": False}
    reasons += ["independent_style_approach_not_supplied",
                "twelve_nonpad_live_cooked_hulls_per_hand_unavailable"]
    return {**result, "seed_sha256": digest(seed_path), "hands": hands,
            "geometry_seed_passed": False, "blocked_reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--native-output", type=Path, default=default_failed_output())
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--style", choices=("pinch", "tripod", "power"))
    parser.add_argument("--style-seed", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.style:
        report = {"style_seed": audit_style_seed(args.project, args.style, args.style_seed)}
    else:
        if args.style_seed:
            parser.error("--style-seed requires --style")
        geometry = load_geometry(args.project)
        report = {"failed_initial_pose": audit_initial_pose(geometry, args.native_output)}
        if args.candidate:
            report["corrected_pose_candidate"] = evaluate_candidate(
                geometry, json.loads(args.candidate.read_text()))
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text)
    if args.style:
        return 0 if report["style_seed"]["geometry_seed_passed"] else 2
    if args.candidate and not report["corrected_pose_candidate"]["geometry_candidate_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
