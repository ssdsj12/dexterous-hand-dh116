"""Build a separate below-axis initialization using USD FK only; never launches Isaac."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
from go2_pvcnn.control.dh116.tabletop_pressure_geometry import pressure_direction

VERSION = "source_fk_below_axis_translation_v1"
TABLE_Z = .30
WIRE_RADIUS = .008
MIN_CLEARANCE = .0003
REQUIRED_PAD_Z = -.0005


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projected_distance(polygon):
    """Exact distance to a convex polygon, not a vertex-only distance."""
    following = np.roll(polygon, -1, axis=0)
    edge = following - polygon
    parameter = np.clip(-(polygon * edge).sum(1) / (edge * edge).sum(1), 0., 1.)
    closest = polygon + parameter[:, None] * edge
    cross = edge[:, 0] * -polygon[:, 1] - edge[:, 1] * -polygon[:, 0]
    if (cross >= -1e-14).all() or (cross <= 1e-14).all():
        return 0.
    return float(np.linalg.norm(closest, axis=1).min())


def load_shapes(source, model, q):
    stage = Usd.Stage.Open(str(source))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    poses = model.poses(q)
    shapes = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):
            continue
        body = max((b for b in poses if str(prim.GetPath()).startswith(b + "/")), key=len)
        vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        relative = (cache.GetLocalToWorldTransform(prim) *
                    cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body)).GetInverse())
        local = (np.c_[vertices, np.ones(len(vertices))] @ np.asarray(relative))[:, :3]
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        shapes.append(dict(body=body, path=str(prim.GetPath()), vertices=local,
                           approximation=str(approximation), point_count=len(vertices),
                           local_vertices_sha256=hashlib.sha256(local.tobytes()).hexdigest()))
    if len(shapes) != 17:
        raise ValueError("expected all 17 source collision meshes per hand")
    pins = {}
    for layer in stage.GetUsedLayers():
        if layer.realPath and Path(layer.realPath).is_file():
            pins[layer.realPath] = sha(layer.realPath)
    return shapes, pins


def world_shapes(model, q, shapes):
    poses = model.poses(q)
    return [(s, s["vertices"] @ poses[s["body"]][:3, :3].T + poses[s["body"]][:3, 3])
            for s in shapes]


def geometry(model, q, shapes, wire):
    points = model.points(q)
    result = {}
    all_vertices = []
    for shape, vertices in world_shapes(model, q, shapes):
        projection = vertices[:, [0, 2]] - wire[[0, 2]]
        polygon = projection[ConvexHull(projection).vertices]
        result[shape["body"].split("/")[-1]] = dict(
            table_clearance_m=float(vertices[:, 2].min() - TABLE_Z),
            infinite_cylinder_surface_gap_m=projected_distance(polygon) - WIRE_RADIUS,
            source_mesh=shape["path"])
        all_vertices.append(vertices)
    vertices = np.concatenate(all_vertices)
    return dict(pad_positions=points.tolist(),
                pad_relative_z_m=(points[:, 2] - wire[2]).tolist(),
                pad_span_m=float(np.linalg.norm(points[0] - points[1])),
                minimum_table_clearance_m=min(v["table_clearance_m"] for v in result.values()),
                minimum_infinite_cylinder_gap_m=min(v["infinite_cylinder_surface_gap_m"]
                                                  for v in result.values()),
                table_body_clearance_m={k: v["table_clearance_m"] for k, v in result.items()},
                wire_infinite_cylinder_radial_gap_m={
                    k: v["infinite_cylinder_surface_gap_m"] for k, v in result.items()},
                per_body=result,
                world_aabb_min_m=vertices.min(0).tolist(),
                world_aabb_max_m=vertices.max(0).tolist())


def hand_separation(left, right):
    lo_l, hi_l = np.array(left["world_aabb_min_m"]), np.array(left["world_aabb_max_m"])
    lo_r, hi_r = np.array(right["world_aabb_min_m"]), np.array(right["world_aabb_max_m"])
    axis_gaps = np.maximum(lo_l - hi_r, lo_r - hi_l)
    return dict(axis_separation_m=axis_gaps.tolist(),
                aabb_distance_lower_bound_m=float(np.linalg.norm(axis_gaps.clip(0))),
                disjoint=bool((axis_gaps > 0).any()))


def build(project, initial, depth_m=.001):
    if not np.isfinite(depth_m) or not .0005 <= depth_m <= .002:
        raise ValueError('initial mean pad depth must be between 0.5 and 2mm')
    required_pad_z = REQUIRED_PAD_Z if depth_m == .001 else -depth_m + .00015
    original = json.loads(initial.read_text())
    new = copy.deepcopy(original)
    manifest_path = project / "assets/fixed_dual_panda_dh116_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    wire = np.array(original["wire_pose"][:3], dtype=float)
    if not np.allclose(wire, [.43, 0., .308], atol=1e-12, rtol=0):
        raise ValueError("this artifact requires the unchanged horizontal 8mm-radius rod seed")
    report = dict(version=VERSION, physics_qualified=False, grasp_qualified=False,
                  simulator_started=False, parent_initialization_sha256=sha(initial),
                  builder_sha256=sha(__file__), manifest_sha256=sha(manifest_path),
                  method="translation only; all original joint_positions and grasp_command retained",
                  radius_m=WIRE_RADIUS, table_top_m=TABLE_Z,
                  required_initial_pad_radial_z_exclusive_m=required_pad_z,
                  minimum_table_clearance_required_m=MIN_CLEARANCE,
                  collision_method="all source collision vertices; convex projected polygon edge distance "
                                   "to infinite horizontal Y cylinder (conservative for finite rod); "
                                   "global hand AABBs establish hand-hand separation",
                  limitations=["No PhysX cooking, force/contact measurements or dynamic qualification.",
                               "Only the INITIAL state is required nonpenetrating. The retained legacy "
                               "closed command is audited separately and can overlap the rod.",
                               "Within-hand geometry unchanged; no new self-collision search."],
                  unchanged=dict(source_contract=True, coupling=True, joint_positions=True,
                                 grasp_command=True, root_rotations=True, rod_pose=True,
                                 mass_geometry_material_physics_guards=True),
                  sides={})
    for index, side in enumerate(("left", "right")):
        source = Path(next(x["path"] for x in manifest["sources"]
                           if x["kind"] == "dh116" and x["side"] == side))
        if sha(source) != original["source_contract"]["source_sha256"][side]:
            raise ValueError("source hash mismatch")
        model = HandPointFK(source, side, original["roots"][index])
        q_initial = np.array(original["joint_positions"][index])[[JOINTS.index(j) for j in ACTIVE]]
        np.testing.assert_allclose(expand(q_initial), original["joint_positions"][index],
                                   atol=1e-7, rtol=0)
        q_closed = np.array(original["grasp_command"][6*index:6*index+6])
        shapes, layer_pins = load_shapes(source, model, q_initial)
        before = {label: geometry(model, q, shapes, wire)
                  for label, q in (("initial", q_initial), ("closed", q_closed))}
        # Aim for the requested average initial radial Z. Also put the retained
        # closed-pad points at least 50um below the requested -0.5mm boundary.
        dz = min(-depth_m - np.mean(before["initial"]["pad_relative_z_m"]),
                 -.00055 - max(before["closed"]["pad_relative_z_m"]))
        if dz >= 0 or dz < -.003:
            raise ValueError("unexpected seed depth")
        polygons = []
        for _, vertices in world_shapes(model, q_initial, shapes):
            points = vertices[:, [0, 2]] - wire[[0, 2]]
            polygons.append(points[ConvexHull(points).vertices])
        def gap(dx):
            return min(projected_distance(poly + np.array([dx, dz])) - WIRE_RADIUS
                       for poly in polygons)
        grid = np.linspace(-.003, .003, 241)
        winner = int(np.argmax([gap(dx) for dx in grid]))
        lower, upper = grid[max(0, winner-1)], grid[min(len(grid)-1, winner+1)]
        fitted = minimize_scalar(lambda dx: -gap(dx), bounds=(lower, upper),
                                 method="bounded", options=dict(xatol=1e-13))
        delta = np.array([float(fitted.x), 0., float(dz)])
        root = np.array(original["roots"][index], dtype=float)
        root[:3] += delta
        new["roots"][index] = root.tolist()
        moved = HandPointFK(source, side, root)
        after = {label: geometry(moved, q, shapes, wire)
                 for label, q in (("initial", q_initial), ("closed", q_closed))}
        if not (np.array(after["initial"]["pad_relative_z_m"]) < required_pad_z).all():
            raise ValueError("pad target infeasible")
        if after["initial"]["minimum_table_clearance_m"] < MIN_CLEARANCE:
            raise ValueError("table margin infeasible")
        if after["initial"]["minimum_infinite_cylinder_gap_m"] < 0:
            raise ValueError("initial mesh-cylinder intersection")
        old_jac = model.jacobian(q_closed)
        new_jac = moved.jacobian(q_closed)
        np.testing.assert_allclose(old_jac, new_jac, atol=2e-9, rtol=0)
        direction, compensation = pressure_direction(new_jac, [1., 0., 0.])
        np.testing.assert_allclose(direction, original["diagnostics"]["hands"][side][
            "closure_direction_active_unit"], atol=1e-6, rtol=0)
        diag = new["diagnostics"]["hands"][side]
        for key in ("pad_positions", "pad_span_m", "minimum_table_clearance_m",
                    "table_body_clearance_m", "wire_infinite_cylinder_radial_gap_m"):
            diag[key] = after["initial"][key]
        diag["closed_pad_positions"] = after["closed"]["pad_positions"]
        diag["closed_minimum_table_clearance_m"] = after["closed"]["minimum_table_clearance_m"]
        diag["closed_wire_infinite_cylinder_radial_gap_m"] = after["closed"][
            "wire_infinite_cylinder_radial_gap_m"]
        diag["closure_direction_active_unit"] = direction.tolist()
        diag["pad_separation_jacobian_m_per_rad"] = ((new_jac[0]-new_jac[1])[0]).tolist()
        diag["initial_pad_relative_radial_z_m"] = after["initial"]["pad_relative_z_m"]
        diag["closed_pad_relative_radial_z_m"] = after["closed"]["pad_relative_z_m"]
        pressure = new["pressure_geometry"]
        pressure["palm_compensation_m_per_rad"][index] = compensation.tolist()
        pressure["derivation"][side].update(
            point_jacobian_m_per_rad=new_jac.tolist(),
            nonlinear_delta_at_offset_008_m=(
                moved.points(q_closed+.08*direction)+.08*compensation-moved.points(q_closed)).tolist(),
            reference_active_rad=q_closed.tolist())
        report["sides"][side] = dict(source_path=str(source), source_sha256=sha(source),
            source_layer_sha256=layer_pins, collision_mesh_count=len(shapes),
            collision_meshes=[{k:v for k,v in s.items() if k != "vertices"} for s in shapes],
            old_root=original["roots"][index], new_root=root.tolist(),
            root_delta_world_m=delta.tolist(), initial_active_rad=q_initial.tolist(),
            closed_active_rad=q_closed.tolist(), before=before, after=after,
            jacobian_max_translation_difference=float(np.abs(new_jac-old_jac).max()),
            direction=direction.tolist(), palm_compensation_m_per_rad=compensation.tolist(),
            nearby_x_search=dict(lower_m=-.003, upper_m=.003, grid_points=len(grid),
                                 selected_initial_min_surface_gap_m=gap(fitted.x)))
    for label in ("initial", "closed"):
        report["hand_hand_" + label] = hand_separation(
            report["sides"]["left"]["after"][label], report["sides"]["right"]["after"][label])
        if not report["hand_hand_" + label]["disjoint"]:
            raise ValueError("hand-hand separation not established")
    new["diagnostics"]["physics_qualified"] = False
    new["diagnostics"]["source_geometry"] = "17 source convex meshes per hand; offline FK; no PhysX start"
    new["diagnostics"]["below_axis_geometry_audit_version"] = VERSION
    new["pressure_geometry"]["parent_initialization_sha256"] = sha(initial)
    new["pressure_geometry"]["source_script_sha256"] = sha(__file__)
    new["below_axis_initialization"] = dict(version=VERSION, parent_sha256=sha(initial),
        target_initial_mean_radial_z_m=-depth_m, physics_qualified=False, grasp_qualified=False,
        reference_command_unchanged=True,
        closed_command_warning="Legacy closed command may overlap cylinder; not an initial contact or grip claim.")
    assert new["source_contract"] == original["source_contract"]
    for key in ("joint_positions", "grasp_command", "wire_pose"):
        assert new[key] == original[key]
    report["initial_static_geometry_passed"] = True
    report["retained_closed_command_nonpenetrating"] = all(
        s["after"]["closed"]["minimum_infinite_cylinder_gap_m"] >= 0
        for s in report["sides"].values())
    return new, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/hand"))
    parser.add_argument("--input", type=Path, default=Path("configs/dh116_training/grasp_initialization_pos64.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/dh116_training/grasp_initialization_below_axis_v1.json"))
    parser.add_argument("--audit", type=Path, default=Path("configs/dh116_training/grasp_initialization_below_axis_v1.audit.json"))
    parser.add_argument("--verify", action="store_true", help="Recompute and compare without writing")
    parser.add_argument("--depth-mm", type=float, default=1., help="Mean initial pad depth below axis, 0.5 to 2mm")
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else args.project / p
    initial, output, audit_path = map(resolve, (args.input, args.output, args.audit))
    if output.resolve() == initial.resolve() or audit_path.resolve() == initial.resolve():
        raise ValueError("must never overwrite parent initialization")
    if not args.verify and (output.exists() or audit_path.exists()):
        raise FileExistsError("new independent output paths required")
    result, report = build(args.project, initial, args.depth_mm / 1000.)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    report["output_initialization_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    if args.verify:
        if json.loads(output.read_text()) != result or json.loads(audit_path.read_text()) != report:
            raise ValueError("saved artifact differs from fresh source-FK rebuild")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(content)
        with audit_path.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(verified=args.verify, output=str(output), audit=str(audit_path),
        physics_qualified=False, initial_static_geometry_passed=True,
        closed_command_nonpenetrating=report["retained_closed_command_nonpenetrating"],
        sides={side: dict(root_delta_world_m=s["root_delta_world_m"],
                         new_root=s["new_root"],
                         initial_pad_z_m=s["after"]["initial"]["pad_relative_z_m"],
                         closed_pad_z_m=s["after"]["closed"]["pad_relative_z_m"],
                         initial_clearance_m=s["after"]["initial"]["minimum_table_clearance_m"],
                         initial_gap_m=s["after"]["initial"]["minimum_infinite_cylinder_gap_m"],
                         closed_gap_m=s["after"]["closed"]["minimum_infinite_cylinder_gap_m"])
               for side, s in report["sides"].items()})))


if __name__ == "__main__":
    main()

