#!/usr/bin/env python3
"""Bake the original right Power palm fan and quantify the isolated grid error."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from scripts import audit_dh116_power_palm_self_intersection as intersections
from scripts import bake_dh116_deformx_pad_sdfs as pad_bake
from scripts.bake_dh116_power_left_palm_sdf import BACKEND_LOCK, BACKEND_LOCK_SHA


SURFACE_EPS_M = 1e-10
CROSSING_REPORT_SHA = "e415122e51214673032ef5ffa37f82f4985040baf1bb49f4ecbc229fc14ec555"
MESH = "right.base_link"


def reference_distances(mesh, points):
    """Independent float64 triangle distance, signed using oriented solid angles."""
    points = np.asarray(points, dtype=np.float64)
    _, distances, _ = trimesh.proximity.closest_point_naive(mesh, points)
    result = np.zeros(len(points))
    off_surface = distances > SURFACE_EPS_M
    corners = mesh.triangles[None] - points[off_surface, None, None, :]
    a, b, c = np.moveaxis(corners, 2, 0)
    lengths = [np.linalg.norm(x, axis=-1) for x in (a, b, c)]
    numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
    denominator = lengths[0] * lengths[1] * lengths[2]
    for u, v, w in ((a, b, 2), (b, c, 0), (c, a, 1)):
        denominator += np.einsum("...i,...i->...", u, v) * lengths[w]
    winding = np.abs(np.arctan2(numerator, denominator).sum(axis=1)) / (2 * np.pi)
    if np.any(np.abs(winding - np.rint(winding)) > 1e-5) or np.any(winding > 1.5):
        raise ValueError("ambiguous winding for original closed surface")
    result[off_surface] = np.where(winding > .5, -1., 1.) * distances[off_surface]
    return result


def audit_grid(mesh, sample, *, spacing_m):
    centre = mesh.triangles_center
    midpoints = mesh.vertices[mesh.edges_unique].mean(axis=1)
    groups = {"surface": np.vstack((mesh.vertices, midpoints, centre))}
    for scale in (.025, .1, .5, 1., 2., 5.):
        offset = scale * spacing_m * mesh.face_normals
        groups[f"face_{scale:g}_voxel"] = np.vstack((centre - offset, centre + offset))
    results = {}
    worst = {"absolute_error_m": -1.}
    for name, points in groups.items():
        reference = reference_distances(mesh, points)
        grid, inside = sample(points)
        if not np.all(inside) or not np.isfinite(grid).all():
            raise ValueError(f"all right palm samples must be inside the lattice: {name}")
        errors = np.abs(reference - grid)
        idx = int(errors.argmax())
        sign_known = np.abs(reference) > SURFACE_EPS_M
        mismatches = sign_known & (np.sign(grid) != np.sign(reference))
        results[name] = dict(count=len(points), signed_samples=int(sign_known.sum()),
                             opposite_sign_count=int(mismatches.sum()), mean_error_m=float(errors.mean()),
                             p95_error_m=float(np.percentile(errors, 95)), max_error_m=float(errors[idx]))
        if errors[idx] > worst["absolute_error_m"]:
            worst = dict(group=name, sample_index=idx, xyz_link_m=points[idx].tolist(),
                         grid_m=float(grid[idx]), reference_m=float(reference[idx]), absolute_error_m=float(errors[idx]))
    return dict(groups=results, worst=worst, scope="sampled_original_fan_error_not_global_bound")


def input_right_palm(capture, sealed_audit):
    expected = pad_bake.checked_json(sealed_audit, CROSSING_REPORT_SHA)
    if intersections.evaluate(capture) != expected or expected["no_self_intersections"] is not True:
        raise ValueError("exact original Power palm triangle audit required")
    path = capture / "gpu" / "native" / "right_base_link.json"
    digest = expected["palms"]["right"]["source_row_sha256"]
    row = pad_bake.checked_json(path, digest)
    if (row["side"], row["link"]) != ("right", "base_link"):
        raise ValueError("right original Power palm identity required")
    hull, basis_delta = intersections._pad_hull(row)
    mesh = trimesh.Trimesh(vertices=hull["vertices"], faces=hull["faces"], process=False)
    if (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.body_count != 1
            or mesh.volume <= 0. or reference_distances(mesh, mesh.vertices.mean(axis=0)[None])[0] >= 0):
        raise ValueError("closed outward fan and certified interior centroid required")
    return {MESH: (row, hull, basis_delta, digest)}, mesh


def verify_weld(mesh, weld):
    welded_vertices, welded_faces = weld(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64))
    if len(welded_vertices) != len(mesh.vertices) or len(welded_faces) != len(mesh.faces):
        raise ValueError("upstream weld must preserve every original right palm vertex and face")
    np.testing.assert_array_equal(welded_vertices[welded_faces], mesh.vertices.astype(np.float32)[mesh.faces])
    metrics = intersections.audit_intersections(welded_vertices, welded_faces)
    if metrics["crossings"]:
        raise ValueError("cast and welded right palm faces self-intersect")
    cast = trimesh.Trimesh(vertices=welded_vertices, faces=welded_faces, process=False)
    if not cast.is_watertight or not cast.is_winding_consistent or np.any(cast.area_faces <= 0.):
        raise ValueError("cast and welded right palm must remain watertight and nondegenerate")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "sealed-audit", "ropesim-package-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.ropesim_package_root.resolve(strict=True)
    lock = pad_bake.checked_json(BACKEND_LOCK, BACKEND_LOCK_SHA)
    packets, mesh = input_right_palm(args.capture.resolve(strict=True), args.sealed_audit)
    for name, digest in lock["backend_files"].items():
        pad_bake.checked_bytes(root / "ropesim" / name, digest)
    if args.output is None:
        print(json.dumps(dict(verified_meshes=list(packets), gpu_started=False)))
        return 0
    import sys
    sys.path.insert(0, str(root))
    from ropesim.isaac_bridge import weld
    from ropesim.sdf import SdfGrid

    postweld = verify_weld(mesh, weld)
    args.output.mkdir()
    result = pad_bake.bake_register(packets, lock, root, args.output)
    result["palms"] = result.pop("pads")
    path = args.output / f"{MESH}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != result["palms"][MESH]["sdf_file_sha256"]:
        raise ValueError("isolated grid bytes differ from the bake receipt")
    with np.load(path, allow_pickle=False) as data:
        grid = SdfGrid(data["values"], data["origin"], float(data["spacing"]))
    result.update(scope="isolated_fixed_right_palm_fan_gpu_sdf_and_error_audit",
                  triangle_audit_sha256=CROSSING_REPORT_SHA, backend_lock_sha256=BACKEND_LOCK_SHA,
                  welded_triangle_audit=postweld, reference_error=audit_grid(mesh, grid.sample, spacing_m=grid.spacing),
                  physx_shape_equivalence_verified=False, native_power_task_registration=False,
                  grasp_verified=False, teacher_admission=False)
    pad_bake.write_json(args.output / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
