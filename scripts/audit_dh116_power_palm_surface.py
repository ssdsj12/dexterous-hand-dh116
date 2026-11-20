#!/usr/bin/env python3
"""Audit the original Power palm triangle surface without changing PhysX planes."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
import trimesh

from scripts.audit_dh116_cooked_convex_hulls import decode_cooked_hull
from scripts import audit_dh116_power_palm_parity as parity
from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull


PARITY_OUTER = {
    "cpu": "f24b231283706510b8a87cf072a96c1fb6f67662ad96b5db01d3a32527f55e17",
    "gpu": "689ac3fb8cdd39c5ea3aa189c17c1595d7f4cbf379785941e95ec75da8bb93da",
}
PARITY_SHA = "c296c2301d9c266693ff5c4956889bb1e279a8e7e6afe9cc988650bc87c381cb"


def audit_surface(hull):
    vertices = np.asarray(hull["vertices"], dtype=float)
    faces = np.asarray(hull["faces"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("finite metre-local native vertices required")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or
            not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("original in-range native triangle indices required")
    scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    tol = scale * 4096 * np.finfo(float).eps
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if scale <= 0 or np.any(lengths <= tol * scale):
        raise ValueError("nondegenerate native triangles required")
    normals /= lengths[:, None]
    residual = np.einsum("fvc,fc->fv", vertices[None, :, :] - tris[:, :1, :], normals)
    violations = np.maximum(0., residual.max(axis=1))
    face = int(np.argmax(violations))
    try:
        boundary = ConvexHull(vertices)
    except QhullError as error:
        raise ValueError("native palms must enclose a nonzero volume") from error
    native_triangles = {tuple(sorted(face)) for face in faces.tolist()}
    convex_triangles = {tuple(sorted(face)) for face in boundary.simplices.tolist()}
    exact_facets = len(native_triangles) == len(faces) and native_triangles == convex_triangles
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    convex_volume = float(boundary.volume)
    signed_volume = float(mesh.volume)
    watertight = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent)
    one_component = mesh.body_count == 1
    volume_equal = abs(signed_volume - convex_volume) <= tol * scale**2
    matches = (watertight and winding and one_component and signed_volume > 0.
               and volume_equal and np.all(violations <= tol))
    return dict(watertight=watertight, winding_consistent=winding, single_component=one_component,
                original_vertex_count=len(vertices), original_triangle_count=len(faces),
                signed_mesh_volume_m3=signed_volume, convex_volume_m3=convex_volume,
                arithmetic_tolerance_m=tol, exact_convex_facets=exact_facets,
                original_facets_not_in_convex_triangulation=len(native_triangles - convex_triangles),
                support_violation_faces=int(np.count_nonzero(violations > tol)),
                worst_support_violation_m=float(violations[face]), worst_triangle_index=face,
                worst_native_vertex_index=int(np.argmax(residual[face])),
                matches_convex_boundary=bool(matches), native_sdf_registered=False)


def evaluate(capture):
    for device, digest in PARITY_OUTER.items():
        if parity.file_sha(capture / device / "outer.json") != digest:
            raise ValueError("sealed CPU/GPU capture receipt SHA required")
    if parity.file_sha(capture / "parity.json") != PARITY_SHA or not parity.compare(
            capture / "cpu", capture / "gpu")["parity_passed"]:
        raise ValueError("exact paired native parity evidence required")
    by_side = {}
    root = capture / "gpu" / "native"
    for side in ("left", "right"):
        path = root / f"{side}_base_link.json"
        row = json.loads(path.read_bytes())
        hull, _ = _pad_hull(row)
        if decode_cooked_hull(hull) != hull:
            raise ValueError("stored triangles must match the native ordered polygon fans")
        by_side[side] = dict(gpu_row_sha256=parity.file_sha(path), **audit_surface(hull))
    return dict(schema="dh116-power-audited-native-palm-fans-v1",
                scope="closed oriented original triangle surface vs support geometry",
                parity_sha256=PARITY_SHA, outer_receipts=PARITY_OUTER, palms=by_side,
                native_sdf_baked=False, native_sdf_registered=False, power_grasp_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.capture.resolve(strict=True))
    parity.write_json(args.output, result)
    print(json.dumps({side: v["matches_convex_boundary"] for side,v in result["palms"].items()}))


if __name__ == "__main__":
    raise SystemExit(main())
