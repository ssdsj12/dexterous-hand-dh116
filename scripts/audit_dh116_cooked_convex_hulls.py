#!/usr/bin/env python3
"""Audit authored DH116 convex cooking and compare labeled hull geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS, read_pad_triangles


_SCHEMA = "PhysxConvexHullCollisionAPI"
_COOK_ATTRIBUTES = ("physxConvexHullCollision:hullVertexLimit", "physxConvexHullCollision:minThickness")
_COLLISION_ATTRIBUTES = ("physics:approximation", "physics:collisionEnabled",
                         "physxCollision:contactOffset", "physxCollision:restOffset", *_COOK_ATTRIBUTES)


def _get(record, key):
    return record[key] if isinstance(record, dict) else getattr(record, key)


def decode_cooked_hull(record):
    """Read the native dictionary or callback struct without inventing a frame."""
    vertices = np.asarray([[v.x, v.y, v.z] if hasattr(v, "x") else v
                           for v in _get(record, "vertices")], dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all()):
        raise ValueError("invalid PhysX hull vertices")
    indices = np.asarray(_get(record, "indices"))
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or not len(indices) or indices.min() < 0 or indices.max() >= len(vertices)):
        raise ValueError("invalid PhysX hull indices")
    polygons, faces = [], []
    for polygon in _get(record, "polygons"):
        start = int(_get(polygon, "index_base"))
        count = int(_get(polygon, "num_vertices"))
        plane = list(_get(polygon, "plane"))
        if start < 0 or count < 3 or start + count > len(indices) or len(plane) != 4:
            raise ValueError("invalid PhysX hull polygon")
        if not np.isfinite(plane).all() or len(set(indices[start:start + count])) != count:
            raise ValueError("invalid PhysX hull polygon")
        polygons.append({"index_base": start, "num_vertices": count, "plane": plane})
        faces.extend((int(indices[start]), int(indices[start + i]), int(indices[start + i + 1]))
                     for i in range(1, count - 1))
    if not faces:
        raise ValueError("PhysX hull has no polygons")
    return {"vertices": vertices.tolist(), "indices": indices.tolist(),
            "polygons": polygons, "faces": [list(face) for face in faces]}


def query_physx_hulls(cooking, paths, *, stage_id, encode_path, valid_result, simulation_running):
    """Capture both PhysX channels. A request result is separate from a live shape."""
    result = {}
    for path in dict.fromkeys(paths):
        if simulation_running:
            count = cooking.get_nb_convex_mesh_data(path)
            if count < 0:
                raise ValueError("invalid live convex shape count")
            live = [decode_cooked_hull(cooking.get_convex_mesh_data(path, i)) for i in range(count)]
        else:
            live = None
        requested = []

        def on_result(status, convexes):
            requested.append((status, [decode_cooked_hull(c) for c in convexes] if status == valid_result else []))

        cooking.request_convex_collision_representation(
            stage_id=stage_id, collision_prim_id=encode_path(path),
            run_asynchronously=False, on_result=on_result,
        )
        if len(requested) != 1:
            raise RuntimeError("synchronous PhysX cooking must return one callback")
        status, hulls = requested[0]
        result[path] = {
            "live_simulation_cooked": {"hulls": live, "coordinate_frame": "unconfirmed",
                                       "readback_attempted": simulation_running},
            "requested_cooking": {"hulls": hulls, "callback_status": str(status),
                                  "coordinate_frame": "unconfirmed_for_dh116_instance_proxy"},
        }
    return result


def read_native_pad_hulls(stage, collision_owner, mesh_path, *, simulation_running):
    """Use in an isolated already-open Isaac stage. Returned frame remains unconfirmed."""
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, UsdUtils

    cache_id = UsdUtils.StageCache.Get().GetId(stage)
    if not cache_id.IsValid():
        raise ValueError("PhysX cooking requires the active USD stage cache ID")
    return query_physx_hulls(
        get_physx_cooking_interface(), (collision_owner, mesh_path), stage_id=cache_id.ToLongInt(),
        encode_path=PhysicsSchemaTools.sdfPathToInt,
        valid_result=PhysxCollisionRepresentationResult.RESULT_VALID,
        simulation_running=simulation_running,
    )


def geometric_hull_estimate(vertices):
    """Offline diagnostic geometry. Does not apply PhysX cooking parameters."""
    from scipy.spatial import ConvexHull

    vertices = np.asarray(vertices, dtype=np.float64)
    hull = ConvexHull(vertices)
    triangles = hull.simplices.copy()
    points = vertices[triangles]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    flip = np.sum(normals * hull.equations[:, :3], axis=1) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    dense_indices = np.full(len(vertices), -1, dtype=np.int64)
    dense_indices[hull.vertices] = np.arange(len(hull.vertices))
    return vertices[hull.vertices], dense_indices[triangles], {
        "source": "scipy_qhull_geometric_estimate",
        "physx_cooking_applied": False,
        "geometry_is_live_physx_cooked": False,
    }


def _surface_samples(mesh, count):
    # Reproducible area-weighted strata and low-discrepancy barycentric samples.
    sequence = np.arange(count) + 0.5
    area = np.cumsum(mesh.area_faces)
    face = np.searchsorted(area, sequence * area[-1] / count, side="right")
    r = np.sqrt(np.mod(sequence * (np.sqrt(2) - 1), 1))
    s = np.mod(sequence * (np.sqrt(3) - 1), 1)
    barycentric = np.stack((1 - r, r * (1 - s), r * s), axis=1)
    return np.einsum("ni,nij->nj", barycentric, mesh.triangles[face])


def compare_surfaces(vertices_a, faces_a, vertices_b, faces_b, *, samples, coverage_threshold_m=0.0005):
    """Compare surfaces in one common metre frame using exact point-to-triangle distance."""
    import trimesh

    if samples < 1 or not np.isfinite(coverage_threshold_m) or coverage_threshold_m <= 0:
        raise ValueError("positive sample count and coverage threshold required")
    meshes = tuple(trimesh.Trimesh(vertices=v, faces=f, process=True)
                   for v, f in ((vertices_a, faces_a), (vertices_b, faces_b)))
    directed = []
    for source, target in (meshes, meshes[::-1]):
        points = _surface_samples(source, samples)
        distances = np.concatenate([trimesh.proximity.closest_point_naive(target, points[i:i+24])[1]
                                    for i in range(0, len(points), 24)])
        directed.append({
            "minimum_m": float(distances.min()), "maximum_m": float(distances.max()),
            "p95_m": float(np.percentile(distances, 95)),
            "coverage_fraction": float(np.mean(distances <= coverage_threshold_m)),
        })
    volumes = [abs(float(mesh.volume)) if mesh.is_watertight and mesh.is_winding_consistent else None
               for mesh in meshes]
    return {
        "method": "deterministic_area_samples_to_exact_target_triangles",
        "samples_per_direction": samples, "coverage_threshold_m": coverage_threshold_m,
        "a_to_b": directed[0], "b_to_a": directed[1],
        "sampled_symmetric_hausdorff_m": max(row["maximum_m"] for row in directed),
        "volume_m3": volumes,
        "volume_delta_m3": volumes[1] - volumes[0] if None not in volumes else None,
    }


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_authored_hulls(project, schema_file, *, samples):
    from pxr import Sdf, Usd, UsdGeom

    schema_layer = Sdf.Layer.FindOrOpen(str(schema_file))
    schema = schema_layer.GetPrimAtPath(f"/{_SCHEMA}")
    defaults = {key: schema.attributes[key].default for key in _COOK_ATTRIBUTES}
    manifest_path = project / "assets/fixed_dual_panda_dh116_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hands = {}
    for side in ("left", "right"):
        row = next(x for x in manifest["sources"] if x["kind"] == "dh116" and x["side"] == side)
        path = manifest_path.parent / "dependencies" / Path(row["path"]).name
        if _hash(path) != row["sha256"]:
            raise ValueError(f"unexpected original {side} USD digest")
        stage = Usd.Stage.Open(str(path))
        root = stage.GetDefaultPrim()
        cache = UsdGeom.XformCache()
        pads = {}
        for link in DISTAL_PAD_LINKS:
            vertices, faces, source = read_pad_triangles(stage, str(root.GetPath().AppendChild(link)))
            roles = {}
            for role in ("body_path", "collision_owner", "mesh_path"):
                prim = stage.GetPrimAtPath(source[role])
                attrs = {}
                for name in _COLLISION_ATTRIBUTES:
                    attr = prim.GetAttribute(name)
                    if attr:
                        attrs[name] = {"value": attr.Get(), "authored": attr.HasAuthoredValueOpinion()}
                roles[role] = {
                    "path": str(prim.GetPath()), "instance_proxy": prim.IsInstanceProxy(),
                    "schemas": prim.GetAppliedSchemas(), "attributes": attrs,
                    "local_to_world_row_major": np.asarray(cache.GetLocalToWorldTransform(prim)).tolist(),
                }
            matrix = np.asarray(roles["body_path"]["local_to_world_row_major"])
            local = np.column_stack((vertices, np.ones(len(vertices))))
            roundtrip = (local @ matrix) @ np.linalg.inv(matrix)
            hull_v, hull_f, estimate = geometric_hull_estimate(vertices)
            pads[link] = {
                **source, "prim_roles": roles, "source_vertices": len(vertices), "source_faces": len(faces),
                "max_link_world_link_roundtrip_m": float(np.linalg.norm(roundtrip[:, :3] - vertices, axis=1).max()),
                "geometric_estimate": {**estimate, "vertices": len(hull_v), "faces": len(hull_f),
                                       "bounds_link_m": [hull_v.min(0).tolist(), hull_v.max(0).tolist()]},
                "source_to_estimate_comparison": compare_surfaces(vertices, faces, hull_v, hull_f, samples=samples),
                "live_physx_cooked": None,
            }
        hands[side] = {"usd_sha256": row["sha256"], "pads": pads}
    return {
        "scope": "offline_authored_convex_hull_audit",
        "installed_schema_sha256": _hash(schema_file),
        "installed_schema_defaults_not_native_effective_values": defaults,
        "hands": hands, "native_cooking_queried": False, "sdf_baked": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-schema", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_authored_hulls(Path(__file__).resolve().parents[1], args.installed_schema.resolve(),
                                  samples=args.samples)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
