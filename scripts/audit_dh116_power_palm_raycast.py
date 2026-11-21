#!/usr/bin/env python3
"""Compare original Power palm triangle surfaces to owned PhysX scene ray hits."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from scripts import audit_dh116_power_palm_parity as palm
from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull, compare_hull_topology


RAY_START_M = .004
RAY_LENGTH_M = .008


def first_triangle_hit(triangles, origin, direction, limit):
    """Nearest surface intersection; direction is normalized by the caller."""
    a, b, c = np.moveaxis(np.asarray(triangles), 1, 0)
    p = np.cross(direction, c - a)
    determinant = np.einsum("ij,ij->i", b - a, p)
    active = np.abs(determinant) > 1e-14
    if not active.any():
        return None
    a, b, c, p, determinant = (x[active] for x in (a, b, c, p, determinant))
    offset = np.asarray(origin) - a
    q = np.cross(offset, b - a)
    u = np.einsum("ij,ij->i", offset, p) / determinant
    v = (q @ direction) / determinant
    distances = np.einsum("ij,ij->i", c - a, q) / determinant
    hit = (u >= 0.) & (v >= 0.) & (u + v <= 1.) & (distances >= 0.) & (distances <= limit)
    return float(distances[hit].min()) if hit.any() else None


def ray_audit(row, reference_row, cast, *, expected_link="base_link"):
    if ((row["side"], row["link"]) != (reference_row["side"], expected_link)
            or reference_row["link"] != expected_link):
        raise ValueError(f"same side and original {expected_link} are required")
    if row["source"]["geometry_sha256"] != reference_row["source"]["geometry_sha256"]:
        raise ValueError("original link collision source differs from the archived reference")
    measured, _ = _pad_hull(row)
    archived, _ = _pad_hull(reference_row)
    if not compare_hull_topology(archived, measured)["parity_passed"]:
        raise ValueError("live link triangles differ from the archived topology")
    vertices = np.asarray(measured["vertices"])
    faces = np.asarray(measured["faces"])
    triangles = vertices[faces]
    convex = vertices[ConvexHull(vertices).simplices]
    xyz = np.asarray(row["sdk_body"]["position_m"])
    rotation = Rotation.from_quat(row["sdk_body"]["quaternion_xyzw"])
    kernels = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = kernels / np.linalg.norm(kernels, axis=1, keepdims=True)
    expected_body = row["source"]["body_path"]
    allowed_colliders = {row["source"][key] for key in ("mesh_path", "collision_owner")}
    results = []
    for index, (centre, normal) in enumerate(zip(triangles.mean(axis=1), normals)):
        start = centre + normal * RAY_START_M
        fan_t = first_triangle_hit(triangles, start, -normal, RAY_LENGTH_M)
        convex_t = first_triangle_hit(convex, start, -normal, RAY_LENGTH_M)
        origin = xyz + rotation.apply(start)
        direction = -rotation.apply(normal)
        hit = cast(origin, direction, RAY_LENGTH_M)
        actual = None
        body = str(hit["rigidBody"]) if hit["hit"] else None
        collider = str(hit["collision"]) if hit["hit"] else None
        if body == expected_body and collider in allowed_colliders:
            hit_point = np.array(tuple(hit["position"]))
            local = rotation.inv().apply(hit_point - xyz)
            actual = float((hit_point - origin) @ direction)
        result = dict(triangle=index, start_link_m=start.tolist(), direction_link=(-normal).tolist(),
                      original_fan_m=fan_t, convex_vertex_boundary_m=convex_t, native_hit_distance_m=actual,
                      native_body=body, native_collider=collider)
        if actual is not None:
            result["native_hit_link_m"] = local.tolist()
            result["native_normal_link"] = rotation.inv().apply(tuple(hit["normal"])).tolist()
        results.append(result)
    return dict(side=row["side"], source_body=expected_body, source_colliders=sorted(allowed_colliders),
                clock=row["clock"], total_rays=len(results), owned_rays=sum(x["native_hit_distance_m"] is not None
                    for x in results), rays=results)


def record_palm_rays(record, references, stage, physx, hands, output, clock):
    from omni.physx import get_physx_scene_query_interface
    from pxr import Gf

    rows = record(stage, physx, hands, output, clock)
    interface = get_physx_scene_query_interface()

    def cast(origin, direction, length):
        return interface.raycast_closest(Gf.Vec3f(*origin), Gf.Vec3f(*direction), length)

    reports = {row["side"]:ray_audit(row, references[row["side"]], cast) for row in rows}
    palm.write_json(output / "palm_rays.json", dict(scope="cooked_palm_face_centre_rays", clock=clock, scene_query_support=True,
        source_shape_equivalence_verified=False, grasp_verified=False, teacher_admission=False, palms=reports))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args(argv)
    reference_root = palm.checked_capture(args.capture.resolve(strict=True) / "gpu", "cuda:0")
    reference_outer = reference_root.parent / "outer.json"
    if palm.file_sha(reference_outer) != "689ac3fb8cdd39c5ea3aa189c17c1595d7f4cbf379785941e95ec75da8bb93da":
        raise ValueError("sealed original GPU palms are required")
    references = {side:json.loads((reference_root / f"{side}_base_link.json").read_text())
                  for side in ("left", "right")}
    if not args.execute_native:
        print(json.dumps(dict(physics_steps=2, rays=sum(len(_pad_hull(x)[0]["faces"])
              for x in references.values()), native_started=False, cameras=False, ppo_updates=0)))
        return 0
    original = sys.argv
    try:
        sys.argv = [original[0], "--output", str(args.output), "--execute-native", "--headless",
                    "--steps", "2", "--device", "cuda:0"]
        recorder = partial(record_palm_rays, palm.probe._record_pads, references)
        with palm.bindings((palm.probe, "DISTAL_PAD_LINKS", palm.LINKS),
                           (palm.probe, "read_pad_triangles", palm.read_palm_source),
                           (palm.probe, "_record_pads", recorder)):
            palm.probe.main(scene_queries=True)
    finally:
        sys.argv = original
    reports = json.loads((args.output / "palm_rays.json").read_bytes())["palms"]
    if not all(data["owned_rays"] for data in reports.values()):
        raise ValueError("both original palms must return owned native ray hits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
