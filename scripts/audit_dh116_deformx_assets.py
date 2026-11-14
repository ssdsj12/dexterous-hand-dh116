#!/usr/bin/env python3
"""Check a pinned DeformX-Assets drop/hang cache without launching Isaac."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) != name:
        raise ValueError(f"invalid relative path: {name}")
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes asset root: {name}")
    if not resolved.is_file():
        raise ValueError(f"missing selected file: {name}")
    return resolved


def verify_files(root, lock):
    manifest_path = _local_file(root, "manifest.json")
    digest = _sha256(manifest_path)
    if digest != lock["manifest_sha256"]:
        raise ValueError("upstream manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {}
    for row in manifest["files"]:
        name = row["path"]
        if name in lock["include_files"] or name.startswith(tuple(lock["include_prefixes"])):
            if name in expected:
                raise ValueError(f"duplicate selected manifest path: {name}")
            expected[name] = (row["sha256"], row["bytes"])
    verified = {"manifest.json": {"sha256": digest, "bytes": manifest_path.stat().st_size}}
    drift = []
    for name in sorted(set(expected) | set(lock["metadata_sha256"])):
        path = _local_file(root, name)
        current = (_sha256(path), path.stat().st_size)
        if name in lock["metadata_sha256"]:
            if current[0] != lock["metadata_sha256"][name]:
                raise ValueError(f"pinned revision metadata digest mismatch: {name}")
            if name in expected and expected[name] != current:
                drift.append(name)
        elif current != expected[name]:
            raise ValueError(f"upstream manifest digest mismatch: {name}")
        verified[name] = {"sha256": current[0], "bytes": current[1]}
    absent = set(lock["include_files"]) - verified.keys()
    if absent:
        raise ValueError(f"required names missing from pinned manifest: {sorted(absent)}")
    return {
        "verified_count": len(verified),
        "verified_bytes": sum(row["bytes"] for row in verified.values()),
        "files": verified,
        "metadata_manifest_drift": drift,
        "upstream_exclusions": manifest.get("excluded", {}),
    }


def audit_usd(root, name, *, expect_mesh):
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdSkel, UsdUtils

    path = _local_file(root, name)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
    local = set()
    external = set()
    for candidate in [*(layer.realPath for layer in layers), *assets]:
        if candidate == str(path):
            continue
        dependency = Path(candidate)
        if dependency.is_absolute() and dependency.is_relative_to(root):
            local.add(dependency.relative_to(root).as_posix())
        else:
            external.add(str(candidate))
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"unreadable USD stage: {name}")
    meshes = []
    skeletons = []
    unresolved_dome_assets = set()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdLux.DomeLight):
            texture = UsdLux.DomeLight(prim).GetTextureFileAttr().Get()
            if texture and not texture.resolvedPath:
                candidate = (path.parent / texture.path).resolve()
                if candidate.is_relative_to(root):
                    unresolved_dome_assets.add(candidate.relative_to(root).as_posix())
        if prim.IsA(UsdSkel.Skeleton):
            skeletons.append({"path": str(prim.GetPath()),
                              "joints": len(UsdSkel.Skeleton(prim).GetJointsAttr().Get() or [])})
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            faces = mesh.GetFaceVertexCountsAttr().Get()
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                    or not np.isfinite(points).all() or faces is None or not len(faces)
                    or min(faces) < 3 or sum(faces) != len(indices)
                    or indices.min() < 0 or indices.max() >= len(points)):
                raise ValueError(f"invalid mesh points or topology: {name}:{prim.GetPath()}")
            meshes.append({"path": str(prim.GetPath()), "vertices": len(points), "faces": len(faces),
                           "local_bounds_stage_units": [points.min(0).tolist(), points.max(0).tolist()]})
    if expect_mesh and (not meshes or not any(row["joints"] for row in skeletons)):
        raise ValueError(f"wire asset missing mesh or skeleton: {name}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError(f"invalid stage unit: {name}")
    unresolved_paths = {Path(name) for name in map(str, unresolved)}
    return {
        "unresolved_dome_dependencies": sorted(unresolved_dome_assets),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": meters_per_unit,
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "meshes": meshes,
        "skeletons": skeletons,
        "local_dependencies": sorted(local),
        "external_dependencies": sorted(external),
        "unresolved_dependencies": sorted(path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
                                          for path in unresolved_paths),
    }


def audit_trajectory(path, *, expected_rods):
    with np.load(path, allow_pickle=False) as data:
        time = data["time_arr"]
        positions = data["pos_arr"]
        directors = data["dir_arr"]
        seeds = data["seed_arr"]
        radii = data["radius_arr"]
        fields = {}
        for key in data.files:
            field = data[key]
            if not np.issubdtype(field.dtype, np.number) or not np.isfinite(field).all():
                raise ValueError(f"non-numeric or non-finite field: {path.name}:{key}")
            fields[key] = {"shape": list(field.shape), "dtype": str(field.dtype)}
        if (time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0)):
            raise ValueError(f"invalid time vector: {path.name}")
        if (positions.ndim != 4 or positions.shape[1:3] != (expected_rods, 3)
                or positions.shape[0] == 0 or positions.shape[-1] < 2):
            raise ValueError(f"invalid position layout: {path.name}")
        records, rods, _, nodes = positions.shape
        if directors.shape != (records, rods, 3, 3, nodes - 1):
            raise ValueError(f"invalid director layout: {path.name}")
        hang_counts = {"hanging_count_arr", "sweep_count_arr"}
        records_per_seed = 1
        if hang_counts.intersection(fields):
            if not hang_counts.issubset(fields):
                raise ValueError(f"incomplete hang layout: {path.name}")
            hanging, sweep = data["hanging_count_arr"], data["sweep_count_arr"]
            if (any(counts.shape != (records,) or not np.issubdtype(counts.dtype, np.integer)
                    or np.any(counts <= 0) for counts in (hanging, sweep))
                    or np.any(hanging + sweep != rods)):
                raise ValueError(f"invalid hang rod counts: {path.name}")
            records_per_seed = 2
        if (seeds.ndim != 1 or len(seeds) * records_per_seed != records
                or not np.issubdtype(seeds.dtype, np.integer)
                or radii.shape != (records, rods) or np.any(radii <= 0)):
            raise ValueError(f"invalid seed or radius layout: {path.name}")
        return {
            "record_count": records,
            "seed_count": len(seeds),
            "rods_per_record": rods,
            "nodes_per_rod": nodes,
            "director_frames_per_rod": nodes - 1,
            "time_vector_samples": len(time),
            "time_vector_endpoints": [float(time[0]), float(time[-1])],
            "radius_range": [float(radii.min()), float(radii.max())],
            "fields": fields,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error("--output must be outside the read-only asset root")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    inventory = verify_files(root, lock)
    wire_usds = sorted(name for name in inventory["files"]
                       if name.startswith("usd/wire_usdc/") and name.endswith(".usdc"))
    if len(wire_usds) != lock["wire_usdc_count"]:
        raise ValueError("pinned wire USD count differs")
    usd = {name: audit_usd(root, name, expect_mesh=name in wire_usds)
           for name in [*lock["scene_usds"], *wire_usds]}
    trajectories = {name: audit_trajectory(_local_file(root, name), expected_rods=rods)
                    for name, rods in lock["trajectory_rods"].items()}
    unresolved = [name for name, row in usd.items()
                  if row["unresolved_dependencies"] or row["external_dependencies"]]
    unexpected = {}
    expected_domes = {}
    for name in unresolved:
        row = usd[name]
        allowed = (set(lock["expected_template_dome_references"].get(name, ()))
                   & set(row["unresolved_dome_dependencies"])
                   & set(row["unresolved_dependencies"]))
        if allowed:
            expected_domes[name] = sorted(allowed)
        other = (set(row["unresolved_dependencies"]) - allowed) | set(row["external_dependencies"])
        if other:
            unexpected[name] = sorted(other)
    report = {
        "repository": lock["repository"],
        "revision": lock["revision"],
        "scope": "pinned_drop_hang_cpu_asset_audit",
        "inventory": inventory,
        "usd": usd,
        "trajectories": trajectories,
        "stages_with_unresolved_or_external_dependencies": unresolved,
        "authored_usd_dependencies_closed": not unresolved,
        "expected_template_dome_references": expected_domes,
        "unexpected_usd_dependencies": unexpected,
        "isaac_launched": False,
        "solver_verified": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return int(bool(unexpected))


if __name__ == "__main__":
    raise SystemExit(main())
