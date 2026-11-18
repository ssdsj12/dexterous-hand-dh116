#!/usr/bin/env python3
"""Read real PhysX convex geometry in one bounded DH116 diagnostic scene."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS, read_pad_triangles
from scripts.audit_dh116_cooked_hull_device_parity import physics_device_matches
from scripts.audit_dh116_cooked_convex_hulls import (
    compare_surfaces, geometric_hull_estimate, read_native_pad_hulls,
)
from scripts.verify_dh116_deformx_rigid_bridge import write_json


_REFERENCE = np.asarray(((0, 0, 0), (.09, 0, 0), (.02, .043, 0), (.01, .01, .032)))
_FACES = (0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3)
_DT = 1 / 480


def classify_reference_frame(points, candidates, *, tolerance_m=0.00005):
    """Confirm exactly one candidate by symmetric vertex-set distance."""
    from scipy.spatial.distance import cdist

    points = np.asarray(points, dtype=float)
    distances = {}
    for name, target in candidates.items():
        target = np.asarray(target, dtype=float)
        if points.shape != target.shape or not np.isfinite(points).all():
            distances[name] = None
            continue
        d = cdist(points, target)
        distances[name] = float(max(d.min(0).max(), d.min(1).max()))
    matching = [name for name, value in distances.items() if value is not None and value <= tolerance_m]
    return {"confirmed_frame": matching[0] if len(matching) == 1 else None,
            "symmetric_vertex_distance_m": distances, "tolerance_m": tolerance_m}


def _sources(project):
    manifest_path = project / "assets/fixed_dual_panda_dh116_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assets = {}
    for side in ("left", "right"):
        row = next(x for x in manifest["sources"] if x["kind"] == "dh116" and x["side"] == side)
        path = manifest_path.parent / "dependencies" / Path(row["path"]).name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"original {side} DH116 USD SHA differs")
        assets[side] = {"path": str(path.resolve()), "sha256": digest}
    return assets


def _create_scene(stage, sources):
    from pxr import Gf, UsdGeom, UsdPhysics

    root = "/World/F4CookedAudit"
    UsdGeom.Xform.Define(stage, root)
    body_path = f"{root}/ReferenceLink"
    mesh_path = f"{body_path}/ReferenceMesh"
    body = UsdGeom.Xform.Define(stage, body_path)
    body.AddTranslateOp().Set(Gf.Vec3d(.41, -.27, .88))
    body.AddRotateXYZOp().Set(Gf.Vec3f(17., -31., 42.))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.2)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.AddTranslateOp().Set(Gf.Vec3d(.14, -.10, .06))
    mesh.AddRotateXYZOp().Set(Gf.Vec3f(23., 32., -19.))
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in _REFERENCE])
    mesh.CreateFaceVertexCountsAttr([3] * 4)
    mesh.CreateFaceVertexIndicesAttr(_FACES)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

    hands = {}
    for side, y in (("left", 1.4), ("right", -1.4)):
        anchor = f"{root}/{side}Placement"
        UsdGeom.Xform.Define(stage, anchor).AddTranslateOp().Set(Gf.Vec3d(0.0, y, .65))
        path = f"{anchor}/Hand"
        stage.OverridePrim(path).GetReferences().AddReference(sources[side]["path"])
        hands[side] = path
    return body_path, mesh_path, hands


def _rigid_pose(physx, path):
    from scipy.spatial.transform import Rotation

    data = physx.get_rigidbody_transformation(path)
    if not data["ret_val"]:
        return {"path": path, "sdk_rigid_body_found": False}
    position = list(data["position"])
    quaternion = list(data["rotation"])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix().T
    transform[3, :3] = position
    return {"path": path, "sdk_rigid_body_found": True, "position_m": position,
            "quaternion_xyzw": quaternion, "link_to_world_row_major": transform.tolist()}


def _reference(stage, physx, body, mesh, *, simulation_running):
    from pxr import UsdGeom

    pose = _rigid_pose(physx, body)
    readback = read_native_pad_hulls(stage, mesh, mesh, simulation_running=simulation_running)
    result = {"pose": pose, "queries": readback, "calibration": {}}
    if not pose["sdk_rigid_body_found"]:
        return result
    cache = UsdGeom.XformCache()
    mesh_to_world = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(mesh)))
    usd_body_to_world = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body)))
    mesh_to_link = mesh_to_world @ np.linalg.inv(usd_body_to_world)
    reference = np.column_stack((_REFERENCE, np.ones(len(_REFERENCE))))
    link_points = reference @ mesh_to_link
    targets = {
        "mesh_local": _REFERENCE,
        "link_local": link_points[:, :3],
        "world": (link_points @ np.asarray(pose["link_to_world_row_major"]))[:, :3],
    }
    result["reference_vertices_m"] = {name: points.tolist() for name, points in targets.items()}
    result["mesh_to_link_row_major"] = mesh_to_link.tolist()
    for name, query in readback[mesh].items():
        hulls = query["hulls"]
        if hulls is not None and len(hulls) == 1:
            result["calibration"][name] = classify_reference_frame(hulls[0]["vertices"], targets)
    return result


def _record_pads(stage, physx, hands, output, clock):
    from pxr import UsdGeom

    rows = []
    cache = UsdGeom.XformCache()
    for side, hand in hands.items():
        for link in DISTAL_PAD_LINKS:
            path = f"{hand}/{link}"
            row = {"side": side, "link": link, "clock": clock}
            try:
                _, _, source = read_pad_triangles(stage, path)
                row["source"] = source
                row["sdk_body"] = _rigid_pose(physx, path)
                row["usd_transforms_row_major"] = {
                    key: np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(source[key]))).tolist()
                    for key in ("body_path", "collision_owner", "mesh_path")
                }
                row["queries"] = read_native_pad_hulls(
                    stage, source["collision_owner"], source["mesh_path"], simulation_running=True,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            write_json(output / f"{side}_{link}.json", row)
            rows.append(row)
    return rows


def _physics_device(physx, requested):
    from isaacsim.core.simulation_manager import SimulationManager

    view = SimulationManager.get_physics_sim_view()
    if view is None:
        raise RuntimeError("live PhysX tensor view is absent")
    return {
        "requested_device": requested,
        "manager_device": SimulationManager.get_physics_sim_device(),
        "tensor_device_ordinal": int(view.device_ordinal),
        "gpu_dynamics_enabled": bool(SimulationManager.is_gpu_dynamics_enabled()),
        "broadphase": str(SimulationManager.get_broadphase_type()),
        "physx_device_override": int(physx.get_overwrite_gpu_setting()),
    }


def _run(sim, output, sources, steps, requested_device):
    import omni.usd
    from omni.physx import get_physx_interface
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("expected the metre Z-up probe stage")
    body, mesh, hands = _create_scene(stage, sources)
    physx = get_physx_interface()
    sim.reset()
    index_before = int(sim.current_time_step_index)
    for _ in range(steps):
        sim.step(render=False)
    index_after = int(sim.current_time_step_index)
    if index_after - index_before != steps or index_after > 10:
        raise RuntimeError(f"native step budget mismatch: {index_before} -> {index_after}")
    physics_device = _physics_device(physx, requested_device)
    write_json(output / "physics_device.json", physics_device)
    if not physics_device_matches(physics_device):
        raise RuntimeError(f"PhysX device evidence differs from the requested backend: {physics_device}")
    clock = {"physics_step": index_after, "dt_s": _DT, "steps_commanded": steps}
    reference = _reference(stage, physx, body, mesh, simulation_running=True)
    write_json(output / "reference.json", {"clock": clock, **reference})
    rows = _record_pads(stage, physx, hands, output, clock)
    write_json(output / "result.json", {
        "scope": "bounded_physx_cooked_geometry_readback", "clock": clock, "source_usds": sources,
        "physics_device": physics_device,
        "reference_frame_calibration": reference["calibration"], "pad_rows": len(rows),
        "sdk_pad_bodies_found": sum(row.get("sdk_body", {}).get("sdk_rigid_body_found", False) for row in rows),
        "pads_with_live_hulls": sum(any(query["live_simulation_cooked"]["hulls"]
                                       for query in row.get("queries", {}).values()) for row in rows),
        "native_pad_frames_qualified": False, "sdf_baked": False,
    })


def compare_native_capture(native_dir, sources, *, samples=192):
    """Compare live hulls only when the archived local bases agree."""
    from pxr import Usd

    result = json.loads((native_dir / "result.json").read_text())
    frame = result["reference_frame_calibration"].get("live_simulation_cooked", {})
    if frame.get("confirmed_frame") != "mesh_local":
        raise ValueError("live readback reference did not confirm mesh-local coordinates")
    rows = {}
    for side, source in sources.items():
        if source["sha256"] != result["source_usds"][side]["sha256"]:
            raise ValueError(f"source SHA changed for {side} hand")
        stage = Usd.Stage.Open(source["path"])
        for link in DISTAL_PAD_LINKS:
            row = json.loads((native_dir / f"{side}_{link}.json").read_text())
            if (row["side"], row["link"], row["clock"]) != (side, link, result["clock"]):
                raise ValueError(f"native identity mismatch for {side} {link}")
            vertices, faces, offline = read_pad_triangles(
                stage, str(stage.GetDefaultPrim().GetPath().AppendChild(link)),
            )
            native = row["source"]
            if offline["geometry_sha256"] != native["geometry_sha256"]:
                raise ValueError(f"native source geometry mismatch for {side} {link}")
            transforms = {name: np.asarray(value)
                          for name, value in row["usd_transforms_row_major"].items()}
            usd_body = transforms["body_path"]
            link_basis = {name: transforms[name] @ np.linalg.inv(usd_body)
                          for name in ("mesh_path", "collision_owner")}
            error = max(float(np.max(np.abs(matrix - np.eye(4)))) for matrix in link_basis.values())
            pose = row["sdk_body"]
            live = row["queries"][native["collision_owner"]]["live_simulation_cooked"]["hulls"]
            if error > 5e-7 or not pose["sdk_rigid_body_found"] or live is None or len(live) != 1:
                raise ValueError(f"expected coincident local bases and one live hull for {side} {link}")
            cooked = live[0]
            points = np.asarray(cooked["vertices"])
            faces_live = np.asarray(cooked["faces"])
            estimate_v, estimate_f, _ = geometric_hull_estimate(vertices)
            matrix = np.asarray(pose["link_to_world_row_major"])
            world = np.column_stack((points, np.ones(len(points)))) @ matrix
            recovered = world @ np.linalg.inv(matrix)
            rows[f"{side}.{link}"] = {
                "live_query_path": native["collision_owner"],
                "live_vertices": len(points), "live_polygons": len(cooked["polygons"]),
                "live_triangulated_faces": len(faces_live),
                "live_bounds_in_coincident_local_frame_m": [points.min(0).tolist(), points.max(0).tolist()],
                "coincident_local_basis_max_matrix_delta": error,
                "sdk_vs_usd_default_body_max_matrix_delta": float(np.max(np.abs(matrix - usd_body))),
                "sdk_pose_link_world_roundtrip_max_m": float(np.linalg.norm(recovered[:, :3] - points, axis=1).max()),
                "source_vs_live": compare_surfaces(vertices, faces, points, faces_live, samples=samples),
                "qhull_estimate_vs_live": compare_surfaces(estimate_v, estimate_f, points, faces_live, samples=samples),
            }
    return {
        "scope": "live_cooked_comparison_for_original_coincident_pad_local_bases",
        "source_run_clock": result["clock"], "reference_calibration": frame,
        "pads": rows, "sdf_baked": False,
    }



def _release_owned_native_view(app, simulation, output):
    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager

    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app

    cls = type(simulation)
    if cls.instance() is not simulation:
        raise RuntimeError("expected the probe-owned SimulationContext")
    cls.clear_instance()
    del simulation
    gc.collect()
    timeline = omni.timeline.get_timeline_interface()
    try:
        timeline.stop()
        app.update()
        if SimulationManager.get_physics_sim_view() is not None:
            raise RuntimeError("owned SDK simulation view survived STOP")
        write_json(output / "simulation_context_cleared.json", {
            "public_api": "SimulationContext.clear_instance then timeline.stop",
            "simulation_manager_tensor_view_released": True,
        })
    finally:
        del timeline
        close_calibration_app(app, output)



def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-native", action="store_true")
    mode.add_argument("--compare-run", type=Path,
                      help="compare an archived native output directory without starting Isaac")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        parser.error("physics steps must be between 1 and 10")
    sources = _sources(Path(__file__).resolve().parents[1])
    if args.compare_run is not None:
        write_json(args.output, compare_native_capture(args.compare_run, sources))
        return
    if not args.execute_native:
        print(json.dumps({"source_usds": sources, "steps": args.steps, "native_started": False}))
        return
    if not args.headless or args.enable_cameras:
        parser.error("the probe requires headless with cameras disabled")
    args.output.mkdir(parents=True, exist_ok=False)
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock

    with native_instance_lock():
        app = AppLauncher(args, fast_shutdown=False).app
        sim = None
        try:
            import isaaclab.sim as sim_utils

            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
                dt=_DT, device=args.device, gravity=(0.0, 0.0, 0.0),
            ))
            _run(sim, args.output, sources, args.steps, args.device)
        except Exception:
            write_json(args.output / "failure.json", {"exception": traceback.format_exc()})
            raise
        finally:
            if sim is None:
                from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app

                close_calibration_app(app, args.output)
            else:
                _release_owned_native_view(app, sim, args.output)


if __name__ == "__main__":
    main()
