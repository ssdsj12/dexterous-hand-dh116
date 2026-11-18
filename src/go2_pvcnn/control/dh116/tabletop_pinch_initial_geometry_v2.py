"""Read-only initial Pinch geometry at copied actual SDK link poses.

Source 17-collider envelopes and five locked F4 hulls provide a geometric
screen only. Existing cooked-data getters remain stored representations, with
no live PxShape attachment or contact-free claim. Never request cooking, step,
author scene data, import Isaac/PhysX interfaces or retain native handles.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import traceback

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    load_geometry, shape_metrics, pose_matrix, transform,
)
from .tabletop_tripod_cooked_geometry import _plain, _validate_hull, _cook_metadata
from .deformx_dh116_colliders import DISTAL_PAD_LINKS

VERSION = "dh116-pinch-initial-actual-geometry-v2"


def _implementation():
    root=Path(__file__).resolve().parents[4]
    files=("src/go2_pvcnn/control/dh116/tabletop_pinch_initial_geometry_v2.py",
        "src/go2_pvcnn/control/dh116/tabletop_tripod_cooked_geometry.py",
        "scripts/audit_dh116_pinch_initial_pose.py","src/go2_pvcnn/control/dh116/usd_surface.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}



def _owned(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=float, copy=True)


def _clock(reader):
    row = json.loads(json.dumps(reader(), allow_nan=False))
    step, time = row["sdk_step"], row["sdk_time_s"]
    if (type(step) is not int or step < 0 or type(time) not in (float, int)
            or not math.isfinite(time)
            or not math.isclose(time, step/480, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("actual same-clock 480Hz SDK step/time required")
    return row


def _checked_geometry(geometry):
    # Reuse the exact source/locked F4 loader, not caller-labelled cooked data.
    actual = load_geometry(geometry["project"], side=geometry["side"])
    if (geometry["source_sha256"] != actual["source_sha256"]
            or geometry["usd"]["sha256"] != actual["usd"]["sha256"]
            or set(geometry["source_colliders"]) != set(actual["source_colliders"])
            or set(geometry["cooked_pads"]) != set(DISTAL_PAD_LINKS)):
        raise ValueError("original source/locked F4 geometry identity differs")
    for group in ("source_colliders", "cooked_pads"):
        for name, row in actual[group].items():
            given = geometry[group][name]
            if not np.array_equal(given["vertices"], row["vertices"]):
                raise ValueError("caller geometry vertices differ from locked source")
            if group == "source_colliders":
                if (given["source"]["geometry_sha256"] != row["source"]["geometry_sha256"]
                        or given["source"]["mesh_to_body"] != row["source"]["mesh_to_body"]):
                    raise ValueError("caller source geometry/offset differs")
            elif (given["capture_sha256"] != row["capture_sha256"]
                  or given["coincident_mesh_body_basis_max_delta"] != row["coincident_mesh_body_basis_max_delta"]):
                raise ValueError("caller locked cooked frame/capture differs")
    return actual


def _stored(cooking, source, actual, matrix, layout, radius):
    result = dict(available=False, reasons=[], query_attempts=[], hulls=[],
        selected_query_path=None, coordinate_frame="mesh_local_hypothesis_not_live_shape_attestation",
        live_shape_attachment_proven=False, included_in_initial_geometry_gate=False)
    if cooking is None:
        result["reasons"].append("cooking_getter_not_available")
        return result
    selected = None
    for role, path in (("actual_collision_owner", actual["collision_owner"]),
                       ("actual_mesh", actual["mesh_path"])):
        attempt = dict(role=role, path=path, count=None, error=None)
        result["query_attempts"].append(attempt)
        try:
            count = cooking.get_nb_convex_mesh_data(path)
            attempt["count"] = _plain(count)
        except Exception as error:
            attempt["error"] = type(error).__name__+": "+str(error)
            continue
        if type(count) is int and count > 0:
            if count != 1 or source["collision_approximation"] != "convexHull":
                result["reasons"].append("stored_count_or_original_approximation_differs")
                return result
            selected = path
            break
    if selected is None:
        result["reasons"].append("both_exact_getter_paths_unavailable_or_empty")
        return result
    result["selected_query_path"] = selected
    try:
        raw = cooking.get_convex_mesh_data(selected, 0)
        validation = _validate_hull(raw)
        plain = _plain(raw)
        hull = dict(raw=plain, validation=validation,
            payload_sha256=hashlib.sha256(json.dumps(plain, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        result["hulls"].append(hull)
        if not validation["closed_convex"]:
            result["reasons"].append("stored_hull_topology_invalid")
            return result
        world = transform(np.asarray(raw["vertices"], dtype=float),
                          matrix @ np.asarray(actual["mesh_to_body"]))
        hull["world_vertices_under_mesh_local_hypothesis"] = world.tolist()
        hull["geometric_metrics_under_frame_hypothesis"] = shape_metrics(
            {"hull": world}, layout, radius)["hull"]
        result["available"] = True
    except Exception as error:
        result["reasons"].append(type(error).__name__+": "+str(error))
    return result


def _capture(stage, hand, geometry, *, cooking_interface, read_clock,
             table_center_m, table_size_m, rod_nodes_w, rod_radius_m,
             minimum_table_gap_m, minimum_rod_gap_m):
    from pxr import Usd, UsdGeom, UsdPhysics
    from .usd_surface import read_surface

    before = _clock(read_clock)
    source = _checked_geometry(geometry)
    side = source["side"]
    view = hand.root_physx_view
    names = list(hand.body_names)
    paths = deepcopy(list(view.link_paths))
    articulation_paths = list(view.prim_paths)
    if (len(set(names)) != len(names) or len(paths) != 1 or len(paths[0]) != len(names)
            or len(articulation_paths) != 1 or not set(source["source_colliders"]).issubset(names)):
        raise ValueError("complete unique actual hand/body mapping required")
    parents = {str(PurePosixPath(p).parent) for p in paths[0]}
    if len(parents) != 1:
        raise ValueError("actual links span different hand containers")
    container = parents.pop()
    root = stage.GetPrimAtPath(container)
    path = articulation_paths[0]
    if (PurePosixPath(container).name != side+"Hand" or not root
            or paths[0] != [container+"/"+name for name in names]
            or not (path == container or
                    (path.startswith(container+"/") and
                     stage.GetPrimAtPath(path).HasAPI(UsdPhysics.ArticulationRootAPI)))):
        raise ValueError("actual articulation/link paths differ from original hand")
    poses = _owned(view.get_link_transforms())
    if (poses.shape != (1, len(names), 7) or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[...,3:],axis=-1),1.,atol=1e-5,rtol=0)):
        raise ValueError("actual finite normalized XYZ/XYZW link poses required")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("metre Z-up actual stage required")
    for value in (rod_radius_m, minimum_table_gap_m, minimum_rod_gap_m):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("explicit finite nonnegative physical thresholds required")
    if rod_radius_m == 0:
        raise ValueError("positive rod radius required")
    layout = dict(table_center=_owned(table_center_m), table_size=_owned(table_size_m),
                  wire_points=_owned(rod_nodes_w))
    layers = {layer.identifier:layer.ExportToString() for layer in stage.GetLayerStack()}
    expected_owners = set()
    source_rows, locked_rows, stored_rows = {}, {}, {}
    matrices = {}
    actual_surfaces = {}
    for name, item in source["source_colliders"].items():
        body = names.index(name)
        actual = read_surface(stage, paths[0][body])
        original = item["source"]
        original_root = str(PurePosixPath(original["body_path"]).parent)
        for key in ("body_path", "collision_owner", "mesh_path"):
            if actual[key] != container+original[key][len(original_root):]:
                raise ValueError("source/actual collider path correspondence differs")
        if (actual["geometry_sha256"] != original["geometry_sha256"]
                or actual["collision_approximation"] != original["collision_approximation"]
                or not np.allclose(actual["mesh_to_body"], original["mesh_to_body"], rtol=0, atol=1e-8)):
            raise ValueError("source/actual mesh geometry or local offset differs")
        expected_owners.add(actual["collision_owner"])
        pose = poses[0,body][[0,1,2,6,3,4,5]]
        matrix = pose_matrix(pose)
        matrices[name] = matrix
        actual_surfaces[name] = actual
        world = transform(item["vertices"], matrix)
        metrics = shape_metrics({name:world}, layout, rod_radius_m)[name]
        source_rows[name] = dict(body_index=body, actual_body_path=paths[0][body],
            actual_collision_owner=actual["collision_owner"], actual_mesh_path=actual["mesh_path"],
            world_from_body_column=matrix.tolist(), mesh_to_body_column=actual["mesh_to_body"],
            geometry_sha256=actual["geometry_sha256"], world_vertices=world.tolist(),
            kind="source_USD_convex_envelope_not_live_cooked", **metrics)
        if name in source["cooked_pads"]:
            cooked = source["cooked_pads"][name]
            world = transform(cooked["vertices"], matrix)
            locked_rows[name] = dict(body_index=body, actual_body_path=paths[0][body],
                world_vertices=world.tolist(), capture_sha256=cooked["capture_sha256"],
                source_geometry_sha256=cooked["geometry_sha256"],
                coincident_mesh_body_basis_max_delta=cooked["coincident_mesh_body_basis_max_delta"],
                kind="locked_F4_historical_cooked_hull_at_current_actual_body_pose",
                **shape_metrics({name:world}, layout, rod_radius_m)[name])
    owners = {str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
              if p.HasAPI(UsdPhysics.CollisionAPI) and
              UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()}
    if len(expected_owners) != 17 or owners != expected_owners:
        raise ValueError("actual enabled collider set differs from all seventeen original meshes")
    for name, item in source["source_colliders"].items():
        actual = actual_surfaces[name]
        stored_rows[name] = _stored(cooking_interface, item["source"], actual, matrices[name],
                                    layout, rod_radius_m)
        stored_rows[name]["cooking_metadata"] = [_cook_metadata(stage.GetPrimAtPath(actual[k]))
                                                for k in ("collision_owner", "mesh_path")]
    if (_clock(read_clock) != before or names != list(hand.body_names)
            or paths != list(view.link_paths) or articulation_paths != list(view.prim_paths)):
        raise ValueError("actual clock or hand instance mapping changed during read-only capture")
    if layers != {layer.identifier:layer.ExportToString() for layer in stage.GetLayerStack()}:
        raise ValueError("actual stage changed during read-only capture")
    if hashlib.sha256(Path(source["usd"]["path"]).read_bytes()).hexdigest() != source["usd"]["sha256"]:
        raise ValueError("source USD changed during capture")

    def passes(rows):
        return all(not r["table"]["intersects"]
            and r["table"]["top_clearance_m"] >= minimum_table_gap_m
            and r["rod"]["capsule_surface_gap_m"] >= minimum_rod_gap_m for r in rows.values())
    source_pass, locked_pass = passes(source_rows), passes(locked_rows)
    reasons = []
    if not source_pass: reasons.append("source17_geometry_gap_gate_failed")
    if not locked_pass: reasons.append("locked_pad5_geometry_gap_gate_failed")
    return _plain(dict(version=VERSION, implementation_sha256=_implementation(), side=side, identity=before,
        source_sha256=source["source_sha256"], actual_articulation_paths=articulation_paths,
        actual_body_names=names, actual_link_paths=paths, actual_link_poses_xyzw=poses,
        source_colliders=source_rows, locked_pads=locked_rows, stored_cooking=stored_rows,
        thresholds=dict(minimum_table_gap_m=minimum_table_gap_m, minimum_rod_gap_m=minimum_rod_gap_m),
        table_center_m=layout["table_center"], table_size_m=layout["table_size"],
        rod_nodes_w=layout["wire_points"], rod_radius_m=rod_radius_m,
        rod_geometry_source="caller_supplied_initial_geometry_not_solver_readback",
        source_geometry_passed=source_pass, locked_pad_geometry_passed=locked_pass,
        initial_geometry_passed=source_pass and locked_pass, reasons=reasons,
        stored_cooking_complete=all(r["available"] for r in stored_rows.values()),
        stored_cooking_scope="stored_representation_getter_not_live_PxShape_attachment",
        native_shape_attachment_proven=False, contact_free_proven=False,
        native_qualification=False, teacher_admission=False, physics_modified=False,
        sdk_steps_executed=0, cooking_requested=False))


def capture_pinch_initial_geometry(stage, hand, geometry, *, cooking_interface, read_clock,
                                   table_center_m, table_size_m, rod_nodes_w, rod_radius_m,
                                   minimum_table_gap_m, minimum_rod_gap_m):
    """Return owned evidence only; release local aliases even on a getter exception.

    read_clock must return actual sdk_step and sdk_time_s. It is checked twice,
    but its authenticity is the caller's responsibility. Initial geometry is
    independent of the subsequent solver construction and contact measurements.
    """
    result, failure = None, None
    try:
        result = _capture(stage, hand, geometry, cooking_interface=cooking_interface,
            read_clock=read_clock, table_center_m=table_center_m, table_size_m=table_size_m,
            rod_nodes_w=rod_nodes_w, rod_radius_m=rod_radius_m,
            minimum_table_gap_m=minimum_table_gap_m, minimum_rod_gap_m=minimum_rod_gap_m)
    except BaseException as error:
        failure = traceback.format_exc()
        # Completed SDK/helper frames may retain native handles. The current
        # executing wrapper is cleared explicitly below; no old exception escapes.
        traceback.clear_frames(error.__traceback__)
    finally:
        stage = hand = geometry = cooking_interface = read_clock = None
    if failure:
        raise RuntimeError(failure) from None
    return result


def _match(observed, expected, label):
    """Strict structure with bounded float round-trip tolerance, never truthy flags."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ValueError(label+" dictionary fields differ")
        for key in expected:
            _match(observed[key], expected[key], label+"."+key)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(observed, (list, tuple)) or len(observed) != len(expected):
            raise ValueError(label+" sequence differs")
        for index, item in enumerate(expected):
            _match(observed[index], item, label+f"[{index}]")
    elif isinstance(expected, (bool, np.bool_)):
        if type(observed) is not bool or observed != bool(expected):
            raise ValueError(label+" boolean differs")
    elif isinstance(expected, (int, float, np.integer, np.floating)):
        if (type(observed) not in (int, float) or not math.isfinite(observed)
                or not math.isclose(observed, float(expected), rel_tol=0, abs_tol=1e-10)):
            raise ValueError(label+" finite numerical value differs")
    elif observed != expected:
        raise ValueError(label+" value differs")


def validate_pinch_initial_geometry(record, geometry, *, table_center_m, table_size_m,
                                   rod_nodes_w, rod_radius_m,
                                   minimum_table_gap_m, minimum_rod_gap_m):
    """Recompute a disk-loaded record from its copied poses and locked geometry.

    Internal consistency is not independent attestation of actual SDK origin.
    Consumers must also bind this record/clock to their owned native ledger.
    Existing stored geometry stays optional and cannot become shape attachment.
    """
    result = dict(valid=False, initial_geometry_passed=False, reasons=[],
                  native_qualification=False, scope="disk_recomputation_not_native_attestation")
    try:
        source = _checked_geometry(geometry)
        _match(record["version"], VERSION, "version")
        _match(record["implementation_sha256"], _implementation(), "implementation_sha256")
        _match(record["source_sha256"], source["source_sha256"], "source_sha256")
        _match(record["side"], source["side"], "side")
        _clock(lambda: record["identity"])
        for field in ("native_shape_attachment_proven","contact_free_proven",
                      "native_qualification","teacher_admission","physics_modified","cooking_requested"):
            _match(record[field], False, field)
        _match(record["sdk_steps_executed"], 0, "sdk_steps_executed")
        _match(record["rod_geometry_source"],
               "caller_supplied_initial_geometry_not_solver_readback", "rod_geometry_source")
        for key, value in (("table_center_m",table_center_m),("table_size_m",table_size_m),
                           ("rod_nodes_w",rod_nodes_w),("rod_radius_m",rod_radius_m)):
            _match(record[key], _plain(value), key)
        thresholds=dict(minimum_table_gap_m=minimum_table_gap_m,minimum_rod_gap_m=minimum_rod_gap_m)
        for value in (*thresholds.values(),rod_radius_m):
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError("explicit valid physical thresholds required")
        _match(record["thresholds"],thresholds,"thresholds")
        names, paths = record["actual_body_names"], record["actual_link_paths"]
        if (len(set(names))!=len(names) or len(paths)!=1 or len(paths[0])!=len(names)
                or not set(source["source_colliders"]).issubset(names)):
            raise ValueError("complete unique recorded source body mapping required")
        containers={str(PurePosixPath(p).parent) for p in paths[0]}
        if len(containers)!=1:
            raise ValueError("recorded body paths span containers")
        container=containers.pop()
        if (PurePosixPath(container).name!=source["side"]+"Hand"
                or paths[0]!=[container+"/"+n for n in names]
                or record["actual_articulation_paths"] not in (
                    [container],[container+"/root_joint"],[container+"/wrist_fixed"])):
            raise ValueError("recorded original articulation/body paths differ")
        poses=np.asarray(record["actual_link_poses_xyzw"],dtype=float)
        if poses.shape!=(1,len(names),7) or not np.isfinite(poses).all():
            raise ValueError("complete finite recorded native poses required")
        layout=dict(table_center=table_center_m,table_size=table_size_m,wire_points=rod_nodes_w)
        expected_source,expected_locked={},{}
        for name,item in source["source_colliders"].items():
            body=names.index(name)
            matrix=pose_matrix(poses[0,body][[0,1,2,6,3,4,5]])
            original=item["source"]
            source_root=str(PurePosixPath(original["body_path"]).parent)
            mapped={key:container+original[key][len(source_root):]
                    for key in ("body_path","collision_owner","mesh_path")}
            world=transform(item["vertices"],matrix)
            expected_source[name]=dict(body_index=body,actual_body_path=paths[0][body],
                actual_collision_owner=mapped["collision_owner"],actual_mesh_path=mapped["mesh_path"],
                world_from_body_column=matrix.tolist(),mesh_to_body_column=original["mesh_to_body"],
                geometry_sha256=original["geometry_sha256"],world_vertices=world.tolist(),
                kind="source_USD_convex_envelope_not_live_cooked",
                **shape_metrics({name:world},layout,rod_radius_m)[name])
            if name in source["cooked_pads"]:
                cooked=source["cooked_pads"][name]
                world=transform(cooked["vertices"],matrix)
                expected_locked[name]=dict(body_index=body,actual_body_path=paths[0][body],
                    world_vertices=world.tolist(),capture_sha256=cooked["capture_sha256"],
                    source_geometry_sha256=cooked["geometry_sha256"],
                    coincident_mesh_body_basis_max_delta=cooked["coincident_mesh_body_basis_max_delta"],
                    kind="locked_F4_historical_cooked_hull_at_current_actual_body_pose",
                    **shape_metrics({name:world},layout,rod_radius_m)[name])
            stored=record["stored_cooking"][name]
            _match(stored["coordinate_frame"],
                   "mesh_local_hypothesis_not_live_shape_attestation","stored.frame")
            _match(stored["live_shape_attachment_proven"],False,"stored.attachment")
            _match(stored["included_in_initial_geometry_gate"],False,"stored.gate")
            _match([m["prim_path"] for m in stored["cooking_metadata"]],
                   [mapped["collision_owner"],mapped["mesh_path"]],"stored.metadata_paths")
            attempts=stored["query_attempts"]
            if len(attempts)>2:
                raise ValueError("extra cooking query attempts")
            for attempt,(role,key) in zip(attempts,(("actual_collision_owner","collision_owner"),
                                                  ("actual_mesh","mesh_path"))):
                _match(attempt["role"],role,"query.role")
                _match(attempt["path"],mapped[key],"query.path")
            for hull in stored["hulls"]:
                raw=hull["raw"]
                sha=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(",",":"),
                                             allow_nan=False).encode()).hexdigest()
                _match(hull["payload_sha256"],sha,"stored.payload_sha256")
                _match(hull["validation"],_plain(_validate_hull(raw)),"stored.topology")
            if stored["available"] is True:
                if (stored["reasons"] or len(stored["hulls"])!=1
                        or not stored["hulls"][0]["validation"]["closed_convex"]):
                    raise ValueError("available stored representation lacks valid closed hull")
                positive=next((a for a in attempts if type(a["count"]) is int and a["count"]>0),None)
                if positive is None or positive["count"]!=1 or stored["selected_query_path"]!=positive["path"]:
                    raise ValueError("selected stored getter query/count differs")
                hull=stored["hulls"][0]
                world=transform(np.asarray(hull["raw"]["vertices"]),matrix@np.asarray(original["mesh_to_body"]))
                _match(hull["world_vertices_under_mesh_local_hypothesis"],world.tolist(),"stored.world_vertices")
                _match(hull["geometric_metrics_under_frame_hypothesis"],
                       shape_metrics({"hull":world},layout,rod_radius_m)["hull"],"stored.metrics")
            elif stored["available"] is not False or not stored["reasons"]:
                raise ValueError("stored unavailable status/reasons differ")
        _match(record["source_colliders"],expected_source,"source17")
        _match(record["locked_pads"],expected_locked,"locked5")
        if set(record["stored_cooking"])!=set(expected_source):
            raise ValueError("stored query set differs from all17")
        def passes(rows):
            return all(not r["table"]["intersects"]
                and r["table"]["top_clearance_m"]>=minimum_table_gap_m
                and r["rod"]["capsule_surface_gap_m"]>=minimum_rod_gap_m for r in rows.values())
        source_pass,locked_pass=passes(expected_source),passes(expected_locked)
        reasons=[]
        if not source_pass:reasons.append("source17_geometry_gap_gate_failed")
        if not locked_pass:reasons.append("locked_pad5_geometry_gap_gate_failed")
        for field,value in (("source_geometry_passed",source_pass),("locked_pad_geometry_passed",locked_pass),
                            ("initial_geometry_passed",source_pass and locked_pass),
                            ("stored_cooking_complete",all(r["available"] for r in record["stored_cooking"].values())),
                            ("reasons",reasons)):
            _match(record[field],value,field)
        result.update(valid=True,initial_geometry_passed=source_pass and locked_pass)
    except (ValueError,KeyError,TypeError,IndexError,AttributeError,RuntimeError) as error:
        result["reasons"].append(type(error).__name__+": "+str(error))
    return result
