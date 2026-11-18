#!/usr/bin/env python3
"""Offline audit of immutable saved hull packets; never launches or repairs PhysX."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_tripod_cooked_geometry import _validate_hull



def diagnose_surface(raw, tolerance):
    """Independent diagnostic checks; best-fit planes never enter validation."""
    result=dict(available=False,native_qualification=False,self_intersections_tested=False,
        geometric_convexity_qualified=False,diagnostic_plane_fits_applied_to_input=False)
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            if tolerance is None or not np.isfinite(tolerance) or tolerance<=0:
                raise ValueError('original numeric tolerance unavailable')
            vertices=np.asarray(raw['vertices'],dtype=float)
            indices=raw['indices'];polygons=raw['polygons']
            if (vertices.shape!=(raw['num_vertices'],3) or not np.isfinite(vertices).all()
                    or len(polygons)!=raw['num_polygons'] or any(type(i) is not int for i in indices)):
                raise ValueError('finite source shape and counts required')
            directed=Counter();used=set();slots=[];rings={};volume=0.
            center=vertices.mean(axis=0);winding=True;nonplanar=[];max_fit=0.
            for index,polygon in enumerate(polygons):
                start=polygon['index_base'];count=polygon['num_vertices']
                if type(start) is not int or type(count) is not int or start<0 or count<3:
                    raise ValueError('integer nondegenerate source polygon range required')
                ids=indices[start:start+count]
                if (len(ids)!=count or len(set(ids))!=count or min(ids)<0
                        or max(ids)>=len(vertices)):
                    raise ValueError('valid source polygon indices required')
                used.update(ids);slots.extend(range(start,start+count))
                points=vertices[ids];area=np.zeros(3)
                plane=np.asarray(polygon['plane'],dtype=float)
                if plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-12:
                    raise ValueError('finite nonzero source plane required')
                for j,a in enumerate(ids):
                    b=ids[(j+1)%count];directed[(a,b)]+=1
                    area+=np.cross(vertices[a]-points[0],vertices[b]-points[0])
                    previous=ids[j-1]
                    rings.setdefault(a,[]).append((previous,b))
                winding=winding and bool(area@(plane[:3]/np.linalg.norm(plane[:3]))>tolerance*tolerance)
                for j in range(1,count-1):
                    a,b,c=points[0]-center,points[j]-center,points[j+1]-center
                    volume+=float(np.dot(a,np.cross(b,c))/6.)
                centered=points-points.mean(axis=0)
                _,_,axes=np.linalg.svd(centered,full_matrices=False)
                fit=float(np.max(abs(centered@axes[-1])))
                max_fit=max(max_fit,fit)
                if fit>tolerance:
                    nonplanar.append(dict(polygon_index=index,num_vertices=count,
                        diagnostic_best_fit_residual_native_units=fit))
            edges={tuple(sorted(edge)) for edge in directed}
            neighbors={v:set() for v in used}
            for a,b in edges:
                neighbors[a].add(b);neighbors[b].add(a)
            def components(graph):
                todo=set(graph);count=0
                while todo:
                    count+=1;stack=[todo.pop()]
                    while stack:
                        current=stack.pop()
                        for neighbor in graph[current]&todo:
                            todo.remove(neighbor);stack.append(neighbor)
                return count
            links_ok=True
            for vertex in used:
                links=rings[vertex];graph={};degree=Counter()
                for a,b in links:
                    graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
                    degree[a]+=1;degree[b]+=1
                if not graph or any(n!=2 for n in degree.values()) or components(graph)!=1:
                    links_ok=False
            paired=all(n==1 and directed[(b,a)]==1 for (a,b),n in directed.items())
            index_cover=sorted(slots)==list(range(len(indices)))
            vertices_used=used==set(range(len(vertices)))
            connected=components(neighbors)
            euler=len(vertices)-len(edges)+len(polygons)
            checks=index_cover and vertices_used and paired and links_ok and connected==1 and euler==2
            result.update(available=True,index_ranges_complete_once=index_cover,
                all_vertices_used=vertices_used,directed_edges_pair_once=paired,
                connected_components=connected,vertex_links_single_cycle=links_ok,
                euler_characteristic=euler,vertices=len(vertices),undirected_edges=len(edges),
                polygons=len(polygons),signed_fan_volume_native_units_cubed=volume,
                positive_volume_at_original_numeric_scale=bool(volume>tolerance**3),
                winding_matches_stored_normals=winding,
                combinatorial_surface_checks_passed=bool(checks),
                nonplanar_polygon_count=len(nonplanar),nonplanar_polygons=nonplanar,
                max_best_fit_face_residual_native_units=max_fit)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError,
            np.linalg.LinAlgError) as error:
        result['error']=type(error).__name__+': '+str(error)
    return result

def analyze_hull(raw):
    """Reapply the original validator and measure source-plane residuals unchanged."""
    result=dict(_validate_hull(raw))
    result.update(polygons_examined=0,bad_plane_count=0,bad_planes=[],
        max_face_residual_native_units=None,max_outside_residual_native_units=None,
        plane_analysis_complete=False)
    tolerance=result.get('numeric_topology_tolerance_native_units')
    result['surface_diagnostics']=diagnose_surface(raw,tolerance)
    if tolerance is None:
        return result
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            vertices=np.asarray(raw['vertices'],dtype=float)
            indices=raw['indices']
            face_max=0.;outside_max=0.
            for index,polygon in enumerate(raw['polygons']):
                start=polygon['index_base'];count=polygon['num_vertices']
                ids=indices[start:start+count]
                if len(ids)!=count or not ids or min(ids)<0 or max(ids)>=len(vertices):
                    raise ValueError('invalid source polygon indices')
                plane=np.asarray(polygon['plane'],dtype=float)
                if plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-12:
                    raise ValueError('invalid source polygon plane')
                unit_plane=plane/np.linalg.norm(plane[:3])
                residual=vertices@unit_plane[:3]+unit_plane[3]
                face=float(np.max(abs(residual[ids])));outside=float(np.max(residual))
                face_max=max(face_max,face);outside_max=max(outside_max,outside)
                result['polygons_examined']+=1
                if face>tolerance or outside>tolerance:
                    result['bad_planes'].append(dict(polygon_index=index,index_base=start,
                        num_vertices=count,vertex_ids=list(ids),
                        max_face_residual_native_units=face,
                        max_outside_residual_native_units=outside))
            result.update(bad_plane_count=len(result['bad_planes']),
                max_face_residual_native_units=face_max,
                max_outside_residual_native_units=outside_max,plane_analysis_complete=True)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:
        result['plane_analysis_error']=type(error).__name__+': '+str(error)
    return result



def _vertex_statistics(distances):
    return dict(p50_m=float(np.percentile(distances,50)),
        p90_m=float(np.percentile(distances,90)),p95_m=float(np.percentile(distances,95)),
        max_m=float(np.max(distances)),count=int(len(distances)))


def _nearest_vertex_distances(points,tree,chunk):
    distances=np.empty(len(points),dtype=float)
    for start in range(0,len(points),chunk):
        stop=min(start+chunk,len(points))
        distances[start:stop]=tree.query(points[start:stop],k=1,workers=1)[0]
    if not np.isfinite(distances).all():raise ValueError("finite nearest-vertex distances required")
    return distances



FAILED_FACE_THRESHOLD_M = 1e-4


def _plane_measurements(raw):
    vertices = np.asarray(raw["vertices"], dtype=float)
    polygons = raw["polygons"]
    indices = raw["indices"]
    if (vertices.shape != (raw["num_vertices"], 3) or not len(vertices)
            or not np.isfinite(vertices).all() or len(polygons) != raw["num_polygons"]
            or any(type(index) is not int for index in indices)):
        raise ValueError("finite counted hull vertices and integer indices required")
    measurements = []
    slots = []
    for polygon in polygons:
        start, count = polygon["index_base"], polygon["num_vertices"]
        if type(start) is not int or type(count) is not int or start < 0 or count < 3:
            raise ValueError("integer polygon range required")
        ids = indices[start:start + count]
        if len(ids) != count or len(set(ids)) != count or min(ids) < 0 or max(ids) >= len(vertices):
            raise ValueError("valid distinct polygon indices required")
        slots.extend(range(start, start + count))
        plane = np.asarray(polygon["plane"], dtype=float)
        if plane.shape != (4,) or not np.isfinite(plane).all():
            raise ValueError("finite stored plane required")
        norm = np.linalg.norm(plane[:3])
        if norm < 1e-12:
            raise ValueError("nonzero stored normal required")
        unit = plane / norm
        residuals = vertices @ unit[:3] + unit[3]
        measurements.append((ids, plane, residuals))
    if sorted(slots) != list(range(len(indices))):
        raise ValueError("complete nonoverlapping polygon ranges required")
    return vertices, measurements


def _plane_maxima(ids, residuals):
    return float(np.max(abs(residuals[ids]))), float(np.max(residuals))


def _plane_threshold_study(raw, original):
    # Counterfactual plane predicate only.
    _, measurements = _plane_measurements(raw)
    if original is None:
        raise ValueError("original validator numeric scale unavailable")
    worst = [max(_plane_maxima(ids, residuals)) for ids, _, residuals in measurements]
    return [dict(name=name, tolerance_m=float(tol), failing_faces=sum(1 for value in worst if value > tol),
                 polygons_examined=len(worst), plane_predicate_only=True, admission_applied=False)
            for name, tol in (("original_numeric_scale", original),
                              ("image_suggested_1e_4_m", FAILED_FACE_THRESHOLD_M))]


def _fan_areas(points):
    offsets = points[1:] - points[0]
    return np.linalg.norm(np.cross(offsets[:-1], offsets[1:]), axis=1) / 2


def _source_vertex_matches(row, coordinates):
    """Bind exact point matches to original record indices, including duplicates."""
    source = row["source_geometry"]
    binding = row["binding"]
    if binding["source_manifest"]["meters_per_unit"] != 1.0:
        raise ValueError("metre source mesh required")
    points = np.asarray(source["original_mesh_points"], dtype=float)
    collision = binding["source_collision"]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite source vertices required")
    digest = _saved_source_geometry_sha(points, source["original_triangles"], collision["orientation"])
    if digest != source["source_geometry_sha256"] or digest != collision["geometry_sha256"]:
        raise ValueError("source content hash differs")
    matches = {}
    for coordinate in coordinates:
        indices = np.flatnonzero(np.all(points == coordinate, axis=1))
        matches[coordinate] = {"first_index": int(indices[0]), "count": len(indices)} if len(indices) else None
    return matches


def _fitted_plane_diagnostics(vertices, ids, stored_normal):
    """Measure a least-squares face plane without replacing the stored plane."""
    points = vertices[ids]
    center = points.mean(axis=0)
    centered = points - center
    _, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    normal = basis[-1]
    face_distances = abs(centered @ normal)
    alignment = float(normal @ stored_normal)
    # A collinear ring or perpendicular reference cannot orient the plane.
    epsilon = np.finfo(float).eps
    outside = None
    if (singular_values[1] > epsilon * max(points.shape) * singular_values[0]
            and abs(alignment) > epsilon * np.linalg.norm(stored_normal)):
        if alignment < 0:
            normal = -normal
        outside = float(np.max((vertices - center) @ normal))
    return dict(best_fit_mean_abs_residual_m=float(face_distances.mean()),
                best_fit_max_residual_m=float(face_distances.max()),
                best_fit_max_outside_residual_m=outside)


def _projection_comparison(faces):
    threshold = 0.00015
    outside = [face["best_fit_max_outside_residual_m"] for face in faces]
    return dict(
        comparison_threshold_m=threshold,
        faces_examined=len(faces),
        fitted_mean_within_threshold=sum(1 for f in faces if f["best_fit_mean_abs_residual_m"] <= threshold),
        fitted_face_max_within_threshold=sum(1 for f in faces if f["best_fit_max_residual_m"] <= threshold),
        fitted_enclosure_above_threshold=sum(1 for value in outside if value is not None and value > threshold),
        fitted_enclosure_unavailable=sum(value is None for value in outside),
        stored_face_or_enclosure_above_threshold=sum(
            1 for f in faces if max(f["max_face_residual_m"], f["max_outside_residual_m"]) > threshold),
    )


def _failed_hull_faces(raw, validation):
    vertices, measurements = _plane_measurements(raw)
    tolerance = validation["numeric_topology_tolerance_native_units"]
    # Reverse storage order without changing any polygon's directed ring.
    reindexed = dict(raw, vertices=raw["vertices"][::-1], polygons=raw["polygons"][::-1],
                     indices=[len(vertices) - 1 - index for index in raw["indices"]])
    _, permuted = _plane_measurements(reindexed)
    reindexed_validation = _validate_hull(reindexed)
    areas = [_fan_areas(vertices[ids]) for ids, _, _ in measurements]
    hull_median_area = float(np.median([fan.sum() for fan in areas]))
    comparisons = []
    failed = []
    for index, ((ids, plane, residuals), (other_ids, _, other_residuals)) in enumerate(
            zip(measurements, reversed(permuted))):
        face_max, outside_max = _plane_maxima(ids, residuals)
        other_max = _plane_maxima(other_ids, other_residuals)
        comparisons.append([max(face_max, outside_max) > limit for limit in
                            (tolerance, FAILED_FACE_THRESHOLD_M)] ==
                           [max(other_max) > limit for limit in (tolerance, FAILED_FACE_THRESHOLD_M)])
        if max(face_max, outside_max) <= FAILED_FACE_THRESHOLD_M:
            continue
        points = vertices[ids]
        projection = _fitted_plane_diagnostics(vertices, ids, plane[:3])
        # All areas use the stored ring's triangle fan. For warped polygons
        # this is the area of that fan, with no claim of a unique planar area.
        fan = areas[index]
        failed.append(dict(
            polygon_index=index, index_base=raw["polygons"][index]["index_base"],
            cooked_vertex_indices=ids, cooked_vertices_mesh_m=points.tolist(),
            stored_normal=plane[:3].tolist(), stored_plane=plane.tolist(),
            face_signed_residuals_m=residuals[ids].tolist(),
            max_face_residual_m=face_max, max_outside_residual_m=outside_max,
            max_outside_vertex_index=int(np.argmax(residuals)),
            stored_mean_abs_face_residual_m=float(np.mean(abs(residuals[ids]))),
            **projection,
            fan_area_m2=float(fan.sum()), min_fan_triangle_area_m2=float(fan.min()),
            hull_median_fan_area_m2=hull_median_area,
            original_winding_area_floor_m2=float(tolerance ** 2),
        ))
    return failed, all(comparisons) and reindexed_validation["closed_convex"] == validation["closed_convex"]


def export_failed_face_evidence(colliders, bodies):
    faces = []
    errors = []
    checked = unchanged = 0
    for row, body in zip(colliders, bodies):
        owner = row["binding"].get("actual_collision_owner_path")
        for hull_index, (item, validation) in enumerate(zip(row.get("cooked_hulls", []), body["hulls"])):
            checked += 1
            identity = dict(side=body["side"], actual_body_path=body["actual_body_path"],
                            collision_owner_path=owner, hull_index=hull_index)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    failed, equivalent = _failed_hull_faces(item["raw"], validation)
                    coordinates = {tuple(point) for face in failed for point in face["cooked_vertices_mesh_m"]}
                    matches = _source_vertex_matches(row, coordinates) if coordinates else {}
                for face in failed:
                    face.update(identity)
                    face["source_exact_vertex_matches"] = [
                        matches[tuple(point)] for point in face["cooked_vertices_mesh_m"]]
                faces.extend(failed)
                unchanged += int(equivalent)
            except (ValueError, KeyError, IndexError, TypeError, OverflowError, FloatingPointError,
                    np.linalg.LinAlgError) as error:
                errors.append(dict(identity, error=str(error)))
    return dict(schema="dh116-cooked-failed-face-evidence-v1", plane_threshold_m=FAILED_FACE_THRESHOLD_M,
                hulls_examined=checked, hulls_unchanged_under_reindex=unchanged, faces=faces, errors=errors,
                projection_comparison=_projection_comparison(faces),
                native_qualification=False)


def _saved_source_geometry_sha(points,triangles,orientation):
    """Recompute existing usd_surface.read_surface byte contract; no USD import."""
    triangles=np.asarray(triangles)
    if (triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles)
            or not np.issubdtype(triangles.dtype,np.integer)
            or triangles.min()<0 or triangles.max()>=len(points)):
        raise ValueError("integer in-range source triangles required")
    if orientation not in ("rightHanded","leftHanded"):
        raise ValueError("explicit saved source orientation required")
    digest=hashlib.sha256()
    digest.update(b"DH116-triangle-surface-v1\0")
    digest.update(np.asarray(points.shape,dtype="<i8").tobytes())
    digest.update(points.astype("<f8").tobytes())
    digest.update(triangles.astype("<i8").tobytes())
    digest.update(orientation.encode("ascii"))
    return digest.hexdigest()

def diagnose_cooked_representation(row,*,query_chunk_size=65536):
    """Read-only SI mesh-frame comparison. Vertex-set distances are NOT surface errors."""
    if type(query_chunk_size) is not int or not 1<=query_chunk_size<=65536:
        raise ValueError("query chunk must be an integer in1..65536")
    result=dict(available=False,query_chunk_size=query_chunk_size,
        method="chunked_bidirectional_unique_vertex_cKDTree",vertex_distances_are_surface_errors=False,
        cooking_abi_alignment_proven=False,live_shape_attachment_proven=False,
        source_geometry_content_hash_recomputed=False,
        input_geometry_modified=False,validator_tolerance_changed=False,native_qualification=False)
    try:
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            from scipy.spatial import cKDTree
            from go2_pvcnn.control.dh116.tabletop_tripod_cooked_geometry import _matrix
            unit=row["binding"]["source_manifest"]["meters_per_unit"]
            if type(unit) not in (int,float) or unit!=1.:
                raise ValueError("explicit original SI metre geometry required; no inferred unit conversion")
            source=row["source_geometry"];frames=row["frames"]
            source_sha=source["source_geometry_sha256"]
            if (type(source_sha) is not str or len(source_sha)!=64
                    or any(c not in "0123456789abcdef" for c in source_sha)
                    or source_sha!=source["live_mesh_geometry_sha256"]):
                raise ValueError("matching saved source/live geometry identity required")
            if frames["convention"]!="column_vectors_world_from_body_SI_metres":
                raise ValueError("explicit original column-vector SI frame convention required")
            points=np.asarray(source["original_mesh_points"],dtype=float)
            if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():
                raise ValueError("finite original Nx3 vertex records required")
            collision=row["binding"]["source_collision"]
            computed=_saved_source_geometry_sha(points,source["original_triangles"],collision["orientation"])
            if computed!=source_sha or collision["geometry_sha256"]!=source_sha:
                raise ValueError("saved source content or binding geometry hash differs")
            result["source_geometry_content_hash_recomputed"]=True
            result["source_content_sha256"]=computed
            body=_matrix(frames["actual_body_link_pose_w"])
            mesh=_matrix(frames["mesh_to_body"]);live_mesh=_matrix(source["live_mesh_to_body"])
            original_mesh=_matrix(source["original_mesh_to_body"])
            if not np.array_equal(mesh,live_mesh):
                raise ValueError("same saved mesh-to-body matrix required")
            usd=_matrix(frames["usd_world_from_mesh_crosscheck_only"])
            world=body@mesh;inverse=np.linalg.inv(world)
            unique=np.unique(points,axis=0);source_tree=cKDTree(unique,copy_data=True)
            hulls=[]
            if not row["cooked_hulls"]:raise ValueError("at least one stored cooked hull required")
            for item in row["cooked_hulls"]:
                raw=item["raw"];vertices=np.asarray(raw["vertices"],dtype=float)
                if (type(raw["num_vertices"]) is not int or vertices.shape!=(raw["num_vertices"],3)
                        or len(vertices)<4 or not np.isfinite(vertices).all()):
                    raise ValueError("finite counted cooked vertices required")
                # Keeping original numbers/order: trees own separate coordinate buffers.
                cooked_tree=cKDTree(vertices,copy_data=True)
                forward=_nearest_vertex_distances(unique,cooked_tree,query_chunk_size)
                reverse=_nearest_vertex_distances(vertices,source_tree,query_chunk_size)
                world_points=vertices@world[:3,:3].T+world[:3,3]
                restored=world_points@inverse[:3,:3].T+inverse[:3,3]
                validation=_validate_hull(raw)
                hulls.append(dict(cooked_vertices=len(vertices),
                    source_unique_to_cooked_vertex=_vertex_statistics(forward),
                    cooked_to_source_unique_vertex=_vertex_statistics(reverse),
                    world_mesh_roundtrip_max_m=float(np.max(np.linalg.norm(restored-vertices,axis=1))),
                    existing_validator_closed_convex=validation["closed_convex"],
                    existing_validator_reasons=validation["reasons"],
                    plane_threshold_study=_plane_threshold_study(raw,validation.get("numeric_topology_tolerance_native_units"))))
            result.update(available=True,source_vertex_records=len(points),source_unique_vertices=len(unique),
                saved_source_live_geometry_sha_equal=True,source_geometry_sha256=source_sha,
                original_live_mesh_to_body_max_abs_entry_delta=float(np.max(abs(original_mesh-live_mesh))),
                world_mesh_usd_crosscheck_max_abs_entry_delta=float(np.max(abs(world-usd))),
                body_rotation_orthogonality_max_abs_delta=float(np.max(abs(body[:3,:3].T@body[:3,:3]-np.eye(3)))),
                body_rotation_determinant=float(np.linalg.det(body[:3,:3])),hulls=hulls)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as error:
        result["error"]=type(error).__name__+": "+str(error)
    return result


def _same_saved_sdk_boundary(before,after):
    def valid(row):
        return (type(row) is dict and set(row)=={"sdk_step","sdk_time_s"}
            and type(row["sdk_step"]) is int and row["sdk_step"]>=0
            and type(row["sdk_time_s"]) in (int,float) and np.isfinite(row["sdk_time_s"])
            and row["sdk_time_s"]>=0)
    return bool(valid(before) and valid(after) and before==after)

def analyze_capture(packet,*,compare_source_vertices=False,export_failed_faces=False):
    capture=packet['capture'];bodies=[];seen=set();duplicates=[]
    hull_count=passed=0
    for row in capture.get('colliders',[]):
        binding=row['binding']
        identity=(binding['side'],binding['actual_body_path'])
        if identity in seen:
            duplicates.append(list(identity))
        seen.add(identity)
        hulls=[]
        for hull in row.get('cooked_hulls',[]):
            validation=analyze_hull(hull['raw'])
            hulls.append(validation);hull_count+=1;passed+=int(validation['closed_convex'])
        bodies.append(dict(side=binding['side'],body_id=binding['body_id'],
            actual_body_path=binding['actual_body_path'],
            stored_complete_flag=row.get('complete'),stored_reasons=row.get('reasons'),
            hulls=hulls))
    result=dict(schema='dh116-offline-cooked-readback-audit-v1',
        capture_claimed_complete=capture.get('complete'),
        collider_rows=len(bodies),unique_body_count=len(seen),duplicate_body_identities=duplicates,
        hulls_returned=hull_count,hulls_passing_existing_validator=passed,
        bad_plane_count=sum(h['bad_plane_count'] for b in bodies for h in b['hulls']),
        polygons_examined=sum(h['polygons_examined'] for b in bodies for h in b['hulls']),
        capture_clock=capture.get('capture_clock'),bodies=bodies,
        input_geometry_modified=False,planes_refitted=False,validator_tolerance_changed=False,
        new_native_runs=0,native_qualification=False)
    if compare_source_vertices:
        for body,row in zip(bodies,capture.get('colliders',[])):
            body['coordinate_diagnostics']=diagnose_cooked_representation(row)
        before,after=packet.get('sdk_clock_before'),packet.get('sdk_clock_after')
        result['schema']='dh116-offline-cooked-readback-audit-v2'
        result['representation_audit']=dict(
            clock_before=before,clock_after=after,same_packet_sdk_boundary=_same_saved_sdk_boundary(before,after),
            clock_source='saved top-level actual SDK clock; missing values remain unknown',
            native_qualification=False,validator_tolerance_changed=False,vertices_refitted=False)
    if export_failed_faces:
        result['failed_face_audit']=export_failed_face_evidence(capture.get('colliders',[]),bodies)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare-source-vertices',action='store_true',
        help='opt-in read-only source/cooked vertex and coordinate diagnostics; never admission')
    parser.add_argument('--export-failed-faces', action='store_true',
        help='export per-face evidence above the diagnostic 0.1 mm plane threshold')
    args=parser.parse_args(argv)
    payload=args.capture.read_bytes()
    result=analyze_capture(json.loads(payload),compare_source_vertices=args.compare_source_vertices,
                           export_failed_faces=args.export_failed_faces)
    result.update(input_path=str(args.capture.resolve()),input_sha256=hashlib.sha256(payload).hexdigest())
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return 0  # Means the offline audit ran; the report carries failed qualification.


if __name__=='__main__':
    raise SystemExit(main())
