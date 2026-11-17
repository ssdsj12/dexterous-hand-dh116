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


def analyze_capture(packet):
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
    return dict(schema='dh116-offline-cooked-readback-audit-v1',
        capture_claimed_complete=capture.get('complete'),
        collider_rows=len(bodies),unique_body_count=len(seen),duplicate_body_identities=duplicates,
        hulls_returned=hull_count,hulls_passing_existing_validator=passed,
        bad_plane_count=sum(h['bad_plane_count'] for b in bodies for h in b['hulls']),
        polygons_examined=sum(h['polygons_examined'] for b in bodies for h in b['hulls']),
        capture_clock=capture.get('capture_clock'),bodies=bodies,
        input_geometry_modified=False,planes_refitted=False,validator_tolerance_changed=False,
        new_native_runs=0,native_qualification=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    payload=args.capture.read_bytes()
    result=analyze_capture(json.loads(payload))
    result.update(input_path=str(args.capture.resolve()),input_sha256=hashlib.sha256(payload).hexdigest())
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return 0  # Means the offline audit ran; the report carries failed qualification.


if __name__=='__main__':
    raise SystemExit(main())
