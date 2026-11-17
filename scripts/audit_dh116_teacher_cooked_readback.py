#!/usr/bin/env python3
"""Offline audit of immutable saved hull packets; never launches or repairs PhysX."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_tripod_cooked_geometry import _validate_hull


def analyze_hull(raw):
    """Reapply the original validator and measure source-plane residuals unchanged."""
    result=dict(_validate_hull(raw))
    result.update(polygons_examined=0,bad_plane_count=0,bad_planes=[],
        max_face_residual_native_units=None,max_outside_residual_native_units=None,
        plane_analysis_complete=False)
    tolerance=result.get('numeric_topology_tolerance_native_units')
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
