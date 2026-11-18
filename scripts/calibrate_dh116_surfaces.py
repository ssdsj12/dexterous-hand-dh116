"""Simulation source-surface reference points; not hardware/cooked-contact approval."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd
from go2_pvcnn.control.dh116.usd_surface import read_surface,layer_pins
from go2_pvcnn.control.dh116.surface_pick import ray_pick
from go2_pvcnn.control.dh116.contact_frames import palm_frame

BODIES=dict(palm='base_link',thumb='finger14_link',index='finger23_link',
            middle='finger33_link',ring='finger43_link',little='finger53_link')


def build(asset,output):
    asset=Path(asset);output=Path(output)
    if output.exists():raise ValueError('refusing existing calibration output')
    stage=Usd.Stage.Open(str(asset.resolve(strict=True)))
    report=dict(schema='dh116-surface-reference-v1',asset=str(asset.resolve()),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),layers=layer_pins(stage),
        hardware_calibrated=False,physics_contact_qualified=False,
        references=[],palm_frames={},method='inspected surface, explicit side-specific first-hit ray')
    for side in ('left','right'):
        for digit,body in BODIES.items():
            path=f'/FixedDualPandaDH116/{side}_arm/dh116/{body}'
            mesh=read_surface(stage,path)
            if mesh['orientation']!='rightHanded':raise ValueError('orientation requires audit')
            points=np.array(mesh['points_body']); faces=np.array(mesh['triangles'])
            lo=points.min(0);hi=points.max(0)
            origin=np.array([0.,lo[1]-.03,hi[2]-.01]); direction=np.array([0.,1.,0.])
            if digit=='palm':origin[2]=.075
            if digit=='thumb':
                if side=='left':origin[2]=lo[2]+.45*(hi[2]-lo[2])
                else:
                    origin=np.array([0.,lo[1]+.6*(hi[1]-lo[1]),lo[2]-.03])
                    direction=np.array([0.,0.,1.])
            ref=ray_pick(points,faces,origin,direction)
            eq=ConvexHull(np.unique(points,axis=0)).equations
            den=eq[:,:3]@direction; entering=den < -1e-12
            hull_t=np.max(-(eq[:,:3]@origin+eq[:,3])[entering]/den[entering])
            gap=float(ref['ray_distance_m']-hull_t)
            if digit!='palm' and abs(gap)>1e-4:raise ValueError('pad ideal hull gap exceeds 0.1 mm')
            ref.update(side=side,digit=digit,body_name=body,body_path=path,
                mesh_path=mesh['mesh_path'],geometry_sha256=mesh['geometry_sha256'],
                mesh_to_body=mesh['mesh_to_body'],collision_approximation=mesh['collision_approximation'],
                ray_origin=origin.tolist(),ray_direction=direction.tolist(),ideal_hull_ray_gap_m=gap,
                triangle_body=points[faces[ref['triangle_index']]].tolist())
            report['references'].append(ref)
            if digit=='palm':
                # Nominal palm plane is base -Y, wrist-to-middle is base +Z.
                # Surface normal is recorded separately: do not use a recessed
                # decorative face as the global palm orientation or contact pad.
                report['palm_frames'][side]=palm_frame(ref['point'],[0.,-1.,0.],[0.,0.,1.]).tolist()
    output.mkdir(parents=True)
    (output/'references.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=build(a.asset,a.output)
    print(json.dumps(dict(references=len(r['references']),hardware_calibrated=False)))
