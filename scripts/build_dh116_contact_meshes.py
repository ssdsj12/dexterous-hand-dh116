"""Pinned simulation convex hulls for DeformX, not PhysX cooked-hull qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import ConvexHull
from pxr import Usd
from go2_pvcnn.control.dh116.usd_surface import read_surface,layer_pins


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(references,output):
    references=Path(references);output=Path(output)
    if output.exists():raise ValueError('refusing existing contact mesh directory')
    refs=json.loads(references.read_text());asset=Path(refs['asset'])
    if sha(asset)!=refs['asset_sha256']:raise ValueError('asset identity mismatch')
    stage=Usd.Stage.Open(str(asset))
    if layer_pins(stage)!=refs['layers']:raise ValueError('layer identity mismatch')
    report=dict(schema='dh116-contact-hulls-v1',reference_sha256=sha(references),
        asset_sha256=refs['asset_sha256'],layers=refs['layers'],
        approximation='source convex hull quantized at 100 nm, not PhysX cooked hull',
        physics_contact_qualified=False,hardware_calibrated=False,meshes={})
    prepared={}
    for side in ('left','right'):
        for digit in ('thumb','index','middle'):
            choices=[r for r in refs['references'] if r['side']==side and r['digit']==digit]
            if len(choices)!=1:raise ValueError('ambiguous reference')
            ref=choices[0];surface=read_surface(stage,ref['body_path'])
            if surface['geometry_sha256']!=ref['geometry_sha256']:raise ValueError('mesh identity mismatch')
            if surface['collision_approximation']!='convexHull':raise ValueError('unsupported collision approximation')
            points=np.unique(np.asarray(surface['points_body']),axis=0)
            hull=trimesh.convex.convex_hull(points)
            center=hull.center_mass.copy()
            # Source STL float noise creates ~1e-10 m near-coplanar facets
            # which Open3D reports as intersections. Quantize a derived hull
            # only, with a recorded <=sqrt(3)*50 nm Hausdorff displacement.
            quantum=1e-7
            local=hull.vertices-center
            rounded=np.round(local/quantum)*quantum
            displacement=float(np.linalg.norm(rounded-local,axis=1).max())
            hull=trimesh.convex.convex_hull(rounded)
            residual=hull.center_mass.copy()
            center+=residual
            hull.apply_translation(-residual)
            eq=ConvexHull(hull.vertices+center).equations
            origin=np.array(ref['ray_origin']);direction=np.array(ref['ray_direction'])
            den=eq[:,:3]@direction;enters=den < -1e-12
            distance=np.max(-(eq[:,:3]@origin+eq[:,3])[enters]/den[enters])
            gap=float(ref['ray_distance_m']-distance)
            if abs(gap)>1e-4:raise ValueError('reference outside contact surface tolerance')
            if not hull.is_watertight or not hull.is_winding_consistent or hull.volume<=0:
                raise ValueError('invalid closed contact hull')
            backend_mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hull.vertices),
                o3d.utility.Vector3iVector(hull.faces))
            if not backend_mesh.is_watertight() or not backend_mesh.is_orientable():
                raise ValueError('contact hull rejected by backend geometry checks')
            channel=f'{side}_{digit}';name=f'{channel}.ply'
            # Preserve the checked geometry in float64, including the recenter.
            lines=['ply','format ascii 1.0',f'element vertex {len(hull.vertices)}',
                'property double x','property double y','property double z',
                f'element face {len(hull.faces)}','property list uchar int vertex_indices','end_header']
            lines.extend(' '.join(format(float(x),'.17g') for x in v) for v in hull.vertices)
            lines.extend('3 '+' '.join(str(int(i)) for i in f) for f in hull.faces)
            payload=('\n'.join(lines)+'\n').encode('ascii')
            prepared[name]=payload
            report['meshes'][channel]=dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),
                body_name=ref['body_name'],body_path=ref['body_path'],source_mesh_path=surface['mesh_path'],
                geometry_sha256=surface['geometry_sha256'],body_local_mesh_com_m=center.tolist(),
                reference_point_body=ref['point'],reference_normal_body=ref['normal'],
                reference_surface_gap_m=gap,quantization_m=quantum,
                convex_hull_displacement_bound_m=displacement,
                vertices=len(hull.vertices),faces=len(hull.faces))
    output.mkdir(parents=True)
    for name,payload in prepared.items():(output/name).write_bytes(payload)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=build(a.references,a.output);print(json.dumps(r,indent=2))
