"""Build a new contact-reference bundle at actual CUDA hull closest points."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
from pxr import Usd

from co_sim.gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod
from go2_pvcnn.control.dh116.collision_scene import SourceCollisionScene
from go2_pvcnn.control.dh116.contact_runtime import BODY_NAMES, load_mesh_manifest
from go2_pvcnn.control.dh116.pad_region_guard import PadRegionGuard
from go2_pvcnn.control.dh116.pose_artifact import load_active_pose


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose',type=Path,required=True)
    parser.add_argument('--source-references',type=Path,required=True)
    parser.add_argument('--contact-manifest',type=Path,required=True)
    parser.add_argument('--asset',type=Path,default=Path('assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    source=SourceCollisionScene(Usd.Stage.Open(str(args.asset)))
    states=load_active_pose(args.pose,digest(args.pose),digest(args.asset),source.layer_pins)
    refs=copy.deepcopy(json.loads(args.source_references.read_text()))
    manifest=copy.deepcopy(json.loads(args.contact_manifest.read_text()))
    records=load_mesh_manifest(args.contact_manifest,digest(args.contact_manifest))
    selected={r['side']+'_'+r['digit']:r for r in refs['references']}
    rod=TorchCudaCosseratRod(GpuRodConfig(n_elem=40,rod_start=(.43,-.25,.308),
        rod_direction=(0.,1.,0.),contact_subsamples=16),device=args.device)
    wire=rod.rod_position.T.cpu().numpy()
    motion_delta=np.r_[np.full(7,.151e-4),np.full(11,.31e-4)]
    report=dict(scope='static CUDA collision-hull calibration; physical grasp unqualified',
        source_pose_sha256=digest(args.pose),source_reference_sha256=digest(args.source_references),
        source_manifest_sha256=digest(args.contact_manifest),channels={},pad_regions={})
    for channel,record in records.items():
        ref=selected[channel]
        model=source.models[ref['side']]
        pose=model.fk(states[ref['side']])[0][model.bodies[ref['body_name']]]
        rotation=pose[:3,:3]
        origin=pose[:3,3]+rotation@np.asarray(record['body_local_mesh_com_m'])
        mesh=o3d.io.read_triangle_mesh(str(record['resolved_path']))
        vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
        rod.register_contact_mesh(channel,vertices,faces)
        if ref['digit'] in ('thumb','index'):
            closest=rod.closest_contact_point(channel,position=origin,rotation=rotation)
            point=closest['point_w'].cpu().numpy()
            rod_point=closest['rod_point_w'].cpu().numpy()
            normal=(rod_point-point)/np.linalg.norm(rod_point-point)
            local=rotation.T@(point-pose[:3,3]);local_normal=rotation.T@normal
            old=np.array(ref['point'])
            ref.update(point=local.tolist(),normal=local_normal.tolist(),
                reference_geometry='derived_closed_collision_hull',
                contact_hull_sha256=record['sha256'],calibration_pose_sha256=digest(args.pose))
            for key in ('triangle_index','weights','ray_distance_m'):
                ref.pop(key,None)
            manifest['meshes'][channel].update(reference_point_body=local.tolist(),
                reference_normal_body=local_normal.tolist(),reference_surface_gap_m=0.)
            report['channels'][channel]=dict(reference_shift_m=float(np.linalg.norm(local-old)),
                point_w=point.tolist(),normal_w=normal.tolist(),initial_gap_m=float(closest['gap_m']))
        guard=PadRegionGuard(vertices,faces,np.array(ref['point'])-
            np.array(record['body_local_mesh_com_m']),ref['normal'])
        path=model.bodies[BODY_NAMES[channel]]
        envelope=float(source.geometry[path]['motion_radii']@motion_delta)
        report['pad_regions'][channel]=guard.check(wire,rotation,origin,.008+envelope)
    report['pad_regions_passed']=all(value['passed'] for value in report['pad_regions'].values())
    report['grasp_qualified']=False
    report_text=json.dumps(report,indent=2)+'\n'
    report_sha=hashlib.sha256(report_text.encode()).hexdigest()
    refs.update(source_report_sha256=report_sha,grasp_qualified=False,
        scope='Actual closest thumb/index collision-hull references for direct-close MPC; physical validation required.')
    manifest.update(reference_sha256=report_sha,physics_contact_qualified=False,
        reference_scope='new calibrated contact-hull points; copied source meshes are byte-identical')
    args.output.mkdir(parents=True,exist_ok=False)
    contact_dir=args.output/'contact';contact_dir.mkdir()
    for channel,record in records.items():
        target=contact_dir/Path(record['resolved_path']).name
        shutil.copy2(record['resolved_path'],target)
        manifest['meshes'][channel]['path']=target.name
        if digest(target)!=record['sha256']:raise RuntimeError('copied contact mesh identity mismatch')
    (args.output/'calibration.json').write_text(report_text)
    (args.output/'point_references.json').write_text(json.dumps(refs,indent=2)+'\n')
    manifest_path=contact_dir/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    load_mesh_manifest(manifest_path,digest(manifest_path))
    print(json.dumps(dict(output=str(args.output),pad_regions_passed=report['pad_regions_passed'],
        contact_manifest_sha256=digest(manifest_path),
        reference_sha256=digest(args.output/'point_references.json'),
        region_failures={c:r['failures'] for c,r in report['pad_regions'].items() if not r['passed']})))
    return 0 if report['pad_regions_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
