"""Refine an offline close initialization toward the unchanged pad centres."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Usd

from go2_pvcnn.control.dh116.arm_ik import align_contact_centres
from go2_pvcnn.control.dh116.close_initialization import load_close_initialization,reframe_close_initialization
from go2_pvcnn.control.dh116.collision_scene import SourceCollisionScene
from go2_pvcnn.control.dh116.contact_runtime import load_mesh_manifest
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.pad_region_guard import PadRegionGuard
from go2_pvcnn.control.dh116.pose_artifact import load_active_pose


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-state',type=Path,required=True)
    parser.add_argument('--initial-state-sha256',required=True)
    parser.add_argument('--point-references',type=Path,required=True)
    parser.add_argument('--contact-manifest',type=Path,required=True)
    parser.add_argument('--asset',type=Path,default=Path('assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('alignment output already exists')
    asset_sha=digest(args.asset);contact_sha=digest(args.contact_manifest)
    payload=load_close_initialization(args.initial_state,args.initial_state_sha256,
        asset_sha256=asset_sha,contact_manifest_sha256=contact_sha)
    scene=SourceCollisionScene(Usd.Stage.Open(str(args.asset)))
    records=load_mesh_manifest(args.contact_manifest,contact_sha)
    references=json.loads(args.point_references.read_text())
    if references['asset_sha256']!=asset_sha or references['layers']!=scene.layer_pins:
        raise ValueError('point-reference asset identity mismatch')
    refs={r['side']+'_'+r['digit']:r for r in references['references']}
    nodes=np.asarray(payload['rod']['dynamic']['rod_position']).T
    fractions=(np.arange(16)+.5)/16
    samples=(nodes[:-1,None,:]*(1-fractions[None,:,None])+nodes[1:,None,:]*fractions[None,:,None]).reshape(-1,3)
    # Same volume-preserving radius update as the restored GPU rod; no state
    # is advanced or edited by this observation-only calculation.
    radius=float(np.max(.008*np.sqrt(np.asarray(payload['rod']['reference']['rest_lengths'])/
                                   np.linalg.norm(np.diff(nodes,axis=0),axis=1))))
    guards={}
    for channel,r in records.items():
        mesh=o3d.io.read_triangle_mesh(str(r['resolved_path']));ref=refs[channel]
        guards[channel]=PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),
            np.asarray(ref['point'])-r['body_local_mesh_com_m'],ref['normal'])
    states={};report=dict(physics_qualified=False,motion_qualified=False,sides={},pad_regions={},
        source_initial_sha256=args.initial_state_sha256,point_reference_sha256=digest(args.point_references),
        scope='offline approach endpoint alignment, no close-phase arm motion or phase credit')
    for side,model in scene.models.items():
        robot=payload['robot'][side]
        seed=np.array([robot['q'][robot['joint_names'].index(n)] for n in
                       [f'{side}_panda_joint{i}' for i in range(1,8)]+list(ACTIVE)])
        contacts=[dict(body=refs[c]['body_name'],reference_point=np.array(refs[c]['point']),
            reference_normal=np.array(refs[c]['normal']),mesh_offset=np.array(records[c]['body_local_mesh_com_m']),
            query=guards[c].surface_query) for c in (side+'_thumb',side+'_index')]
        result=align_contact_centres(model,seed,contacts,samples,scene.limits[side][:,0],scene.limits[side][:,1])
        report['sides'][side]=result;states[side]=np.asarray(result['q'])
    delta={s:np.r_[np.full(7,.151e-4),np.full(11,.31e-4)] for s in states}
    poses={}
    for side,model in scene.models.items():poses.update(model.fk(states[side])[0])
    measured={path:poses[path] for path in scene.geometry}
    report['source_screen']=scene.check(states,nodes,permit_pad_contact=True,
        measured_body_poses=measured,physical_motion_delta=delta,wire_radius=radius)
    nominal=np.c_[np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.308)]
    report['nominal_initial_screen']=scene.check(states,nominal,permit_pad_contact=True)
    for channel,guard in guards.items():
        side=refs[channel]['side'];path=scene.models[side].bodies[refs[channel]['body_name']]
        t=poses[path];offset=np.array(records[channel]['body_local_mesh_com_m'])
        motion=float(scene.geometry[path]['motion_radii']@delta[side])
        report['pad_regions'][channel]=guard.check(nodes,t[:3,:3],t[:3,3]+t[:3,:3]@offset,radius+motion)
    passed=bool(all(r['converged'] for r in report['sides'].values())
        and report['source_screen']['source_clearance_passed']
        and report['nominal_initial_screen']['source_clearance_passed']
        and all(r['passed'] for r in report['pad_regions'].values()))
    report['static_alignment_passed']=passed
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'alignment.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:
        print(json.dumps(dict(output=str(args.output),static_alignment_passed=False)))
        return 1
    aligned=reframe_close_initialization(payload,states,alignment=report)
    pose=dict(schema='dh116-static-pose-v1',asset_sha256=asset_sha,layers=scene.layer_pins,
        states={s:q.tolist() for s,q in states.items()},source=dict(path=str(args.initial_state),
            sha256=args.initial_state_sha256,operation=report['scope']),grasp_qualified=False)
    pose_path=args.output/'pose.json';pose_path.write_text(json.dumps(pose,indent=2)+'\n')
    init_path=args.output/'initialization.json';init_path.write_text(json.dumps(aligned,allow_nan=False))
    load_active_pose(pose_path,digest(pose_path),asset_sha,scene.layer_pins)
    load_close_initialization(init_path,digest(init_path),asset_sha256=asset_sha,contact_manifest_sha256=contact_sha)
    print(json.dumps(dict(output=str(args.output),static_alignment_passed=True,
        pose_sha256=digest(pose_path),initial_state_sha256=digest(init_path),
        palm_shifts_m={s:r['palm_delta_world_m'] for s,r in report['sides'].items()},
        max_gap_changes_m={s:r['max_gap_change_m'] for s,r in report['sides'].items()})))
    return 0


if __name__=='__main__':raise SystemExit(main())
