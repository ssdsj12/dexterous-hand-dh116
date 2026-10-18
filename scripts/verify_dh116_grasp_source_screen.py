"""Recompute pinned DH116 candidate IK/source clearance; never command motion."""
import argparse,json,hashlib
from pathlib import Path
from itertools import product
import numpy as np
from pxr import Usd
from go2_pvcnn.control.dh116.collision_scene import SourceCollisionScene
from go2_pvcnn.control.dh116.grasp_geometry import cylinder_residual
from go2_pvcnn.control.dh116.arm_ik import solve_arm_pose
from go2_pvcnn.control.dh116.usd_surface import read_surface
from go2_pvcnn.control.dh116.contact_frames import surface_reference
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--asset',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--grasp-span-m',type=float,default=.4)
args=p.parse_args();asset=args.asset;input_path=args.candidates;out=args.output
if out.exists():raise ValueError('refusing existing evidence')
if not np.isfinite(args.grasp_span_m) or not .2<=args.grasp_span_m<=.42:raise ValueError('unsupported grasp span')
stage=Usd.Stage.Open(str(asset));scene=SourceCollisionScene(stage)
source=json.loads(input_path.read_text());candidates=source['candidates'] if isinstance(source,dict) else source
wire=np.c_[np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.308)]
for c in candidates:
    side=c['side'];model=scene.models[side];q=np.array(c['ik']['q']);poses=model.fk(q)[0]
    centers=[]
    for body,ref in zip(('finger14_link','finger23_link'),c['references'],strict=True):
        path=model.bodies[body];mesh=read_surface(stage,path)
        triangle=np.asarray(mesh['points_body'])[np.asarray(mesh['triangles'])[ref['triangle_index']]]
        point,normal=surface_reference(triangle,ref['weights'])
        if not np.allclose(point,ref['point'],atol=1e-9,rtol=0) or not np.allclose(normal,ref['normal'],atol=1e-7,rtol=0):
            raise ValueError('candidate reference no longer matches source surface')
        if np.max((scene.geometry[path]['vertices']-point)@normal)>1e-4:
            raise ValueError('source point is not a supporting front surface')
        t=poses[path];centers.append(t[:3,:3]@(point+.008*normal)+t[:3,3])
    old_center=np.mean(centers,axis=0);desired_y=args.grasp_span_m/2*(1 if side=='left' else -1)
    target=np.array(c['target']);target[:3,3]+=np.array([.43,desired_y,.308])-old_center
    bounds=scene.limits[side];seed=c['ik']['q'][:7]
    for attempt in range(3):
        fit=solve_arm_pose(scene.models[side],target,c['hand'],seed,bounds[:,0],bounds[:,1]);seed=fit['q'][:7]
        if fit['converged']:break
    c['target']=target.tolist();c['ik']=fit;c['grasp_center_y']=desired_y
records=[]
for li,ri in product([i for i,c in enumerate(candidates) if c['side']=='left'],[i for i,c in enumerate(candidates) if c['side']=='right']):
    pair=[candidates[li],candidates[ri]];states={c['side']:c['ik']['q'] for c in pair}
    if not all(c['ik']['converged'] for c in pair):continue
    for c in pair:
        k=scene.models[c['side']];poses=k.fk(states[c['side']])[0];points=[];normals=[]
        for body,ref in zip(('finger14_link','finger23_link'),c['references']):
            t=poses[k.bodies[body]];points.append(t[:3,:3]@ref['point']+t[:3,3]);normals.append(t[:3,:3]@ref['normal'])
        error=cylinder_residual(points,normals,.008,c['opposition'])
        if np.max(np.abs(error))>1e-5:raise ValueError('actual IK contact mismatch')
    r=scene.check(states,wire,permit_pad_contact=True)
    records.append(dict(indices=[li,ri],states=states,result=r))
    print('PAIR',li,ri,json.dumps(r),flush=True)
out.mkdir()
report=dict(asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),layers=scene.layer_pins,
    input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),candidates=candidates,pairs=records,
    motion_qualified=False,grasp_qualified=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
