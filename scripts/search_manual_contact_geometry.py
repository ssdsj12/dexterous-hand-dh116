"""Offline convex geometry search; fixed palms' orientation, bounded finger pose.

This is NOT a grasp success or force validation. It searches the collision
surfaces' closest points, rather than the source's selected pad reference.
"""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from build_dh116_below_axis_initialization import load_shapes,world_shapes
from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
from go2_pvcnn.control.dh116.linear_coupling import active_limits


def closest(polygon):
    edge=np.roll(polygon,-1,axis=0)-polygon
    t=np.clip(-(polygon*edge).sum(1)/(edge*edge).sum(1),0.,1.)
    p=polygon+t[:,None]*edge
    cross=edge[:,0]*-polygon[:,1]-edge[:,1]*-polygon[:,0]
    if (cross>=-1e-14).all() or (cross<=1e-14).all():return np.zeros(2)
    return p[np.linalg.norm(p,axis=1).argmin()]


parser=argparse.ArgumentParser();parser.add_argument('--joint-budget-rad',type=float,default=.08)
parser.add_argument('--wrist-range-mm',type=float,default=3.)
parser.add_argument('--report-name',default='manual_joint_geometry_search.json')
options=parser.parse_args()
root=Path('/root/hand');seed=json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json').read_text())
manifest=json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
wire=np.array(seed['wire_pose'][:3]);low,high=active_limits();rng=np.random.default_rng(20261029)
report=dict(physics_qualified=False,learning_performed=False,palm_orientation_changed=False,
            joint_budget_rad=options.joint_budget_rad,wrist_range_mm=options.wrist_range_mm,sides={})
for hi,side in enumerate(('left','right')):
    source=Path(next(s['path'] for s in manifest['sources'] if s['kind']=='dh116' and s['side']==side))
    fk=HandPointFK(source,side,seed['roots'][hi]);reference=np.array(seed['grasp_command'][hi*6:hi*6+6])
    shapes,_=load_shapes(source,fk,reference)
    for s in shapes:s['vertices']=s['vertices'][ConvexHull(s['vertices']).vertices]
    thumb=next(i for i,s in enumerate(shapes) if s['body'].endswith('/finger14_link'))
    index=next(i for i,s in enumerate(shapes) if s['body'].endswith('/finger23_link'))
    cache={}
    def metrics(x):
        key=tuple(x)
        if key in cache:return cache[key]
        q=reference.copy();q[:3]+=x[:3];q=q.clip(low,high)
        shifted=[v+x[3:]*.001 for _,v in world_shapes(fk,q,shapes)]
        gaps=[];contact_z=[];points=[]
        for v in shifted:
            p=v[:,[0,2]]-wire[[0,2]]
            poly=p[ConvexHull(p).vertices];point=closest(poly);distance=np.linalg.norm(point)
            gaps.append(distance-.008)
            contact_z.append(.008*point[1]/max(distance,1e-9))
            points.append(point.tolist())
        other=[g for i,g in enumerate(gaps) if i not in (thumb,index)]
        out=dict(gaps=np.array(gaps)[[thumb,index]],z=np.array(contact_z)[[thumb,index]],
                 clearance=min(v[:,2].min() for v in shifted)-.30,other=min(other),points=points,
                 q=q)
        cache[key]=out;return out
    def constraints(x):
        m=metrics(x)
        return np.r_[(m['clearance']-.0003)*1000,(m['other']+.00002)*1000,
                     (m['gaps']+.00020)*1000,(-.00010-m['gaps'])*1000,
                     (m['z']+.0023)*1000,(-.0017-m['z'])*1000]
    budget=options.joint_budget_rad;wr=options.wrist_range_mm
    bounds=list(zip(np.maximum(-budget,low[:3]-reference[:3]),np.minimum(budget,high[:3]-reference[:3])))+[(-wr,wr),(-.01,.01),(-4.,wr)]
    candidates=[]
    starts=[np.zeros(6)]+[np.r_[rng.uniform(-.03,.03,3),rng.uniform(-1.,1.),0.,rng.uniform(-1.,1.)] for _ in range(7)]
    for start in starts:
        fit=minimize(lambda x:float((x[:3]/budget)@(x[:3]/budget)+.01*(x[3:]@x[3:])),start,
                     bounds=bounds,constraints=[dict(type='ineq',fun=constraints)],method='SLSQP',
                     options=dict(maxiter=150,ftol=1e-8))
        m=metrics(fit.x);margin=float(constraints(fit.x).min());valid=margin>=-1e-5
        row=dict(feasible=valid,minimum_constraint_margin_mm=margin,objective=float(fit.fun),
                 message=str(fit.message),joint_delta_rad=fit.x[:3].tolist(),joint_command_rad=m['q'].tolist(),
                 palm_offsets_m=(fit.x[3:]*.001).tolist(),target_collision_gap_mm=(m['gaps']*1000).tolist(),
                 target_contact_radial_z_mm=(m['z']*1000).tolist(),table_clearance_mm=m['clearance']*1000,
                 other_link_gap_mm=m['other']*1000)
        candidates.append(row)
        print('GEOMETRY_CASE',side,json.dumps(row),flush=True)
        if valid:break
    best=sorted(candidates,key=lambda r:(not r['feasible'],-r['minimum_constraint_margin_mm'] if not r['feasible'] else r['objective']))[0]
    report['sides'][side]=dict(best=best,candidates=candidates)
(root/'runs'/options.report_name).write_text(json.dumps(report,indent=2))
