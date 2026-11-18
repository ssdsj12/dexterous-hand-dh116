"""Offline feasibility search; writes diagnostics only, never a training seed."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from build_dh116_below_axis_initialization import load_shapes, world_shapes, projected_distance
from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS

project=Path('/root/hand')
seed=json.loads((project/'configs/dh116_training/grasp_initialization_below_axis_v1.json').read_text())
manifest=json.loads((project/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
wire=np.array(seed['wire_pose'][:3])
report={}
for index,side in enumerate(('left','right')):
    source=Path(next(s['path'] for s in manifest['sources'] if s['kind']=='dh116' and s['side']==side))
    model=HandPointFK(source,side,seed['roots'][index])
    q=np.array(seed['joint_positions'][index])[[JOINTS.index(j) for j in ACTIVE]]
    shapes,_=load_shapes(source,model,q)
    clouds=[v[ConvexHull(v).vertices] for _,v in world_shapes(model,q,shapes)]
    pads=model.points(q); pivot=pads.mean(0)
    def moved(x):
        rot=Rotation.from_rotvec(np.r_[x[:2],0.]).as_matrix()
        shift=np.array([x[2]*.001,0.,wire[2]-.0035-pivot[2]])
        return [(v-pivot)@rot.T+pivot+shift for v in clouds],(pads-pivot)@rot.T+pivot+shift
    def metrics(x):
        vertices,p=moved(x)
        clearance=min(v[:,2].min() for v in vertices)-.30
        gaps=[]
        for v in vertices:
            poly=v[:,[0,2]]-wire[[0,2]]
            poly=poly[ConvexHull(poly).vertices]
            gaps.append(projected_distance(poly)-.008)
        z=p[:,2]-wire[2]
        return np.r_[(clearance-.0003)*1000,(min(gaps)-.00005)*1000,
                     (z+.004)*1000,(-.003-z)*1000]
    best=None
    for rx in (-.5,-.25,0.,.25,.5):
        fit=minimize(lambda x: float(x[:2]@x[:2]+.002*x[2]**2),[rx,0.,0.],
                     method='SLSQP',bounds=[(-.65,.65),(-.10,.10),(-3.,3.)],
                     constraints=[dict(type='ineq',fun=metrics)],options=dict(maxiter=150,ftol=1e-10))
        valid=bool(metrics(fit.x).min()>=-1e-5)
        if best is None or (valid and not best['valid']) or (valid==best['valid'] and fit.fun<best['objective']):
            v,p=moved(fit.x)
            best=dict(valid=valid,objective=float(fit.fun),x=fit.x.tolist(),
                      constraint_margins_mm=metrics(fit.x).tolist(),pad_z_mm=((p[:,2]-wire[2])*1000).tolist(),message=fit.message)
    report[side]=best
    print(side,json.dumps(best),flush=True)
(project/'deep_grasp_geometry_probe.json').write_text(json.dumps(report,indent=2))
