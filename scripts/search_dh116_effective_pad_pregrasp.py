"""Auxiliary CAD-pad-only search, with static wrench screening after geometry.

No simulator or policy update. Candidate force vectors are predictions, never
measured forces; geometrically feasible and static-feasible remain separate.
"""
from pathlib import Path
import importlib.util,json,sys,hashlib
import numpy as np
import torch
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import _surface,SurfaceHandModel,transform
from go2_pvcnn.control.dh116.tabletop_effective_pad import _patch
from go2_pvcnn.control.dh116.tabletop_effective_pad_runtime import _on_patch
from go2_pvcnn.control.dh116.tabletop_surface_force_sharing import surface_targets
from go2_pvcnn.control.dh116.tabletop_gravity_feasibility import gravity_feasibility

ROOT=Path(sys.argv[sys.argv.index('--root')+1]) if '--root' in sys.argv else Path(__file__).resolve().parents[1]
DZ=.06;CENTER=np.array([.43,0.,.368])
spec=importlib.util.spec_from_file_location('aux_surface_base',ROOT/'scripts/generate_dh116_multifinger_pregrasp.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.CENTER=tuple(CENTER)
old_decode=base.decode;old_evaluate=base.evaluate


def decode(model,x,roots):
    shifted=roots.clone();shifted[:,2]+=DZ
    return old_decode(model,x,shifted)


def strict_samples(model,full):
    rows=[]
    for link in base.DISTALS:
        source=next(s for s in model.spec['shapes'] if s['body'].endswith('/'+link))
        atlas=_patch(source['vertices'],model.side,link)
        region={k:model.t(atlas[k]) for k in ('center_local_m','tangent_u','tangent_v','half_widths_m','planes_local')}
        region.update(surface_tolerance_m=atlas['surface_tolerance_m'],normal_cosine_min=atlas['normal_cosine_min'])
        p,n,_=_surface(np.asarray(source['vertices']),count=8192,seed=517)
        p,n=model.t(p),model.t(n)
        accepted=_on_patch(p,-n,region)
        ids=accepted.nonzero().flatten()
        if len(ids)<3:raise ValueError('no verified CAD pad samples: '+link)
        if not full:ids=ids[torch.linspace(0,len(ids)-1,min(24,len(ids))).long()]
        rows.append(dict(body=source['body'],points=p[ids],normals=n[ids]))
    # The evaluator stacks five digit clouds. Keep every sample in the same
    # audited patch while giving the clouds a common deterministic size.
    size=min(len(row['points']) for row in rows)
    for row in rows:
        take=torch.linspace(0,len(row['points'])-1,size).long()
        row['points']=row['points'][take];row['normals']=row['normals'][take]
    return rows


def evaluate(*args,**kwargs):
    g=old_evaluate(*args,**kwargs);points=g['collision_points'];clear=[]
    for y in (-.243,.243):
        d=(points-points.new_tensor([.43,y,.33])).abs()-points.new_tensor([.012,.005,.03])
        sdf=d.clamp_min(0).norm(dim=-1)+d.amax(-1).clamp_max(0)
        clear.append(sdf.amin(-1))
    closest=torch.stack(clear,-1).amin(-1)
    depth=(-closest).clamp_min(0)*1000
    g['table']=torch.maximum(g['table'],depth)
    g['minimum_clearance_m']=torch.minimum(g['minimum_clearance_m'],closest)
    weights=torch.exp(-g['patch_distance_m']/.008)
    down=(-g['hand_normals'][...,2]).clamp_min(0).square()
    g['score']=g['score']+60*depth.square()+2*(weights*down).sum(-1)/weights.sum(-1).clamp_min(1e-12)
    return g


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def screen(row,models):
    geometries=row['geometry']['per_hand']
    loads=torch.zeros(1,2,5,dtype=torch.float64);normals=torch.tensor([[g['hand_normals'] for g in geometries]],dtype=torch.float64)
    for h,g in enumerate(geometries):
        for name in g['contact_digits']:loads[0,h,base.DIGITS.index(name)]=1.
    faces=surface_targets(loads,normals)
    if not faces['paired'].all():return dict(static_feasible=False,status='unpaired_effective_surfaces',dynamic_success=False)
    points=[];norm=[];jac=[];hands=[];digits=[];labels=[]
    for h,model in enumerate(models):
        q=model.t(row['active_angles'][h*6:(h+1)*6])[None];root=model.t(row['wrist_pose_w'][h])[None]
        poses=model.fk(q,root)
        for d,name in enumerate(base.DIGITS):
            if name not in geometries[h]['contact_digits']:continue
            body=next(s['body'] for s in model.spec['shapes'] if s['body'].endswith('/'+base.DISTALS[d]))
            point=model.t(geometries[h]['contact_points'][d]);pose=poses[body][0]
            local=pose[:3,:3].T@(point-pose[:3,3]);derivative=np.zeros((3,12))
            for j in range(6):
                plus=q.clone();minus=q.clone();plus[0,j]+=1e-5;minus[0,j]-=1e-5
                p=model.fk(plus,root)[body];m=model.fk(minus,root)[body]
                derivative[:,6*h+j]=((transform(p,local[None])[0,0]-transform(m,local[None])[0,0])/2e-5).numpy()
            points.append(point.numpy());norm.append(geometries[h]['hand_normals'][d]);jac.append(derivative)
            hands.append(h);digits.append(d);labels.append(int(faces['groups'][0,h,d]))
    return gravity_feasibility(np.array(points),np.array(norm),np.array(jac),CENTER,.12,.8,np.full(12,.2),
        hand_ids=np.array(hands),digit_ids=np.array(digits),face_ids=np.array(labels))


base.decode=decode;base.distal_samples=strict_samples;base.evaluate=evaluate
if '--self-test' in sys.argv:
    base.CENTER=(.43,0.,.308)
base.main()
if '--self-test' not in sys.argv:
    path=Path(sys.argv[sys.argv.index('--output')+1]);report=json.loads(path.read_text())
    models=[SurfaceHandModel.from_usd(ROOT/'assets/fixed_dual_panda_dh116_v1/dependencies'/name,side)
        for name,side in [('d134680ab619_DH116_left.usd','left'),('3046ad2ed855_DH116_right.usd','right')]]
    original=report['candidates'];retained=[]
    for row in original:
        row['static_wrench']=jsonable(screen(row,models))
        if row['static_wrench']['static_feasible']:retained.append(row)
    report.update(geometry_candidates=original,candidates=retained,static_feasible_count=len(retained),
        geometry_feasible_count=len(original),geometry_only=True,qualified=False,
        effective_pad_search_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        support_clearance_m=DZ,static_friction_for_screen=.8,force_goal_range_n=None,
        support_collision_checked=True,effective_patch_source='same_CAD_atlas_as_native_acceptance',
        status='static_feasible_candidates' if retained else 'no_static_feasible_candidate')
    path.write_text(json.dumps(report,indent=2,allow_nan=False));print('STATIC_WRENCH_RESULT',len(retained),flush=True)
