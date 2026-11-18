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

ROOT=Path(__file__).resolve().parents[1];DZ=.06;CENTER=np.array([.43,0.,.368])
UPWARD_BIAS='--upward-bias' in sys.argv
if UPWARD_BIAS:sys.argv.remove('--upward-bias')
warm_path=None
if '--warm-from' in sys.argv:
    wi=sys.argv.index('--warm-from');warm_path=Path(sys.argv[wi+1]);del sys.argv[wi:wi+2]
warm=json.loads(warm_path.read_text()) if warm_path else None
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


import itertools
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import pose_matrix

def center_samples(model,full):
    result=[]
    for link in base.DISTALS:
        source=next(s for s in model.spec['shapes'] if s['body'].endswith('/'+link))
        atlas=_patch(source['vertices'],model.side,link)
        p=model.t(atlas['center_local_m']);n=model.t(atlas['reference_outward_normal_local'])
        # Three identical geometric references satisfy the existing evaluator
        # layout; they never stand in for three actual contacts or forces.
        result.append(dict(body=source['body'],points=p[None].repeat(3,1),normals=n[None].repeat(3,1)))
    return result

def center_decode(model,x,roots):
    if warm is not None:
        if not hasattr(model,'warm_delta'):
            h=0 if model.side=='left' else 1
            target=model.t([row['normalized_variables'] for row in warm['hand_diagnostics'][h]])
            if target.shape!=x.shape:raise ValueError('warm start count must match initial search')
            model.warm_delta=target-x.detach()
        z=(x+model.warm_delta).clamp(-1,1)
    else:z=x.clamp(-1,1)
    model.effective_variables=z
    q=model.lower+(z[:,:6]+1)/2*(model.upper-model.lower)
    offset=.020*z[:,6:9];angles=z[:,9:]
    angles=angles/torch.maximum(angles.norm(dim=-1,keepdim=True),torch.ones_like(angles[:,:1]))
    shifted=roots.clone();shifted[:,2]+=DZ
    quat=compose_rotation(shifted[:,3:],angles)
    r0=pose_matrix(shifted)[:,:3,:3]
    rn=pose_matrix(torch.cat((shifted[:,:3],quat),-1))[:,:3,:3]
    delta=rn@r0.transpose(-1,-2)
    pivot=shifted[:,:3].clone();pivot[:,0]=.43;pivot[:,2]=.368
    position=pivot+torch.einsum('nij,nj->ni',delta,shifted[:,:3]-pivot)+offset
    pose=torch.cat((position,quat),-1)
    return q,pose,offset,angles

def center_evaluate(*args,**kwargs):
    g=evaluate(*args,**kwargs);near=g['patch_distance_m'];ndot=g['patch_normal_dot'];costs=[]
    for a,b in itertools.combinations(range(5),2):
        rd=(g['radial_directions'][:,a]*g['radial_directions'][:,b]).sum(-1)
        nd=(g['hand_normals'][:,a]*g['hand_normals'][:,b]).sum(-1)
        score=30*(near[:,[a,b]]/.008).square().mean(-1)+6*(1+ndot[:,[a,b]]).square().mean(-1)
        score=score+3*(rd+.98).clamp_min(0).square()+3*(nd+.98).clamp_min(0).square()
        if UPWARD_BIAS:
            below=(g['contact_points'][:,[a,b],2]-(.368-.002))/.008
            down=(-g['hand_normals'][:,[a,b],2]).clamp_min(0)
            score=score+2.5*below.square().mean(-1)+6*down.square().mean(-1)
        else:
            above=(g['contact_points'][:,[a,b],2]-.368).clamp_min(0)/.008
            score=score+.4*above.square().mean(-1)
        costs.append(score)
    grasp,subset=torch.stack(costs,-1).min(-1)
    collision=35*((g['penetration']-.08).clamp_min(0).square()+(g['object_in_hand']-.08).clamp_min(0).square()
                  +(g['self']-.08).clamp_min(0).square())+60*g['table'].square()
    g.update(score=grasp+collision,grasp_score=grasp,selected_subset=subset,
             effective_variables=args[0].effective_variables)
    return g

def center_check(g):
    eligible=[i for i in range(5) if g['patch_distance_m'][i]<=.0005 and g['patch_normal_dot'][i]<=-.9]
    pairs=[];reasons=[]
    for a,b in itertools.combinations(eligible,2):
        rd=float(np.dot(g['radial_directions'][a],g['radial_directions'][b]))
        nd=float(np.dot(g['hand_normals'][a],g['hand_normals'][b]))
        if rd<=-.65 and nd<=-.65:pairs.append(dict(digits=[base.DIGITS[a],base.DIGITS[b]],radial_dot=rd,normal_dot=nd))
    for key in ('penetration','object_in_hand','self'):
        if g[key]>.3:reasons.append(key+'_exceeds_0.3mm')
    if g['minimum_clearance_m']<.00029:reasons.append('clearance_below_0.29mm')
    if len(eligible)<2:reasons.append('fewer_than_two_central_aligned_distals')
    if not pairs:reasons.append('no_opposed_central_pad_pair')
    return not reasons,[base.DIGITS[i] for i in eligible],pairs,reasons

def self_test():
    g=dict(penetration=.1,object_in_hand=.1,self=0,minimum_clearance_m=.01,
        patch_distance_m=[.0003,.1,.0003,.1,.1],patch_normal_dot=[-1.]*5,
        radial_directions=[[1,0,0],[1,0,0],[-1,0,0],[1,0,0],[1,0,0]],
        hand_normals=[[-1,0,0],[-1,0,0],[1,0,0],[-1,0,0],[-1,0,0]])
    assert center_check(g)[0]
    g['patch_distance_m'][2]=.000501;assert not center_check(g)[0]
    g['patch_distance_m'][2]=.0003;g['patch_normal_dot'][0]=-.89;assert not center_check(g)[0]
    g['patch_normal_dot'][0]=-1.;g['penetration']=.30001;assert not center_check(g)[0]
    print('CENTER_SELF_TEST_PASS: two_or_more_actual_source_center_references;strict_geometry_and_normals')

base.decode=center_decode;base.distal_samples=center_samples;base.evaluate=center_evaluate
base.contact_check=center_check;base.self_test=self_test
old_as_row=base.as_hand_row
def as_row(g,j,*args,**kwargs):
    record=old_as_row(g,j,*args,**kwargs)
    record['normalized_variables']=g['effective_variables'][j].detach().cpu().tolist()
    return record
base.as_hand_row=as_row
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
        auxiliary_harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        support_clearance_m=DZ,static_friction_for_screen=.8,force_goal_range_n=None,
        support_collision_checked=True,effective_patch_source='same_CAD_atlas_as_native_acceptance',
        status='static_feasible_candidates' if retained else 'no_static_feasible_candidate')
    report['contract'].update(any_three_of_five=False,minimum_aligned_digits=2,
        geometric_targets='exact_CAD_pad_centroids;not_nearest_boundary_samples',
        wrist_local_rotation_vector_norm_rad=1.,contact_distance_m=.0005,contact_hand_object_normal_dot_max=-.9,
        wrist_rotation_pivot='rod_axis_at_reset_wrist_Y;not_wrist_base',
        translation_per_axis_m=.02,root_displacement_includes_pivot_rotation=True)
    report['contract']['upward_soft_preference']=UPWARD_BIAS
    report['warm_from_sha256']=hashlib.sha256(warm_path.read_bytes()).hexdigest() if warm_path else None
    path.write_text(json.dumps(report,indent=2,allow_nan=False));print('STATIC_WRENCH_RESULT',len(retained),flush=True)
