"""Bounded multi-start native grasp optimization; outputs are not teachers."""
import math
import time
import torch
from .linear_coupling import active_limits
from .tabletop_free_wrist_grasp import compose_rotation
from .tabletop_surface_kinematics import pose_matrix,transform
from .tabletop_surface_objectives import surface_terms

LOWER=(-.03,-.03,-.004)
UPPER=(.03,.03,.05)
CONTRACT=dict(version='dh116_multisurface_candidates_v1',active_per_hand=6,
    variables_per_hand='6_active_angles_3_external_translations_3_external_angles',
    follower_coordinates='existing_exact_affine_mimic',qualified=False,
    optimization_proxy='inscribed_48_vertex_hulls;full_planes_rechecked_after_optimization',
    table_clearance_m=.0003,geometry_only_is_contact=False)


def encode_variables(q,offset,angles):
    low,high=(q.new_tensor(x) for x in active_limits())
    lo=q.new_tensor(LOWER);hi=q.new_tensor(UPPER)
    return torch.cat((2*(q-low)/(high-low)-1,2*(offset.to(q)-lo)/(hi-lo)-1,angles.to(q)/(math.pi/2)),-1)


def decode_variables(x,seed_roots):
    if x.ndim!=3 or tuple(x.shape[1:])!=(2,12) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('finite N x 2 x 12 normalized search variables required')
    seed_roots=seed_roots.to(x)
    if seed_roots.shape!=(2,7):raise ValueError('two seed wrist poses required')
    x=x.clamp(-1,1);low,high=(x.new_tensor(a) for a in active_limits())
    q=low+(x[:,:,:6]+1)/2*(high-low)
    lo=x.new_tensor(LOWER);hi=x.new_tensor(UPPER)
    offset=lo+(x[:,:,6:9]+1)/2*(hi-lo);angles=x[:,:,9:]*(math.pi/2)
    root=seed_roots[None].expand(len(x),-1,-1)
    pose=torch.cat((root[:,:,:3]+offset,compose_rotation(root[:,:,3:],angles)),-1)
    return q,pose,offset,angles


def rank_diverse(rows,maximum=8):
    selected=[]
    for row in sorted(rows,key=lambda x:x['score']):
        q=torch.tensor(row['active_angles'],dtype=torch.float64)
        root=torch.tensor(row['wrist_pose_w'],dtype=torch.float64)
        duplicate=False
        for other in selected:
            other_q=torch.tensor(other['active_angles'],dtype=torch.float64)
            other_root=torch.tensor(other['wrist_pose_w'],dtype=torch.float64)
            near_q=(q-other_q).abs().max()<.035
            near_p=(root[:,:3]-other_root[:,:3]).norm(dim=-1).max()<.001
            alignment=(root[:,3:]*other_root[:,3:]).sum(-1).abs().clamp(0,1)
            near_r=(2*torch.acos(alignment)).max()<.08
            if near_q and near_p and near_r:duplicate=True;break
        if not duplicate:selected.append(dict(row,qualified=False))
        if len(selected)>=maximum:break
    return selected


def object_surface_points(center,quaternion,radius=.008,length=.5):
    angle=torch.arange(24,device=center.device,dtype=center.dtype)*(2*math.pi/24)
    y=torch.linspace(-length/2,length/2,129,device=center.device,dtype=center.dtype)
    theta,axial=torch.meshgrid(angle,y,indexing='ij')
    points=torch.stack((radius*theta.cos(),axial,radius*theta.sin()),-1).flatten(0,1)
    return transform(pose_matrix(torch.cat((center,quaternion),-1)),points)


def evaluate(models,x,seed_roots,center,quaternion,*,fast=True,include_self=True):
    q,pose,offset,angles=decode_variables(x,seed_roots)
    samples=[model.sample(q[:,h],pose[:,h]) for h,model in enumerate(models)]
    objects=object_surface_points(center,quaternion)
    inside=torch.stack([m.object_penetration(q[:,h],pose[:,h],objects,fast=fast) for h,m in enumerate(models)],-1).amax(-1)
    self_depth=(torch.stack([m.self_penetration(q[:,h],pose[:,h],fast=fast) for h,m in enumerate(models)],-1).amax(-1)
        if include_self else q.sum((1,2))*0)
    terms=surface_terms(torch.stack([s['patches'] for s in samples],1),
        torch.cat([s['collision_points'] for s in samples],1),center,quaternion,
        patch_normals=torch.stack([s['patch_normals'] for s in samples],1),
        self_penetration_m=self_depth,object_hand_penetration_m=inside)
    # Encourage opposed contacts, without prescribing an axial height or pose.
    patch_centers=torch.stack([s['patches'].mean(2) for s in samples],1)
    qw,qx,qy,qz=quaternion.unbind(-1)
    axis=torch.stack((2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)),-1)
    relative=patch_centers-center[:,None,None]
    radial=relative-(relative*axis[:,None,None]).sum(-1,keepdim=True)*axis[:,None,None]
    direction=radial/radial.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    opposition=((direction[:,:,0]*direction[:,:,1]).sum(-1)+.8).clamp_min(0).mean(-1)
    score=(10*terms['reach']+terms['normal_alignment']+2*opposition
        +30*(terms['penetration']+terms['object_in_hand']+terms['self'])+40*terms['table'])
    terms.update(opposition=opposition,score=score,root_pose=pose,active=q,
        minimum_clearance_m=torch.cat([s['collision_points'][:,:,2] for s in samples],1).amin(-1)-.30)
    return terms


def optimize(models,seed_roots,seed_active,object_pose,*,starts=24,steps=240,maximum=8,wall_seconds=240.,seed=17,progress=None):
    if starts<2 or steps<1:raise ValueError('multiple starts and a positive step budget required')
    device=models[0].device;dtype=models[0].dtype
    roots=torch.as_tensor(seed_roots,device=device,dtype=dtype)
    active=torch.as_tensor(seed_active,device=device,dtype=dtype).reshape(2,6)
    obj=torch.as_tensor(object_pose,device=device,dtype=dtype)
    generator=torch.Generator(device=device);generator.manual_seed(seed)
    initial=encode_variables(active,torch.zeros(2,3,device=device,dtype=dtype),torch.zeros(2,3,device=device,dtype=dtype))
    x=initial[None].repeat(starts,1,1)
    x[:,:,0:3]+=torch.randn(starts,2,3,device=device,dtype=dtype,generator=generator)*.35
    x[:,:,3:6]+=torch.randn(starts,2,3,device=device,dtype=dtype,generator=generator)*.12
    x[:,:,6:8]+=torch.randn(starts,2,2,device=device,dtype=dtype,generator=generator)*.15
    x[:,:,8]+=.1*torch.rand(starts,2,device=device,dtype=dtype,generator=generator)
    x[:,:,9:]=torch.randn(starts,2,3,device=device,dtype=dtype,generator=generator)*.2
    x[0]=initial;x=torch.nn.Parameter(x.clamp(-1,1))
    optimizer=torch.optim.Adam([x],lr=.012)
    center=obj[None,:3].expand(starts,-1);quat=obj[None,3:].expand(starts,-1)
    best=torch.full((starts,),float('inf'),device=device,dtype=dtype);best_x=x.detach().clone()
    begin=time.monotonic();completed=0
    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        terms=evaluate(models,x,roots,center,quat,fast=True,include_self=True)
        loss=terms['score'];better=loss<best
        best=torch.minimum(best,loss.detach());best_x=torch.where(better[:,None,None],x.detach(),best_x)
        loss.sum().backward()
        if x.grad is None or not torch.isfinite(x.grad).all():raise RuntimeError('invalid surface optimization gradient')
        torch.nn.utils.clip_grad_norm_([x],10.);optimizer.step()
        with torch.no_grad():x.clamp_(-1,1)
        completed=iteration+1
        if progress is not None and iteration%20==0:
            progress(dict(iteration=completed,best_score=float(best.min()),elapsed_s=time.monotonic()-begin))
        if time.monotonic()-begin>=wall_seconds:break
    # Rank with the full original convex halfspaces, never the inner proxy.
    with torch.no_grad():final=evaluate(models,best_x,roots,center,quat,fast=False,include_self=True)
    rows=[]
    for i in range(starts):
        geometry={key:final[key][i].cpu().tolist() for key in ('reach','penetration','table','normal_alignment','self','object_in_hand','patch_distance_m','minimum_clearance_m','opposition')}
        feasible=(max(final['patch_distance_m'][i].flatten())<=.0008
            and final['penetration'][i]<=.3 and final['object_in_hand'][i]<=.3 and final['self'][i]<=.3
            and final['minimum_clearance_m'][i]>=.00029)
        rows.append(dict(candidate_index=i,score=float(final['score'][i]),qualified=False,
            geometry_feasible=bool(feasible),geometry=geometry,
            active_angles=final['active'][i].flatten().cpu().tolist(),wrist_pose_w=final['root_pose'][i].cpu().tolist(),
            normalized_variables=best_x[i].cpu().tolist()))
    rows.sort(key=lambda row:(not row['geometry_feasible'],row['score']))
    feasible_rows=[row for row in rows if row['geometry_feasible']]
    candidates=rank_diverse(feasible_rows or rows,maximum=maximum)
    return dict(contract=CONTRACT,iterations=completed,starts=starts,elapsed_s=time.monotonic()-begin,
                geometry_feasible_count=len(feasible_rows),qualified=False,candidates=candidates,all_candidates=rows)
