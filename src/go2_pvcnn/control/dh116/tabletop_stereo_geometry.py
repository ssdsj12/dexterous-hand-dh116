"""Optical pixel centres -> world -> actual palm FK, with explicit missing masks."""
import torch


def backproject(depth_m,K,valid):
    z=torch.as_tensor(depth_m)
    if z.ndim!=3 or not z.is_floating_point() or valid.shape!=z.shape or valid.dtype!=torch.bool:
        raise ValueError('depth/valid requires float/bool [B,H,W]')
    n,h,w=z.shape;k=torch.as_tensor(K,device=z.device,dtype=z.dtype)
    if k.shape==(3,3):k=k[None].expand(n,-1,-1)
    if k.shape!=(n,3,3) or not torch.isfinite(k).all() or (torch.linalg.det(k).abs()<1e-12).any():
        raise ValueError('finite invertible per-batch K required')
    y,x=torch.meshgrid(torch.arange(h,device=z.device,dtype=z.dtype)+.5,torch.arange(w,device=z.device,dtype=z.dtype)+.5,indexing='ij')
    pixel=torch.stack((x,y,torch.ones_like(x)),-1).reshape(1,h*w,3).expand(n,-1,-1)
    rays=pixel@torch.linalg.inv(k).transpose(-1,-2)
    p=(rays*z.reshape(n,-1,1)).reshape(n,h,w,3)
    good=valid&torch.isfinite(z)&(z>0)
    return torch.where(good[...,None],p,float('nan'))


def transform_points(points,T_world_camera):
    p=torch.as_tensor(points)
    if p.ndim<3 or p.shape[-1]!=3 or not p.is_floating_point():raise ValueError('batched 3D points required')
    n=p.shape[0];t=torch.as_tensor(T_world_camera,device=p.device,dtype=p.dtype)
    if t.shape==(4,4):t=t[None].expand(n,-1,-1)
    if t.shape!=(n,4,4) or not torch.isfinite(t).all():raise ValueError('finite world transform required')
    if not torch.allclose(t[:,3],t.new_tensor([0,0,0,1]).expand(n,-1),atol=1e-6,rtol=0.):raise ValueError('homogeneous transform required')
    r=t[:,:3,:3]
    if (not torch.allclose(r.transpose(-1,-2)@r,torch.eye(3,device=p.device,dtype=p.dtype).expand(n,-1,-1),atol=1e-5,rtol=0.)
            or not torch.allclose(torch.linalg.det(r),torch.ones(n,device=p.device,dtype=p.dtype),atol=1e-5,rtol=0.)):
        raise ValueError('rigid rotation required')
    flat=p.reshape(n,-1,3)@t[:,:3,:3].transpose(-1,-2)+t[:,None,:3,3]
    return flat.reshape_as(p)


def world_to_palms(points,palm_poses):
    p=torch.as_tensor(points);pose=torch.as_tensor(palm_poses,device=p.device,dtype=p.dtype)
    if p.ndim<3 or p.shape[-1]!=3 or not p.is_floating_point() or pose.shape!=(p.shape[0],2,7) or not torch.isfinite(pose).all():
        raise ValueError('world points and actual two-palm FK xyz+wxyz required')
    q=pose[...,3:];scale=q.abs().amax(-1,keepdim=True)
    if (scale<=0).any():raise ValueError('nonzero palm quaternion required')
    q=q/scale;q=q/q.norm(dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1)
    rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),
                          2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,2,3,3)
    flat=p.reshape(p.shape[0],-1,3)
    local=(flat[:,None]-pose[:,:,None,:3])@rotation
    return local.reshape(p.shape[0],2,*p.shape[1:])


def order_polyline(points,valid,adjacency):
    """Order an explicitly associated simple path; never invent hidden cable shape.

    adjacency must come from sensor segmentation/skeleton association. Branches,
    loops and disconnected observed fragments raise instead of claiming a full
    centreline. Padding stays NaN/false. No rigid-axis sorting or truth fallback.
    """
    p=torch.as_tensor(points)
    if (p.ndim!=3 or p.shape[-1]!=3 or valid.shape!=p.shape[:2] or valid.dtype!=torch.bool
            or adjacency.shape!=(*p.shape[:2],p.shape[1]) or adjacency.dtype!=torch.bool):
        raise ValueError('points/mask/simple path topology shape mismatch')
    n,k,_=p.shape
    out=torch.full_like(p,float('nan'));mask=torch.zeros_like(valid)
    order=torch.full((n,k),-1,device=p.device,dtype=torch.long)
    arc=torch.full((n,k),float('nan'),device=p.device,dtype=p.dtype)
    for row in range(n):
        keep=valid[row].nonzero(as_tuple=True)[0]
        if not len(keep):continue
        if not torch.isfinite(p[row,keep]).all():raise ValueError('valid path points must be finite')
        a=adjacency[row][keep][:,keep]
        if (a.diag().any() or not torch.equal(a,a.T) or (adjacency[row][~valid[row]].any())
                or adjacency[row][:,~valid[row]].any()):raise ValueError('invalid sensor path topology')
        degree=a.sum(1)
        if len(keep)==1:walk=[0]
        else:
            ends=(degree==1).nonzero(as_tuple=True)[0]
            if len(ends)!=2 or (degree>2).any() or (degree==0).any():raise ValueError('branch/loop/disconnected path topology')
            walk=[int(ends[0])]
            while len(walk)<len(keep):
                nexts=[int(v) for v in a[walk[-1]].nonzero(as_tuple=True)[0] if int(v) not in walk]
                if len(nexts)!=1:raise ValueError('disconnected path topology')
                walk.append(nexts[0])
        ids=keep[torch.tensor(walk,device=p.device)]
        count=len(ids);ordered=p[row,ids]
        out[row,:count]=ordered;mask[row,:count]=True;order[row,:count]=ids
        arc[row,0]=0.
        if count>1:arc[row,1:count]=(ordered[1:]-ordered[:-1]).norm(dim=-1).cumsum(0)
    return dict(points=out,valid=mask,order=order,arc_length_m=arc,association_valid=mask.clone())
