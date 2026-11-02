"""Read-only native 24D correction proposals at Student-visited states.

The oracle reads measured contacts/Jacobians and solves bounded velocities.
It neither installs a Teacher controller nor writes any simulation/task target.
Bounds establish kinematic validity only; labels remain candidates until a
separate native correction validation admits them for auxiliary BC.
"""
import torch
from .linear_coupling import ACTIVE,MIMIC
from .tabletop_manual_joint_contact import bounded_dls
from .tabletop_surface_contract import validate_actions


@torch.no_grad()
def correction(env,student_action,dt=1/60):
    validate_actions(student_action)
    t,w=env.task,env.world
    m=w.measure();n=t.n
    loads=m['effective_pad_loads_n'];points=m['effective_pad_points_w']
    ids=m['effective_pad_body_ids'];contact=m['effective_pad_valid']
    jac=points.new_zeros((n,2,5,3,12));jac[...,:3]=torch.eye(3,device=t.device,dtype=points.dtype)
    rows=torch.arange(n,device=t.device)
    for h,hand in enumerate(w.hands):
        fold=points.new_zeros((len(hand.joint_names),9))
        for j,k in enumerate(w.w.mounts[h].angular_ids):fold[k,j]=1
        for j,name in enumerate(ACTIVE):fold[hand.joint_names.index(name),3+j]=1
        for follower,master,ratio,_ in MIMIC:fold[hand.joint_names.index(follower),3+ACTIVE.index(master)]=ratio
        native=hand.root_physx_view.get_jacobians().to(points)
        for d in range(5):
            bid=ids[:,h,d].clamp(1,len(hand.body_names)-1)
            j=native[rows,bid-1]
            com=hand.data.body_com_pos_w.to(points)[rows,bid]
            lever=(points[:,h,d]-com)[:,None].expand(-1,j.shape[-1],-1)
            shifted=j[:,:3]+torch.linalg.cross(j[:,3:].transpose(-1,-2),lever).transpose(-1,-2)
            jac[:,h,d,:,3:]=shifted@fold
    center=w.wp[:,None,None]
    radial=center-points
    direction=radial/radial.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    # Preserve observed load allocation. Acquisition outside valid native pad
    # pairs is rejected; no fabricated contact or force measurement is used.
    total=loads.sum(-1,keepdim=True)
    target=total/contact.sum(-1,keepdim=True).clamp_min(1)
    normal=(.0006*(target-loads)).clamp(-.001,.001)
    tangent=w.wv[:,:3][:,None,None].expand(-1,2,5,-1).clone()
    tangent[...,2]+=student_action[:,[14,17]][:,:,None].clamp(0,1)*.003
    desired=tangent+direction*normal[...,None]
    a=jac*contact[...,None,None]
    b=desired*contact[...,None]
    a=a.reshape(n,2,15,12);b=b.reshape(n,2,15)
    low=points.new_full((n,2,12),-.3);high=-low
    low[:,:,:3]=((points.new_tensor([-.03,-.03,-.004])-w.palms)/dt).clamp(-.01,.01)
    high[:,:,:3]=((points.new_tensor([.03,.03,.05])-w.palms)/dt).clamp(-.01,.01)
    low[:,:,2]=torch.maximum(low[:,:,2],-(w.clearances-.0003).clamp_min(0)/dt)
    low[:,:,3:6]=((-torch.pi/2-w.wrist_angles)/dt).clamp(-.5,.5)
    high[:,:,3:6]=((torch.pi/2-w.wrist_angles)/dt).clamp(-.5,.5)
    low[:,:,6:]=((t.low.reshape(2,6)-t.command.reshape(n,2,6))/dt).clamp(-.3,.3)
    high[:,:,6:]=((t.high.reshape(2,6)-t.command.reshape(n,2,6))/dt).clamp(-.3,.3)
    if not all(bool(torch.isfinite(x).all()) for x in (a,b,low,high)) or bool((low>high).any()):
        raise ValueError('invalid actual native correction model')
    scale=points.new_tensor([.001]*3+[.02]*9)
    velocity=bounded_dls(a*scale,b,torch.zeros_like(low),low/scale,high/scale,1e-5)*scale
    velocity[:,:,:3]/=(velocity[:,:,:3].norm(dim=-1,keepdim=True)/.01).clamp_min(1)
    velocity[:,:,3:6]/=(velocity[:,:,3:6].norm(dim=-1,keepdim=True)/.5).clamp_min(1)
    action=torch.cat(((velocity[:,:,6:]/.3).flatten(1),(velocity[:,:,:3]/.01).flatten(1),
        (velocity[:,:,3:6]/.5).flatten(1)),1).clamp(-1,1)
    from .tabletop_surface_force_sharing import surface_targets
    paired=surface_targets(loads,m['effective_pad_vectors_w'],valid=contact)['paired'].all(-1)
    usable=(paired&(m['effective_pad_depth_m'].amax((1,2))<=.0008)
        &(m['rejected_contact_loads_n'].sum(-1)<=1e-6)&(t.failure==0))
    return dict(actions=action,kinematically_valid=usable,native_correction_qualified=False,
        snapshot=dict(q=m['q'].clone(),object_pose=torch.cat((w.wp,w.wq),-1).clone(),
            wrist_pose=m['wrist_pose'].clone(),loads=loads.clone(),body_ids=ids.clone()))
