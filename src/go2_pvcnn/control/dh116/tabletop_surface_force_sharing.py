"""Allocation by opposing grasp surfaces. No fixed total-force goal."""
import torch

CONTRACT=dict(version='opposed_surface_sharing_v1',force_target_range_n=None,
    total_force_source='current_measured_effective_pad_load;policy_explores_joint_targets',
    allocation='per_hand_two_opposed_surfaces_half_each;each_surface_equal_among_bearing_digits',
    digit_names_fixed=False,contact_threshold_n=.05,minimum_opposition_cosine=-.25,
    same_face_max_min_ratio=1.5,opposed_face_max_min_ratio=1.5,
    no_contacts_are_not_balanced=True,measurement_mutation=False)


def surface_targets(loads,normals,*,valid=None,previous_axis=None):
    """Allocate [...,5] real pad forces via [...,5,3] measured surface normals.

    Face labels are interchangeable. Their axis follows the most opposed
    bearing pair, with sign continuity when the previous axis is available.
    Every bearing contact must be assigned; ambiguous or single-face rows are
    returned unchanged and never count as a balanced grasp. Subthreshold loads
    stay unchanged so redistribution never creates a fixed pressure request.
    """
    if (not isinstance(loads,torch.Tensor) or not loads.is_floating_point()
            or loads.ndim<1 or loads.shape[-1]!=5 or normals.shape!=(*loads.shape,3)
            or normals.dtype!=loads.dtype or normals.device!=loads.device
            or not torch.isfinite(loads).all() or (loads<0).any()
            or not torch.isfinite(normals).all()):
        raise ValueError('finite matching nonnegative five-digit loads and normals required')
    if valid is None:valid=torch.ones_like(loads,dtype=torch.bool)
    if valid.shape!=loads.shape or valid.dtype!=torch.bool or valid.device!=loads.device:
        raise ValueError('matching boolean effective-contact validity required')
    length=normals.norm(dim=-1)
    unit=normals/length.clamp_min(1e-12)[...,None]
    bearing=(loads>.05)&valid
    usable=bearing&(length>1e-12)
    dot=unit@unit.transpose(-1,-2)
    pairs=usable[..., :,None]&usable[...,None,:]
    score=dot.masked_fill(~pairs,2.)
    selected=score.flatten(-2).argmin(-1)
    a=selected//5;b=selected%5
    take=lambda ids:unit.gather(-2,ids[...,None,None].expand(*ids.shape,1,3)).squeeze(-2)
    axis=take(a)-take(b);axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    major=axis.abs().argmax(-1,keepdim=True)
    sign=torch.where(axis.gather(-1,major).squeeze(-1)<0,-1.,1.)
    if previous_axis is not None:
        if (previous_axis.shape!=axis.shape or previous_axis.dtype!=axis.dtype
                or previous_axis.device!=axis.device or not torch.isfinite(previous_axis).all()):
            raise ValueError('matching finite previous surface axis required')
        known=previous_axis.norm(dim=-1)>1e-12
        sign=torch.where(known,torch.where((axis*previous_axis).sum(-1)<0,-1.,1.),sign)
    axis=axis*sign[...,None]
    projected=(unit*axis[...,None,:]).sum(-1)
    assigned=usable&(projected.abs()>.2)
    groups=torch.where(assigned,(projected<0).long(),-1)
    masks=torch.stack((groups==0,groups==1),-2)
    counts=masks.sum(-1)
    face_loads=(masks*loads[...,None,:]).sum(-1)
    opposed=score.flatten(-2).amin(-1)<=-.25
    paired=opposed&(counts>0).all(-1)&(~bearing|assigned).all(-1)
    total=face_loads.sum(-1)
    face_per_digit=(total[...,None]/2)/counts.clamp_min(1)
    proposed=(masks*face_per_digit[..., :,None]).sum(-2)
    desired=torch.where(paired[...,None]&assigned,proposed,loads)
    face_error=(face_loads[...,0]-face_loads[...,1]).abs()/total.clamp_min(1e-12)
    within=((desired-loads).abs()*assigned).sum(-1)/total.clamp_min(1e-12)
    # Separation of face and within-face tolerances permits 2:1 across a
    # one-vs-two grasp while still requiring equality within the two-digit face.
    small=torch.where(masks,loads[...,None,:],float('inf')).amin(-1)
    large=torch.where(masks,loads[...,None,:],0.).amax(-1)
    face_ok=face_loads.amax(-1)<=1.5*face_loads.amin(-1)
    within_ok=(large<=1.5*small).all(-1)
    balanced=paired&face_ok&within_ok
    return dict(desired=desired,groups=groups,axis=torch.where(paired[...,None],axis,0.),
        paired=paired,balanced=balanced,face_loads=face_loads,face_counts=counts,
        face_imbalance=torch.where(paired,face_error,0.),
        within_imbalance=torch.where(paired,within,0.))
