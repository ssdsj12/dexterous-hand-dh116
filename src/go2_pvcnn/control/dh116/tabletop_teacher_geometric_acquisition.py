"""Bounded CAD sidewall references for a privileged simulation Teacher.

These references do not create contacts, predict forces, transport wrists or
manage the caller's cumulative acquisition budget.
"""
import math

import torch


REASON_CODES = dict(eligible=0, digit_disabled=1, bearing_load=2,
    invalid_pad=3, invalid_object=4, outside_sidewall=5, inside_or_on_surface=6,
    gap_too_large=7, wrong_facing=8, budget_exhausted=9, invalid_budget=10)


@torch.no_grad()
def geometric_acquisition_targets(points, pad_outward, object_pose, loads, contact_valid, *,
        cylinder_radius_m, cylinder_length_m, horizon=4, dt=1/60,
        max_speed_m_s=.001, max_gap_m=.010, minimum_facing_cosine=.8,
        remaining_distance_m=None):
    """Build targets at times 1..H for missing thumb/index/middle contacts.

    Geometry is [N,2,5,3], pose [N,7] is world xyz + wxyz, and loads/valid
    are [N,2,5]. Numeric tensors share a floating dtype and device. The finite
    cylinder's local Y axis defines its length. Only points outside its curved
    sidewall, within the gap bound and facing the object, receive weights.

    A load > .05 N blocks acquisition even if contact_valid is false. At or
    below that threshold geometry may supply a reference regardless of the
    contact-valid bit; that bit is never changed or manufactured. Ring/little
    fingers always have zero weight. Bad pads are isolated; bad object poses
    disable their whole environment. Ineligible targets stay at finite input
    points (nonfinite points use zero placeholders with zero weight).

    remaining_distance_m is None, a finite nonnegative scalar, or a matching
    [N,2,5] tensor. Zero exhausts the budget; bad tensor entries disable only
    their pad. requested_distance_m is the first-step requested distance.
    Every output tensor owns its values and all numeric outputs are finite.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    parameters = dict(cylinder_radius_m=cylinder_radius_m,
        cylinder_length_m=cylinder_length_m, dt=dt,
        max_speed_m_s=max_speed_m_s, max_gap_m=max_gap_m)
    for name, value in parameters.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if max_speed_m_s > .001 or max_gap_m > .010:
        raise ValueError('acquisition requires speed <= .001 m/s and gap <= .010 m')
    if (type(minimum_facing_cosine) not in (int, float)
            or not math.isfinite(minimum_facing_cosine) or not 0 <= minimum_facing_cosine <= 1):
        raise ValueError('minimum_facing_cosine must be finite in [0,1]')
    if (not isinstance(points, torch.Tensor) or not points.is_floating_point()
            or points.ndim != 4 or tuple(points.shape[1:]) != (2,5,3) or not len(points)):
        raise ValueError('floating points [N,2,5,3] required')
    maximum = torch.finfo(points.dtype).max
    try:
        duration = horizon * dt
    except OverflowError as error:
        raise ValueError('horizon duration must be representable') from error
    if not math.isfinite(duration) or duration > maximum or any(v > maximum for v in parameters.values()):
        raise ValueError('geometry and horizon duration must be representable in points dtype')
    shape = points.shape[:-1]
    numeric = [('pad_outward', pad_outward, points.shape),
        ('object_pose', object_pose, (len(points),7)), ('loads', loads, shape)]
    if isinstance(remaining_distance_m, torch.Tensor):
        numeric.append(('remaining_distance_m', remaining_distance_m, shape))
    elif remaining_distance_m is not None:
        if (type(remaining_distance_m) not in (int,float)
                or not math.isfinite(remaining_distance_m) or not 0 <= remaining_distance_m <= maximum):
            raise ValueError('remaining_distance_m scalar must be finite and nonnegative')
    for name, value, expected in numeric:
        if (not isinstance(value,torch.Tensor) or value.shape != expected
                or value.dtype != points.dtype or value.device != points.device):
            raise ValueError(name + ' requires matching shape, floating dtype and device')
    if (not isinstance(contact_valid,torch.Tensor) or contact_valid.shape != shape
            or contact_valid.dtype != torch.bool or contact_valid.device != points.device):
        raise ValueError('contact_valid must be matching boolean [N,2,5]')

    q = object_pose[:,3:]
    qnorm = q.norm(dim=-1)
    object_ok = torch.isfinite(object_pose).all(-1) & torch.isfinite(qnorm) & (qnorm > 0)
    safe_q = torch.where(object_ok[:,None], q, 0.)
    safe_q = safe_q / torch.where(object_ok,qnorm,1.)[:,None]
    w,x,y,z = safe_q.unbind(-1)
    axis = torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)[:,None,None]
    center = torch.where(object_ok[:,None],object_pose[:,:3],0.)[:,None,None]
    point_ok = torch.isfinite(points).all(-1)
    origin = torch.where(point_ok[...,None],points,0.)
    relative = origin-center
    axial = (relative*axis).sum(-1)
    radial = relative-axial[...,None]*axis
    radial_length = radial.norm(dim=-1)
    normal_length = pad_outward.norm(dim=-1)
    pad_ok = (point_ok & torch.isfinite(pad_outward).all(-1)
        & torch.isfinite(normal_length) & (normal_length > 0)
        & torch.isfinite(loads) & (loads >= 0)
        & torch.isfinite(radial).all(-1) & torch.isfinite(radial_length)
        & torch.isfinite(axial))
    radial_ok = pad_ok & (radial_length > 0) & object_ok[:,None,None]
    radial_unit = torch.where(radial_ok[...,None],radial,0.) / torch.where(radial_ok,radial_length,1.)[...,None]
    normal = torch.where(pad_ok[...,None],pad_outward,0.) / torch.where(pad_ok,normal_length,1.)[...,None]
    gap = torch.where(radial_ok,radial_length-cylinder_radius_m,0.)
    facing = (normal * -radial_unit).sum(-1).clamp(-1,1)
    sidewall = axial.abs() < cylinder_length_m/2
    budget = (torch.full_like(loads,max_gap_m) if remaining_distance_m is None
        else remaining_distance_m.clone() if isinstance(remaining_distance_m,torch.Tensor)
        else torch.full_like(loads,remaining_distance_m))
    budget_ok = torch.isfinite(budget) & (budget >= 0)
    eligible = (radial_ok & sidewall & (gap > 0) & (gap <= max_gap_m)
        & (facing >= minimum_facing_cosine) & (loads <= .05) & budget_ok & (budget > 0))
    enabled = torch.zeros_like(eligible);enabled[...,:3] = True
    eligible &= enabled
    inward = torch.where(eligible[...,None],-radial_unit,0.)
    distance = torch.where(eligible,torch.minimum(gap,budget),0.)
    time = torch.arange(1,horizon+1,dtype=points.dtype,device=points.device)*dt
    path_distance = torch.minimum(distance[:,None],(max_speed_m_s*time)[None,:,None,None])
    target = origin[:,None]+path_distance[...,None]*inward[:,None]
    closest = origin-torch.where((radial_ok&sidewall)[...,None],radial_unit*gap[...,None],0.)
    closest_finite = torch.isfinite(closest).all(-1)
    closest = torch.where(closest_finite[...,None],closest,origin)
    reason = torch.full(shape,REASON_CODES['eligible'],dtype=torch.long,device=points.device)
    for mask,name in ((facing < minimum_facing_cosine,'wrong_facing'),
            (gap > max_gap_m,'gap_too_large'),(gap <= 0,'inside_or_on_surface'),
            (~sidewall,'outside_sidewall'),(budget == 0,'budget_exhausted'),
            (~budget_ok,'invalid_budget'),(loads > .05,'bearing_load'),
            (~enabled,'digit_disabled'),(~pad_ok,'invalid_pad'),
            (~object_ok[:,None,None],'invalid_object')):
        reason = torch.where(mask,REASON_CODES[name],reason)
    return dict(geometry_eligible=eligible, gap_m=gap, facing_cos=facing,
        inward=inward, closest_surface_points=closest, target_points=target,
        weights=eligible[:,None].expand(-1,horizon,-1,-1).to(points.dtype).clone(),
        requested_distance_m=path_distance[:,0].clone(), reason_code=reason,
        reason_codes=REASON_CODES.copy(),mode='cad_cylinder_sidewall_acquisition',force_prediction=False)
