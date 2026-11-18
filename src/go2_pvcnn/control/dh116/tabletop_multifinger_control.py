"""Small-batch five-digit pressure allocation using the original joint drives."""
import math
import torch
from .tabletop_manual_joint_contact import bounded_dls


def balanced_targets(total, eligible):
    if (total.ndim!=2 or total.shape[1]!=2 or not total.is_floating_point()
            or eligible.shape!=(*total.shape,5) or eligible.dtype!=torch.bool
            or eligible.device!=total.device or not torch.isfinite(total).all() or (total<0).any()):
        raise ValueError('finite nonnegative N x 2 totals and N x 2 x 5 eligibility required')
    return total[:,:,None]*eligible/eligible.sum(-1,keepdim=True).clamp_min(1)


def sharing_targets(total, loads, gap, eligible):
    """Share each hand's budget among load-bearing digits; seek independently.

    Loads are measured mean normal forces. Retain every digit above 0.05 N,
    adding the nearest eligible digits by absolute gap only if fewer than two
    bear load. Unloaded eligible digits keep seeking without diluting an
    established group's budget. A lone sharing digit gets at most half.
    """
    if (total.ndim!=2 or total.shape[1]!=2 or not total.is_floating_point()
            or loads.shape!=(*total.shape,5) or gap.shape!=loads.shape
            or eligible.shape!=loads.shape or eligible.dtype!=torch.bool
            or eligible.device!=total.device):
        raise ValueError('N x 2 totals and matching N x 2 x 5 digit arrays required')
    for value in (total,loads,gap):
        if (not value.is_floating_point() or value.dtype!=total.dtype
                or value.device!=total.device or not torch.isfinite(value).all()):
            raise ValueError('matching finite floating arrays required')
    if (total<0).any() or (loads<0).any():
        raise ValueError('nonnegative totals and mean loads required')
    bearing=loads>.05
    seeking=eligible&~bearing
    order=gap.abs().masked_fill(~seeking,float('inf')).argsort(-1)
    rank=torch.empty_like(order).scatter(-1,order,torch.arange(5,device=total.device).expand_as(order))
    needed=(2-bearing.sum(-1,keepdim=True)).clamp_min(0)
    sharing=bearing|(seeking&(rank<needed))
    desired=total[:,:,None]*sharing/sharing.sum(-1,keepdim=True).clamp_min(2)
    return desired,sharing,seeking


def compliance_from_physx_material(properties):
    """Decode native negative-restitution soft stiffness; hard contact has zero compliance."""
    if (not isinstance(properties,torch.Tensor) or properties.ndim<2 or properties.shape[-1]!=3
            or not properties.is_floating_point() or not torch.isfinite(properties).all()):
        raise ValueError('finite native material properties ending in three coefficients required')
    stiffness=(-properties[...,2]).clamp_min(0)
    result=torch.zeros_like(stiffness);soft=stiffness>0
    result[soft]=stiffness[soft].reciprocal()
    if not torch.isfinite(result).all():raise ValueError('finite native contact compliance required')
    return result


def projected_contact_stiffness_count(normals,forces,direction):
    """Dimensionless parallel stiffness along a unit aggregate normal.

    Active oblique springs can have a projected sum below one. Only an
    unloaded patch uses the unit fallback; loaded patches retain that sum.
    """
    if (normals.ndim!=2 or normals.shape[-1]!=3 or forces.shape!=(len(normals),)
            or direction.shape!=(3,) or not normals.is_floating_point()
            or any(v.dtype!=normals.dtype or v.device!=normals.device for v in (forces,direction))
            or not all(torch.isfinite(v).all() for v in (normals,forces,direction))):
        raise ValueError('finite matching native contact normals, forces and direction required')
    loaded=forces>1e-8
    projected=((normals@direction).square()*loaded).sum()
    return torch.where(loaded.any(),projected.clamp_min(1e-12),projected.new_tensor(1.))


def force_joint_rates(normal_jac, loads, desired, active, lower, upper, dt, *, retreat=None, seeking=None,
                      contact_compliance_m_n=None):
    """Return bounded six-joint rates per hand; no wrist or extra effort action.

    Joint-drive compliance J J.T / 10 plus optional diagonal pad compliance
    converts force error into normal displacement. The two springs act in
    series; omitting the pad term can greatly under-correct at small lever arms.
    Unloaded eligible fingers seek slowly along their actual approach gradient.
    Explicit seeking also permits zero desired force; omission keeps the legacy
    unloaded positive-target rule. Retreat takes priority over either rule.
    Mean load feedback is for force control; raw stability is checked separately.
    """
    if not isinstance(dt,(int,float)) or not math.isfinite(dt) or dt<=0:
        raise ValueError('positive finite control dt required')
    n=normal_jac.shape[0]
    if (normal_jac.shape!=(n,2,5,6) or loads.shape!=(n,2,5) or desired.shape!=loads.shape
            or active.shape!=loads.shape or active.dtype!=torch.bool
            or lower.shape!=(n,2,6) or upper.shape!=lower.shape):
        raise ValueError('matching five-digit/six-joint arrays required')
    for value in (normal_jac,loads,desired,lower,upper):
        if (not value.is_floating_point() or value.device!=normal_jac.device
                or value.dtype!=normal_jac.dtype or not torch.isfinite(value).all()):
            raise ValueError('matching finite floating arrays required')
    if (loads<0).any() or (desired<0).any() or (lower>upper).any():
        raise ValueError('nonnegative loads and consistent bounds required')
    jac=normal_jac*active[...,None]
    compliance=(jac@jac.transpose(-1,-2))/10.
    if contact_compliance_m_n is not None:
        pad=contact_compliance_m_n
        if (not isinstance(pad,torch.Tensor) or pad.shape!=loads.shape or pad.dtype!=loads.dtype
                or pad.device!=loads.device or not torch.isfinite(pad).all() or (pad<0).any()):
            raise ValueError('matching finite nonnegative contact compliance required')
        compliance=compliance+torch.diag_embed(pad*active)
    velocity=(.10/dt*(compliance@(desired-loads)[...,None]).squeeze(-1)).clamp(-.001,.001)
    if seeking is None:
        seeking=active&(loads<.05)&(desired>0)
    elif seeking.shape!=active.shape or seeking.dtype!=torch.bool or seeking.device!=active.device:
        raise ValueError('matching digit seeking mask required')
    velocity=torch.where(seeking,torch.full_like(velocity,.0003),velocity)
    if retreat is not None:
        if retreat.shape!=active.shape or retreat.dtype!=torch.bool or retreat.device!=active.device:
            raise ValueError('matching digit retreat mask required')
        velocity=torch.where(retreat,torch.full_like(velocity,-.0003),velocity)
    velocity=velocity*active
    bound_low=lower.clamp_min(-.3);bound_high=upper.clamp_max(.3)
    if (bound_low>bound_high).any():raise ValueError('joint bounds exclude the actuator velocity range')
    bias=torch.zeros_like(lower)
    return bounded_dls(jac,velocity,bias,bound_low,bound_high,1e-5)


def table_safe_rates(rates, z_jac, clearance, dt, floor=.0003):
    """Uniformly scale joint rates to obey the local table-clearance derivative.

    This is a linear prediction, not continuous collision proof. Existing
    physical collision response and wrist clearance restoration remain active.
    """
    if dt<=0 or not math.isfinite(dt):raise ValueError('positive finite dt required')
    if z_jac.shape[:-1]!=clearance.shape or z_jac.shape[:-2]!=rates.shape[:-1] or z_jac.shape[-1]!=6:
        raise ValueError('matching six-joint table Jacobians required')
    if not all(torch.isfinite(v).all() for v in (rates,z_jac,clearance)):
        raise ValueError('finite table state required')
    downward=(-torch.einsum('nhsk,nhk->nhs',z_jac,rates)).clamp_min(0)
    allowance=(clearance-floor).clamp_min(0)/dt
    fraction=torch.where(downward>0,allowance/downward.clamp_min(1e-12),torch.ones_like(downward))
    scale=fraction.amin(-1).clamp(0,1)
    return rates*scale[...,None]
