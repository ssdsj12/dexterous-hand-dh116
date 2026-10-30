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


def force_joint_rates(normal_jac, loads, desired, active, lower, upper, dt, *, retreat=None):
    """Return bounded six-joint rates per hand; no wrist or extra effort action.

    Implicit-drive compliance J J.T / 10 converts force error into normal
    displacement. Unlike a fixed Cartesian gain, this accounts for lever arms.
    Unloaded eligible fingers seek slowly along their actual approach gradient.
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
    velocity=(.10/dt*(compliance@(desired-loads)[...,None]).squeeze(-1)).clamp(-.001,.001)
    velocity=torch.where(active&(loads<.05)&(desired>0),torch.full_like(velocity,.0003),velocity)
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
