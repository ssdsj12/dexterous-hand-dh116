"""Optional read-only PhysX sensor reads, without ContactSensor update/data.

Unsupported metadata/layout returns None before touching any sensor state.
The caller owns sensor clocks, fallback and all effective-contact decisions.
"""
import torch

from .tabletop_effective_pad_batch import _owners_batched


def _layout(raw, friction, matrices, n):
    """Inspect tensor metadata only: one body/filter and uniform capacities."""
    if any(len(v) != 6 for v in raw) or any(len(v) != 4 for v in friction):
        return None
    if any(not isinstance(t, torch.Tensor) for values in (*raw, *friction) for t in values):
        return None
    if any(not isinstance(t, torch.Tensor) for t in matrices):
        return None
    first=raw[0][0]
    device,dtype=first.device,first.dtype
    capacity=first.numel()
    friction_capacity=friction[0][0].numel()//3
    if not first.is_floating_point():
        return None
    for values in raw:
        f,p,normal,sep,count,start=values
        if (f.numel()!=capacity or sep.numel()!=capacity or
                p.shape!=(capacity,3) or normal.shape!=(capacity,3) or
                count.numel()!=n or start.numel()!=n or
                any(v.device!=device for v in values) or
                any(v.dtype!=dtype for v in values[:4])):
            return None
    for values in friction:
        force,point,count,start=values
        if (force.shape!=(friction_capacity,3) or point.shape!=force.shape or
                count.numel()!=n or start.numel()!=n or
                any(v.device!=device for v in values) or
                any(v.dtype!=dtype for v in values[:2])):
            return None
    for matrix in matrices:
        if (matrix.ndim<2 or matrix.shape[0]!=n or matrix.shape[-1]!=3 or
                matrix.numel()!=n*3 or matrix.device!=device or matrix.dtype!=dtype):
            return None
    return capacity,friction_capacity


def _ownership(values, count_index, n, capacity):
    count=torch.stack([v[count_index].reshape(n) for v in values])
    start=torch.stack([v[count_index+1].reshape(n) for v in values])
    ci,si=count.long(),start.long()
    invalid=(~torch.isfinite(count) | ~torch.isfinite(start) | (count!=ci) | (start!=si))
    ci=torch.where(invalid,-1,ci)
    si=torch.where(invalid,-1,si)
    env,active,invalid=_owners_batched(ci,si,capacity)
    return ci,env,active,invalid


def _sum_owned(data, env, active, n):
    result=data.new_zeros((len(data),n,3))
    result.scatter_add_(1,env[...,None].expand(-1,-1,3),
                        torch.where(active[...,None],data,0.))
    return result


def _bad_owned(finite, env, active, n):
    faults=torch.zeros((len(env),n),dtype=torch.long,device=env.device)
    faults.scatter_add_(1,env,(active & ~finite).long())
    return faults.any(0)


@torch.no_grad()
def read_native_sensors(world):
    """Read 34 ordered single-body, object-filtered native sensors or return None.

    Accept a TensorWorld (world.w.sensors) or its underlying world.sensors.
    Return raw_contacts (34 native six-tuples with snapshotted indices), vectors, friction,
    contact_positions (N,2,17,3 float64), and bad_env (N bool). Forces retain
    force_matrix values; friction is summed; points are an UNWEIGHTED mean
    including finite zero/negative normal-scalar releases, empty groups NaN.
    Arithmetic uses the original buffer dtype before conversion to float64,
    matching the old ContactSensor aggregation. Native padding never owns data.
    No update(), .data, clock/cache mutation, masking, or physics writes occur.
    """
    w=getattr(world,'w',world)
    n=getattr(world,'n',0)
    sensors=getattr(w,'sensors',())
    if n<1 or len(sensors)!=34:
        return None
    views=[getattr(sensor,'contact_physx_view',None) for sensor in sensors]
    if any(view is None or getattr(view,'filter_count',None)!=1 or
           any(not callable(getattr(view,name,None)) for name in
               ('get_contact_force_matrix','get_contact_data','get_friction_data')) for view in views):
        return None
    matrices=[v.get_contact_force_matrix(dt=world.dt) for v in views]
    raw=[]
    for view in views:
        values=view.get_contact_data(dt=world.dt)
        if isinstance(values,(tuple,list)) and len(values)==6:
            # PhysX reuses _contact_count_buffer/_start_indices_buffer for
            # friction. Snapshot BEFORE get_friction_data overwrites them.
            values=tuple(x.clone() if i>=4 and isinstance(x,torch.Tensor) else x
                for i,x in enumerate(values))
        raw.append(values)
    friction=[v.get_friction_data(dt=world.dt) for v in views]
    if any(not isinstance(v,(tuple,list)) for v in (*raw,*friction)):
        return None
    layout=_layout(raw,friction,matrices,n)
    if layout is None:
        return None
    capacity,friction_capacity=layout
    count,env,active,invalid=_ownership(raw,4,n,capacity)
    forces=torch.stack([v[0].reshape(capacity) for v in raw])
    points=torch.stack([v[1] for v in raw])
    normals=torch.stack([v[2] for v in raw])
    separation=torch.stack([v[3].reshape(capacity) for v in raw])
    finite=(torch.isfinite(forces) & torch.isfinite(points).all(-1) &
            torch.isfinite(normals).all(-1) & torch.isfinite(separation))
    point_sum=_sum_owned(points,env,active,n)
    contact_positions=torch.where(((count>0) & ~invalid)[...,None],
                                  point_sum/count.clamp_min(1)[...,None],float('nan'))
    bad=invalid.any(0) | _bad_owned(finite,env,active,n) | ~torch.isfinite(point_sum).all((0,2))
    _,friction_env,friction_active,friction_invalid=_ownership(friction,2,n,friction_capacity)
    friction_forces=torch.stack([v[0] for v in friction])
    friction_points=torch.stack([v[1] for v in friction])
    finite_friction=torch.isfinite(friction_forces).all(-1) & torch.isfinite(friction_points).all(-1)
    friction_sum=_sum_owned(friction_forces,friction_env,friction_active,n)
    bad |= friction_invalid.any(0) | _bad_owned(finite_friction,friction_env,friction_active,n)
    bad |= ~torch.isfinite(friction_sum).all((0,2))
    vectors=torch.stack([v.reshape(n,3) for v in matrices])
    bad |= ~torch.isfinite(vectors).all((0,2))
    def ordered(value):
        return value.transpose(0,1).reshape(n,2,17,3).double()
    return dict(raw_contacts=[tuple(v) for v in raw],vectors=ordered(vectors),
                friction=ordered(friction_sum),contact_positions=ordered(contact_positions),bad_env=bad)
