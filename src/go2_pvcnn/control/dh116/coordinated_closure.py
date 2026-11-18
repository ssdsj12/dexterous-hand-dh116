"""Simulation-only velocity allocation; not a collision or grasp certificate.

Columns are Panda1..7, DH116 finger11/finger12/finger21. Other fingers remain
outside this controller. It commands no tangential pad motion and no palm
rotation by default. An explicit common world velocity permits transport;
the caller must qualify the whole physical trajectory independently.
"""
import numpy as np


def orient_normals_toward_wire(points, normals, rod_position):
    """Orient pad normals toward the measured wire centerline.

    USD surface normals are geometric outward normals and do not encode which
    side of the pad the current wire occupies.  Closure uses the normal as a
    *pad-to-wire* command direction, so the sign is calibrated once when the
    close stage starts from the measured rod state.  The returned normals are
    unit vectors and are intended for both velocity allocation and normal-load
    sign evaluation.
    """
    p = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    rod = np.asarray(rod_position, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or p.shape[0] not in (4, 6)
            or n.shape != p.shape or rod.ndim != 2
            or rod.shape[0] != 3 or rod.shape[1] < 2
            or any(not np.isfinite(x).all() for x in (p, n, rod))):
        raise ValueError('invalid wire-normal orientation evidence')
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('normal vectors must be nonzero')
    unit = n / lengths[:, None]
    nearest = rod.T[np.argmin(np.linalg.norm(rod.T[:, None, :] - p[None, :, :], axis=2), axis=0)]
    direction = nearest - p
    signs = np.where(np.einsum('ij,ij->i', unit, direction) < 0., -1., 1.)
    return unit * signs[:, None], signs.astype(float)


def wire_approach_normals(points, rod_position):
    """Return measured pad-to-wire directions for active closure."""
    p = np.asarray(points, dtype=float)
    rod = np.asarray(rod_position, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or p.shape[0] not in (4, 6)
            or rod.ndim != 2 or rod.shape[0] != 3
            or rod.shape[1] < 2
            or any(not np.isfinite(x).all() for x in (p, rod))):
        raise ValueError('invalid wire approach direction evidence')
    nearest = rod.T[np.argmin(np.linalg.norm(
        rod.T[:, None, :] - p[None, :, :], axis=2), axis=0)]
    direction = nearest - p
    lengths = np.linalg.norm(direction, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError('wire approach direction is degenerate')
    return direction / lengths[:, None]


def _velocity_candidate(point_jacobians,normals,palm_angular_jacobian,
                        loads_n,dt,*,target_loads_n,transport_velocity_w=None,normal_speeds_m_s=None,
                        normal_speed_limit_m_s=.002,force_speed_limit_m_s=.002,
                        active_mask=None):
    if (not np.isfinite([normal_speed_limit_m_s,force_speed_limit_m_s]).all()
            or not 0<normal_speed_limit_m_s<=.05 or not 0<force_speed_limit_m_s<=.002):
        raise ValueError('invalid simulation normal speed limits')
    J=np.asarray(point_jacobians,dtype=float)
    n=np.asarray(normals,dtype=float)
    W=np.asarray(palm_angular_jacobian,dtype=float)
    load=np.asarray(loads_n,dtype=float);target=np.asarray(target_loads_n,dtype=float)
    transport=np.zeros(3) if transport_velocity_w is None else np.asarray(transport_velocity_w,dtype=float)
    if (transport.shape!=(3,) or not np.isfinite(transport).all()
            or np.linalg.norm(transport)>.01+1e-12):
        raise ValueError('invalid coordinated transport velocity')
    if (J.ndim != 3 or J.shape[1:]!=(3,10) or n.shape!=(J.shape[0],3) or W.shape!=(3,10)
            or load.shape!=(J.shape[0],) or target.shape!=(J.shape[0],)
            or any(not np.isfinite(a).all() for a in (J,n,W,load,target))
            or not np.isfinite(dt) or not 0<dt<=1e-4
            or np.any(load<0) or np.any(target<=0)
            or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6,rtol=0)
            or not np.allclose(W[:,7:],0.,atol=1e-10,rtol=0)):
        raise ValueError('invalid coordinated closure evidence')
    if active_mask is not None:
        active=np.asarray(active_mask)
        if active.shape!=(J.shape[0],) or active.dtype.kind!='b':
            raise ValueError('invalid active pad mask')
    else:
        active=np.ones(J.shape[0],dtype=bool)
    error=target-load
    speed=np.clip(.01*error,-force_speed_limit_m_s,force_speed_limit_m_s)
    speed[np.abs(error)<=.03]=0.
    if normal_speeds_m_s is not None:
        speed=np.asarray(normal_speeds_m_s,dtype=float)
        if speed.shape!=(J.shape[0],) or not np.isfinite(speed).all() or np.any(np.abs(speed)>normal_speed_limit_m_s):
            raise ValueError('invalid explicit pad normal speed')
    rows=[];rhs=[]
    for i in range(J.shape[0]):
        if active[i]:
            rows.extend(J[i])
            rhs.extend(n[i]*speed[i]+transport)
    A=np.vstack((np.asarray(rows),.05*W))
    b=np.r_[np.asarray(rhs),np.zeros(3)]
    if np.linalg.svd(A,compute_uv=False)[-1]<1e-5:
        raise ValueError('coordinated closure Jacobian is singular')
    velocity=np.linalg.lstsq(A,b,rcond=None)[0]
    if not np.isfinite(velocity).all():raise ValueError('nonfinite closure velocity')
    if normal_speeds_m_s is not None and not np.allclose(A@velocity,b,atol=1e-10,rtol=1e-7):
        raise ValueError('touch-first point hold is not achievable')
    return velocity


def coordinated_closure_increment(point_jacobians,normals,palm_angular_jacobian,
                                  loads_n,dt,*,target_loads_n,transport_velocity_w=None):
    velocity=_velocity_candidate(point_jacobians,normals,palm_angular_jacobian,loads_n,dt,
        target_loads_n=target_loads_n,transport_velocity_w=transport_velocity_w)
    scale=max(1.,float(np.max(np.abs(velocity[:7])))/.05,
                 float(np.max(np.abs(velocity[7:])))/.02)
    return velocity*(dt/scale)


def coordinated_bimanual_increment(point_jacobians,normals,palm_angular_jacobians,
                                   loads_n,dt,*,target_loads_n,transport_velocity_w,normal_speeds_m_s=None,
                                   normal_speed_limit_m_s=.002,force_speed_limit_m_s=.002,
                                   active_pad_mask=None,return_diagnostics=False):
    """Common (3,) or bilateral (2,3) transport; one shared joint-rate scale.

    Per-hand transport enables measured synchronization by the caller. It does
    not itself qualify contact, symmetry, collision clearance or a real lift.
    """
    if type(return_diagnostics) is not bool:raise ValueError('diagnostics flag must be bool')
    J=np.asarray(point_jacobians,dtype=float);n=np.asarray(normals,dtype=float)
    W=np.asarray(palm_angular_jacobians,dtype=float)
    loads=np.asarray(loads_n,dtype=float);targets=np.asarray(target_loads_n,dtype=float)
    if (J.ndim != 4 or J.shape[0] != 2 or J.shape[2:]!=(3,10)
            or J.shape[1] not in (2,3) or n.shape!=(2,J.shape[1],3) or W.shape!=(2,3,10)
            or loads.shape!=(2,J.shape[1]) or targets.shape!=(2,J.shape[1])):
        raise ValueError('invalid bimanual closure evidence shape')
    transport=np.asarray(transport_velocity_w,dtype=float)
    if transport.shape==(3,):transport=np.broadcast_to(transport,(2,3))
    if (transport.shape!=(2,3) or not np.isfinite(transport).all()
            or np.any(np.linalg.norm(transport,axis=1)>.01+1e-12)):
        raise ValueError('invalid bimanual transport velocity')
    speeds=None if normal_speeds_m_s is None else np.asarray(normal_speeds_m_s,dtype=float)
    if speeds is not None and (speeds.shape!=(2,J.shape[1]) or not np.isfinite(speeds).all()
                               or np.any(np.abs(speeds)>normal_speed_limit_m_s)):
        raise ValueError('invalid bimanual explicit pad speeds')
    active=None if active_pad_mask is None else np.asarray(active_pad_mask)
    if active is not None and (active.shape!=(2,J.shape[1]) or active.dtype.kind!='b'):
        raise ValueError('invalid bimanual active pad mask')
    velocity=np.array([_velocity_candidate(J[i],n[i],W[i],loads[i],dt,
        target_loads_n=targets[i],transport_velocity_w=transport[i],
        normal_speeds_m_s=None if speeds is None else speeds[i],
        normal_speed_limit_m_s=normal_speed_limit_m_s,force_speed_limit_m_s=force_speed_limit_m_s,
        active_mask=None if active is None else active[i]) for i in range(2)])
    scale=max(1.,float(np.max(np.abs(velocity[:,:7])))/.05,
                 float(np.max(np.abs(velocity[:,7:])))/.02)
    increment=velocity*(dt/scale)
    if not return_diagnostics:return increment
    return increment,dict(joint_rate_scale=scale,
        requested_pad_velocity_w_m_s=np.einsum('shij,sj->shi',J,velocity).tolist(),
        allocated_pad_velocity_w_m_s=np.einsum('shij,sj->shi',J,velocity/scale).tolist(),
        arm_joint_speed_cap_rad_s=.05,hand_joint_speed_cap_rad_s=.02,
        scope='Jacobian allocation, not measured physical velocity')
