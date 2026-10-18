"""Simulation-only velocity allocation; not a collision or grasp certificate.

Columns are Panda1..7, DH116 finger11/finger12/finger21. Other fingers remain
outside this controller. It commands no tangential pad motion and no palm
rotation by default. An explicit common world velocity permits transport;
the caller must qualify the whole physical trajectory independently.
"""
import numpy as np


def _velocity_candidate(point_jacobians,normals,palm_angular_jacobian,
                        loads_n,dt,*,target_loads_n,transport_velocity_w=None,normal_speeds_m_s=None):
    J=np.asarray(point_jacobians,dtype=float)
    n=np.asarray(normals,dtype=float)
    W=np.asarray(palm_angular_jacobian,dtype=float)
    load=np.asarray(loads_n,dtype=float);target=np.asarray(target_loads_n,dtype=float)
    transport=np.zeros(3) if transport_velocity_w is None else np.asarray(transport_velocity_w,dtype=float)
    if (transport.shape!=(3,) or not np.isfinite(transport).all()
            or np.linalg.norm(transport)>.01+1e-12):
        raise ValueError('invalid coordinated transport velocity')
    if (J.shape!=(2,3,10) or n.shape!=(2,3) or W.shape!=(3,10)
            or load.shape!=(2,) or target.shape!=(2,)
            or any(not np.isfinite(a).all() for a in (J,n,W,load,target))
            or not np.isfinite(dt) or not 0<dt<=1e-4
            or np.any(load<0) or np.any(target<=0)
            or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6,rtol=0)
            or not np.allclose(W[:,7:],0.,atol=1e-10,rtol=0)):
        raise ValueError('invalid coordinated closure evidence')
    error=target-load
    speed=np.clip(.01*error,-.002,.002)
    speed[np.abs(error)<=.03]=0.
    if normal_speeds_m_s is not None:
        speed=np.asarray(normal_speeds_m_s,dtype=float)
        if speed.shape!=(2,) or not np.isfinite(speed).all() or np.any(np.abs(speed)>.002):
            raise ValueError('invalid explicit pad normal speed')
    A=np.vstack((J.reshape(6,10),.05*W))
    b=np.r_[(n*speed[:,None]+transport).reshape(6),np.zeros(3)]
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
                                   loads_n,dt,*,target_loads_n,transport_velocity_w,normal_speeds_m_s=None):
    """Left/right candidates share one rate scale; no runtime safety authority."""
    J=np.asarray(point_jacobians,dtype=float);n=np.asarray(normals,dtype=float)
    W=np.asarray(palm_angular_jacobians,dtype=float)
    loads=np.asarray(loads_n,dtype=float);targets=np.asarray(target_loads_n,dtype=float)
    if (J.shape!=(2,2,3,10) or n.shape!=(2,2,3) or W.shape!=(2,3,10)
            or loads.shape!=(2,2) or targets.shape!=(2,2)):
        raise ValueError('invalid bimanual closure evidence shape')
    speeds=None if normal_speeds_m_s is None else np.asarray(normal_speeds_m_s,dtype=float)
    if speeds is not None and (speeds.shape!=(2,2) or not np.isfinite(speeds).all()
                               or np.any(np.abs(speeds)>.002)):
        raise ValueError('invalid bimanual explicit pad speeds')
    velocity=np.array([_velocity_candidate(J[i],n[i],W[i],loads[i],dt,
        target_loads_n=targets[i],transport_velocity_w=transport_velocity_w,
        normal_speeds_m_s=None if speeds is None else speeds[i]) for i in range(2)])
    scale=max(1.,float(np.max(np.abs(velocity[:,:7])))/.05,
                 float(np.max(np.abs(velocity[:,7:])))/.02)
    return velocity*(dt/scale)
