"""Bounded simulation force-feedback increment, not a hardware controller.

Positive normal speed moves the pad toward the wire. Load is the measured
compressive normal force, not the magnitude of an arbitrary reaction vector.
"""
import numpy as np

TARGET_LOAD_N=.6
LOAD_DEADBAND_N=.03
FORCE_TO_SPEED_M_NS=.001
MAX_NORMAL_SPEED_M_S=.0002
MAX_ACTIVE_SPEED_RAD_S=.02


def closure_increment(normal_jacobian,loads_n,dt,*,target_loads_n=(TARGET_LOAD_N,TARGET_LOAD_N),
                      force_to_speed=FORCE_TO_SPEED_M_NS,max_normal_speed=MAX_NORMAL_SPEED_M_S):
    J=np.asarray(normal_jacobian,dtype=float);load=np.asarray(loads_n,dtype=float)
    target=np.asarray(target_loads_n,dtype=float)
    if (J.shape!=(2,3) or load.shape!=(2,) or not np.isfinite(J).all() or not np.isfinite(load).all()
            or target.shape!=(2,) or not np.isfinite(target).all() or np.any(target<=0)
            or not np.isfinite([force_to_speed,max_normal_speed]).all()
            or force_to_speed<=0 or max_normal_speed<=0
            or not np.isfinite(dt) or not 0<dt<=.0001):
        raise ValueError('invalid closure evidence or unvalidated contact timestep')
    if np.linalg.svd(J,compute_uv=False)[-1]<1e-5:
        raise ValueError('normal closure Jacobian is singular')
    error=target-load
    speed=np.clip(force_to_speed*error,-max_normal_speed,max_normal_speed)
    speed[np.abs(error)<=LOAD_DEADBAND_N]=0.
    velocity=np.linalg.lstsq(J,speed,rcond=None)[0]
    scale=max(1.,float(np.max(np.abs(velocity)))/MAX_ACTIVE_SPEED_RAD_S)
    return velocity*(dt/scale)
