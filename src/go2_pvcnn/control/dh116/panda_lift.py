"""Panda-only Cartesian lift increments for mounted DH116 hands."""
from __future__ import annotations

import numpy as np


def panda_feedback_increment(
    point_jacobians,
    palm_angular_jacobians,
    transport_velocity_w,
    dt,
    *,
    palm_target_positions_w=None,
    palm_actual_positions_w=None,
    position_gain=2.0,
    damping_lambda=1e-5,
    speed_limit_rad_s=0.05,
    joint_positions=None,
    joint_lower=None,
    joint_upper=None,
    minimum_singular_value=1e-8,
    residual_tolerance=1e-4,
):
    """Closed-loop Panda transport with measured palm-position feedback.

    The DH116 coordinates never enter this solve.  The target velocity is
    ``v_transport + Kp * (p_target - p_actual)`` and the Panda command is a
    damped least-squares solution of the measured pad/palm Jacobian.  The
    returned report makes reachability and residual checks observable to the
    caller instead of silently accepting a rank-deficient least-squares result.
    """
    J=np.asarray(point_jacobians,dtype=np.float64)
    W=np.asarray(palm_angular_jacobians,dtype=np.float64)
    velocity=np.asarray(transport_velocity_w,dtype=np.float64)
    if velocity.shape==(3,):velocity=np.broadcast_to(velocity,(2,3)).copy()
    target=None if palm_target_positions_w is None else np.asarray(palm_target_positions_w,dtype=np.float64)
    actual=None if palm_actual_positions_w is None else np.asarray(palm_actual_positions_w,dtype=np.float64)
    if (J.shape!=(2,3,3,7) or W.shape!=(2,3,7) or velocity.shape!=(2,3)
            or target is None or actual is None or target.shape!=(2,3) or actual.shape!=(2,3)
            or not all(np.isfinite(x).all() for x in (J,W,velocity,target,actual))
            or not np.isfinite([dt,position_gain,damping_lambda,speed_limit_rad_s,
                               minimum_singular_value,residual_tolerance]).all()
            or not 0. < dt <= .01 or position_gain < 0. or damping_lambda <= 0.
            or speed_limit_rad_s <= 0. or minimum_singular_value <= 0.
            or residual_tolerance <= 0.):
        raise ValueError("invalid Panda feedback lift inputs")
    supplied_limits=(joint_positions is not None or joint_lower is not None or joint_upper is not None)
    if supplied_limits:
        if joint_positions is None or joint_lower is None or joint_upper is None:
            raise ValueError("joint positions and both Panda limits are required together")
        q=np.asarray(joint_positions,dtype=np.float64);lower=np.asarray(joint_lower,dtype=np.float64)
        upper=np.asarray(joint_upper,dtype=np.float64)
        if (q.shape!=lower.shape or q.shape!=upper.shape or q.shape!=(2,7)
                or not all(np.isfinite(x).all() for x in (q,lower,upper)) or np.any(lower>=upper)
                or np.any(q<lower) or np.any(q>upper)):
            raise ValueError("invalid Panda joint limits")
    increments=np.zeros((2,7),dtype=np.float64);reports=[]
    for side in range(2):
        point_rows=J[side].reshape(9,7)
        A=np.vstack((point_rows,0.05*W[side]))
        commanded=velocity[side]+position_gain*(target[side]-actual[side])
        b=np.concatenate((np.tile(commanded,3),np.zeros(3)))
        singular=np.linalg.svd(A,compute_uv=False)
        if len(singular)<3 or singular[2] < minimum_singular_value:
            raise ValueError(f"Panda feedback Jacobian is singular for side {side}")
        H=A.T@A+(damping_lambda**2)*np.eye(7)
        speed=np.linalg.solve(H,A.T@b)
        if not np.isfinite(speed).all():raise ValueError('nonfinite Panda feedback solution')
        unclipped=speed.copy()
        peak=float(np.max(np.abs(speed)))
        if peak>speed_limit_rad_s:speed*=speed_limit_rad_s/peak
        residual=float(np.linalg.norm(A@speed-b))
        if supplied_limits:
            speed=np.minimum(np.maximum(speed,(lower[side]-q[side])/dt),(upper[side]-q[side])/dt)
        increments[side]=speed*dt
        reports.append(dict(commanded_velocity_w=commanded.copy(),position_error_m=float(np.linalg.norm(target[side]-actual[side])),
                            residual_norm=residual,residual_tolerance=float(residual_tolerance),
                            reachable=bool(residual<=residual_tolerance),
                            minimum_singular_value=float(singular[2]),
                            speed_rad_s=speed.copy(),speed_capped=bool(np.any(np.abs(unclipped-speed)>1e-12))))
    return increments,reports


def panda_only_transport_increment(
    point_jacobians,
    palm_angular_jacobians,
    transport_velocity_w,
    dt,
    *,
    speed_limit_rad_s=0.05,
):
    """Map bilateral fingertip transport velocities to Panda-only increments.

    ``point_jacobians`` has shape ``(2, 3, 3, 7)``: two arms, three pad
    points, XYZ rows, seven Panda joint columns. No DH116 columns are accepted.
    ``palm_angular_jacobians`` has shape ``(2, 3, 7)``.
    """
    J = np.asarray(point_jacobians, dtype=np.float64)
    W = np.asarray(palm_angular_jacobians, dtype=np.float64)
    velocity = np.asarray(transport_velocity_w, dtype=np.float64)
    if (J.shape != (2, 3, 3, 7) or W.shape != (2, 3, 7)
            or velocity.shape != (2, 3)
            or not np.isfinite(J).all() or not np.isfinite(W).all()
            or not np.isfinite(velocity).all()
            or not np.isfinite([dt, speed_limit_rad_s]).all()
            or not 0.0 < dt <= 0.01 or speed_limit_rad_s <= 0.0):
        raise ValueError("invalid Panda-only lift Jacobian, velocity, or timestep")

    zeros=np.zeros((2,3),dtype=np.float64)
    increments,_=panda_feedback_increment(J,W,velocity,dt,
        palm_target_positions_w=zeros,palm_actual_positions_w=zeros,
        position_gain=0.,speed_limit_rad_s=speed_limit_rad_s,
        residual_tolerance=1e-4)
    return increments


__all__ = ["panda_feedback_increment", "panda_only_transport_increment"]
