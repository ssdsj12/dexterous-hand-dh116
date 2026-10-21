"""Panda-only Cartesian lift increments for mounted DH116 hands."""
from __future__ import annotations

import numpy as np


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

    increments = np.zeros((2, 7), dtype=np.float64)
    for side in range(2):
        point_rows = J[side].reshape(9, 7)
        A = np.vstack((point_rows, 0.05 * W[side]))
        b = np.concatenate((np.tile(velocity[side], 3), np.zeros(3)))
        speed, _residuals, rank, _singular_values = np.linalg.lstsq(A, b, rcond=None)
        if rank < 3 or not np.isfinite(speed).all():
            raise ValueError(f"Panda lift Jacobian is singular for side {side}")
        peak = float(np.max(np.abs(speed)))
        if peak > speed_limit_rad_s:
            speed *= speed_limit_rad_s / peak
        increments[side] = speed * dt
    return increments


__all__ = ["panda_only_transport_increment"]
