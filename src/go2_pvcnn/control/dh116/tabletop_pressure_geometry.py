"""Local pressure geometry; feedforward compensation is not a grasp guarantee."""
import numpy as np

from .tabletop_adaptive_backend import PALM_LIMIT_M


def pressure_direction(jacobians, span_axis):
    """Close two pads along their diameter while compensating midpoint motion.

    Jacobians are world-frame point Jacobians, already folded through the
    selected coupling model. This is a first-order relation at the seed pose.
    """
    jac = np.asarray(jacobians, dtype=float)
    axis = np.asarray(span_axis, dtype=float)
    if (jac.shape != (2, 3, 6) or axis.shape != (3,)
            or not np.isfinite(jac).all() or not np.isfinite(axis).all()
            or not np.isclose(np.linalg.norm(axis), 1., atol=1e-8, rtol=0)):
        raise ValueError('finite two-pad Jacobians and unit span axis required')
    relative = jac[0] - jac[1]
    if np.linalg.matrix_rank(relative) != 3 or np.linalg.cond(relative) > 100.:
        raise ValueError('relative pad Jacobian lacks a well-conditioned rank three')
    direction = np.linalg.lstsq(relative, -axis, rcond=None)[0]
    direction /= np.linalg.norm(direction)
    compensation = -jac.mean(axis=0) @ direction
    return direction, compensation


def compensated_palm_action(action, pressure_offset, compensation):
    """Combine policy and pressure targets within the SAME total palm envelope.

    Supports (2,3) and (N,2,3); the backend still enforces actual clearance,
    tracking gates and rate limits. Return clipping diagnostics, not a promise
    that the physical palm has reached this requested compensation.
    """
    action = np.asarray(action, dtype=float)
    offset = np.asarray(pressure_offset, dtype=float)
    compensation = np.asarray(compensation, dtype=float)
    if (action.ndim not in (2, 3) or action.shape[-2:] != (2, 3)
            or offset.shape != action.shape[:-1] or compensation.shape not in ((2, 3), action.shape)
            or not all(np.isfinite(x).all() for x in (action, offset, compensation))
            or np.any(abs(action) > 1.) or np.any(abs(compensation) > .1)):
        raise ValueError('finite bounded palm actions and pressure compensation required')
    target = action * PALM_LIMIT_M + offset[..., None] * compensation
    clipped = abs(target) > PALM_LIMIT_M
    return np.clip(target / PALM_LIMIT_M, -1., 1.), clipped


def seed_pressure_compensation(payload):
    """Legacy seeds opt out; corrected seeds must carry an explicit derivation."""
    value = payload.get('pressure_geometry')
    if value is None:
        return None
    if value.get('version') != 'diametric_midpoint_feedforward_v1':
        raise ValueError('unknown pressure geometry version')
    compensation = np.asarray(value['palm_compensation_m_per_rad'], dtype=float)
    compensated_palm_action(np.zeros((2, 3)), np.zeros(2), compensation)
    return compensation.copy()
