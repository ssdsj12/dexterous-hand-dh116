"""Small measured-contact update for online hand-only MPC task progress."""
from __future__ import annotations

import numpy as np


def estimate_tracking_bias(records,*,through_step):
    """Causal initialization from measured, full previous control intervals.

    No future/failed/short terminal interval or learned-policy label contributes.
    This calibrates the reduced predictor, not rod material or measured force.
    """
    bias=np.zeros(12)
    for row in records:
        outcome=row.get('outcome',{});prediction=row.get('prediction') or {}
        if (row.get('phase')!='close' or row.get('owner')!='hand_mpc'
                or outcome.get('passed') is not True or outcome['step']>through_step
                or prediction.get('model')!='linearized_pad_kinematics'
                or not np.isclose(outcome['elapsed_s'],row['control_dt_s'],rtol=0,atol=1e-12)):
            continue
        q=np.asarray(row['observation']['hand_q']).reshape(12)
        after=np.asarray(outcome['state']['hand_q']).reshape(12)
        analytic=np.asarray(prediction.get('analytic_load_offset_rad',
                                         prediction['actuator_load_offset_rad'])).reshape(12)
        alpha=float(prediction.get('actuator_response',.5))
        nominal=q+alpha*(np.asarray(row['action'])-analytic-q)
        measured=(nominal-after)/alpha
        if not np.isfinite(measured).all():raise ValueError('finite measured model error required')
        jac=np.asarray(row['observation']['pad_jacobians'])
        affected=(np.abs(jac).sum((1,2))>1e-12).reshape(12)
        clearance=np.asarray(row['observation']['clearance_jacobians']).reshape(-1,12)
        affected|=np.abs(clearance).sum(0)>1e-12
        bias=np.clip(.75*bias+.25*measured,-np.pi/180,np.pi/180)*affected
    return bias


def initial_active_action(value: float, hand_count: int = 2) -> np.ndarray:
    """Return the 12D two-hand MPC reference matching the reset joint state."""
    if not np.isfinite(value) or not isinstance(hand_count, int) or hand_count <= 0:
        raise ValueError("initial joint value and positive hand count are required")
    return np.full(6 * hand_count, float(value), dtype=np.float32)


def advance_progress(grasp: float, lift: float, *, contact_force_n: float,
                     force_abort_n: float = 30.0, contact_threshold_n: float = 1.0,
                     increment: float = 0.08,
                     contact_latched: bool = False) -> tuple[float, float]:
    """Advance or back off MPC task progress from the latest measured contact.

    No contact advances closure; moderate contact holds closure progress and
    advances the finger-only lift/hold phase; excessive contact backs off the
    closure target.  Progress values are normalized to [0, 1].
    """
    values = np.asarray([grasp, lift, contact_force_n, force_abort_n,
                         contact_threshold_n, increment], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("MPC feedback inputs must be finite")
    if contact_force_n < 0 or force_abort_n <= contact_threshold_n or not 0 < increment <= 1:
        raise ValueError("invalid force limit or progress increment")
    if contact_force_n > force_abort_n:
        grasp = max(0.0, grasp - increment)
    elif contact_force_n >= contact_threshold_n or contact_latched:
        if grasp < 1.0:
            grasp = min(1.0, grasp + increment)
        else:
            lift = min(1.0, lift + increment)
    else:
        grasp = min(1.0, grasp + increment)
    return float(np.clip(grasp, 0.0, 1.0)), float(np.clip(lift, 0.0, 1.0))


def trajectory_action_index(grasp: float, lift: float, *, grasp_steps: int,
                            lift_steps: int) -> int:
    """Select a step from a preplanned grasp-then-lift trajectory.

    Progress is measured from contact feedback, so a slow or paused phase
    reuses its current reference instead of running MPC again. The final grasp
    reference is held at the boundary before advancing into the lift segment.
    """
    values = np.asarray([grasp, lift], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("grasp and lift progress must be finite and in [0,1]")
    if (isinstance(grasp_steps, bool) or not isinstance(grasp_steps, int) or grasp_steps <= 0
            or isinstance(lift_steps, bool) or not isinstance(lift_steps, int) or lift_steps <= 0):
        raise ValueError("grasp_steps and lift_steps must be positive integers")
    if grasp < 1.0 or lift == 0.0:
        return int(np.clip(np.ceil(grasp * grasp_steps) - 1, 0, grasp_steps - 1))
    return grasp_steps + int(np.clip(np.ceil(lift * lift_steps) - 1, 0, lift_steps - 1))


def to_numpy(value: object) -> np.ndarray:
    """Explicitly transfer planner outputs from CUDA before NumPy use."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def top_down_hand_pose(x: float, z: float) -> np.ndarray:
    """Return a fixed hand root pose with palm above target and fingers down.

    Quaternion uses Isaac's ``(w, x, y, z)`` order and rotates the local hand
    frame by pi around X, preserving the left/right X separation.
    """
    if not np.isfinite([x, z]).all():
        raise ValueError("hand root pose must be finite")
    return np.asarray([x, 0.0, z, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def peak_body_force(body_names: list[str] | tuple[str, ...], forces: object) -> tuple[str, float]:
    """Return the body name and magnitude of the largest measured contact force."""
    values = np.asarray(forces, dtype=np.float64)
    if values.shape != (len(body_names), 3) or not np.isfinite(values).all() or not body_names:
        raise ValueError("body forces must be a finite (number-of-bodies, 3) array")
    magnitudes = np.linalg.norm(values, axis=1)
    index = int(np.argmax(magnitudes))
    return str(body_names[index]), float(magnitudes[index])


__all__ = ["advance_progress", "initial_active_action", "peak_body_force",
           "trajectory_action_index",
           "to_numpy", "top_down_hand_pose"]
