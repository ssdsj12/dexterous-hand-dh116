"""Small, gravity-free simulation probe, not DH116 hardware parameters."""
import numpy as np
from .linear_coupling import ACTIVE, JOINTS, expand

STIFFNESS = 1.0
DAMPING = .05
EFFORT_LIMIT = .1
VELOCITY_LIMIT = .3


def reference(t, active_index=2):
    if isinstance(active_index, bool) or not isinstance(active_index, int) or not 0 <= active_index < 6:
        raise ValueError('expected active joint index 0..5')
    if not np.isfinite(t) or t < 0:
        raise ValueError('expected finite nonnegative time')
    q = np.full(6, .2)
    if 2 < t < 6:
        q[active_index] += .05*(1-np.cos(np.pi*(t-2)/2))
    return q


def contact_metrics(forces, expected_bodies):
    """Net normal-force alarm; does not prove pairwise contact clearance."""
    f = np.asarray(forces)
    if f.shape != (expected_bodies, 3) or expected_bodies < 1 or not np.isfinite(f).all():
        return dict(passed=False, reason='invalid_contact_data')
    peak = float(np.max(np.linalg.norm(f,axis=-1)))
    return dict(passed=peak <= .01, max_net_contact_n=peak)


def metrics(full, velocity, target):
    full, velocity = np.asarray(full), np.asarray(velocity)
    if full.shape != (11,) or velocity.shape != (11,) or not np.isfinite(full).all() or not np.isfinite(velocity).all():
        return dict(passed=False, reason='invalid_state')
    active = full[[JOINTS.index(n) for n in ACTIVE]]
    try:
        expected = expand(active)
    except ValueError:
        return dict(passed=False, reason='active_limits')
    coupling = float(np.max(np.abs(full-expected)))
    tracking = float(np.max(np.abs(full-expand(target))))
    speed = float(np.max(np.abs(velocity)))
    return dict(passed=bool(coupling <= np.deg2rad(.5) and tracking <= np.deg2rad(1) and speed <= .31),
                coupling_rad=coupling, tracking_rad=tracking, speed_rad_s=speed)
