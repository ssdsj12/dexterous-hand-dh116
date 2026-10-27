"""Opt-in solver headroom candidate, not a bound on measured joint motion."""

ACTIVE_VELOCITY_LIMIT_RAD_S = .3
LEGACY_PASSIVE_VELOCITY_LIMIT_RAD_S = 3.
PASSIVE_VELOCITY_HEADROOM_RAD_S = 2.97
CLOSE_COMMAND_RATE_RAD_S = .20

VELOCITY_HEADROOM_CONTRACT = dict(
    version='passive_velocity_headroom_candidate_v1',
    passive_velocity_limit_rad_s=PASSIVE_VELOCITY_HEADROOM_RAD_S,
    nominal_guard_limit_rad_s=3.,
    qualification='controlled_simulation_candidate',
    guaranteed_measured_velocity_bound=False,
)


def passive_velocity_limit(control_mode, requested=LEGACY_PASSIVE_VELOCITY_LIMIT_RAD_S):
    """Keep existing modes and defaults; accept only the reviewed candidate."""
    if (isinstance(requested, bool) or not isinstance(requested, (int, float))
            or requested not in (LEGACY_PASSIVE_VELOCITY_LIMIT_RAD_S,
                                 PASSIVE_VELOCITY_HEADROOM_RAD_S)):
        raise ValueError('passive velocity limit must be 3.0 or the 2.97 rad/s candidate')
    if control_mode in ('joint_target', 'adaptive_grasp'):
        return float(requested)
    return ACTIVE_VELOCITY_LIMIT_RAD_S
