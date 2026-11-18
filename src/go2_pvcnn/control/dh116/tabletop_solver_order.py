"""Explicit PhysX articulation ordering for contact/velocity convergence."""

CONTRACT = dict(version='articulation_contact_and_velocity_last_v1',
    solve_articulation_contact_last=True,
    purpose='resolve_contact_and_joint_velocity_after_other_constraints',
    measured_safety_guards='unchanged',
    guaranteed_measured_velocity_bound=False)


def verify_solver_order(physics_context, requested):
    """Fail before stepping if the scene did not receive the requested order."""
    if type(requested) is not bool:
        raise RuntimeError('articulation solver order requested must be bool')
    getter = getattr(physics_context, 'get_solve_articulation_contact_last', None)
    if getter is None:
        if requested:
            raise RuntimeError('this Isaac Sim version cannot enable or verify articulation contact-last ordering')
        return dict(requested=False, actual=None, verified=False,
                    reason='API unavailable; using engine default ordering')
    actual = getter()
    if type(requested) is not bool or type(actual) is not bool or actual != requested:
        raise RuntimeError(f'articulation solver order mismatch: requested={requested!r}, actual={actual!r}')
    return dict(requested=requested, actual=actual, verified=True)
