"""CPU static wrench proposals; never measured force or hardware calibration.

Each contact has a supplied object-side force ray and a measured capacity.
``force_to_joint_torque_nm_per_n[j, c]`` must project one Newton along the
NORMALIZED ray through the actual contact Jacobian. Original motor budgets
are required. No default equal hand share or inferred motor capacity exists.
"""
import numpy as np


def _numeric_array(value):
    # Inspect before float conversion erases bools in mixed lists/object arrays.
    pending, seen = [np.asarray(value, dtype=object)], set()
    while pending:
        item = pending.pop()
        if isinstance(item, (bool, np.bool_)):
            raise ValueError('boolean is not a measured numeric value')
        if isinstance(item, (np.ndarray, tuple, list)) and id(item) not in seen:
            seen.add(id(item))
            pending.extend(item.flat if isinstance(item, np.ndarray) else item)
    return np.asarray(value, dtype=float)


def _bounded_residual(equilibrium, desired, capacity, torque, limits, weight, lever_arm):
    """Minimize dimensionless L-infinity residual under the ORIGINAL bounds.

    Forces are divided by measured weight; moments by weight times the largest
    measured COM lever arm. With zero lever arm every moment row is identically
    zero and is left unscaled; no artificial length/material parameter is used.
    This second LP is diagnostic only and never supplies a command/proposal.
    """
    from scipy.optimize import linprog
    count = len(capacity)
    moment_scale = weight * lever_arm if lever_arm > 0. else None
    scale = np.array([weight] * 3 + [moment_scale if moment_scale is not None else 1.] * 3)
    scaled, target = equilibrium / scale[:, None], desired / scale
    bounded_torque = np.concatenate((torque, -torque))
    constraints = np.concatenate((
        np.column_stack((bounded_torque, np.zeros(len(bounded_torque)))),
        np.column_stack((scaled, -np.ones(6))),
        np.column_stack((-scaled, -np.ones(6)))))
    upper = np.concatenate((limits, limits, target, -target))
    diagnostic = linprog(np.r_[np.zeros(count), 1.], A_ub=constraints, b_ub=upper,
        bounds=list(zip(np.zeros(count), capacity)) + [(0., None)], method='highs')
    if not diagnostic.success:
        return None
    magnitude = diagnostic.x[:count]
    residual = equilibrium @ magnitude - desired
    joint_torque = torque @ magnitude
    normalized = residual / scale
    if (not all(np.isfinite(v).all() for v in (magnitude, residual, joint_torque, normalized))
            or (magnitude < -1e-8).any() or (magnitude > capacity + 1e-8).any()
            or (np.abs(joint_torque) > limits + 1e-8).any()
            or abs(np.max(np.abs(normalized)) - diagnostic.x[-1]) > 1e-7):
        return None
    return dict(evidence_kind='bounded_minimum_wrench_residual_not_command',
        force_scale_n=float(weight), characteristic_lever_arm_m=float(lever_arm),
        moment_scale_nm=float(moment_scale) if moment_scale is not None else None,
        moment_rows_identically_zero=lever_arm == 0.,
        bounded_candidate_contact_magnitude_n=magnitude.tolist(),
        bounded_candidate_joint_torque_nm=joint_torque.tolist(),
        force_residual_n=residual[:3].tolist(), moment_residual_nm=residual[3:].tolist(),
        normalized_residual=normalized.tolist(),
        normalized_linf_residual=float(np.max(np.abs(normalized))), solver_status=int(diagnostic.status))


def allocate_support_wrench(*, points_w_m, object_com_w_m, directions_w,
                            capacities_n, object_weight_n, hand_ids,
                            force_to_joint_torque_nm_per_n,
                            original_effort_limit_nm):
    """Solve force AND COM-moment equilibrium with unilateral bounded rays.

    Each input contact owns its capacity once; this minimal interface does not
    expand several rays per pad into a friction cone with shared capacities.
    Capacities and directions are inputs, not new material parameters.
    Infeasible or unknown inputs return no proposed forces; a clipped solution
    is never labelled balanced. Infeasible equilibrium includes a separately
    labelled bounded residual diagnostic, in N and Nm with explicit dimensionless
    normalization. This does not command any simulator actuator.
    """
    result = dict(evidence_kind='allocation_proposal_not_measured_force',
        support_bearing_attested=False, hardware_calibrated=False, feasible=False,
        reason=None, target_hand_fz_n=None, contact_force_w_n=None,
        contact_magnitude_n=None, target_joint_torque_nm=None,
        force_residual_n=None, moment_residual_nm=None, infeasibility_diagnostic=None)
    if force_to_joint_torque_nm_per_n is None or original_effort_limit_nm is None:
        return dict(result, reason='unknown_motor_capacity')
    try:
        points, com, direction, capacity, torque, limits, weight = [_numeric_array(v)
            for v in (points_w_m, object_com_w_m, directions_w, capacities_n,
                      force_to_joint_torque_nm_per_n, original_effort_limit_nm, object_weight_n)]
        count = len(points)
        if (points.shape != (count, 3) or count < 1 or com.shape != (3,)
                or direction.shape != points.shape or capacity.shape != (count,)
                or limits.ndim != 1 or not len(limits)
                or torque.shape != (len(limits), count)
                or len(hand_ids) != count or any(type(h) is not int or h not in (0, 1) for h in hand_ids)
                or not all(np.isfinite(v).all() for v in (points, com, direction, capacity, torque, limits))
                or weight.shape != () or not np.isfinite(weight) or weight <= 0
                or (capacity < 0).any() or (limits <= 0).any()):
            return dict(result, reason='invalid_measurement')
        # Scale each ray before taking its norm: even a finite 1e308 ray has a
        # perfectly well-defined unit direction and must not silently become 0.
        largest = np.max(np.abs(direction), axis=-1)
        if (largest == 0.).any():
            return dict(result, reason='invalid_measurement')
        unit_scale = direction / largest[:, None]
        length = np.linalg.norm(unit_scale, axis=-1)
        if (largest <= 1e-12 / length).any():
            return dict(result, reason='invalid_measurement')
        direction = unit_scale / length[:, None]
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            levers = points - com
            equilibrium = np.concatenate((direction.T, np.cross(levers, direction).T), axis=0)
            lever_arm = float(np.max(np.linalg.norm(levers, axis=-1)))
            weight = float(weight)
            if not np.isfinite(weight * lever_arm):
                return dict(result, reason='invalid_measurement')
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return dict(result, reason='invalid_measurement')
    desired = np.array([0., 0., weight, 0., 0., 0.])
    from scipy.optimize import linprog
    solution = linprog(np.ones(count), A_ub=np.concatenate((torque, -torque)),
        b_ub=np.concatenate((limits, limits)), A_eq=equilibrium, b_eq=desired,
        bounds=list(zip(np.zeros(count), capacity)), method='highs')
    if not solution.success:
        # A numerical/solver failure is not evidence of physical infeasibility.
        if solution.status != 2:
            return dict(result, reason='equilibrium_solver_failed', solver_status=int(solution.status))
        diagnostic = _bounded_residual(equilibrium, desired, capacity, torque, limits, weight, lever_arm)
        return dict(result, reason='equilibrium_infeasible', solver_status=int(solution.status),
            force_residual_n=diagnostic['force_residual_n'] if diagnostic is not None else None,
            moment_residual_nm=diagnostic['moment_residual_nm'] if diagnostic is not None else None,
            infeasibility_diagnostic=diagnostic)
    magnitude = solution.x
    residual = equilibrium @ magnitude - desired
    joint_torque = torque @ magnitude
    # Independently check the solver output; never publish a numerically bad LP.
    if (not np.isfinite(magnitude).all() or np.max(np.abs(residual)) > 1e-7
            or (magnitude < -1e-8).any() or (magnitude > capacity + 1e-8).any()
            or (np.abs(joint_torque) > limits + 1e-8).any()):
        return dict(result, reason='solver_residual_rejected',
                    force_residual_n=residual[:3].tolist(), moment_residual_nm=residual[3:].tolist())
    forces = magnitude[:, None] * direction
    return dict(result, feasible=True, reason='equilibrium_proposal',
        target_hand_fz_n=[float(forces[np.asarray(hand_ids) == h, 2].sum()) for h in (0, 1)],
        contact_force_w_n=forces.tolist(), contact_magnitude_n=magnitude.tolist(),
        target_joint_torque_nm=joint_torque.tolist(), force_residual_n=residual[:3].tolist(),
        moment_residual_nm=residual[3:].tolist(), solver_status=int(solution.status))
