"""Explicit teacher velocity allocation with hard normal-task priority.

No Isaac/PPO integration. Inputs use float64 and shapes (..., rows, dof), with
matching batch axes (typically two environments by two hands). Bias and box
vectors may broadcast over those batch axes. Joint/wrist scaling belongs to
the caller and must be applied consistently to both task matrices and bounds.
"""
import math

import torch

from .tabletop_manual_joint_contact import bounded_dls


OK = 0
PRIMARY_TARGET_UNMET = 1
PRIMARY_DIRECTION_INFEASIBLE = 2
SECONDARY_FALLBACK = 3


def _row_max(value):
    if value.shape[-1] == 0:
        return value.new_zeros(value.shape[:-1])
    return value.abs().amax(-1)


@torch.no_grad()
def prioritized_velocity(normal_J, normal_v, secondary_A, secondary_b,
                         bias, lower, upper, damping, *,
                         equality_tolerance=1e-8, direction_tolerance=1e-12):
    """Return ``(velocity, info)`` without changing the caller's velocity box.

    Primary is exactly the existing bounded_dls(normal_J, normal_v, bias,
    lower, upper, damping). Secondary steps preserve the ACHIEVED primary
    task, including inactive/zero rows, using an undamped Moore-Penrose null
    projector. Damping is only used for the primary/secondary least squares,
    never for that projector. The secondary loop takes at most dof+1 steps.

    Per-batch info fields:
      primary, primary_achieved, primary_error: original primary solution/task.
      primary_target_met: max absolute primary target error <= equality_tolerance.
      primary_direction_infeasible: a nonzero requested correction is reversed
        by the RETURNED primary solution. No secondary step is attempted for it.
        This is not a proof of global box infeasibility: bounded_dls is the
        existing primary allocator. A blocked zero correction is target-unmet.
      status: OK=0, PRIMARY_TARGET_UNMET=1, PRIMARY_DIRECTION_INFEASIBLE=2,
        SECONDARY_FALLBACK=3 (failed numerical preservation/box check).
      fallback_to_primary, normal_preservation_error, secondary_iterations.

    Reversed primary results are returned explicitly, not replaced by zero
    (which might violate a positive wrist lower bound). The teacher must
    handle that status: e.g. relax its wrist ascent floor within the original
    physical box and retry, then stop/record the hand if still reversed.
    ``OK`` concerns this mathematical solve, not measured force or motion.
    """
    values = (normal_J, normal_v, secondary_A, secondary_b, bias, lower, upper)
    if (not all(isinstance(x, torch.Tensor) for x in values)
            or normal_J.ndim < 2 or normal_J.shape[-1] == 0):
        raise ValueError('matching float64 task matrices and velocity vectors required')
    if any(x.dtype != torch.float64 or x.device != normal_J.device for x in values):
        raise ValueError('all tasks and box vectors must be float64 on the same device')
    if not all(bool(torch.isfinite(x).all()) for x in values):
        raise ValueError('finite tasks, bias and box required')
    if (not math.isfinite(damping) or damping <= 0
            or not 0 < equality_tolerance <= 1e-8
            or not math.isfinite(direction_tolerance) or direction_tolerance < 0):
        raise ValueError('positive damping, equality tolerance <=1e-8 and nonnegative direction tolerance required')
    batch = normal_J.shape[:-2]
    rows, dof = normal_J.shape[-2:]
    if (normal_v.shape != batch+(rows,) or secondary_A.ndim != normal_J.ndim
            or secondary_A.shape[:-2] != batch or secondary_A.shape[-1] != dof
            or secondary_b.shape != secondary_A.shape[:-1]):
        raise ValueError('task batch, row and dof shapes must match')
    try:
        bias, lower, upper = (torch.broadcast_to(x, batch+(dof,)) for x in (bias, lower, upper))
    except RuntimeError as error:
        raise ValueError('bias and box must broadcast to the task velocity shape') from error
    if bool((lower > upper).any()):
        raise ValueError('velocity box lower bounds exceed upper bounds')

    primary = bounded_dls(normal_J, normal_v, bias, lower, upper, damping)
    if not bool(torch.isfinite(primary).all()):
        raise ValueError('primary solve returned nonfinite velocity')
    achieved = (normal_J @ primary.unsqueeze(-1)).squeeze(-1)
    primary_error = achieved-normal_v
    target_met = _row_max(primary_error) <= equality_tolerance
    reversed_primary = ((normal_v.abs() > direction_tolerance)
                        & (achieved*normal_v.sign() < -direction_tolerance)).any(-1)
    status = torch.where(target_met, OK, PRIMARY_TARGET_UNMET).to(torch.int64)
    status = torch.where(reversed_primary, PRIMARY_DIRECTION_INFEASIBLE, status)
    fallback = torch.zeros_like(reversed_primary)
    iterations = torch.zeros_like(status)
    result = primary.clone()

    count = primary.numel()//dof
    ns = normal_J.reshape(count, rows, dof)
    matrices = secondary_A.reshape(count, secondary_A.shape[-2], dof)
    goals = secondary_b.reshape(count, secondary_A.shape[-2])
    starts = primary.reshape(count, dof)
    lows, highs = lower.reshape(count, dof), upper.reshape(count, dof)
    results = result.reshape(count, dof)
    flat_status, flat_fallback, flat_iterations = status.reshape(-1), fallback.reshape(-1), iterations.reshape(-1)

    for slot in range(count):
        if bool(reversed_primary.reshape(-1)[slot]) or matrices.shape[-2] == 0:
            continue
        n, a, b = ns[slot], matrices[slot], goals[slot]
        start, lo, hi = starts[slot], lows[slot], highs[slot]
        current = start.clone()
        free = lo < hi  # Equality bounds are fixed from the outset.
        failed = False
        try:
            for _ in range(dof+1):
                if not bool(free.any()):
                    break
                flat_iterations[slot] += 1
                # P has zero rows/columns at pinned DOFs. There is deliberately
                # no damping in pinv: a soft inverse would leak normal motion.
                nf = n*free
                projector = torch.diag(free.to(n.dtype))-torch.linalg.pinv(nf) @ nf
                projector = projector*free[:, None]*free[None, :]
                projected_a = a @ projector
                residual = b-a @ current
                eye = torch.eye(a.shape[-2], dtype=a.dtype, device=a.device)
                dual = torch.linalg.solve(projected_a @ projected_a.T+damping**2*eye, residual)
                delta = projector @ projected_a.T @ dual
                # In finite precision an ill-conditioned normal matrix leaves
                # tiny non-null eigenvalues in P; the secondary solve can
                # amplify them. P is mathematically idempotent, so applying it
                # again removes that roundoff without a damped normal inverse.
                delta = projector @ delta
                if not bool(torch.isfinite(delta).all()):
                    failed = True
                    break
                if float(delta.abs().max()) <= 1e-12:
                    break
                ratios = torch.where(delta > 0, (hi-current)/delta,
                    torch.where(delta < 0, (lo-current)/delta, float('inf')))
                alpha = ratios.min().clamp(0., 1.)
                # Only roundoff is clipped: alpha already stops at the FIRST
                # boundary. Final normal equality is checked after this clip.
                current = (current+alpha*delta).maximum(lo).minimum(hi)
                if float(alpha) >= 1.:
                    break
                hit = free & (ratios <= alpha+1e-12)
                if not bool(hit.any()):
                    failed = True
                    break
                free &= ~hit
        except torch.linalg.LinAlgError:
            failed = True

        normal_before, normal_after = n @ start, n @ current
        leaked = bool(_row_max(normal_after-normal_before) > equality_tolerance)
        changed_sign = bool(((normal_before.abs() > direction_tolerance)
                            & (normal_after*normal_before.sign() < -direction_tolerance)).any())
        outside = bool(((current < lo) | (current > hi)).any())
        if (failed or leaked or changed_sign or outside
                or not bool(torch.isfinite(current).all())):
            flat_fallback[slot] = True
            flat_status[slot] = SECONDARY_FALLBACK
        else:
            results[slot] = current

    preservation_error = _row_max((normal_J @ result.unsqueeze(-1)).squeeze(-1)-achieved)
    return result, dict(primary=primary, primary_achieved=achieved, primary_error=primary_error,
        primary_target_met=target_met, primary_direction_infeasible=reversed_primary,
        status=status, fallback_to_primary=fallback,
        normal_preservation_error=preservation_error, secondary_iterations=iterations)
