"""URDF-linear DH116 V2 approximation, in radians. Not a hardware model.

The caller must validate the USD joint names and mimic coefficients before use.
No actuator gains, effort, velocity or default drive parameters are supplied.
"""
import numpy as np

ACTIVE = ('finger11', 'finger12', 'finger21', 'finger31', 'finger41', 'finger51')
JOINTS = ('finger11', 'finger12', 'finger13', 'finger21', 'finger22',
          'finger31', 'finger32', 'finger41', 'finger42', 'finger51', 'finger52')
MIMIC = (('finger13', 'finger12', .88178789, -.00697725445),
         ('finger22', 'finger21', .9206, .011072),
         ('finger32', 'finger31', .9206, .011072),
         ('finger42', 'finger41', .9206, .011072),
         ('finger52', 'finger51', .9206, .011072))

# Source V2 URDF limits in JOINTS order; not measured actuator capability.
_SOURCE_LOWER = (0., 0., 0., 0., .011072, 0., .011072, 0., .011072, 0., .011072)
_SOURCE_UPPER = (1.588, 1.030, 1.13, 1.57, 1.950714, 1.57, 1.950714,
                 1.57, 1.950714, 1.57, 1.950714)
# DH116 V2.0 manual section 3.3, NOT the older DH116 or DH2116 manual.
_MANUAL_UPPER_DEG = (80., 90., 70., 70., 70., 70.)


def active_limits():
    """Return fresh lower/upper rad arrays for this linear simulation model.

    Intersect V2 manual active limits, source joint limits, and all affine
    follower limits. These are not collision clearance or hardware approval.
    The Excel table domain does not apply to this independent URDF model.
    """
    indices = [JOINTS.index(name) for name in ACTIVE]
    lower = np.array(_SOURCE_LOWER)[indices]
    upper = np.minimum(np.array(_SOURCE_UPPER)[indices],
                       np.deg2rad(_MANUAL_UPPER_DEG))
    for follower, master, ratio, offset in MIMIC:
        i, j = ACTIVE.index(master), JOINTS.index(follower)
        # All five source ratios are positive.
        lower[i] = max(lower[i], (_SOURCE_LOWER[j]-offset)/ratio)
        upper[i] = min(upper[i], (_SOURCE_UPPER[j]-offset)/ratio)
    return lower, upper


def coupling_matrix(names=JOINTS):
    """dq_full / dq_active: named full-joint rows, six active columns."""
    if len(names) != 11 or len(set(names)) != 11 or set(names) != set(JOINTS):
        raise ValueError('expected exactly the eleven DH116 joints')
    matrix = np.zeros((11, 6))
    for i, name in enumerate(names):
        if name in ACTIVE:
            matrix[i, ACTIVE.index(name)] = 1.
        else:
            _, master, ratio, _ = next(m for m in MIMIC if m[0] == name)
            matrix[i, ACTIVE.index(master)] = ratio
    return matrix


def expand(q):
    """Expand six active radians; reject infeasible states without clamping.

    In particular the thumb master cannot be zero: its affine offset would
    violate the follower's zero lower limit. Caller chooses a feasible state.
    """
    q = np.asarray(q, dtype=np.float64)
    lower, upper = active_limits()
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('expected six finite active radians')
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError('active state violates DH116 coupled limits')
    full = coupling_matrix() @ q
    for name, _, _, offset in MIMIC:
        full[JOINTS.index(name)] += offset
    return full


def fold_jacobian(full_jacobian, joint_names):
    """Fold (..., spatial_rows, 11) to (..., spatial_rows, 6), same frame.

    Input must already refer to the desired link/contact point, not an
    unshifted COM Jacobian. This function does not change the point or frame.
    """
    j = np.asarray(full_jacobian, dtype=np.float64)
    if j.ndim < 2 or j.shape[-1] != 11 or j.shape[-2] not in (3, 6) or not np.isfinite(j).all():
        raise ValueError('expected finite (...,3|6,11) Jacobian')
    return j @ coupling_matrix(joint_names)
