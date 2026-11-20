"""Opt-in exact-scope distance certification for one convex solid and finite segment."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from scripts.audit_dh116_pinch_initial_pose import finite_array


def _certify_segment_distance(vertices, start, axis, planes, estimate, scale):
    """Check a feasible upper bound and a full-vertex separating lower bound."""
    point = estimate[:3].copy()
    endpoint = start + np.clip(estimate[3], 0., 1.) * axis
    normals, offsets = planes[:, :3], planes[:, 3]
    if np.any(offsets >= 0.):  # The constructed interior is the origin.
        return None
    heights = normals @ point
    positive = heights > 0.
    if positive.any():
        factor = min(1., float(np.min(-offsets[positive] / heights[positive])))
        point *= factor * (1. - 1e-12)
    if np.any(normals @ point + offsets > 0.):
        return None

    delta = point - endpoint
    norm = float(np.linalg.norm(delta))
    rounding_m = 1e-12
    lower = 0.
    if norm > 0.:
        # A non-expansive direction makes support separation a distance bound.
        normal = delta / (norm * (1. + 1e-12))
        separation = (vertices @ normal).min() - max(start @ normal, (start + axis) @ normal)
        lower = max(0., float(separation) * scale - rounding_m)
    upper = norm * scale + rounding_m
    # Keep the original capsule contact comparison's nanometre precision.
    if not 0. <= upper - lower <= 1e-9:
        return None
    return point, endpoint, lower


def segment_hull_distance(vertices, start, end):
    """Resolve status 8 only when independent bounds match within one nanometre."""
    v, a = finite_array(vertices), finite_array(start, (3,))
    axis = finite_array(end, (3,)) - a
    origin = v.mean(0)
    scale = max(float(np.ptp(v, axis=0).max()), float(np.linalg.norm(axis)), .001)
    v, a, axis = (v - origin) / scale, (a - origin) / scale, axis / scale
    planes = ConvexHull(v).equations
    normals, offsets = planes[:, :3], planes[:, 3]
    constraints = np.column_stack([normals, np.zeros(len(planes))])

    def residual(x):
        return x[:3] - a - x[3] * axis

    def objective(x):
        d = residual(x)
        return float(d @ d)

    def jacobian(x):
        d = residual(x)
        return np.r_[2 * d, -2 * axis @ d]

    result = minimize(objective, np.r_[v.mean(0), .5], jac=jacobian,
        bounds=[(None, None)] * 3 + [(0, 1)],
        constraints=dict(type="ineq", fun=lambda x: -constraints @ x - offsets,
                         jac=lambda x: -constraints),
        method="SLSQP", options=dict(ftol=1e-14, maxiter=200))
    estimate = result.x
    if (np.isfinite(estimate).all() and -.00000001 <= estimate[3] <= 1.00000001
            and np.max(constraints @ estimate + offsets) <= 1e-8):
        if result.success:
            return scale * float(np.linalg.norm(residual(estimate)))
        if result.status == 8:
            checked = _certify_segment_distance(v, a, axis, planes, estimate, scale)
            if checked is not None:
                return checked[2]
    raise ValueError("continuous patch/convex-hull distance did not converge")
