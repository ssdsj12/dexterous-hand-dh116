"""Opt-in exact-scope distance certification for one convex solid and finite segment."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from scripts.audit_dh116_pinch_initial_pose import finite_array


_DISTANCE_ROUNDING_M = 1e-12


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
    rounding_m = _DISTANCE_ROUNDING_M
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


def _polish_segment_distance(vertices, start, axis, planes, estimate, scale):
    """Try at most 64 nearby feature sets; only the original certificate accepts."""
    from itertools import chain, combinations, islice
    constraints = np.vstack([
        np.column_stack([planes[:, :3], np.zeros(len(planes))]),
        [0., 0., 0., -1.], [0., 0., 0., 1.],
    ])
    offsets = np.r_[planes[:, 3], 0., -1.]
    slack = np.abs(constraints @ estimate + offsets)
    # Candidate selection only. This does not relax feasibility or distance bounds.
    active = np.flatnonzero(slack <= 1e-6)
    active = active[np.argsort(slack[active], kind="stable")]
    features = chain.from_iterable(combinations(active, size) for size in range(1, 5))
    residual_matrix = np.column_stack([np.eye(3), -axis])
    attempts = 0
    for selected in islice(features, 64):
        attempts += 1
        equality, values = constraints[list(selected)], offsets[list(selected)]
        kkt = np.block([
            [residual_matrix.T @ residual_matrix, equality.T],
            [equality, np.zeros((len(equality), len(equality)))],
        ])
        rhs = np.r_[residual_matrix.T @ start, -values]
        try:
            candidate = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:4]
        except np.linalg.LinAlgError:
            continue
        if (not np.isfinite(candidate).all()
                or np.max(constraints @ candidate + offsets) > 1e-8):
            continue
        checked = _certify_segment_distance(vertices, start, axis, planes, candidate, scale)
        if checked is not None:
            return checked, attempts
    return None, attempts


def segment_hull_bounds(vertices, start, end):
    """Certify distance bounds for a convex solid and one finite centreline segment.

    Optimizer success is not a certificate. Both accepted stop statuses require
    feasible witnesses and full-vertex support bounds within one nanometre.
    These are unsigned distances, not penetration depth or multi-frame coverage.
    """
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
        if (result.success and result.status == 0) or result.status == 8:
            checked = _certify_segment_distance(v, a, axis, planes, estimate, scale)
            method, attempts = "direct_support_bounds", 0
            if checked is None:
                checked, attempts = _polish_segment_distance(v, a, axis, planes, estimate, scale)
                method = "active_feature_polish"
            if checked is not None:
                point, endpoint, lower = checked
                return dict(
                    lower_m=lower,
                    upper_m=float(np.linalg.norm(point - endpoint)) * scale + _DISTANCE_ROUNDING_M,
                    hull_point_m=(point * scale + origin).tolist(),
                    segment_point_m=(endpoint * scale + origin).tolist(),
                    optimizer_status=int(result.status),
                    certificate_method=method, polish_candidates=attempts,
                )
    raise ValueError("continuous patch/convex-hull distance did not converge")


def segment_hull_distance(vertices, start, end):
    """Return only the independently certified lower bound for legacy callers."""
    return segment_hull_bounds(vertices, start, end)["lower_m"]


ENVELOPE_REFINEMENT_VERSION = "dh116-continuous-envelope-support-v1"


def validate_envelope_version(value):
    if value is not None and (type(value) is not str or value != ENVELOPE_REFINEMENT_VERSION):
        raise ValueError("explicit registered envelope refinement version required")


def rod_envelope_clearance(vertices, node_samples, *, max_penetration_m, max_qp_calls=40):
    """Certify separation from all 20 convex T-by-3-node envelopes plus 5 mm radius.

    Segment QP witnesses generate directions only. Acceptance always uses every
    vertex and every frame in each original group, never a subset upper bound.
    No native eligibility, contact force or trajectory-tracking claim is made.
    """
    from scripts.screen_dh116_tripod_supported_reset import archived_tube

    v = np.asarray(vertices, dtype=float)
    nodes = np.asarray(node_samples, dtype=float)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all()
            or nodes.ndim != 3 or nodes.shape[1:] != (41, 3) or not 1 <= len(nodes) <= 9
            or not np.isfinite(nodes).all() or not 0. < max_penetration_m <= .00015
            or type(max_qp_calls) is not int or not 0 <= max_qp_calls <= 40):
        raise ValueError("bounded finite full-envelope geometry and original contact limit required")
    # Work in one translated frame, including the QP witnesses.
    origin = v.mean(axis=0)
    v, nodes = v-origin, nodes-origin
    normals, upper = archived_tube(nodes)
    lower = np.max(np.min(v @ normals.T, axis=0)[None, :] - upper, axis=1)
    lower -= _DISTANCE_ROUNDING_M
    fast = float(lower.min())
    calls, witnesses = 0, []
    subset_upper = None

    def result(reason, error=None):
        report = dict(method=ENVELOPE_REFINEMENT_VERSION, passed=reason == "certified",
                      reason=reason, fast_lower_m=fast, lower_m=float(lower.min()),
                      group_lower_m=lower.tolist(), frame_count=len(nodes),
                      radius_m=.005, max_penetration_m=max_penetration_m,
                      qp_calls=calls, subset_upper_m=subset_upper, witnesses=witnesses)
        if error is not None:
            report["error"] = error
        return report

    for group in np.flatnonzero(lower < -max_penetration_m):
        start = int(group)*2
        support_nodes = nodes[:, start:start+3].reshape(-1, 3)
        for segment in (start, start+1):
            if calls >= max_qp_calls:
                return result("qp_budget_exhausted")
            calls += 1
            try:
                bound = segment_hull_bounds(v, nodes[-1, segment], nodes[-1, segment+1])
            except ValueError as exc:
                return result("distance_certificate_failed", str(exc))
            upper_gap = bound["upper_m"] - .005
            subset_upper = upper_gap if subset_upper is None else min(subset_upper, upper_gap)
            if upper_gap < -max_penetration_m:
                return result("subset_collision")
            delta = np.asarray(bound["hull_point_m"]) - np.asarray(bound["segment_point_m"])
            length = float(np.linalg.norm(delta))
            if not np.isfinite(length) or length == 0.:
                return result("unresolved_direction")
            normal = delta / (length*(1.+1e-12))
            support_lower = float(np.min(v @ normal) - np.max(support_nodes @ normal) - .005
                                  - _DISTANCE_ROUNDING_M)
            lower[group] = max(lower[group], support_lower)
            witnesses.append(dict(group=int(group), segment=segment, source_frame=len(nodes)-1,
                                  normal=normal.tolist(), support_lower_m=support_lower,
                                  segment_upper_m=upper_gap))
            if lower[group] >= -max_penetration_m:
                break
        if lower[group] < -max_penetration_m:
            return result("full_envelope_unresolved")
    return result("certified")
