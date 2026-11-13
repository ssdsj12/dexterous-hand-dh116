"""Continuous piecewise-trilinear line audit in the original SDF local frame.

This evaluates the float64 reference interpolant of the stored lattice. It is
not a CUDA bitwise proof or a bound on physical rod motion. Outside the half-open
lattice the actual solver sampler returns false; that is not mesh clearance.
"""
from itertools import product

import numpy as np
from numpy.polynomial import polynomial as poly

VERSION = "dh116-sdf-segment-extrema-reference-v1"


def _inputs(grid, start, end, radius):
    values = np.asarray(grid["values"], dtype=float)
    origin = np.asarray(grid["origin"], dtype=float)
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    h, r = float(grid["spacing"]), float(radius)
    if (values.ndim != 3 or min(values.shape) < 2 or origin.shape != (3,)
            or a.shape != (3,) or b.shape != (3,) or not np.isfinite(h) or h <= 0
            or not np.isfinite(r) or r < 0
            or not all(np.isfinite(x).all() for x in (values, origin, a, b))):
        raise ValueError("finite 3D grid, local endpoints, positive spacing and nonnegative radius required")
    return values, (a-origin)/h, (b-origin)/h, r


def _clip(a, b, high):
    lo, hi = 0., 1.
    for x, d, upper in zip(a, b-a, high):
        if d == 0:
            if not 0 <= x < upper:
                return None
        else:
            t0, t1 = -x/d, (upper-x)/d
            lo, hi = max(lo, min(t0,t1)), min(hi, max(t0,t1))
    if lo > hi:
        return None
    # A singleton on the excluded upper boundary has no active sample.
    if lo == hi and not np.all((a+(b-a)*lo >= 0) & (a+(b-a)*lo < high)):
        return None
    return lo, hi


def _coefficients(values, cell, p, delta):
    result = np.zeros(4)
    for bits in product((0,1), repeat=3):
        term = np.array([1.])
        for k, bit in enumerate(bits):
            term = poly.polymul(term, [p[k], delta[k]] if bit else [1-p[k], -delta[k]])
        result[:len(term)] += float(values[tuple(cell+bits)])*term
    return result


def segment_extrema(grid, start_local, end_local, *, radius=0.):
    """Evaluate all lattice boundaries and interior stationary points, once.

    Each segment polynomial uses u in [0,1]; its fraction interval refers to the
    caller's complete line. Boundary minima may be infima on an open lattice face.
    No discretely sampled minimum is promoted to a continuous safety certificate.
    """
    values, a, b, r = _inputs(grid, start_local, end_local, radius)
    high = np.array(values.shape)-1
    result = dict(version=VERSION, segments=[], minimum_gap_m=None,
                  minimum_at_fraction=None, minimum_cell=None, domain_fraction=0.,
                  numeric_model="stored_lattice_float64_reference_interpolant",
                  outside_domain_semantics="solver_sdf_sample_returns_false_not_physical_clearance",
                  gpu_bitwise_verified=False, native_contact_verified=False)
    clipped = _clip(a,b,high)
    if clipped is None:
        return result
    lo, hi = clipped
    result["domain_fraction"] = float(hi-lo)
    cuts = [lo,hi]
    for x,d,n in zip(a,b-a,high):
        if d != 0:
            cuts.extend(t for k in range(1,int(n)) if lo < (t:=(k-x)/d) < hi)
    cuts = np.unique(cuts).tolist()
    intervals = list(zip(cuts[:-1],cuts[1:])) if len(cuts)>1 else [(lo,hi)]
    for left,right in intervals:
        middle = a+(b-a)*((left+right)/2)
        cell = np.clip(np.floor(middle).astype(int),0,high-1)
        p, delta = a+(b-a)*left-cell, (b-a)*(right-left)
        coeff = _coefficients(values,cell,p,delta)
        derivative = poly.polyder(coeff)
        roots = poly.polyroots(derivative) if np.any(derivative != 0) else []
        stationary = sorted(float(x.real) for x in roots
                            if abs(x.imag) <= 1e-10 and 0 < x.real < 1)
        candidates = [0.,1.,*stationary]
        phi = [float(poly.polyval(u,coeff)) for u in candidates]
        index = int(np.argmin(phi))
        fraction = float(left+(right-left)*candidates[index])
        gap = phi[index]-r
        row = dict(start_fraction=float(left),end_fraction=float(right),cell=cell.tolist(),
                   phi_coefficients_ascending=coeff.tolist(),
                   stationary_fractions=[float(left+(right-left)*u) for u in stationary],
                   minimum_gap_m=gap,minimum_at_fraction=fraction)
        result["segments"].append(row)
        if result["minimum_gap_m"] is None or gap < result["minimum_gap_m"]:
            result.update(minimum_gap_m=gap,minimum_at_fraction=fraction,minimum_cell=cell.tolist())
    return result
