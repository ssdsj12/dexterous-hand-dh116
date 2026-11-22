"""Certified gap against the actual finite material segments of a free wire."""
import numpy as np

from scripts.audit_dh116_certified_segment_distance import segment_hull_bounds
from .tabletop_wire_live_palm_approach import wire_tube


NONCONTACT_GAP_M = .00025
SELECTED_PENETRATION_M = .00015
ROUNDING_M = 1e-12
MAX_FRAMES = 9


class WireClearance:
    """Use the inexpensive full-hull projection before checking individual segments."""

    def __init__(self, nodes, *, radius_m):
        self.nodes = np.array(nodes, dtype=float, copy=True)
        if self.nodes.ndim != 3 or self.nodes.shape[1:] != (41, 3) or not 1 <= len(self.nodes) <= MAX_FRAMES:
            raise ValueError("one to nine complete sets of 41 material nodes required")
        self.normals, self.upper = wire_tube(self.nodes, radius_m=radius_m)
        self.radius_m = float(radius_m)

    def certify(self, vertices, *, selected, reserve_m=0.):
        """Certify the original gap plus an optional nonnegative displacement reserve."""
        v = np.asarray(vertices, dtype=float)
        if (v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 4
                or not np.isfinite(v).all() or type(selected) is not bool):
            raise ValueError("finite link hull vertices and explicit contact status required")
        if (isinstance(reserve_m, (bool, np.bool_))
                or not isinstance(reserve_m, (int, float, np.integer, np.floating))
                or not np.isfinite(reserve_m) or reserve_m < 0.):
            raise ValueError("finite nonnegative motion reserve required")
        threshold = -SELECTED_PENETRATION_M if selected else NONCONTACT_GAP_M
        required = threshold + float(reserve_m)
        evidence = {"reserve_m": float(reserve_m)} if reserve_m else {}

        def accepted(gap):
            return gap >= required if selected else gap > required

        support = (v @ self.normals.T).min(axis=0)
        lower = (support[None] - self.upper).max(axis=1) - ROUNDING_M
        fast = float(lower.min())
        checked = 0
        terminal = None
        for group, bound in enumerate(lower):
            if accepted(bound):
                continue
            lower_segment = np.inf
            for frame in self.nodes:
                for edge in (2*group, 2*group+1):
                    result = segment_hull_bounds(v, frame[edge], frame[edge+1])
                    margin = result["lower_m"] - self.radius_m - ROUNDING_M
                    checked += 1
                    lower_segment = min(lower_segment, margin)
                    if not accepted(margin):
                        terminal = {"group": group, "segment": edge, "upper_m": result["upper_m"]-self.radius_m}
                        break
                if terminal is not None:
                    break
            if terminal is not None:
                return dict(passed=False, lower_m=float(min(fast, lower_segment)), fast_lower_m=fast,
                            segments_checked=checked, threshold_m=threshold, failed_segment=terminal, **evidence)
            lower[group] = max(bound, lower_segment)
        return dict(passed=True, lower_m=float(lower.min()), fast_lower_m=fast,
                    segments_checked=checked, threshold_m=threshold, failed_segment=None, **evidence)
