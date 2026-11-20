"""Conservative separation for a convex envelope and a finite flexible rod."""

import numpy as np

from scripts.audit_dh116_tripod_fixed_root_path import certify_path
from .tabletop_polyline_support_certificate import NUMERIC_GUARD, directions


def _unit_directions():
    azimuth = np.arange(64) * (2 * np.pi / 64)
    axial = np.array([0., -.2, .2, -.5, .5, -.8, .8])[:, None]
    radial = np.sqrt(1. - axial ** 2)
    rings = np.stack(np.broadcast_arrays(radial * np.cos(azimuth), axial,
                                         radial * np.sin(azimuth)), axis=-1)
    directions = np.vstack((rings.reshape(-1, 3), [0., 1., 0.], [0., -1., 0.]))
    # Shrink rather than risk roundoff giving a direction norm above one.
    return directions / (1. + 1e-12)


_DIRECTIONS = _unit_directions()


class RodCapsuleSeparation:
    """Bound each segment separately over a declared envelope of node motions.

    The motion array is supplied by the caller. Zero bounds compare only to
    the fixed snapshot. A predicted bound is not verified dynamic tracking.
    """

    def __init__(self, nodes_m, radius_m, *, node_motion_bound_m):
        points = np.asarray(nodes_m, dtype=float)
        motion = np.asarray(node_motion_bound_m, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all()
                or not np.isfinite(radius_m) or radius_m <= 0.):
            raise ValueError("finite rod centerline and positive radius required")
        if motion.shape != (len(points),) or not np.isfinite(motion).all() or np.any(motion < 0.):
            raise ValueError("finite nonnegative motion bound required for every rod node")
        self._origin = points[0].copy()
        self._nodes = points - self._origin
        self._motion = motion[:, None].copy()
        self._radius = float(radius_m)
        self._obstacle_support = self._segment_support(_DIRECTIONS)

    def _segment_support(self, normals):
        support = self._nodes @ normals.T + self._motion
        # A convex segment attains the largest projection at one of its ends.
        return np.maximum(support[:-1], support[1:]) + self._radius

    @staticmethod
    def _lower(vertices, normals, obstacle_support):
        hull_support = (vertices @ normals.T).min(axis=0)
        separation = hull_support[None, :] - obstacle_support
        # For each segment pick the strongest valid projection. The rod is
        # the union of its segment capsules, so the least gap controls.
        return float(separation.max(axis=1).min() - NUMERIC_GUARD)

    def gap_lower_m(self, hull_vertices_m):
        vertices = np.asarray(hull_vertices_m, dtype=float)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all()):
            raise ValueError("nonempty finite convex envelope vertices required")
        vertices = vertices - self._origin
        lower = self._lower(vertices, _DIRECTIONS, self._obstacle_support)
        if lower < 0.:
            refined, _ = directions(vertices, self._nodes)
            lower = max(lower, self._lower(vertices, refined, self._segment_support(refined)))
        return lower

    def certify_path(self, world_vertices, travel_m, *, margin_m):
        if not callable(world_vertices) or not np.isfinite(margin_m) or margin_m < 0.:
            raise ValueError("convex trajectory and finite nonnegative separation margin required")

        def sample(fraction):
            return {name: self.gap_lower_m(vertices)
                    for name, vertices in world_vertices(fraction).items()}

        return certify_path(sample, travel_m, dict.fromkeys(travel_m, float(margin_m)),
                            max_depth=8, max_samples=129)
