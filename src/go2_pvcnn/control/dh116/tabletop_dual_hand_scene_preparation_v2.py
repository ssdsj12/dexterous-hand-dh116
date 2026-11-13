"""Opt-in legacy external-wire layout metadata; original v1 source is frozen.

Enter only around the separately gated post-App builder. This module creates
no scene, native resource, physics step or rigid wire. The authored point mean
is bookkeeping for the original spawn function, not the solver's rod seed.
"""
from contextlib import contextmanager


@contextmanager
def external_wire_layout():
    """Temporarily supply wire_center to the unchanged spawn_parameters path."""
    from . import tabletop_dual_hand_scene_preparation as scene
    import numpy as np
    previous=scene.spawn_parameters
    def spawn_parameters_with_wire_metadata(project,prepared):
        args,layout,sources=previous(project,prepared)
        points=np.asarray(layout["wire_points"],dtype=float)
        if points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<2 or not np.isfinite(points).all():
            raise ValueError("finite authored wire_points with shape (N>=2,3) required")
        with np.errstate(over="ignore",invalid="ignore"):
            center=points.mean(axis=0)
        if not np.isfinite(center).all():
            raise ValueError("authored wire_points mean must be finite")
        if "wire_center" in layout:
            old=np.asarray(layout["wire_center"],dtype=float)
            if old.shape!=(3,) or not np.allclose(old,center,rtol=0,atol=1e-12):
                raise ValueError("existing wire_center differs from authored wire_points mean")
        return args,dict(layout,wire_center=center.copy()),sources
    try:
        scene.spawn_parameters=spawn_parameters_with_wire_metadata
        yield
    finally:
        scene.spawn_parameters=previous
        scene=previous=spawn_parameters_with_wire_metadata=None
