"""Capture independent camera readbacks without moving the control boundary."""

from copy import deepcopy

import numpy as np


def capture_at_boundary(snapshot, render_once, read_clock, readers):
    """Copy all camera products from one observed render epoch and one physical state."""
    if not readers:
        raise ValueError("capture boundary has no render products")
    before = deepcopy(snapshot())
    if not {"physics_tick", "rod_step", "state_sha256"}.issubset(before):
        raise ValueError("capture boundary state identity is incomplete")
    previous = {side: read_clock(side) for side in readers}
    render_once()
    current = {side: read_clock(side) for side in readers}
    clock = next(iter(current.values()))
    if (not isinstance(clock, tuple) or not clock
            or any(not isinstance(stamp, tuple) or not stamp or len(stamp) != len(clock)
            for stamp in (*previous.values(), *current.values()))
            or any(stamp != clock or stamp[0] <= previous[side][0] for side, stamp in current.items())):
        raise ValueError("capture boundary render epoch is stale or differs by camera")
    before_read = deepcopy(snapshot())
    if before_read != before:
        raise ValueError("capture boundary state changed during render")
    captures = {
        side: {key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
               for key, value in read().items()}
        for side, read in readers.items()
    }
    after = {side: read_clock(side) for side in readers}
    after_read = deepcopy(snapshot())
    if any(stamp != clock for stamp in after.values()) or after_read != before:
        raise ValueError("capture boundary state or render clock changed during readback")
    timings = {side: {"before_render": previous[side], "before_read": current[side], "after_read": after[side]}
               for side in readers}
    return {
        "state": before, "render_clock": clock, "camera_render_clocks": timings, "captures": captures,
        "physical_state_trace": {"before_render": before, "before_read": before_read, "after_read": after_read},
    }


def pinhole_radial_to_z(radial_distance_m, intrinsics):
    """Convert only an undistorted pinhole distance image at continuous pixel centres."""
    radial = np.asarray(radial_distance_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if (radial.ndim != 2 or not radial.size or k.shape != (3, 3) or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0 or not np.array_equal(k[2], [0., 0., 1.])):
        raise ValueError("pinhole depth requires a nonempty image and valid intrinsics")
    rows, cols = np.indices(radial.shape, dtype=np.float64)
    pixels = np.stack((cols + .5, rows + .5, np.ones_like(rows)), axis=0)
    rays = np.linalg.solve(k, pixels.reshape(3, -1))
    ray_norm = np.linalg.norm(rays / rays[2], axis=0).reshape(radial.shape)
    valid = np.isfinite(radial) & (radial > 0)
    z = np.full(radial.shape, np.nan)
    np.divide(radial, ray_norm, out=z, where=valid)
    return z, valid


def audit_depth_pair(radial_distance_m, optical_z_m, intrinsics):
    """Compare native optical Z to radial distance projected using the pinned pinhole K."""
    converted, valid = pinhole_radial_to_z(radial_distance_m, intrinsics)
    z = np.asarray(optical_z_m, dtype=np.float64)
    if z.shape != converted.shape:
        raise ValueError("radial and optical-Z images are not aligned")
    valid &= np.isfinite(z) & (z > 0)
    if not valid.any():
        raise ValueError("radial and optical-Z images have no common valid pixels")
    h, w = z.shape
    y, x = np.indices(z.shape, dtype=np.float64)
    offset = np.maximum(np.abs((y + .5) / h - .5), np.abs((x + .5) / w - .5))
    center = valid & (offset <= .2)
    edge = valid & (offset >= .3)
    error = np.abs(z - converted)
    return {
        "valid_pixels": int(valid.sum()),
        "center_pixels": int(center.sum()),
        "edge_pixels": int(edge.sum()),
        "p95_abs_error_m": float(np.quantile(error[valid], .95)),
        "center_p95_abs_error_m": float(np.quantile(error[center], .95)) if center.any() else None,
        "edge_p95_abs_error_m": float(np.quantile(error[edge], .95)) if edge.any() else None,
    }


def lifecycle_status(*, artifact_integrity, synchronized, optical_depth, app_close_returned, outer_exit_code):
    """Keep observed data, cleanup and external process results independently visible."""
    if type(outer_exit_code) is not int:
        raise ValueError("outer process exit code must be a recorded integer")
    return {
        "artifact_integrity_verified": artifact_integrity,
        "capture_synchronization_verified": synchronized,
        "depth_semantics_verified": optical_depth,
        "app_close_returned": app_close_returned,
        "process_exit_verified": outer_exit_code == 0 and app_close_returned is True,
        "restart_verified": False,
        "outer_exit_code": outer_exit_code,
    }
