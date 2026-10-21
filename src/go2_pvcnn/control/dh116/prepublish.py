"""Shared pre-publication checks extracted unchanged from the legacy verifier."""

CHANNELS = (
    "left_thumb", "left_index", "left_middle",
    "right_thumb", "right_index", "right_middle",
)

def prepublish_contact_gates(snapshot, mesh_bodies, motions, *, impulses, mode,
                             selected_channel, max_speed_m_s,
                             max_pad_penetration_m, radius_m):
    """Evaluate speed and signed-distance pad depth before any external submit."""
    import numpy as np
    position = np.asarray(snapshot["rod_position"], dtype=float)
    velocity = np.asarray(snapshot["rod_velocity"], dtype=float)
    if position.ndim != 2 or position.shape[0] != 3 or velocity.shape != position.shape:
        raise ValueError("invalid rod state shape")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("rod state must be finite before publish")
    speed = float(np.linalg.norm(velocity, axis=0).max())
    midpoints = .5 * (position[:, 1:] + position[:, :-1])
    radii = np.asarray(radius_m, dtype=float)
    if radii.ndim == 0:
        radii = np.full(midpoints.shape[1], float(radii))
    if radii.shape != (midpoints.shape[1],) or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("rod radii must be positive finite per-element values")
    depths = {}
    for channel in CHANNELS:
        if channel not in mesh_bodies or channel not in motions:
            raise ValueError("mesh/motion channels are incomplete")
        result = mesh_bodies[channel].query_closest_points(midpoints.T)
        distances = np.asarray(result[1], dtype=float)
        if distances.shape != (midpoints.shape[1],) or not np.isfinite(distances).all():
            raise ValueError("pad signed distances must be finite per element")
        depths[channel] = float(np.maximum(radii - distances, 0.).max(initial=0.))
    maximum = max(depths.values())
    failures = []
    if speed > max_speed_m_s: failures.append("speed gate exceeded before publish")
    if maximum > max_pad_penetration_m: failures.append("pad penetration gate exceeded before publish")
    def fields(value):
        if isinstance(value, dict):
            return value.get("linear_w", ()), value.get("angular_com_w", ()), value.get("peak_force_n"), value.get("contact_duration_s")
        return value.linear_w, value.angular_com_w, value.peak_force_n, value.contact_duration_s
    if not isinstance(impulses, dict) or tuple(impulses) != CHANNELS:
        raise ValueError("pre-publish impulses must have exact channels")
    allowed = {selected_channel} if mode == "single-pad" else set()
    for channel, impulse in impulses.items():
        if channel in allowed:
            continue
        linear, angular, peak, duration = fields(impulse)
        values = np.asarray(tuple(linear) + tuple(angular) + (peak, duration), dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("pre-publish impulse evidence must be finite")
        if np.any(values != 0.):
            failures.append("unexpected channel reaction before publish")
            break
    return failures, {"max_wire_speed_m_s": speed, "max_pad_penetration_m": maximum,
                      "pad_penetration_m": depths}
