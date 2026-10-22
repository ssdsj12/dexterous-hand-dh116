"""Simulation-only compliant contact settings for the DH116 grasp probe."""
from __future__ import annotations

import math


def compliant_contact_parameters(stiffness_n_per_m: float, damping_n_s_per_m: float) -> dict[str, float]:
    """Return validated PhysX settings; these are not measured hardware values."""
    stiffness = float(stiffness_n_per_m)
    damping = float(damping_n_s_per_m)
    if not math.isfinite(stiffness) or not math.isfinite(damping) or stiffness <= 0.0 or damping <= 0.0:
        raise ValueError("simulation contact stiffness and damping must be finite and positive")
    return {
        "compliant_contact_stiffness": stiffness,
        "compliant_contact_damping": damping,
    }


def table_center_z_for_wire(wire_center_z_m: float, wire_radius_m: float,
                            table_thickness_m: float) -> float:
    """Place a horizontal table so its top touches a cylindrical wire bottom."""
    values = tuple(float(value) for value in
                   (wire_center_z_m, wire_radius_m, table_thickness_m))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wire and table dimensions must be finite")
    center, radius, thickness = values
    if center <= 0.0 or radius <= 0.0 or thickness <= 0.0 or center <= radius:
        raise ValueError("wire center, radius, and table thickness must define a positive placement")
    return center - radius - thickness / 2.0


__all__ = ["compliant_contact_parameters", "table_center_z_for_wire"]
