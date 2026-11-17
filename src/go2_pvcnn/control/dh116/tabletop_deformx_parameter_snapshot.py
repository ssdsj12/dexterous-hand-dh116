"""Validate owned native parameter snapshots; never qualifies an experiment or reward."""
from __future__ import annotations
import math
import numpy as np

def validate_snapshot(packet, *, expected_num_envs, expected_nodes, expected_time_s):
    p = packet
    if (p.get("schema") != "dh116.native_parameter_snapshot.v1"
            or p.get("metadata_source") != "native_host_metadata"
            or p.get("segment_source") != "cuda_device_copy"
            or p.get("hardware_measured") is not False):
        raise ValueError("native parameter provenance mismatch")
    n, v = expected_num_envs, expected_nodes
    if (type(n) is not int or type(v) is not int or n < 1 or v < 2
            or type(p.get("num_envs")) is not int or type(p.get("nodes_per_env")) is not int
            or (p["num_envs"], p["nodes_per_env"]) != (n, v)):
        raise ValueError("parameter layout differs from batch identity")
    if not math.isfinite(expected_time_s) or not math.isfinite(p["time_s"]) or p["time_s"] != expected_time_s:
        raise ValueError("parameter snapshot is not at the expected solver time")
    fields = {}

    def read(key, shape, unit, source, lower=None, strict=False, frame=None):
        x = np.array(p[key], dtype=np.float64, copy=True)
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError(f"nonfinite or invalid shape: {key}")
        if lower is not None and (np.any(x <= lower) if strict else np.any(x < lower)):
            raise ValueError(f"invalid parameter: {key}")
        fields[key] = {"value": x.tolist(), "unit": unit, "frame": frame,
                       "source": source, "valid": True, "uncertainty": None,
                       "hardware_measured": False}
        return x

    host = "native_host_metadata"
    device = "cuda_device_copy"
    derived = "derived_from_cuda_device_copy"
    read("contact_radius_m", (), "m", host, 0, True)
    read("gravity_world_m_s2", (3,), "m/s^2", host, frame="world")
    for key in ("contact_mu_static", "contact_mu_kinetic"):
        read(key, (), "1", host, 0)
    read("contact_v_slip_m_s", (), "m/s", host, 0, True)
    read("contact_kappa", (), "1", host, 0, True)
    rest = read("rest_lengths_m", (n, v-1), "m", device, 0, True)
    k = read("stored_k_stretch_n_m", (n, v-1), "N*m", device, 0, True)
    q = read("rest_quaternion_norm_n_m", (n, v-1), "N*m", device, 0, True)
    ea = read("stretch_ea_n", (n, v-1), "N", derived, 0, True)
    ei = read("bend_ei_n_m2", (n, v-1), "N*m^2", derived, 0, True)
    inverse = read("inverse_masses_kg_inv", (n, v), "1/kg", device, 0)
    if not np.allclose(ea, k / rest, rtol=2e-6, atol=0):
        raise ValueError("EA does not reconcile with device coefficients")
    if not np.allclose(ei, q * rest / 4, rtol=2e-6, atol=0):
        raise ValueError("EI does not reconcile with device coefficients")
    return {"schema": p["schema"], "time_s": float(p["time_s"]), "fields": fields,
            "fixed_node_indices": np.argwhere(inverse == 0).tolist(),
            "hardware_measured": False, "teacher_qualified": False,
            "native_experiment_qualified": False,
            "notes": "Contact radius is native host metadata; constitutive values are device-derived. No physical calibration is asserted."}
