"""Audit host copies of DeformX aggregates without inferring contact pairs."""
from collections.abc import Mapping
import math

import numpy as np


def audit_solver_force_row(row, *, physics_evidence):
    """Check units, slot provenance, and clock for the current locked solver."""
    if type(physics_evidence) is not bool:
        raise ValueError("explicit evidence scope is required")
    if row.get("rod_mesh_contact_pairs") is not None:
        raise ValueError("locked solver aggregates cannot supply rod-mesh contact pairs")
    for field in ("env_id", "reset_id", "physics_step", "sdk_step", "rod_tick", "rod_substep"):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError(f"missing or invalid clock field {field}")
    for field in ("episode_uid", "rod_uid"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"missing physical identity {field}")
    dt = row["dt_s"]
    time = row["timestamp_s"]
    if (not math.isclose(dt, 1 / 480, rel_tol=0, abs_tol=1e-12)
            or row["physics_step"] != row["sdk_step"]
            or row["rod_tick"] != row["rod_substep"]
            or not math.isclose(time, row["physics_step"] * dt, rel_tol=0, abs_tol=2e-7)
            or not math.isclose(time, row["sdk_time_s"], rel_tol=0, abs_tol=2e-7)):
        raise ValueError("solver evidence clock mismatch")

    nodes = np.asarray(row["rod_nodes_w"], dtype=float)
    forces = np.asarray(row["rod_node_contact_forces_w"], dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0 or forces.shape != nodes.shape
            or not np.isfinite(nodes).all() or not np.isfinite(forces).all()):
        raise ValueError("expected a finite host vector per rod node")
    active_ids = np.flatnonzero(np.linalg.norm(forces, axis=1) > 1e-5).tolist()
    if row["contact_node_ids"] != active_ids:
        raise ValueError("active net-force node IDs disagree with the force buffer")

    bindings = row["actual_pad_bindings"]
    wrenches = row["rod_mesh_wrenches_w"]
    if (not isinstance(bindings, Mapping) or not bindings or "table" in bindings
            or not isinstance(wrenches, Mapping) or set(wrenches) != set(bindings) | {"table"}):
        raise ValueError("expected a bound wrench slot for every pad and one table slot")
    slots, body_indices, paths, measured = set(), set(), set(), {}
    for name, wrench in wrenches.items():
        v = np.asarray(wrench, dtype=float)
        if v.shape != (6,) or not np.isfinite(v).all():
            raise ValueError(f"{name} requires finite world force and torque")
        if name == "table":
            slot, path = len(bindings), None
        else:
            b = bindings[name]
            slot, path, body = b.get("mesh_index"), b.get("body_path"), b.get("body_index")
            if (type(slot) is not int or not 0 <= slot < len(bindings) or slot in slots
                    or type(body) is not int or body < 0 or body in body_indices
                    or not isinstance(path, str) or not path.startswith("/") or path in paths
                    or path.rsplit("/", 1)[-1] != name):
                raise ValueError(f"{name} has ambiguous mesh slot or PhysX link identity")
            slots.add(slot)
            paths.add(path)
            body_indices.add(body)
        measured[name] = dict(mesh_index=slot, body_path=path, force_n=v[:3].tolist(),
                              torque_nm_about_mesh_origin=v[3:].tolist())

    total = forces.sum(axis=0) + np.sum([m["force_n"] for m in measured.values()], axis=0)
    residual = float(np.linalg.norm(total))
    if residual > 1e-3 or not math.isclose(
            residual, row["pair_force_residual_n"], rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("aggregate force residual differs from the measured vectors")
    return {
        "native_readback": physics_evidence,
        "source": "locked DeformX host copies, step-averaged in world frame",
        "on_rod_node_forces_world_n": forces.tolist(),
        "active_rod_node_ids": active_ids,
        "mesh_wrenches_on_mesh_world": measured,
        "net_force_residual_n": residual,
        "rod_mesh_contact_pairs": None,
        "table_contact": None,
        "valid_for_r0_contact_or_slip": False,
        "valid_for_r2": False,
    }
