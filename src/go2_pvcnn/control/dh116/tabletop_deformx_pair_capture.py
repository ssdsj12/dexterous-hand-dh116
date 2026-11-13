"""Validate complete solver pair samples before exposing contact evidence."""
import math

import numpy as np


def validate_pair_capture(packet, *, dt_s, node_forces, mesh_wrenches):
    """Reconcile a one-rod, mesh-only capture against the solver's own totals."""
    if not math.isclose(dt_s, 1 / 480, rel_tol=0, abs_tol=1e-12):
        raise ValueError("expected the 480 Hz exchange clock")
    substeps, meshes, capacity, attempted = (
        packet[k] for k in ("substeps", "mesh_count", "capacity", "attempted"))
    if (any(type(x) is not int for x in (substeps, meshes, capacity, attempted))
            or not 1 <= substeps <= 64 or not 1 <= meshes <= 64
            or not 1 <= capacity <= 1048576 or not 0 <= attempted <= capacity
            or packet.get("overflow") is not False):
        raise ValueError("complete bounded nonoverflowing capture required")
    ids, vectors = np.asarray(packet["indices"]), np.asarray(packet["vectors"], dtype=float)
    nodes, wrenches = np.asarray(node_forces, dtype=float), np.asarray(mesh_wrenches, dtype=float)
    if (ids.shape != (attempted, 3) or ids.dtype.kind not in "iu" or vectors.shape != (attempted, 12)
            or nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 1
            or wrenches.shape != (meshes, 6)
            or not all(np.isfinite(a).all() for a in (vectors, nodes, wrenches))):
        raise ValueError("invalid pair or aggregate arrays")
    if (np.any(ids < 0) or np.any(ids >= [substeps, len(nodes), meshes])
            or len(np.unique(ids, axis=0)) != attempted):
        raise ValueError("pair indices out of range or duplicated within a substep")
    point, normal, force, torque = np.split(vectors, 4, axis=1)
    normal_force = np.sum(force * normal, axis=1)
    if (not np.allclose(np.linalg.norm(normal, axis=1), 1, atol=2e-5, rtol=0)
            or np.any(normal_force < -1e-5)):
        raise ValueError("invalid outward normal or compressive force")
    by_node, by_mesh = np.zeros_like(nodes), np.zeros_like(wrenches)
    np.add.at(by_node, ids[:, 1], force / substeps)
    np.add.at(by_mesh, ids[:, 2], np.concatenate((-force, torque), axis=1) / substeps)
    if not all(np.allclose(observed, expected, atol=3e-4, rtol=5e-5)
               for observed, expected in ((by_node, nodes), (by_mesh, wrenches))):
        raise ValueError("pair samples do not reconcile with step-averaged solver totals")
    order = np.lexsort((ids[:, 2], ids[:, 1], ids[:, 0]))
    return {
        "scope": "rod_node_to_sdf_mesh_substep_force",
        "substep_dt_s": dt_s / substeps,
        "pairs": [
            dict(substep=int(ids[i, 0]), node_index=int(ids[i, 1]), mesh_index=int(ids[i, 2]),
                 contact_point_world_m=point[i].tolist(), outward_normal_world=normal[i].tolist(),
                 force_on_rod_world_n=force[i].tolist(),
                 normal_force_on_rod_n=float(normal_force[i]),
                 tangential_force_on_rod_world_n=(force[i] - normal_force[i] * normal[i]).tolist(),
                 impulse_on_rod_ns=(force[i] * (dt_s / substeps)).tolist(),
                 torque_on_mesh_about_origin_world_nm=torque[i].tolist())
            for i in order
        ],
    }


def read_pair_capture(solver, *, dt_s, physics_step, rod_tick, mesh_names):
    """Read one atomic solver packet at an explicitly supplied exchange boundary."""
    if (type(physics_step) is not int or physics_step < 1 or type(rod_tick) is not int
            or rod_tick < 1 or isinstance(mesh_names, str)):
        raise ValueError("explicit exchange clock and ordered mesh names required")
    names = tuple(mesh_names)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("distinct ordered mesh names required")
    packet = solver.contact_pair_samples()
    if len(names) != packet["mesh_count"]:
        raise ValueError("mesh binding does not match the captured slots")
    result = validate_pair_capture(packet, dt_s=dt_s,
        node_forces=packet["node_forces_on_rod_world_n"],
        mesh_wrenches=packet["mesh_wrenches_on_mesh_world"])
    for pair in result["pairs"]:
        pair["mesh_name"] = names[pair["mesh_index"]]
    return {**result, "physics_step": physics_step, "rod_tick": rod_tick,
            "physx_contact_pairs": None, "native_qualification": False}
