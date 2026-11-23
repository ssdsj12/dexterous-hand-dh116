"""Compact env-indexed contact reductions for the shared DeformX capture ABI."""
import math

import numpy as np
import torch


def compact_pair_capture(packet, *, dt_s, device):
    """Validate one atomic host capture and retain compact tensors on the selected device.

    Force is ON rod. Mesh force is its opposite, with the captured mesh-origin
    torque unchanged. Reductions use the legacy step-average and tolerances.
    No cached packet or per-contact Python dictionaries are retained.
    """
    if not math.isclose(dt_s, 1 / 480, abs_tol=1e-12, rel_tol=0):
        raise ValueError("expected the 480 Hz exchange clock")
    lanes = packet.get("lanes", ())
    if packet.get("schema") != "dh116.sdf_pair_samples_batch.v1" or not 2 <= len(lanes) <= 512:
        raise ValueError("complete shared solver batch required")
    nodes = torch.as_tensor(packet["node_forces_on_rod_world_n"], dtype=torch.float64, device=device)
    wrenches = torch.as_tensor(packet["mesh_wrenches_on_mesh_world"], dtype=torch.float64, device=device)
    n = len(lanes)
    if (nodes.ndim != 3 or nodes.shape[0] != n or nodes.shape[2] != 3 or nodes.shape[1] < 1
            or wrenches.ndim != 3 or wrenches.shape[0] != n or wrenches.shape[2] != 6
            or not 1 <= wrenches.shape[1] <= 64):
        raise ValueError("invalid aggregate array shapes")
    substeps = lanes[0]["substeps"]
    clock = lanes[0]["owner_time_s_before"]
    if (type(substeps) is not int or not 1 <= substeps <= 64
            or not math.isfinite(clock) or clock < 0):
        raise ValueError("finite completed owner clock and bounded substeps required")
    counts, indices, vectors, env_ids = [], [], [], []
    for env_id, lane in enumerate(lanes):
        count, capacity = lane["attempted"], lane["capacity"]
        if (lane.get("schema") != "dh116.sdf_pair_samples_lane.v1"
                or type(lane.get("env_id")) is not int or lane["env_id"] != env_id
                or type(count) is not int or type(capacity) is not int
                or not 1 <= capacity <= 1048576 or not 0 <= count <= capacity
                or lane["overflow"] is not False or lane["last_error"] != 0
                or lane["substeps"] != substeps or lane["mesh_count"] != wrenches.shape[1]
                or lane["owner_time_s_before"] != clock or lane["owner_time_s_after"] != clock):
            raise ValueError("mixed identity, incomplete bucket or moving owner clock")
        ids = np.asarray(lane["indices"])
        values = np.asarray(lane["vectors"])
        if ids.shape != (count, 3) or ids.dtype.kind not in "iu" or values.shape != (count, 12):
            raise ValueError("invalid compact pair arrays")
        indices.append(torch.as_tensor(ids, dtype=torch.int64, device=device))
        vectors.append(torch.as_tensor(values, dtype=torch.float64, device=device))
        env_ids.append(torch.full((count,), env_id, dtype=torch.int64, device=device))
        counts.append(count)
    ids = torch.cat(indices)
    values = torch.cat(vectors)
    owners = torch.cat(env_ids)
    bounds = torch.tensor([substeps, nodes.shape[1], wrenches.shape[1]], device=device)
    if (not bool(torch.isfinite(values).all() & torch.isfinite(nodes).all() & torch.isfinite(wrenches).all())
            or bool(((ids < 0) | (ids >= bounds)).any())):
        raise ValueError("nonfinite data or pair index out of bounds")
    keys = ((owners * substeps + ids[:, 0]) * nodes.shape[1] + ids[:, 1]) * wrenches.shape[1] + ids[:, 2]
    if torch.unique(keys).numel() != keys.numel():
        raise ValueError("duplicated contact within an environment and substep")
    normal, force = values[:, 3:6], values[:, 6:9]
    normal_force = (normal * force).sum(-1)
    if (not bool(torch.isclose(torch.linalg.vector_norm(normal, dim=-1),
                              torch.ones_like(normal_force), atol=2e-5, rtol=0).all())
            or bool((normal_force < -1e-5).any())):
        raise ValueError("invalid outward normal or compressive force")
    by_node = torch.zeros_like(nodes).reshape(-1, 3)
    by_mesh = torch.zeros_like(wrenches).reshape(-1, 6)
    by_node.index_add_(0, owners * nodes.shape[1] + ids[:, 1], force / substeps)
    by_mesh.index_add_(0, owners * wrenches.shape[1] + ids[:, 2],
                       torch.cat((-force, values[:, 9:12]), dim=-1) / substeps)
    by_node = by_node.reshape_as(nodes)
    by_mesh = by_mesh.reshape_as(wrenches)
    if (not torch.allclose(by_node, nodes, atol=3e-4, rtol=5e-5)
            or not torch.allclose(by_mesh, wrenches, atol=3e-4, rtol=5e-5)):
        raise ValueError("pair samples do not reconcile with solver totals per environment")
    return dict(env_ids=owners, counts=torch.tensor(counts, dtype=torch.int64, device=device),
                indices=ids, vectors=values, normal_force=normal_force,
                node_forces=by_node, mesh_wrenches=by_mesh)


def read_compact_pair_capture(solver, *, physics_step, rod_ticks, mesh_names,
                              expected_owner_time_s, device):
    """Read once and bind the capture to the caller's current shared boundary.

    The native API reports solver owner time, independently of SDK time.
    Local Reset can leave different rod ticks in different environments.
    There is no retained cache to reuse after a failed step or Reset.
    """
    ticks, names = tuple(rod_ticks), tuple(mesh_names)
    if (type(physics_step) is not int or physics_step < 1
            or not ticks or any(type(tick) is not int or tick < 1 for tick in ticks)
            or not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or not math.isfinite(expected_owner_time_s)
            or expected_owner_time_s < 0 or solver.last_error != 0):
        raise ValueError("current shared SDK boundary and distinct mesh names required")
    raw = solver.contact_pair_samples_batched_v1()
    if (solver.last_error != 0 or len(raw["lanes"]) != len(ticks)
            or any(lane["mesh_count"] != len(names)
                   or lane["owner_time_s_before"] != expected_owner_time_s
                   for lane in raw["lanes"])):
        raise ValueError("stale capture, changed solver status or lane binding")
    tensors = compact_pair_capture(raw, dt_s=1/480, device=device)
    return dict(tensors, physics_step=physics_step, rod_ticks=ticks, mesh_names=names,
                owner_time_s=expected_owner_time_s, dt_s=1/480)
