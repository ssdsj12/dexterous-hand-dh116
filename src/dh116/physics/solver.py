"""Native free-rod solver/build/profile, parameter snapshots, packed/flat contact ABI reductions and read-once reset ownership."""
import argparse, hashlib as nativ_build__hashlib, json as nativ_build__json, os as nativ_build__os, shutil, subprocess, sys
import numpy as nativ_solve__np
from pathlib import Path as nativ_verif__Path
import math as batch_pair_captu__math
import math as defor_solve_clock__math
import math as defor_param_snaps__math
from dataclasses import dataclass
import math as tenso_pair_captu__math
from functools import wraps
from pathlib import Path as nativ_build__Path
nativ_solve__DT = 1 / 480
import os as nativ_verif__os, json as nativ_verif__json, hashlib as nativ_verif__hashlib, importlib.util
import numpy as batch_pair_captu__np
import numpy as defor_solve_clock__np
import numpy as defor_param_snaps__np
from pathlib import Path as tenso_backe_profi__Path
import torch as tenso_pair_captu__torch
import math as tenso_rod_owner__math
from dh116.env.contracts import runti_io__ROOT as nativ_build__ROOT

def material_mapping(cfg):
    r = cfg['rod']
    return {'density_unit': 'kg/m', 'constructor': {'density': r['linear_density_kg_m'], 'k_stretch': r['axial_stiffness_n'], 'k_bend': r['bend_stiffness_n_m2'], 'max_h': nativ_solve__DT / r['solver_substeps'], 'num_itr': r['solver_iterations']}, 'young_modulus': {'status': 'not_identifiable_from_independent_EA_EI'}, 'shear_modulus': {'status': 'unsupported_independent_parameter'}, 'native_getters': {'mass': True, 'damping': True, 'drag': True, 'num_itr': True, 'gravity': False, 'contact': False, 'EA': False, 'EI': False}, 'material_status': cfg['material_status']}
import numpy as nativ_verif__np
import torch as batch_pair_captu__torch
defor_solve_clock__DT = 1 / 480

def validate_snapshot(packet, *, expected_num_envs, expected_nodes, expected_time_s):
    p = packet
    if p.get('schema') != 'dh116.native_parameter_snapshot.v1' or p.get('metadata_source') != 'native_host_metadata' or p.get('segment_source') != 'cuda_device_copy' or (p.get('hardware_measured') is not False):
        raise ValueError('native parameter provenance mismatch')
    (n, v) = (expected_num_envs, expected_nodes)
    if type(n) is not int or type(v) is not int or n < 1 or (v < 2) or (type(p.get('num_envs')) is not int) or (type(p.get('nodes_per_env')) is not int) or ((p['num_envs'], p['nodes_per_env']) != (n, v)):
        raise ValueError('parameter layout differs from batch identity')
    if not defor_param_snaps__math.isfinite(expected_time_s) or not defor_param_snaps__math.isfinite(p['time_s']) or p['time_s'] != expected_time_s:
        raise ValueError('parameter snapshot is not at the expected solver time')
    fields = {}

    def read(key, shape, unit, source, lower=None, strict=False, frame=None):
        x = defor_param_snaps__np.array(p[key], dtype=defor_param_snaps__np.float64, copy=True)
        if x.shape != shape or not defor_param_snaps__np.isfinite(x).all():
            raise ValueError(f'nonfinite or invalid shape: {key}')
        if lower is not None and (defor_param_snaps__np.any(x <= lower) if strict else defor_param_snaps__np.any(x < lower)):
            raise ValueError(f'invalid parameter: {key}')
        fields[key] = {'value': x.tolist(), 'unit': unit, 'frame': frame, 'source': source, 'valid': True, 'uncertainty': None, 'hardware_measured': False}
        return x
    host = 'native_host_metadata'
    device = 'cuda_device_copy'
    derived = 'derived_from_cuda_device_copy'
    read('contact_radius_m', (), 'm', host, 0, True)
    read('gravity_world_m_s2', (3,), 'm/s^2', host, frame='world')
    for key in ('contact_mu_static', 'contact_mu_kinetic'):
        read(key, (), '1', host, 0)
    read('contact_v_slip_m_s', (), 'm/s', host, 0, True)
    read('contact_kappa', (), '1', host, 0, True)
    rest = read('rest_lengths_m', (n, v - 1), 'm', device, 0, True)
    k = read('stored_k_stretch_n_m', (n, v - 1), 'N*m', device, 0, True)
    q = read('rest_quaternion_norm_n_m', (n, v - 1), 'N*m', device, 0, True)
    ea = read('stretch_ea_n', (n, v - 1), 'N', derived, 0, True)
    ei = read('bend_ei_n_m2', (n, v - 1), 'N*m^2', derived, 0, True)
    inverse = read('inverse_masses_kg_inv', (n, v), '1/kg', device, 0)
    if not defor_param_snaps__np.allclose(ea, k / rest, rtol=2e-06, atol=0):
        raise ValueError('EA does not reconcile with device coefficients')
    if not defor_param_snaps__np.allclose(ei, q * rest / 4, rtol=2e-06, atol=0):
        raise ValueError('EI does not reconcile with device coefficients')
    return {'schema': p['schema'], 'time_s': float(p['time_s']), 'fields': fields, 'fixed_node_indices': defor_param_snaps__np.argwhere(inverse == 0).tolist(), 'hardware_measured': False, 'teacher_qualified': False, 'native_experiment_qualified': False, 'notes': 'Contact radius is native host metadata; constitutive values are device-derived. No physical calibration is asserted.'}
import json as tenso_backe_profi__json

def tensor_pair_capture(raw, *, n, vertices, meshes, substeps, device):
    if raw.get('schema') != 'dh116.sdf_pair_device.v1' or raw['substeps'] != substeps or raw['num_envs'] != n or (raw['mesh_count'] != meshes) or (not 2 <= n <= 512) or (raw['last_error'] != 0) or (raw['owner_time_s_before'] != raw['owner_time_s_after']) or (not tenso_pair_captu__math.isfinite(raw['owner_time_s_before'])):
        raise ValueError('source-qualified current device contact ABI required')
    counts = tenso_pair_captu__torch.as_tensor(raw['counts'], device=device)
    samples = tenso_pair_captu__torch.as_tensor(raw['samples'], device=device)
    if counts.shape != (n,) or counts.dtype != tenso_pair_captu__torch.int32 or samples.dtype != tenso_pair_captu__torch.float32 or (samples.ndim != 3) or (samples.shape != (n, raw['capacity'], 24)) or bool(((counts < 0) | (counts > raw['capacity'])).any()):
        raise ValueError('bounded complete native per-lane pair buckets required')
    mask = tenso_pair_captu__torch.arange(samples.shape[1], device=device)[None, :] < counts[:, None]
    values = samples[mask].clone()
    owners = tenso_pair_captu__torch.arange(n, device=device)[:, None].expand_as(mask)[mask]
    ids = values[:, :3].contiguous().view(tenso_pair_captu__torch.int32).long()
    vectors = values[:, 3:15].double()
    limits = ids.new_tensor([substeps, vertices, meshes])
    if bool(((ids < 0) | (ids >= limits)).any()) or not bool(tenso_pair_captu__torch.isfinite(values[:, 3:]).all()):
        raise ValueError('actual pair indices or floating fields invalid')
    key = ((owners * substeps + ids[:, 0]) * vertices + ids[:, 1]) * meshes + ids[:, 2]
    if tenso_pair_captu__torch.unique(key).numel() != key.numel():
        raise ValueError('duplicate actual contact pair')
    normal = vectors[:, 3:6]
    force = vectors[:, 6:9]
    normal_force = (normal * force).sum(-1)
    if not bool(tenso_pair_captu__torch.isclose(normal.norm(dim=-1), tenso_pair_captu__torch.ones_like(normal_force), atol=2e-05, rtol=0).all()) or bool((normal_force < -1e-05).any()):
        raise ValueError('outward unit normal and compressive actual force required')
    nodes = tenso_pair_captu__torch.zeros(n * vertices, 3, dtype=tenso_pair_captu__torch.float64, device=device)
    wrench = tenso_pair_captu__torch.zeros(n * meshes, 6, dtype=tenso_pair_captu__torch.float64, device=device)
    nodes.index_add_(0, owners * vertices + ids[:, 1], force / substeps)
    wrench.index_add_(0, owners * meshes + ids[:, 2], tenso_pair_captu__torch.cat((-force, vectors[:, 9:12]), -1) / substeps)
    nodes = nodes.reshape(n, vertices, 3)
    wrench = wrench.reshape(n, meshes, 6)
    reference_nodes = tenso_pair_captu__torch.as_tensor(raw['node_forces'], device=device).double()
    reference_wrench = tenso_pair_captu__torch.as_tensor(raw['mesh_wrenches'], device=device)[:, :meshes].double()
    if not tenso_pair_captu__torch.allclose(nodes, reference_nodes, atol=0.0003, rtol=5e-05) or not tenso_pair_captu__torch.allclose(wrench, reference_wrench, atol=0.0003, rtol=5e-05):
        raise ValueError('actual pair reductions disagree with native totals')
    return dict(counts=counts.long().clone(), env_ids=owners, indices=ids, vectors=vectors, normal_force=normal_force, node_forces=nodes, mesh_wrenches=wrench)
import numpy as tenso_rod_owner__np

def sha(path):
    return nativ_build__hashlib.sha256(path.read_bytes()).hexdigest()

def translated_static_pose(style, entry, origin):
    """Translate the measured static collider pose into its own Isaac lane."""
    pose_key = {'pinch': 'table_pose', 'tripod': 'static_pose', 'power': 'static_pose'}.get(style)
    pose = entry.get(pose_key)
    if pose is None:
        raise ValueError('grasp-specific archived static pose required')
    (position, quaternion) = pose
    return (nativ_solve__np.asarray(position) + origin, nativ_solve__np.asarray(quaternion).copy())
import torch as nativ_verif__torch

def compact_pair_capture(packet, *, dt_s, device):
    """Validate one atomic host capture and retain compact tensors on the selected device.

    Force is ON rod. Mesh force is its opposite, with the captured mesh-origin
    torque unchanged. Reductions use the legacy step-average and tolerances.
    No cached packet or per-contact Python dictionaries are retained.
    """
    if not batch_pair_captu__math.isclose(dt_s, 1 / 480, abs_tol=1e-12, rel_tol=0):
        raise ValueError('expected the 480 Hz exchange clock')
    lanes = packet.get('lanes', ())
    if packet.get('schema') != 'dh116.sdf_pair_samples_batch.v1' or not 2 <= len(lanes) <= 512:
        raise ValueError('complete shared solver batch required')
    nodes = batch_pair_captu__torch.as_tensor(packet['node_forces_on_rod_world_n'], dtype=batch_pair_captu__torch.float64, device=device)
    wrenches = batch_pair_captu__torch.as_tensor(packet['mesh_wrenches_on_mesh_world'], dtype=batch_pair_captu__torch.float64, device=device)
    n = len(lanes)
    if nodes.ndim != 3 or nodes.shape[0] != n or nodes.shape[2] != 3 or (nodes.shape[1] < 1) or (wrenches.ndim != 3) or (wrenches.shape[0] != n) or (wrenches.shape[2] != 6) or (not 1 <= wrenches.shape[1] <= 64):
        raise ValueError('invalid aggregate array shapes')
    substeps = lanes[0]['substeps']
    clock = lanes[0]['owner_time_s_before']
    if type(substeps) is not int or not 1 <= substeps <= 64 or (not batch_pair_captu__math.isfinite(clock)) or (clock < 0):
        raise ValueError('finite completed owner clock and bounded substeps required')
    (counts, indices, vectors, env_ids) = ([], [], [], [])
    for (env_id, lane) in enumerate(lanes):
        (count, capacity) = (lane['attempted'], lane['capacity'])
        if lane.get('schema') != 'dh116.sdf_pair_samples_lane.v1' or type(lane.get('env_id')) is not int or lane['env_id'] != env_id or (type(count) is not int) or (type(capacity) is not int) or (not 1 <= capacity <= 1048576) or (not 0 <= count <= capacity) or (lane['overflow'] is not False) or (lane['last_error'] != 0) or (lane['substeps'] != substeps) or (lane['mesh_count'] != wrenches.shape[1]) or (lane['owner_time_s_before'] != clock) or (lane['owner_time_s_after'] != clock):
            raise ValueError('mixed identity, incomplete bucket or moving owner clock')
        ids = batch_pair_captu__np.asarray(lane['indices'])
        values = batch_pair_captu__np.asarray(lane['vectors'])
        if ids.shape != (count, 3) or ids.dtype.kind not in 'iu' or values.shape != (count, 12):
            raise ValueError('invalid compact pair arrays')
        indices.append(batch_pair_captu__torch.as_tensor(ids, dtype=batch_pair_captu__torch.int64, device=device))
        vectors.append(batch_pair_captu__torch.as_tensor(values, dtype=batch_pair_captu__torch.float64, device=device))
        env_ids.append(batch_pair_captu__torch.full((count,), env_id, dtype=batch_pair_captu__torch.int64, device=device))
        counts.append(count)
    ids = batch_pair_captu__torch.cat(indices)
    values = batch_pair_captu__torch.cat(vectors)
    owners = batch_pair_captu__torch.cat(env_ids)
    bounds = batch_pair_captu__torch.tensor([substeps, nodes.shape[1], wrenches.shape[1]], device=device)
    if not bool(batch_pair_captu__torch.isfinite(values).all() & batch_pair_captu__torch.isfinite(nodes).all() & batch_pair_captu__torch.isfinite(wrenches).all()) or bool(((ids < 0) | (ids >= bounds)).any()):
        raise ValueError('nonfinite data or pair index out of bounds')
    keys = ((owners * substeps + ids[:, 0]) * nodes.shape[1] + ids[:, 1]) * wrenches.shape[1] + ids[:, 2]
    if batch_pair_captu__torch.unique(keys).numel() != keys.numel():
        raise ValueError('duplicated contact within an environment and substep')
    (normal, force) = (values[:, 3:6], values[:, 6:9])
    normal_force = (normal * force).sum(-1)
    if not bool(batch_pair_captu__torch.isclose(batch_pair_captu__torch.linalg.vector_norm(normal, dim=-1), batch_pair_captu__torch.ones_like(normal_force), atol=2e-05, rtol=0).all()) or bool((normal_force < -1e-05).any()):
        raise ValueError('invalid outward normal or compressive force')
    by_node = batch_pair_captu__torch.zeros_like(nodes).reshape(-1, 3)
    by_mesh = batch_pair_captu__torch.zeros_like(wrenches).reshape(-1, 6)
    by_node.index_add_(0, owners * nodes.shape[1] + ids[:, 1], force / substeps)
    by_mesh.index_add_(0, owners * wrenches.shape[1] + ids[:, 2], batch_pair_captu__torch.cat((-force, values[:, 9:12]), dim=-1) / substeps)
    by_node = by_node.reshape_as(nodes)
    by_mesh = by_mesh.reshape_as(wrenches)
    if not batch_pair_captu__torch.allclose(by_node, nodes, atol=0.0003, rtol=5e-05) or not batch_pair_captu__torch.allclose(by_mesh, wrenches, atol=0.0003, rtol=5e-05):
        raise ValueError('pair samples do not reconcile with solver totals per environment')
    return dict(env_ids=owners, counts=batch_pair_captu__torch.tensor(counts, dtype=batch_pair_captu__torch.int64, device=device), indices=ids, vectors=values, normal_force=normal_force, node_forces=by_node, mesh_wrenches=by_mesh)

def solver_clock_matches(before, after, *, steps):
    """Compare SDK-observed solver seconds to the native repeated float sum.

    A float64 backend or CPU fixture may agree with the real-valued sum instead.
    Match either representation within a fraction of one step. The caller still
    verifies actual SDK and integer rod ticks independently.
    """
    if type(steps) is not int or not 1 <= steps <= 1000000 or any((type(value) not in (int, float) or not defor_solve_clock__math.isfinite(value) or value < 0 or (value > defor_solve_clock__np.finfo(defor_solve_clock__np.float32).max / 2) for value in (before, after))) or (after <= before):
        return False
    rounded = defor_solve_clock__np.float32(before)
    delta = defor_solve_clock__np.float32(defor_solve_clock__DT)
    for _ in range(steps):
        advanced = rounded + delta
        if advanced <= rounded:
            return False
        rounded = advanced
    ulp = float(defor_solve_clock__np.spacing(defor_solve_clock__np.float32(after)))
    if ulp >= defor_solve_clock__DT / 4:
        # Beyond this point an ULP tolerance may accept a skipped native tick.
        return before == float(defor_solve_clock__np.float32(before)) and after == float(rounded)
    tolerance = max(1e-08, ulp)
    exact = before + steps * defor_solve_clock__DT
    return min(abs(after - exact), abs(after - float(rounded))) <= tolerance
from dh116.env.contracts import runti_io__ROOT as tenso_backe_profi__ROOT

def flat_cuda_pair_capture(raw, *, n, vertices, meshes, substeps, device, expected_owner_time_s, strict_audit=True):
    """Validate owned compact tensors from the read-once flat ABI."""
    if raw.get('schema') != 'dh116.sdf_pair_flat_cuda.v1' or raw['substeps'] != substeps or raw['num_envs'] != n or (raw['mesh_count'] != meshes) or (raw['owner_time_s_before'] != expected_owner_time_s) or (raw['owner_time_s_after'] != expected_owner_time_s):
        raise ValueError('current complete flat CUDA boundary required')
    ids = raw['indices_gpu']
    vectors = raw['vectors_gpu']
    meta = raw['lane_meta_gpu']
    if not all((isinstance(x, tenso_pair_captu__torch.Tensor) and x.device == tenso_pair_captu__torch.device(device) for x in (ids, vectors, meta, raw['node_forces_gpu'], raw['mesh_wrenches_gpu']))) or ids.ndim != 2 or ids.shape[1] != 4 or (ids.dtype != tenso_pair_captu__torch.int32) or (vectors.shape != (len(ids), 12)) or (vectors.dtype != tenso_pair_captu__torch.float32) or (meta.shape != (n, 4)) or (meta.dtype != tenso_pair_captu__torch.int32):
        raise ValueError('owned finite-capacity actual flat CUDA tensors required')
    node_forces, mesh_wrenches = raw['node_forces_gpu'], raw['mesh_wrenches_gpu']
    if node_forces.shape != (n, vertices, 3) or mesh_wrenches.shape != (n, meshes, 6) or not all(
            x.dtype in (tenso_pair_captu__torch.float32, tenso_pair_captu__torch.float64)
            for x in (node_forces, mesh_wrenches)):
        raise ValueError('owned native force/wrench reductions have invalid shape or dtype')
    invalid_lane = ((meta[:, 0] != 0) | (meta[:, 1] != 0) | (meta[:, 2] < 1)
                    | (meta[:, 3] < 0) | (meta[:, 3] > meta[:, 2]))
    index = ids.long()
    limits = index.new_tensor([n, substeps, vertices, meshes])
    if not strict_audit:
        # The source-locked native ABI creates owned pair buffers. Keep the
        # device health mask and safe gather indices until the window gate.
        valid_indices = ((index >= 0) & (index < limits)).all()
        index = index.clamp_min(0).minimum(limits - 1)
        val = vectors.double()
        nf = (val[:, 3:6] * val[:, 6:9]).sum(-1)
        valid = ~invalid_lane & valid_indices & (meta[:, 3].sum() == len(ids))
        valid &= tenso_pair_captu__torch.isfinite(val).all() & (nf >= -1e-5).all()
        valid &= tenso_pair_captu__torch.isfinite(node_forces).flatten(1).all(1)
        valid &= tenso_pair_captu__torch.isfinite(mesh_wrenches).flatten(1).all(1)
        balance = node_forces.double().sum(1) + mesh_wrenches[..., :3].double().sum(1)
        valid &= balance.norm(dim=-1) <= 0.001
        return dict(counts=meta[:, 3].long(), env_ids=index[:, 0], indices=index[:, 1:],
                    vectors=val, normal_force=nf, node_forces=node_forces,
                    mesh_wrenches=mesh_wrenches, pair_lane_meta=meta, valid=valid,
                    pair_owner_time_s=expected_owner_time_s)
    if bool(invalid_lane.any()) or not bool(meta[:, 3].sum() == len(ids)):
        raise ValueError('finite capacity and matching owned lane counts required')
    env_id = index[:, 0]
    pair = index[:, 1:]
    val = vectors.double()
    if bool(((index < 0) | (index >= limits)).any()) or not bool(tenso_pair_captu__torch.isfinite(val).all()):
        raise ValueError('invalid flat actual pair')
    counts = tenso_pair_captu__torch.bincount(env_id, minlength=n)
    if not tenso_pair_captu__torch.equal(counts, meta[:, 3].long()):
        raise ValueError('flat pair lane count differs')
    key = ((env_id * substeps + pair[:, 0]) * vertices + pair[:, 1]) * meshes + pair[:, 2]
    if tenso_pair_captu__torch.unique(key).numel() != len(key):
        raise ValueError('duplicate flat actual pair')
    normal = val[:, 3:6]
    force = val[:, 6:9]
    nf = (normal * force).sum(-1)
    if not bool(tenso_pair_captu__torch.isclose(normal.norm(dim=-1), tenso_pair_captu__torch.ones_like(nf), atol=2e-05, rtol=0).all()) or bool((nf < -1e-05).any()):
        raise ValueError('invalid flat contact normal/force')
    nodes = val.new_zeros(n * vertices, 3)
    wrench = val.new_zeros(n * meshes, 6)
    nodes.index_add_(0, env_id * vertices + pair[:, 1], force / substeps)
    wrench.index_add_(0, env_id * meshes + pair[:, 2], tenso_pair_captu__torch.cat((-force, val[:, 9:12]), -1) / substeps)
    nodes = nodes.reshape(n, vertices, 3)
    wrench = wrench.reshape(n, meshes, 6)
    if not tenso_pair_captu__torch.allclose(nodes, raw['node_forces_gpu'].double(), atol=0.0003, rtol=5e-05) or not tenso_pair_captu__torch.allclose(wrench, raw['mesh_wrenches_gpu'].double(), atol=0.0003, rtol=5e-05):
        raise ValueError('flat pair native totals differ')
    return dict(counts=counts, env_ids=env_id, indices=pair, vectors=val, normal_force=nf, node_forces=nodes, mesh_wrenches=wrench, pair_lane_meta=meta.clone(), pair_owner_time_s=expected_owner_time_s)
import torch as tenso_rod_owner__torch

def verify_source(root=nativ_build__ROOT):
    root = nativ_build__Path(root)
    lock = nativ_build__json.loads((root / 'configs/runtime/ropesim_source_lock.json').read_text())
    for (name, expected) in lock['source_sha256'].items():
        relative = nativ_build__Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('root-relative source inventory required')
        source = root / 'vendor/ropesim' / relative
        if source.is_symlink() or not source.is_file() or sha(source) != expected:
            raise ValueError('vendor source SHA differs: ' + name)
    return lock

def make_solver(backend, plan, reset, offsets, *, style, capacity, empty_lane=True):
    nodes = nativ_solve__np.asarray(reset['rod_reset']['node_positions'], dtype=nativ_solve__np.float32)
    initial = nodes + offsets[:, None, :]
    if empty_lane:
        initial[0, :, 0] += 2.0
    cfg = plan['cfg']
    rod = cfg['rod']
    solver = backend.RopeBatch.from_positions(initial, **material_mapping(cfg)['constructor'], pinned=nativ_solve__np.zeros(initial.shape[:-1], dtype=bool), fix_root_orientation=False, straight_rest=True, deterministic=True)
    solver.set_gravity(nativ_solve__np.asarray(rod['gravity_world_m_s2'], dtype=nativ_solve__np.float32))
    solver.set_contact(radius=rod['radius_m'], mu_s=rod['mu_s'], mu_k=rod['mu_k'], v_slip=rod['v_slip_m_s'], kappa=rod['kappa'])
    solver.set_rope_contact(rod['self_collision'])
    for key in ('damping', 'drag', 'drag_quad', 'drag_quad_t'):
        setattr(solver, key, rod[key])
    for (slot, entry) in enumerate(plan['entries']):
        if entry['name'] in reset['pad_poses_xyzw']:
            (position, quaternion) = reset['pad_poses_xyzw'][entry['name']]
        else:
            (position, quaternion) = translated_static_pose(style, entry, nativ_solve__np.zeros(3))
        xyz = nativ_solve__np.asarray(position, dtype=nativ_solve__np.float32) + offsets
        rotation = nativ_solve__np.broadcast_to(quaternion, (len(offsets), 4)).astype(nativ_solve__np.float32)
        actual = solver.add_mesh(entry['values'], entry['origin'], entry['spacing'], pos=xyz, quat=rotation, mu_s=entry['mu_s'], mu_k=entry['mu_k'])
        if actual != slot:
            raise ValueError('actual native mesh slot changed')
    solver.enable_contact_forces(True)
    if capacity is not None:
        if len(offsets) == 1:
            solver.enable_contact_pair_capture(capacity)
        else:
            solver.enable_batched_contact_pair_capture(capacity)
    return solver
from dh116.env.contracts import runti_io__ROOT as nativ_verif__ROOT
nativ_verif__flat_cuda_pair_capture = flat_cuda_pair_capture

def read_compact_pair_capture(solver, *, physics_step, rod_ticks, mesh_names, expected_owner_time_s, device):
    """Read once and bind the capture to the caller's current shared boundary.

    The native API reports solver owner time, independently of SDK time.
    Local Reset can leave different rod ticks in different environments.
    There is no retained cache to reuse after a failed step or Reset.
    """
    (ticks, names) = (tuple(rod_ticks), tuple(mesh_names))
    if type(physics_step) is not int or physics_step < 1 or (not ticks) or any((type(tick) is not int or tick < 1 for tick in ticks)) or (not names) or any((not isinstance(name, str) or not name for name in names)) or (len(set(names)) != len(names)) or (not batch_pair_captu__math.isfinite(expected_owner_time_s)) or (expected_owner_time_s < 0) or (solver.last_error != 0):
        raise ValueError('current shared SDK boundary and distinct mesh names required')
    raw = solver.contact_pair_samples_batched_v1()
    if solver.last_error != 0 or len(raw['lanes']) != len(ticks) or any((lane['mesh_count'] != len(names) or lane['owner_time_s_before'] != expected_owner_time_s for lane in raw['lanes'])):
        raise ValueError('stale capture, changed solver status or lane binding')
    tensors = compact_pair_capture(raw, dt_s=1 / 480, device=device)
    return dict(tensors, physics_step=physics_step, rod_ticks=ticks, mesh_names=names, owner_time_s=expected_owner_time_s, dt_s=1 / 480, buffer_metadata=tuple(({key: lane[key] for key in ('substeps', 'mesh_count', 'capacity', 'attempted', 'overflow')} for lane in raw['lanes'])))
from dh116.env.contracts import checked_file
tenso_rod_owner__solver_clock_matches = solver_clock_matches
tenso_rod_owner__flat_cuda_pair_capture = flat_cuda_pair_capture

def poison_failure(operation):

    @wraps(operation)
    def run(self, *args, **kwargs):
        try:
            return operation(self, *args, **kwargs)
        except BaseException:
            self._failed = True
            self._reaction = None
            raise
    return run

def new_output(root, output):
    from dh116.env.contracts import ProjectPaths
    path = ProjectPaths(root).output(output)
    if path.relative_to(root).parts[0] != 'build' or path.exists():
        raise ValueError('new project build directory required')
    return path

def step_thumb_physics(sim, expected_sdk_step, *, budget_end):
    before = sim.current_time_step_index
    if before != expected_sdk_step:
        raise RuntimeError(f'unexpected PhysX clock before measurement: {before}, expected {expected_sdk_step}')
    if before >= budget_end:
        raise RuntimeError('PhysX budget exhausted before measurement')
    if not sim.is_playing():
        raise RuntimeError('PhysX timeline stopped before measurement')
    sim.step(render=False)
    after = sim.current_time_step_index
    if after != before + 1:
        raise RuntimeError(f'unexpected PhysX clock advance during measurement: {before} -> {after}')
    return after
from dh116.env.scene import load_task_scene
nativ_verif__compact_pair_capture = compact_pair_capture
from dh116.env.contracts import runti_io__digest as digest

class TensorRodOwner:

    def __init__(self, solver, *, run_uuid, physics_step, episodes, mesh_names, solver_substeps=12, device='cuda:0'):
        n = solver.num_envs
        if not 2 <= n <= 512 or solver.last_error != 0 or len(episodes) != n or (len(set(episodes)) != n) or (len(set(mesh_names)) != len(mesh_names)) or (solver_substeps != 12) or (type(physics_step) is not int) or (physics_step < 0):
            raise ValueError('complete healthy source-owned shared solver required')
        self.solver = solver
        self.run_uuid = run_uuid
        self.physics_step = physics_step
        self.mesh_names = tuple(mesh_names)
        self.episodes = tuple(episodes)
        self.substeps = solver_substeps
        self.resets = [1] * n
        self._reset_step = [physics_step] * n
        self._time = float(solver.time)
        self.device = tenso_rod_owner__torch.device(device)
        self._initial = tenso_rod_owner__torch.as_tensor(solver.positions(), device=device).detach().clone()
        if self._initial.shape != (n, 41, 3) or not bool(tenso_rod_owner__torch.isfinite(self._initial).all()):
            raise ValueError('actual complete free-rod initial nodes required')
        self._failed = False
        self._consumed = True
        self._reaction = None
        self._initial_step = physics_step
        self._initial_used = False
        self.strict_audit = True

    @property
    def ticks(self):
        return tuple(((self.physics_step - step) * self.substeps for step in self._reset_step))

    def _healthy(self):
        if self._failed or self.solver.last_error != 0 or float(self.solver.time) != self._time:
            self._failed = True
            raise ValueError('actual shared solver boundary changed or failed')

    def advance(self, *, physics_step):
        try:
            self._healthy()
            if physics_step != self.physics_step + 1 or not self._consumed:
                raise ValueError('one SDK step and consumed old reaction required')
            old = self._time
            self.solver.step_free(1 / 480, self.substeps)
            new = float(self.solver.time)
            if self.solver.last_error != 0 or not tenso_rod_owner__solver_clock_matches(old, new, steps=1):
                raise ValueError('actual shared CUDA solver clock failed')
            packet = tenso_rod_owner__flat_cuda_pair_capture(
                self.solver.contact_pair_samples_flat_cuda_v1(), n=self.solver.num_envs, vertices=41,
                meshes=len(self.mesh_names), substeps=self.substeps, device=self.device,
                expected_owner_time_s=new, strict_audit=self.strict_audit,
            )
            if self.strict_audit:
                residual = packet['node_forces'].sum(1) + packet['mesh_wrenches'][..., :3].sum(1)
                if bool((residual.norm(dim=-1) > 0.001).any()):
                    raise ValueError('original rod/mesh force balance guard failed')
            self.physics_step = physics_step
            self._time = new
            self._consumed = False
            self._reaction = packet
            if self.strict_audit:
                return {k: v.clone() if isinstance(v, tenso_rod_owner__torch.Tensor) else v for (k, v) in packet.items()}
            return packet
        except BaseException:
            self._failed = True
            self._reaction = None
            raise

    def consume(self, *, physics_step):
        self._healthy()
        if self._consumed or self._reaction is None or physics_step != self.physics_step + 1:
            self._failed = True
            raise ValueError('immediate next-SDK exclusive reaction ownership required')
        self._consumed = True
        result = self._reaction
        self._reaction = None
        return self._source_packet(result, physics_step)

    def _source_packet(self, packet, applied_step):
        return dict(packet, source_solver_id=id(self.solver), parent_num_envs=self.solver.num_envs, run_uuid=self.run_uuid, source_physics_step=self.physics_step, applied_physics_step=applied_step, rod_ticks=self.ticks, episode_uids=self.episodes, reset_ids=tuple(self.resets), solver_time_s=self._time)

    @poison_failure
    def initial_reaction(self, *, physics_step):
        """One explicitly empty pending reaction before any solver advance."""
        self._healthy()
        if self._initial_used or self.physics_step != self._initial_step or physics_step != self.physics_step + 1 or (not self._consumed) or any(self.ticks) or any((r != 1 for r in self.resets)):
            raise ValueError('only the original unadvanced empty pending reaction is available')
        self._initial_used = True
        return self._source_packet(dict(bootstrap_empty=True, mesh_wrenches=tenso_rod_owner__torch.zeros(self.solver.num_envs, len(self.mesh_names), 6, device=self.device, dtype=tenso_rod_owner__torch.float64)), physics_step)

    @poison_failure
    def reset_lanes(self, env_ids, *, episode_uids):
        """Only the task owner may call at a completed window with pending reaction."""
        self._healthy()
        ids = tuple(env_ids)
        if not ids or len(set(ids)) != len(ids) or len(ids) != len(episode_uids) or any((type(i) is not int or not 0 <= i < self.solver.num_envs for i in ids)) or any((not isinstance(u, str) or not u or u in self.episodes for u in episode_uids)) or (len(set(episode_uids)) != len(ids)) or self._consumed or (self._reaction is None):
            raise ValueError('distinct selected Reset IDs/epochs with pending reaction required')
        self.park_rods(ids)
        epochs = list(self.episodes)
        for (i, u) in zip(ids, episode_uids):
            epochs[i] = u
            self.resets[i] += 1
            self._reset_step[i] = self.physics_step
        self.episodes = tuple(epochs)

    @poison_failure
    def park_rods(self, ids):
        """Restore selected mechanics at fixed global time without starting new episodes."""
        self._healthy()
        before = tenso_rod_owner__torch.as_tensor(self.solver.positions(), device=self.device).clone()
        self.solver.reset_envs(tenso_rod_owner__np.asarray(ids, dtype=tenso_rod_owner__np.int32))
        after = tenso_rod_owner__torch.as_tensor(self.solver.positions(), device=self.device)
        selected = tenso_rod_owner__torch.tensor(ids, device=self.device)
        keep = tenso_rod_owner__torch.ones(self.solver.num_envs, dtype=tenso_rod_owner__torch.bool, device=self.device)
        keep[selected] = False
        if self.solver.last_error != 0 or float(self.solver.time) != self._time or (not tenso_rod_owner__torch.equal(before[keep], after[keep])) or (not tenso_rod_owner__torch.equal(after[selected], self._initial[selected])):
            self._failed = True
            raise ValueError('zero-time selected rod Reset mutated peer or time')
        packet = self._reaction
        if packet is None:  # Already consumed for the immediately following SDK step.
            return
        retained = ~tenso_rod_owner__torch.isin(packet['env_ids'], selected)
        for k in ('env_ids', 'indices', 'vectors', 'normal_force'):
            packet[k] = packet[k][retained].clone()
        for k in ('counts', 'node_forces', 'mesh_wrenches'):
            packet[k][selected] = 0
        packet['pair_lane_meta'][selected, 3] = 0

def build(output, *, root=nativ_build__ROOT, jobs=8):
    lock = verify_source(root)
    output = new_output(root, output)
    tool = root / 'build/toolchain'
    cmake = tool / 'cmake/bin/cmake'
    cuda = tool / 'cuda-12.8'
    nvcc = cuda / 'bin/nvcc'
    version = subprocess.check_output([str(nvcc), '--version'], text=True)
    if 'V12.8.61' not in version:
        raise ValueError('current CUDA12.8.61 toolchain required')
    if lock.get('cuda_architectures') != [89] or lock.get('physics_kernel_changes') is not False:
        raise ValueError('source-bound unchanged-physics SM89 profile required')
    output.mkdir(parents=True, exist_ok=False)
    commands = [[str(cmake), '-S', str(root / 'vendor/ropesim'), '-B', str(output / 'cmake'), '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=89', '-DCMAKE_CUDA_COMPILER=' + str(nvcc), '-DCUDAToolkit_ROOT=' + str(cuda), '-Dpybind11_DIR=' + str(tool / 'pybind11/share/cmake/pybind11'), '-DPYTHON_EXECUTABLE=' + sys.executable, '-DCMAKE_INSTALL_RPATH=' + str(cuda / 'lib64')], [str(cmake), '--build', str(output / 'cmake'), '--parallel', str(jobs)], [str(cmake), '--install', str(output / 'cmake'), '--prefix', str(output / 'python')]]
    env = dict(nativ_build__os.environ, TMPDIR=str(output), CUDA_CACHE_PATH=str(root / '.cache/cuda'))
    with (output / 'build.log').open('w') as stream:
        for command in commands:
            stream.write(nativ_build__json.dumps(command) + '\n')
            stream.flush()
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    package = output / 'python/ropesim'
    for name in ('__init__.py', 'sdf.py', 'isaac_bridge.py'):
        shutil.copy2(root / 'vendor/ropesim/ropesim' / name, package / name)
    (binary,) = package.glob('_ropesim*.so')
    result = dict(schema='dh116-root-ropesim-rebuild-v1', binary_path=str(binary.relative_to(root)), binary_sha256=sha(binary), source_lock_sha256=sha(root / 'configs/runtime/ropesim_source_lock.json'), nvcc_version=version, commands=commands, system_install=False, teacher_admission=False)
    (output / 'build_manifest.json').write_text(nativ_build__json.dumps(result, indent=2) + '\n')
    return result
nativ_verif__make_solver = make_solver

def flat_backend_parity(output):
    output = nativ_verif__Path(output)
    if output.exists():
        raise ValueError('fresh parity output required')
    config = nativ_verif__json.loads((nativ_verif__ROOT / 'configs/runtime/backend.json').read_text())
    binary = nativ_verif__ROOT / config['binary_path']
    spec = importlib.util.spec_from_file_location('_ropesim', binary)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    (archive, reset) = load_task_scene('pinch')
    plan = archive.mesh_plan
    capacity = 41 * len(plan['entries']) * 12
    solver = nativ_verif__make_solver(backend, plan, reset, nativ_verif__np.array([[0, 0, 0], [3, 0, 0]], dtype=nativ_verif__np.float32), style='pinch', capacity=capacity)
    saved = []
    packets = []
    errors = []
    for step in range(2):
        solver.step_free(1 / 480, 12)
        old = solver.contact_pair_samples_batched_v1()
        ref = nativ_verif__compact_pair_capture(old, dt_s=1 / 480, device='cuda:0')
        if step == 0:
            with nativ_verif__torch.cuda.stream(nativ_verif__torch.cuda.Stream()):
                try:
                    solver.contact_pair_samples_flat_cuda_v1()
                except ValueError:
                    pass
                else:
                    raise AssertionError('unsafe nondefault stream accepted')
        packet = solver.contact_pair_samples_flat_cuda_v1()
        actual = nativ_verif__flat_cuda_pair_capture(packet, n=2, vertices=41, meshes=len(plan['entries']), substeps=12, device='cuda:0', expected_owner_time_s=float(solver.time))
        for key in ('counts', 'env_ids', 'indices', 'vectors', 'normal_force'):
            nativ_verif__torch.testing.assert_close(actual[key], ref[key], rtol=0, atol=0)
        for key in ('node_forces', 'mesh_wrenches'):
            delta = float((actual[key] - ref[key]).abs().max())
            errors.append(delta)
            nativ_verif__torch.testing.assert_close(actual[key], ref[key], rtol=0, atol=1e-12)
        for (prior, expected) in zip(packets, saved):
            for key in prior:
                nativ_verif__torch.testing.assert_close(prior[key], expected[key], rtol=0, atol=0)
        packets.append({k: v for (k, v) in packet.items() if isinstance(v, nativ_verif__torch.Tensor)})
        assert actual['counts'][0] == 0 and actual['counts'][1] > 0
        try:
            solver.contact_pair_samples_flat_cuda_v1()
        except ValueError:
            pass
        else:
            raise AssertionError('flat read-once not enforced')
        saved.append({k: v.clone() for (k, v) in packet.items() if isinstance(v, nativ_verif__torch.Tensor)})
        print('STEP', step, actual['counts'].tolist(), flush=True)
    for key in saved[0]:
        if key in packet:
            pass
    assert not nativ_verif__torch.equal(saved[0]['node_forces_gpu'], saved[1]['node_forces_gpu'])
    solver.reset_envs(nativ_verif__np.array([0], nativ_verif__np.int32))
    try:
        solver.contact_pair_samples_flat_cuda_v1()
    except ValueError:
        pass
    else:
        raise AssertionError('stale post-reset flat contact')
    from copy import deepcopy
    empty_reset = deepcopy(reset)
    empty_reset['rod_reset']['node_positions'] = nativ_verif__np.asarray(empty_reset['rod_reset']['node_positions']) + nativ_verif__np.array([2.0, 0.0, 0.0])
    empty = nativ_verif__make_solver(backend, plan, empty_reset, nativ_verif__np.array([[0, 0, 0], [3, 0, 0]], dtype=nativ_verif__np.float32), style='pinch', capacity=capacity, empty_lane=False)
    empty.step_free(1 / 480, 12)
    empty_raw = empty.contact_pair_samples_flat_cuda_v1()
    empty_capture = nativ_verif__flat_cuda_pair_capture(empty_raw, n=2, vertices=41, meshes=len(plan['entries']), substeps=12, device='cuda:0', expected_owner_time_s=float(empty.time))
    assert empty_capture['counts'].sum() == 0 and len(empty_capture['indices']) == 0
    overflow = nativ_verif__make_solver(backend, plan, reset, nativ_verif__np.array([[0, 0, 0], [3, 0, 0]], dtype=nativ_verif__np.float32), style='pinch', capacity=1)
    overflow.step_free(1 / 480, 12)
    try:
        nativ_verif__flat_cuda_pair_capture(overflow.contact_pair_samples_flat_cuda_v1(), n=2, vertices=41, meshes=len(plan['entries']), substeps=12, device='cuda:0', expected_owner_time_s=float(overflow.time))
    except ValueError:
        pass
    else:
        raise AssertionError('overflow accepted')
    _nativ_verif_import__TensorRodOwner = TensorRodOwner
    owned_solver = nativ_verif__make_solver(backend, plan, reset, nativ_verif__np.array([[0, 0, 0], [3, 0, 0]], dtype=nativ_verif__np.float32), style='pinch', capacity=capacity)
    owner = _nativ_verif_import__TensorRodOwner(owned_solver, run_uuid='flat-rod-probe', physics_step=2, episodes=('a', 'b'), mesh_names=tuple((x['name'] for x in plan['entries'])), device='cuda:0')
    owner.advance(physics_step=3)
    owner.consume(physics_step=4)
    owner.advance(physics_step=4)
    peer = nativ_verif__torch.as_tensor(owned_solver.positions())[1].clone()
    owner.reset_lanes((0,), episode_uids=('a2',))
    assert nativ_verif__torch.equal(nativ_verif__torch.as_tensor(owned_solver.positions())[1], peer)
    owner.consume(physics_step=5)
    owner.advance(physics_step=5)
    owner.consume(physics_step=6)
    assert owner.ticks == (12, 36)
    from dh116.physics.geometry import TensorMeshBridge
    snapshot = owned_solver.mesh_state_snapshot()
    M = len(plan['entries'])
    pos = nativ_verif__torch.tensor(snapshot['current_positions_world_m'][:, :M], device='cuda:0')
    quat = nativ_verif__torch.tensor(snapshot['current_quaternions_xyzw'][:, :M], device='cuda:0')
    bindings = plan['bindings']
    bridge = TensorMeshBridge(tuple((x['name'] for x in plan['entries'])), bindings, pos, quat)
    links = nativ_verif__torch.zeros(2, 2, 23, 7, device='cuda:0')
    links[..., 6] = 1
    for (key, row) in bindings.items():
        j = bridge.names.index(key)
        s = 0 if row['side'] == 'left' else 1
        b = row['body_index']
        links[:, s, b, :3] = pos[:, j]
        links[:, s, b, 3:] = quat[:, j]
    bridge.submit(owned_solver, links)
    actual = owned_solver.mesh_state_snapshot()
    nativ_verif__np.testing.assert_array_equal(actual['target_positions_world_m'][:, :M], pos.cpu().numpy())
    result = dict(binary_sha256=nativ_verif__hashlib.sha256(binary.read_bytes()).hexdigest(), steps=7, environments=2, tensor_owner_reset_continuation=True, gpu_all_mesh_submission=True, nondefault_stream_rejected=True, all_empty_batch=True, overflow_rejected=True, owned_outputs_stable=True, raw_indices_vectors_exact=True, reductions_max_abs_error=max(errors), reductions_atol=1e-12, empty_lane=True, contacting_lane=True, read_once=True, reset_rejected=True, isaac_steps=0, ppo_updates=0, device=nativ_verif__os.environ['CUDA_VISIBLE_DEVICES'])
    output.write_text(nativ_verif__json.dumps(result, indent=2) + '\n')
    print(nativ_verif__json.dumps(result), flush=True)
    return result
from dh116.env.contracts import ProjectPaths

def build_geometry(output, *, root=nativ_build__ROOT):
    from dh116.env.contracts import ProjectPaths
    destination = ProjectPaths(root).output(output)
    if destination.exists():
        raise ValueError('new CUDA geometry build required')
    destination.mkdir(parents=True, exist_ok=False)
    source = root / 'src/dh116/cuda_measurements'
    nvcc = root / 'build/toolchain/cuda-12.8/bin/nvcc'
    binary = destination / 'libdh116_hull_support.so'
    command = [str(nvcc), '-O3', '--shared', '-Xcompiler=-fPIC', '-arch=sm_89', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', '--ftz=false', str(source / 'hull_support.cu'), str(source / 'pad_support.cu'), '-o', str(binary)]
    with (destination / 'build.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = dict(binary=str(binary.relative_to(root)), binary_sha256=sha(binary), command=command, fp64=True, sm=89, fast_math=False, fmad=False, system_install=False)
    (destination / 'build_manifest.json').write_text(nativ_build__json.dumps(result, indent=2) + '\n')
    return result
LOCK = tenso_backe_profi__ROOT / 'configs/runtime/backend.json'

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('ropesim', 'geometry', 'flat-parity'))
    parser.add_argument('--output', type=nativ_build__Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 32:
        parser.error('jobs must be 1..32')
    if args.operation == 'ropesim':
        result = build(args.output, jobs=args.jobs)
    elif args.operation == 'geometry':
        result = build_geometry(args.output)
    else:
        _nativ_build_import__flat_backend_parity = flat_backend_parity
        result = _nativ_build_import__flat_backend_parity(args.output)
    print(nativ_build__json.dumps(result, indent=2))

@dataclass(frozen=True, slots=True)
class TensorBackendProfile:
    binary_path: str
    binary_sha256: str
    source_tree_sha256: str
    patch_sha256: str
    digest: str
    manifest_path: str

    @classmethod
    def load(cls, path=LOCK):
        path = tenso_backe_profi__Path(path)
        if not path.is_absolute():
            path = tenso_backe_profi__ROOT / path
        lock = tenso_backe_profi__json.loads(path.read_text())
        if lock.get('schema') != 'dh116-runtime-backend-v2' or lock.get('upstream_commit') != '96b8514c7c55c724f1abba1fbf6ed6642b7db72b' or lock.get('flat_abi') != 'contact_pair_samples_flat_cuda_v1' or (lock.get('teacher_admission') is not False) or (lock.get('native_mechanical_receipt') is not None) or (lock.get('physics_kernel_changes') is not False) or (lock.get('cuda_architectures') != [89]):
            raise ValueError('source-bound unqualified SM89 flat backend required')
        paths = ProjectPaths(tenso_backe_profi__ROOT)

        def local(name):
            return paths._relative(name)
        source = tenso_backe_profi__json.loads(checked_file(local(lock['source_lock']), lock['source_lock_sha256']))
        inventory = source['source_sha256']
        if len(inventory) != 72 or source['upstream_commit'] != lock['upstream_commit']:
            raise ValueError('complete current solver source inventory required')
        for (name, expected) in inventory.items():
            checked_file(local('vendor/ropesim/' + name), expected)
        patch = {n: inventory[n] for n in ('csrc/sdf.h', 'csrc/batch.h', 'csrc/bindings.cpp')}
        if digest(patch) != lock['patch_sha256']:
            raise ValueError('prepared flat ABI source identity differs')
        binary = local(lock['binary_path'])
        checked_file(binary, lock['binary_sha256'])
        build = tenso_backe_profi__json.loads(checked_file(local(lock['build_manifest']), lock['build_manifest_sha256']))
        if build['binary_path'] != lock['binary_path'] or build['binary_sha256'] != lock['binary_sha256'] or build['source_lock_sha256'] != lock['source_lock_sha256'] or (build['system_install'] is not False) or (build['teacher_admission'] is not False) or ('V12.8.61' not in build['nvcc_version']):
            raise ValueError('actual local CUDA12.8 build identity differs')
        proof = lock['parity_evidence']
        parity = tenso_backe_profi__json.loads(checked_file(local(proof['path']), proof['sha256']))
        if parity['binary_sha256'] != lock['binary_sha256'] or parity['reductions_max_abs_error'] > 1e-12 or (not all((parity[k] is True for k in ('raw_indices_vectors_exact', 'all_empty_batch', 'overflow_rejected', 'owned_outputs_stable', 'read_once', 'reset_rejected', 'tensor_owner_reset_continuation', 'gpu_all_mesh_submission', 'nondefault_stream_rejected')))) or (parity['isaac_steps'] != 0) or (parity['ppo_updates'] != 0):
            raise ValueError('actual flat-interface parity required; not Isaac qualification')
        return cls(str(binary), lock['binary_sha256'], digest(inventory), lock['patch_sha256'], digest(lock), str(path))

    def verify(self):
        if self != type(self).load(self.manifest_path):
            raise ValueError('live backend source identity changed')
        return self
