"""Optional end-supported DeformX candidate; no Isaac/solver imports at import time.

Static supports are ordinary collisions throughout the rollout, never attachments.
Legacy mesh order and 24D action meaning are unchanged. CPU qualification is not
native execution evidence. Zero support wrench alone never proves lift eligibility.
"""
from copy import deepcopy
import numpy as np
from .tabletop_end_supported_scene import plan_end_support_scene, author_end_supports

MESH_CONTRACT = dict(left_pad_indices=list(range(5)), table_index=5,
                     right_pad_indices=list(range(6, 11)), support_indices=[11, 12],
                     mesh_count=13, wrench_capacity=64, quaternion_order='xyzw')


def _array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def plan_flexible_end_supported_scene(layout, seeds_by_style, clearance_m=.06, *,
                                      wire_length_m=None, wire_radius_m=None):
    """Translate independent seed copies only along Z; preserve joints and sources."""
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError('candidate support clearance must be finite and positive')
    configured, scene = plan_end_support_scene(
        layout, clearance_m, wire_length_m=wire_length_m, wire_radius_m=wire_radius_m)
    seeds = deepcopy(seeds_by_style)
    for style, seed in seeds.items():
        nodes = np.array(seed['nodes'], dtype=float, copy=True)
        roots = np.array(seed['roots'], dtype=float, copy=True)
        joints = np.asarray(seed['joints'])
        if (nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2
                or roots.shape != (2, 7) or joints.ndim != 2 or joints.shape[0] != 2
                or not all(np.isfinite(a).all() for a in (nodes, roots, joints))):
            raise ValueError(f'invalid nodes/joints/roots for {style}')
        nodes[:, 2] += scene['dz_m']
        roots[:, 2] += scene['dz_m']
        seed['nodes'], seed['roots'] = nodes, roots
    return dict(layout=configured, scene=scene, seeds_by_style=seeds,
                mesh_contract=deepcopy(MESH_CONTRACT),
                calibration_status='candidate_not_measured')


def _support_poses(scene, env_origins):
    origins = np.asarray(_array(env_origins), dtype=np.float32)
    if origins.ndim != 2 or origins.shape[1] != 3 or not np.isfinite(origins).all():
        raise ValueError('finite [N,3] environment origins required')
    supports = scene['supports']
    if len(supports) != 2:
        raise ValueError('exactly two supports required')
    centers = np.asarray([s['center_m'] for s in supports], np.float32)
    sizes = np.asarray([s['size_m'] for s in supports], np.float32)
    if (centers.shape != (2, 3) or sizes.shape != (2, 3)
            or not np.isfinite([centers, sizes]).all() or np.any(sizes <= 0)):
        raise ValueError('finite positive support boxes required')
    pos = centers[None] + origins[:, None]
    quat = np.zeros((len(origins), 2, 4), np.float32); quat[:, :, 3] = 1
    return pos, quat


def extend_mesh_pose_buffers(mesh_pos, mesh_quat, scene, env_origins):
    """Return fresh 13-slot buffers; keep legacy 0:11 poses untouched (numpy/torch)."""
    n = len(env_origins)
    if tuple(mesh_pos.shape) != (n, 11, 3) or tuple(mesh_quat.shape) != (n, 11, 4):
        raise ValueError('expected legacy 11-slot mesh pose buffers')
    pos, quat = _support_poses(scene, env_origins)
    if hasattr(mesh_pos, 'detach'):
        import torch
        if not hasattr(mesh_quat, 'detach'):
            raise ValueError('mesh buffers must share backend')
        return (torch.cat((mesh_pos, torch.as_tensor(pos, device=mesh_pos.device, dtype=mesh_pos.dtype)), dim=1),
                torch.cat((mesh_quat, torch.as_tensor(quat, device=mesh_quat.device, dtype=mesh_quat.dtype)), dim=1))
    return (np.concatenate((mesh_pos, pos.astype(mesh_pos.dtype)), axis=1),
            np.concatenate((mesh_quat, quat.astype(mesh_quat.dtype)), axis=1))


def make_support_sdf_grids(scene, *, spacing_m=.005, margin_m=.025):
    """Real ropesim exact box SDF; default resolution matches retained table."""
    if not np.isfinite([spacing_m, margin_m]).all() or spacing_m <= 0 or margin_m <= 0:
        raise ValueError('finite positive SDF spacing/margin required')
    if margin_m < float(scene['wire_radius_m']) + 2 * spacing_m:
        raise ValueError('SDF margin must cover rope radius plus two cells')
    from ropesim import sdf
    grids = []
    for support in scene['supports']:
        half = np.asarray(support['size_m'], dtype=float) / 2
        grids.append(sdf.from_function(sdf.sdf_box(half), -half-margin_m, half+margin_m, spacing_m))
    return grids


def _bind_support_material(stage, env_prefix, paths, mu_s, mu_k):
    """Independent per-environment candidate material; never edit table/hand."""
    from pxr import Sdf, UsdPhysics, UsdShade
    material_path = env_prefix + '/EndSupports/PhysicsMaterial'
    material = UsdShade.Material.Define(stage, material_path)
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(float(mu_s))
    physical.CreateDynamicFrictionAttr(float(mu_k))
    physical.CreateRestitutionAttr(0.)
    prim = material.GetPrim()
    prim.AddAppliedSchema('PhysxMaterialAPI')
    prim.CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('average')
    prim.CreateAttribute('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token).Set('average')
    prim.CreateAttribute('dh116:calibrationStatus', Sdf.ValueTypeNames.Token).Set('candidate_not_measured')
    for path in paths:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
    return dict(path=material_path, mu_s=float(mu_s), mu_k=float(mu_k),
                restitution=0., friction_combine_mode='average',
                restitution_combine_mode='average', calibration_status='candidate_not_measured')


def bind_native_end_supports(rope, stage, prefix, scene, env_origins, *,
                             spacing_m=.005, margin_m=.025, mu_s=.5, mu_k=.4,
                             existing_mesh_indices=tuple(range(11))):
    """Append real meshes 11/12 and static USD boxes for ALL environments.

    Call once after original 11 meshes and set_contact(), before stepping/cloning
    finalization. Caller must use returned poses in every set_mesh_poses_all call.
    This helper creates only independent support materials; never changes existing
    table/hand materials, creates a solver, pins nodes, or steps.
    USD authoring is explicit per-env; prefix must be /World/envs. A failed index
    assertion invalidates this newly assembled scene; do not continue simulation.
    """
    if prefix != '/World/envs' or tuple(existing_mesh_indices) != tuple(range(11)):
        raise ValueError('require /World/envs and original left5/table/right5 ordering')
    if (not np.isfinite([mu_s, mu_k]).all() or mu_s < 0 or mu_k < 0 or mu_k > mu_s):
        raise ValueError('invalid support friction')
    positions, quaternions = _support_poses(scene, env_origins)
    grids = make_support_sdf_grids(scene, spacing_m=spacing_m, margin_m=margin_m)
    indices = []
    for j, grid in enumerate(grids):
        idx = rope.add_mesh(*grid.as_args(), pos=positions[:, j].copy(),
                            quat=quaternions[:, j].copy(), mu_s=mu_s, mu_k=mu_k)
        if idx != 11 + j:
            raise RuntimeError(f'end-support mesh index {idx}; expected {11+j}; discard scene')
        indices.append(int(idx))
    records = []
    paths = []
    for i, origin in enumerate(_array(env_origins)):
        shifted = deepcopy(scene)
        for support in shifted['supports']:
            support['center_m'] = (np.asarray(support['center_m']) + origin).tolist()
        env_paths = author_end_supports(stage, f'{prefix}/env_{i}', shifted)
        support_material = _bind_support_material(stage, f'{prefix}/env_{i}', env_paths, mu_s, mu_k)
        paths.extend(env_paths)
        records.append(dict(env_id=i, support_mesh_indices=indices.copy(),
                            physx_paths=list(env_paths), deformx_bound=True, physx_bound=True,
                            support_material=support_material))
    rope.enable_contact_forces(True)
    return dict(mesh_indices=indices, mesh_count=13, num_envs=len(positions),
                positions=positions, quat_xyzw=quaternions, usd_paths=paths,
                binding_records=records, mesh_contract=deepcopy(MESH_CONTRACT),
                calibration_status='candidate_not_measured')


def support_load_records(wrenches, bindings, *, contact_data_valid, step_id):
    """Capture actual last-step [N,64,6] support loads; validity is explicit.

    contact_data_valid must describe fresh native contact-force collection after a
    completed solver step, not an inference from zeros. Missing/NaN/invalid data
    remains NA. Force and torque retain ropesim's world-frame conventions.
    """
    w = np.asarray(_array(wrenches))
    n = bindings['num_envs']
    if (w.ndim != 3 or w.shape[0] != n or w.shape[1] < 13 or w.shape[2] != 6
            or list(bindings['mesh_indices']) != [11, 12]
            or len(bindings['binding_records']) != n):
        raise ValueError('invalid support binding/wrench buffer contract')
    validity = np.asarray(contact_data_valid, dtype=bool)
    if validity.ndim == 0:
        validity = np.repeat(validity, n)
    if validity.shape != (n,):
        raise ValueError('contact validity must be scalar or [N]')
    rows = []
    for i, bound in enumerate(bindings['binding_records']):
        load = w[i, 11:13].astype(float)
        valid = bool(validity[i] and np.isfinite(load).all() and step_id is not None
                     and bound.get('env_id') == i and bound.get('deformx_bound')
                     and bound.get('physx_bound') and bound.get('support_mesh_indices') == [11, 12]
                     and len(bound.get('physx_paths', [])) == 2)
        rows.append(dict(env_id=i, step_id=step_id, mesh_indices=[11, 12],
                         binding_record=deepcopy(bound), contact_data_valid=valid,
                         support_wrenches_world=load.tolist() if np.isfinite(load).all() else None,
                         total_support_force_norm_n=float(np.linalg.norm(load[:, :3], axis=1).sum()) if valid else None,
                         wrench_convention='rope_on_support_world_force_torque_about_mesh_origin'))
    return rows


def _segments_intersect_box(nodes, low, high):
    # Conservative swept-sphere test: each full centerline segment against a box
    # inflated by rod radius. Corner false positives keep qualification closed.
    start, delta = nodes[:-1], np.diff(nodes, axis=0)
    enter = np.zeros(len(start)); leave = np.ones(len(start))
    for axis in range(3):
        moving = np.abs(delta[:, axis]) > 1e-15
        outside = ~moving & ((start[:, axis] < low[axis]) | (start[:, axis] > high[axis]))
        enter[outside] = np.inf
        a = np.full(len(start), -np.inf); b = np.full(len(start), np.inf)
        a[moving] = (low[axis] - start[moving, axis]) / delta[moving, axis]
        b[moving] = (high[axis] - start[moving, axis]) / delta[moving, axis]
        enter = np.maximum(enter, np.minimum(a, b))
        leave = np.minimum(leave, np.maximum(a, b))
    return bool(np.any(enter <= leave))


def evaluate_support_clearance(nodes, radius_m, scene, contact_record, *,
                               initial_min_z_m, force_tolerance_n=1e-6,
                               clearance_tolerance_m=1e-5, expected_step_id=None):
    """Qualification only, not a Hold accumulator. Nodes are environment-local.

    Net lift subtracts the settled INITIAL lowest centerline node height;
    support height is never lift. Caller must rebase initial_min_z_m after settling
    and may accumulate Hold only while eligible.
    Any unknown contact/load/freshness evidence returns NA.
    """
    na = dict(status='NA', eligible=False, net_lift_m=None)
    nodes = np.asarray(_array(nodes), dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2
            or not np.isfinite(nodes).all() or not np.isfinite([radius_m, initial_min_z_m,
            force_tolerance_n, clearance_tolerance_m]).all() or radius_m <= 0
            or force_tolerance_n < 0 or clearance_tolerance_m < 0):
        return dict(na, reason='invalid_geometry_or_reference')
    if (not contact_record or not contact_record.get('contact_data_valid')
            or contact_record.get('step_id') is None
            or contact_record.get('mesh_indices') != [11, 12]
            or (expected_step_id is not None and contact_record['step_id'] != expected_step_id)):
        return dict(na, reason='missing_or_invalid_native_contact_data')
    load = contact_record.get('total_support_force_norm_n')
    if load is None or not np.isfinite(load) or load < 0:
        return dict(na, reason='invalid_support_load')
    inflated_radius = radius_m + clearance_tolerance_m
    intersections = []
    for support in scene['supports']:
        center = np.asarray(support['center_m'])
        half = np.asarray(support['size_m']) / 2 + inflated_radius
        intersections.append(_segments_intersect_box(nodes, center-half, center+half))
    # Leaving sideways and falling onto the retained table is not a clean lift.
    table_clear = bool(np.min(nodes[:, 2]) - radius_m > scene['table_top_m'] + clearance_tolerance_m)
    eligible = not any(intersections) and table_clear and load <= force_tolerance_n
    return dict(status='clear' if eligible else 'supported', eligible=bool(eligible),
                net_lift_m=max(0., float(np.min(nodes[:, 2]) - initial_min_z_m)) if eligible else None,
                support_intersections=intersections, support_force_n=float(load),
                table_clear=table_clear, reason='qualified' if eligible else 'geometry_or_load_not_clear')


def power_surface_qualification(surface_evidence):
    """Require actual palm/proximal surfaces in BOTH engines for both hands.

    Geometry readiness does not prove palm enclosure/five-finger closure during
    a grasp. Five distal pads alone are always blocked.
    """
    missing = []
    for side in ('left', 'right'):
        evidence = surface_evidence.get(side, {})
        for surface in ('palm', 'proximal'):
            item = evidence.get(surface, {})
            for key in ('physx_collision', 'deformx_sdf_registered', 'geometry_source'):
                if not item.get(key):
                    missing.append(f'{side}.{surface}.{key}')
    return dict(status='blocked' if missing else 'geometry_ready',
                missing=missing, grasp_execution_qualified=False,
                required_behavior='palm_enclosure_and_five_fingers_closed',
                reason='missing_native_palm_or_proximal_surfaces' if missing else 'runtime_grasp_evidence_still_required')