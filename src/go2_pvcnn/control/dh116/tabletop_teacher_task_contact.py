"""Same-step task contact diagnostics from independent PhysX report readback."""
from __future__ import annotations

import math

import numpy as np

from .tabletop_stereo_contact_calibration import _report_pairs


def route_object_reports(report, *, objects, digits):
    """Preserve both actor sides and keep contact impulses distinct from anchors."""
    by_env = {env_id: [] for env_id in objects.values()}
    if len(by_env) != len(objects):
        raise ValueError('exactly one object actor per environment required')
    for pair in _report_pairs({'full_report': report}):
        for side in (0, 1):
            object_actor = pair[f'actor{side}']
            if object_actor not in objects:
                continue
            env_id = objects[object_actor]
            partner_actor = pair[f'actor{1 - side}']
            digit = digits.get(partner_actor)
            if digit is None:
                scope = 'unmapped_partner'
            elif digit[0] != env_id:
                scope = 'cross_environment_digit'
            else:
                scope = 'same_environment_digit'
            by_env[env_id].append(dict(
                object_actor_index=side, object_actor=object_actor,
                object_collider=pair[f'collider{side}'], partner_actor=partner_actor,
                partner_collider=pair[f'collider{1 - side}'], stage_id=pair['stage_id'],
                partner_scope=scope, digit=digit[1:] if scope == 'same_environment_digit' else None,
                contact_data_offset=pair['contact_data_offset'],
                friction_anchors_offset=pair['friction_anchors_offset'],
                source_contact_impulse_world_n_s=[point['impulse'] for point in pair['contacts']],
                source_friction_impulse_world_n_s=pair['friction_impulses'],
                object_force_world_n=None))
    return by_env


def probe_inventory(bridge, stage, usd_physics, physx_schema):
    """Resolve actual PhysX actor paths and record potential external sources."""
    from .tabletop_effective_pad import DISTAL_LINKS

    world = bridge.world
    object_paths = list(bridge.device_world.wire.root_physx_view.prim_paths)
    if len(object_paths) != bridge.n or len(set(object_paths)) != bridge.n:
        raise ValueError('unique native object actor path required per environment')
    objects = {path: i for i, path in enumerate(object_paths)}
    digits = {}
    digit_names = ('thumb', 'index', 'middle', 'ring', 'little')
    for hand_index, side in enumerate(('left','right')):
        for link, digit in zip(DISTAL_LINKS, digit_names):
            sensor = world.sensors[hand_index * len(world.sensor_links) + world.sensor_links.index(link)]
            paths = list(sensor.body_physx_view.prim_paths)
            if len(paths) != bridge.n:
                raise ValueError('native digit actor path count differs from environment count')
            for env_id, path in enumerate(paths):
                if path in digits or not path.startswith(world.paths[env_id]+'/'):
                    raise ValueError('duplicated or out-of-environment digit actor path')
                digits[path] = (env_id, side, digit)
    for env_id, path in enumerate(object_paths):
        if not path.startswith(world.paths[env_id]+'/'):
            raise ValueError('object actor does not belong to its declared native environment')
    report_actors = []
    collision_prims = []
    authored_force_prims = []
    attachment_and_joint_prims = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(usd_physics.CollisionAPI):
            collision_prims.append(path)
        if prim.HasAPI(physx_schema.PhysxContactReportAPI):
            report_actors.append(path)
        if prim.HasAPI(physx_schema.PhysxForceAPI):
            authored_force_prims.append(path)
        if prim.IsA(usd_physics.Joint) or 'attachment' in str(prim.GetTypeName()).lower():
            attachment_and_joint_prims.append(dict(path=path, type=str(prim.GetTypeName()),
                relationships={rel.GetName(): [str(target) for target in rel.GetTargets()]
                    for rel in prim.GetRelationships() if rel.GetTargets()}))
    return dict(objects=objects, digits=digits, report_actors=report_actors,
                collision_prims=collision_prims, authored_force_prims=authored_force_prims,
                attachment_and_joint_prims=attachment_and_joint_prims,
                external_source_complete=False)


def install_probe_world(stack, gpu_isaac, configure_readback):
    """Set the report flag before the isolated scene is built or first reset."""
    from unittest.mock import patch

    original = gpu_isaac.GpuWorld

    class ContactProbeWorld(original):
        def __init__(self, *args, **kwargs):
            self.contact_probe_before_reset = configure_readback()
            super().__init__(*args, **kwargs)

    stack.enter_context(patch.object(gpu_isaac, 'GpuWorld', ContactProbeWorld))


def momentum_comparison(pairs, *, before, after, mass_kg, gravity_world_m_s2, dt_s):
    """Compare both report sign hypotheses against measured COM momentum."""
    velocity_before, velocity_after, gravity = [
        np.asarray(value, dtype=float) for value in (before, after, gravity_world_m_s2)]
    if (not math.isfinite(mass_kg) or mass_kg <= 0 or not math.isfinite(dt_s) or dt_s <= 0
            or any(v.shape != (3,) or not np.isfinite(v).all() for v in (
                velocity_before, velocity_after, gravity))):
        raise ValueError('finite actual COM velocities, mass, gravity and positive dt required')
    measured = mass_kg * (velocity_after - velocity_before - dt_s * gravity)
    contact, friction = np.zeros(3), np.zeros(3)
    for pair in pairs:
        sign = (1, -1)[pair['object_actor_index']]
        contact += sign * np.asarray(pair['source_contact_impulse_world_n_s']).reshape(-1, 3).sum(0)
        friction += sign * np.asarray(pair['source_friction_impulse_world_n_s']).reshape(-1, 3).sum(0)
    inferred = contact + friction
    return dict(non_gravity_impulse_world_n_s=measured.tolist(),
        actor0_contact_impulse_world_n_s=contact.tolist(), actor0_friction_impulse_world_n_s=friction.tolist(),
        raw_on_actor0_residual_n_s=float(np.linalg.norm(measured-inferred)),
        raw_on_actor1_residual_n_s=float(np.linalg.norm(measured+inferred)),
        source_qualified=False)
