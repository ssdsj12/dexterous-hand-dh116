"""Same-step task contact diagnostics from independent PhysX report readback."""
from __future__ import annotations

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
                source_normal_impulse_world_n_s=[point['impulse'] for point in pair['contacts']],
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
