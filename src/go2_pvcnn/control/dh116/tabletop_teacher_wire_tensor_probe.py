"""Read-only Wire-side contact Tensor views in an existing CUDA simulation."""
from numbers import Integral

import numpy as np

from .tabletop_bimanual_mpc_scene_measurements import (
    ComposedUSDScene, _array, _finite, create_passive_contact_views,
)


def _paths(value):
    return list(value)


def _check_routing(view, sensors, filters):
    sensors = list(sensors)
    expected = [list(row) for row in filters]
    returned = _paths(view.filter_paths)
    width = len(expected[0])
    if returned == [path for row in expected for path in row]:
        returned = expected
    elif returned and not isinstance(returned[0], str):
        returned = [list(row) for row in returned]
    if (view.check() is not True or view.sensor_count != len(sensors)
            or _paths(view.sensor_paths) != sensors or view.filter_count != width or returned != expected):
        raise ValueError('native sensor/filter routing differs from exact configured actors or colliders')


def _owned_cells(snapshot, *, normal, n, width, capacity):
    widths = (1, 3, 3, 1) if normal else (3, 3)
    if len(snapshot) != len(widths) + 2:
        raise ValueError('installed SDK normal/friction tuple layout differs')
    *values, counts, starts = snapshot
    if (counts.shape != (n, width) or starts.shape != (n, width)
            or counts.dtype.kind not in 'iu' or starts.dtype.kind not in 'iu'
            or any(array.shape != (capacity, size) for array, size in zip(values, widths))):
        raise ValueError('native contact buffer or capacity layout mismatch')
    if int(counts.sum(dtype=np.int64)) >= capacity:
        raise ValueError('contact capacity saturated; completeness is unknown')
    occupied = set()
    output = np.zeros((n, width, 3))
    cells = []
    for i in range(n):
        row = []
        for j in range(width):
            start, count = int(starts[i,j]), int(counts[i,j])
            if start < 0 or count < 0 or start + count > capacity:
                raise ValueError('owned slot lies outside contact capacity')
            slots = set(range(start, start + count))
            if occupied.intersection(slots):
                raise ValueError('native owned slot belongs to multiple filter cells')
            occupied.update(slots)
            blocks = [array[start:start + count] for array in values]
            if any(not np.isfinite(block).all() for block in blocks):
                raise ValueError('owned native slot contains nonfinite values')
            vectors = blocks[0]
            if normal:
                if not np.allclose(np.linalg.norm(blocks[2], axis=-1), 1., rtol=1e-5, atol=1e-5):
                    raise ValueError('owned normal contact vector is not unit length')
                vectors = vectors * blocks[2]
            output[i,j] = vectors.sum(axis=0)
            row.append(dict(count=count, start=start, buffers=[block.tolist() for block in blocks]))
        cells.append(row)
    return output, cells


def read_filtered(view, sensors, filters, *, dt):
    _check_routing(view, sensors, filters)
    n, width = len(sensors), len(filters[0])
    capacity = view.max_contact_data_count
    if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 1:
        raise ValueError('positive actual filtered capacity required')
    matrix = _finite(_array(view.get_contact_force_matrix(dt=dt)), (n, width, 3))
    # The installed friction getter overwrites the normal getter's count/start
    # allocations. Take ownership of every normal buffer before that call.
    normal = tuple(_array(item) for item in view.get_contact_data(dt=dt))
    friction = tuple(_array(item) for item in view.get_friction_data(dt=dt))
    normal_sum, normal_cells = _owned_cells(normal, normal=True, n=n, width=width, capacity=capacity)
    friction_sum, friction_cells = _owned_cells(friction, normal=False, n=n, width=width, capacity=capacity)
    if not np.allclose(matrix, normal_sum, rtol=1e-4, atol=1e-5):
        raise ValueError('normal contact reduction differs from SDK force matrix')
    return dict(capacity=int(capacity), sensor_paths=list(sensors), filter_paths=[list(row) for row in filters],
        normal_pair_n=normal_sum.tolist(), friction_pair_n=friction_sum.tolist(), matrix_n=matrix.tolist(),
        normal_cells=normal_cells, friction_cells=friction_cells,
        upstream_overflow_excluded=False, source_side_frame_calibrated=False)


def net_residual(net, groups):
    net = _finite(net, (len(net), 3))
    normal = np.zeros_like(net)
    friction = np.zeros_like(net)
    for group in groups.values():
        shape = (len(net), len(group['filter_paths'][0]), 3)
        normal += _finite(group['normal_pair_n'], shape).sum(axis=1)
        friction += _finite(group['friction_pair_n'], shape).sum(axis=1)
    return dict(unfiltered_sdk_net_n=net.tolist(), enumerated_filtered_normal_n=normal.tolist(),
        enumerated_filtered_friction_n=friction.tolist(), net_minus_filtered_normal_n=(net-normal).tolist(),
        net_minus_filtered_normal_and_friction_n=(net-normal-friction).tolist(),
        net_contains_friction=None, unknown_partner_coverage_qualified=False, upstream_overflow_excluded=False)


class WireTensorProbe:
    """Post-reset, opt-in views on the original Torch CUDA SimulationView."""
    def __init__(self, bridge, stage, settings, *, capacity_per_wire=256):
        from pxr import PhysxSchema, UsdPhysics
        from .tabletop_teacher_task_contact import probe_inventory

        world = bridge.device_world
        sensors = tuple(world.w.sensors)
        if len(sensors) != 34:
            raise ValueError('original 34 hand actor contact sensors required')
        sim_view = sensors[0]._physics_sim_view
        if sim_view.device_ordinal < 0 or any(sensor._physics_sim_view is not sim_view for sensor in sensors):
            raise ValueError('original shared CUDA SimulationView required')
        self.world = world
        self.dt = world.dt
        self.wires = list(world.wire.root_physx_view.prim_paths)
        self.hand_views = [sensor.contact_physx_view for sensor in sensors]
        self.hand_actor_columns = [list(view.sensor_paths) for view in self.hand_views]
        self.hand_filters = [list(row) for row in zip(*self.hand_actor_columns)]
        if (len(self.hand_filters) != bridge.n or any(len(set(row)) != 34 for row in self.hand_filters)
                or any(not actor.startswith(wire.rsplit('/',1)[0]+'/')
                       for wire, row in zip(self.wires, self.hand_filters) for actor in row)):
            raise ValueError('unique actual hand actor routes for each Wire environment required')
        for view, actors in zip(self.hand_views, self.hand_actor_columns):
            _check_routing(view, actors, [[wire] for wire in self.wires])

        self.scene = ComposedUSDScene(stage=stage, settings=settings)
        self.passive = create_passive_contact_views(scene=self.scene, simulation_view=sim_view,
            rigid_body_view=world.wire.root_physx_view, enabled=True, capacity_per_wire=capacity_per_wire)
        self.hand = sim_view.create_rigid_contact_view(pattern=self.wires, filter_patterns=self.hand_filters,
            max_contact_data_count=bridge.n * capacity_per_wire)
        self.net = sim_view.create_rigid_contact_view(pattern=self.wires)
        _check_routing(self.hand, self.wires, self.hand_filters)
        _check_routing(self.net, self.wires, [[] for _ in self.wires])

        inventory = probe_inventory(bridge, stage, UsdPhysics, PhysxSchema)
        covered = {actor for row in self.hand_filters for actor in row}
        covered.update(path for row in self.passive.filter_paths.values() for path in row)
        covered.update(self.wires)
        unlisted = sorted(path for path in inventory['collision_prims']
                          if not any(path == candidate or path.startswith(candidate+'/') for candidate in covered))
        self.inventory = dict(inventory, wire_actors=self.wires, hand_actors_per_wire=self.hand_filters,
            exact_passive_colliders=dict(self.passive.filter_paths), unlisted_collision_prim_paths=unlisted,
            device_ordinal=sim_view.device_ordinal, normal_and_friction_api_scope='installed PhysX tensors',
            external_source_complete=False, no_usd_or_setting_write=True)

    def _hand_mirror(self, hand):
        normal, friction = [], []
        for view, actors in zip(self.hand_views, self.hand_actor_columns):
            expected = [[wire] for wire in self.wires]
            _check_routing(view, actors, expected)
            n, capacity = len(self.wires), view.max_contact_data_count
            normal.append(_finite(_array(view.get_contact_force_matrix(dt=self.dt)), (n, 1, 3))[:,0])
            snapshot = tuple(_array(item) for item in view.get_friction_data(dt=self.dt))
            reduced, _ = _owned_cells(snapshot, normal=False, n=n, width=1, capacity=capacity)
            friction.append(reduced[:,0])
        normal = np.stack(normal, axis=1)
        friction = np.stack(friction, axis=1)
        return dict(hand_sensor_normal_n=normal.tolist(), hand_sensor_friction_n=friction.tolist(),
            wire_plus_hand_sensor_normal_n=(np.asarray(hand['normal_pair_n'])+normal).tolist(),
            wire_plus_hand_sensor_friction_n=(np.asarray(hand['friction_pair_n'])+friction).tolist())

    def _wrench(self):
        result = {}
        for name in ('instantaneous_wrench_composer', 'permanent_wrench_composer'):
            composer = getattr(self.world.wire, name, None)
            if composer is None:
                result[name] = None
            else:
                result[name] = dict(active=bool(composer.active),
                    force_body=_array(composer.composed_force_as_torch).tolist(),
                    torque_body=_array(composer.composed_torque_as_torch).tolist())
        return result

    def sample(self):
        hand = read_filtered(self.hand, self.wires, self.hand_filters, dt=self.dt)
        groups = {'hand': hand}
        for name, view in self.passive.views.items():
            paths = [[path] for path in self.passive.filter_paths[name]]
            groups[name] = read_filtered(view, self.wires, paths, dt=self.dt)
        _check_routing(self.net, self.wires, [[] for _ in self.wires])
        net = _array(self.net.get_net_contact_forces(dt=self.dt))
        return dict(filtered=groups, aggregate=net_residual(net, groups),
            mirror=self._hand_mirror(hand), post_step_wire_wrench_composers=self._wrench(),
            external_wrench_absence_proven=False, force_qualification=False)
