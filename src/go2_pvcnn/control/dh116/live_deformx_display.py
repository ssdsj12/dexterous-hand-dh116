"""Presentation-only adapter for already captured DeformX physical endpoints.

The owner must capture under its exclusive read phase. Clock checks detect drift;
they do not provide a lock. The snapshot retains no native or exchange references.
"""
from copy import deepcopy
import math
import numpy as np

from .tabletop_bimanual_table_contacts import _clock
from .tabletop_cosserat_material_marker_reference import _rotation
from .tabletop_native_teacher_profile import BackendProfile


def _profile(profile):
    if type(profile) is not BackendProfile:
        raise ValueError('actual source-loaded BackendProfile required')
    profile.verify()
    return profile


def _radius(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.floating)):
        raise ValueError('finite positive radius required')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive radius required')
    # DeformX stores contactRadius as float32: compare canonical native values.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        value = float(np.float32(value))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('radius must be representable as positive float32')
    return value


def capture_snapshot(exchange, *, source_instance_id, read_clock, backend_profile):
    """Capture owned CPU geometry from a completed physics exchange."""
    try:
        rod, packet = exchange['rod'], exchange['material_director']
    except (KeyError, TypeError) as exc:
        raise ValueError('missing completed physics exchange') from exc
    return _capture(rod, packet, phase='physics', source_instance_id=source_instance_id,
                    read_clock=read_clock, backend_profile=backend_profile)


def capture_reset_snapshot(rod_reset, material_director, *, source_instance_id, read_clock, backend_profile):
    """Capture the real reset receipt and its reader.capture_reset packet, with no contact claim."""
    return _capture(rod_reset, material_director, phase='reset', source_instance_id=source_instance_id,
                    read_clock=read_clock, backend_profile=backend_profile)


def _capture(rod, packet, *, phase, source_instance_id, read_clock, backend_profile):
    if type(source_instance_id) is not int or source_instance_id <= 0:
        raise ValueError('actual rod instance identity required')
    try:
        profile = _profile(backend_profile)
        from . import tabletop_mesh_observation_source as source
        identity = _clock(lambda: packet['identity'])
        basis = source.source_basis()
        boundary = 'completed_reset_endpoint' if phase == 'reset' else 'completed_physics_endpoint'
        if phase == 'reset' and (identity['rod_tick'] != 0 or identity['env_id'] != 0
                                 or identity['reset_id'] < 1 or rod['contact_valid'] is not False):
            raise ValueError('actual reset endpoint with no contact sample required')
        if (_clock(lambda: rod) != identity or _clock(read_clock) != identity
                or packet['source_instance_id'] != source_instance_id
                or packet['version'] != source.PARAMETER_VERSION
                or packet['sources'] != basis
                or basis['backend_sha256'] != profile.binary_sha256
                or basis['backend_path'] != profile.binary_path
                or basis['source_tree_sha256'] != profile.source_tree_sha256
                or packet['boundary'] != boundary
                or not isinstance(packet['material_reference_uid'], str)
                or not packet['material_reference_uid']):
            raise ValueError('native source or completed boundary identity differs')
        field = packet['native_parameters']['fields']['contact_radius_m']
        radius = _radius(field['value'])
        if (packet['raw_native_parameters']['time_s'] != identity['solver_time_s']
                or packet['native_parameters']['time_s'] != identity['solver_time_s']
                or field['source'] != 'native_host_metadata' or field['valid'] is not True
                or _radius(packet['raw_native_parameters']['contact_radius_m']) != radius):
            raise ValueError('native parameter radius differs')
        nodes = np.array(packet['node_positions_m'], copy=True)
        quats = np.array(packet['segment_quaternions_xyzw'], copy=True)
        if (nodes.shape != (1, 41, 3) or quats.shape != (1, 40, 4)
                or nodes.dtype.kind != 'f' or quats.dtype.kind != 'f'
                or not np.isfinite(nodes).all() or not np.isfinite(quats).all()
                or not np.array_equal(nodes, np.asarray(rod['node_positions']))):
            raise ValueError('finite same-boundary 41-node and 40-director data required')
        if np.any(np.linalg.norm(np.diff(nodes[0], axis=0), axis=1) <= 1e-10):
            raise ValueError('distinct rod nodes required')
        directors = np.stack([_rotation(q)[:, [1, 2, 0]].T for q in quats[0]], axis=-1)
        if _clock(read_clock) != identity:
            raise ValueError('clock changed while copying endpoint')
        return dict(rod_position=nodes[0].T.copy(), rod_director=directors,
                    identity=deepcopy(identity), source_instance_id=source_instance_id,
                    material_reference_uid=packet['material_reference_uid'],
                    backend_profile_digest=profile.digest, radius_m=radius, phase=phase)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('missing or malformed native endpoint') from exc


class LiveDeformxDisplay:
    """Own only a visual publisher, tied to one owner, profile and radius."""
    def __init__(self, publisher, set_visible, *, radius, source_instance_id, backend_profile):
        self._publisher = publisher
        self._set_visible = set_visible
        self._set_visible(False)
        self._radius = _radius(radius)
        if type(source_instance_id) is not int or source_instance_id <= 0:
            raise ValueError('actual rod instance identity required')
        self._source_instance_id = source_instance_id
        self._profile = _profile(backend_profile)

    @classmethod
    def for_mesh(cls, mesh, radius, *, source_instance_id, backend_profile):
        from pxr import UsdGeom
        from .wire_mesh_publisher import WireMeshPublisher
        imageable = UsdGeom.Imageable(mesh.GetPrim())
        def visibility(visible):
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        return cls(WireMeshPublisher(mesh, radius), visibility, radius=radius,
                   source_instance_id=source_instance_id, backend_profile=backend_profile)

    def hide(self):
        self._set_visible(False)

    def publish_snapshot(self, snapshot, *, read_clock):
        """Publish owned data only if current owner, epoch, clock and radius match."""
        self.hide()
        try:
            profile = _profile(self._profile)
            identity = _clock(lambda: snapshot['identity'])
            if (snapshot['phase'] not in ('reset', 'physics')
                    or (snapshot['phase'] == 'reset' and identity['rod_tick'] != 0)
                    or snapshot['source_instance_id'] != self._source_instance_id
                    or snapshot['backend_profile_digest'] != profile.digest
                    or _radius(snapshot['radius_m']) != self._radius
                    or _clock(read_clock) != identity):
                raise ValueError('snapshot owner, profile, radius or clock differs')
            self._publisher(snapshot)
            if _clock(read_clock) != identity:
                raise ValueError('clock changed while publishing mesh')
            self._set_visible(True)
            return snapshot
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('missing or malformed owned snapshot') from exc

    def publish(self, exchange, *, source_instance_id, read_clock):
        self.hide()
        if source_instance_id != self._source_instance_id:
            raise ValueError('rod owner changed')
        snapshot = capture_snapshot(exchange, source_instance_id=source_instance_id,
                                    read_clock=read_clock, backend_profile=self._profile)
        return self.publish_snapshot(snapshot, read_clock=read_clock)
