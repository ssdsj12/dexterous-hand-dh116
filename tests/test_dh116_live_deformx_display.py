import copy
import numpy as np
import pytest
from go2_pvcnn.control.dh116.live_deformx_display import capture_snapshot, LiveDeformxDisplay


@pytest.fixture(autouse=True)
def profile_fixture(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_profile as profile_module
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    global PROFILE
    PROFILE = object.__new__(profile_module.BackendProfile)
    object.__setattr__(PROFILE, 'binary_sha256', source.BACKEND_SHA)
    object.__setattr__(PROFILE, 'digest', 'fixture-profile')
    object.__setattr__(PROFILE, 'binary_path', str(source.BACKEND))
    object.__setattr__(PROFILE, 'source_tree_sha256', source.TREE_SHA)
    monkeypatch.setattr(profile_module.BackendProfile, 'verify', lambda self: self)
    monkeypatch.setattr(source, 'source_basis', lambda: {'backend_sha256': source.BACKEND_SHA, 'backend_path': str(source.BACKEND), 'source_tree_sha256': source.TREE_SHA})


def sample():
    clock = dict(run_uuid='run', env_id=0, episode_uid='ep', reset_id=2, rod_uid='rod', physics_step=12, rod_tick=12, timestamp_s=12/480, solver_time_s=.025, clock_valid=True)
    nodes = np.zeros((1, 41, 3), np.float32)
    nodes[0, :, 0] = np.arange(41) * .01
    quat = np.tile(np.array([0, 0, 0, 1], np.float32), (1, 40, 1))
    packet = dict(version='dh116-mesh-observation-parameter-endpoints-v1', sources=__import__('go2_pvcnn.control.dh116.tabletop_mesh_observation_source', fromlist=['source_basis']).source_basis(), raw_native_parameters={'contact_radius_m': .005, 'time_s': .025}, native_parameters={'time_s': .025, 'fields': {'contact_radius_m': {'value': .005, 'source': 'native_host_metadata', 'valid': True}}}, identity=copy.deepcopy(clock), source_instance_id=123, boundary='completed_physics_endpoint', material_reference_uid='material', node_positions_m=nodes, segment_quaternions_xyzw=quat)
    return dict(rod=dict(clock, node_positions=nodes.copy()), material_director=packet), clock


def test_snapshot_is_owned_and_maps_x_tangent_to_row_basis():
    row, clock = sample()
    # Rotation about X by 90 degrees: Y -> Z, Z -> -Y, X -> X.
    row['material_director']['segment_quaternions_xyzw'][0, :, :] = [2**-.5, 0, 0, 2**-.5]
    snap = capture_snapshot(row, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)
    assert np.allclose(snap['rod_director'][:, :, 0], [[0, 0, 1], [0, -1, 0], [1, 0, 0]], atol=1e-6)
    row['material_director']['node_positions_m'][:] = 7
    assert snap['rod_position'][0, 1] == pytest.approx(.01)
    assert snap['identity'] == clock


@pytest.mark.parametrize('field', ['source', 'episode', 'reset', 'tick', 'nan', 'missing', 'rod_nodes', 'clock'])
def test_invalid_snapshot_never_shows_mesh(field):
    row, clock = sample()
    if field == 'source': row['material_director']['source_instance_id'] = 124
    if field == 'episode': row['material_director']['identity']['episode_uid'] = 'other'
    if field == 'reset': row['material_director']['identity']['reset_id'] += 1
    if field == 'tick': row['material_director']['identity']['rod_tick'] += 1
    if field == 'nan': row['material_director']['segment_quaternions_xyzw'][0,0,0] = np.nan
    if field == 'missing': del row['material_director']['node_positions_m']
    if field == 'rod_nodes': row['rod']['node_positions'][0,1,0] = 5
    if field == 'clock': clock['physics_step'] += 1
    events = []
    display = LiveDeformxDisplay(lambda snapshot: events.append('publish'), lambda visible: events.append(visible), radius=.005, source_instance_id=123, backend_profile=PROFILE)
    events.clear()
    with pytest.raises((ValueError, KeyError)):
        display.publish(row, source_instance_id=123, read_clock=lambda: clock)
    assert events == [False]


def test_clock_change_during_publication_hides_mesh():
    row, clock = sample()
    events = []
    def publish(snapshot):
        events.append('publish')
        clock['rod_tick'] += 1
    display = LiveDeformxDisplay(publish, lambda visible: events.append(visible), radius=.005, source_instance_id=123, backend_profile=PROFILE)
    events.clear()
    with pytest.raises(ValueError):
        display.publish(row, source_instance_id=123, read_clock=lambda: clock)
    assert events == [False, False, 'publish']


def test_unproven_packet_is_rejected():
    row, clock = sample()
    del row['material_director']['sources']
    with pytest.raises(ValueError):
        capture_snapshot(row, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)


def test_real_surface_uses_cross_section_not_tangent():
    from go2_pvcnn.control.fixed_dual_panda_o6.wire_visual import rod_surface
    row, clock = sample()
    snap = capture_snapshot(row, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)
    vertices, counts, indices = rod_surface(snap['rod_position'], snap['rod_director'], .002)
    assert np.allclose(vertices[0], [0, .002, 0])
    assert np.allclose(vertices[4], [0, 0, .002], atol=1e-8)
    assert len(counts) == 40*16+2 and len(indices) > 0


def test_real_profile_expected_backend_can_capture(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_profile as profile_module
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    profile = object.__new__(profile_module.BackendProfile)
    object.__setattr__(profile, 'binary_sha256', '8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340')
    object.__setattr__(profile, 'digest', 'fixture-profile')
    object.__setattr__(profile, 'binary_path', str(source.BACKEND))
    object.__setattr__(profile, 'source_tree_sha256', source.TREE_SHA)
    monkeypatch.setattr(profile_module.BackendProfile, 'verify', lambda self: self)
    basis = {'backend_sha256': profile.binary_sha256, 'backend_path': profile.binary_path, 'source_tree_sha256': profile.source_tree_sha256}
    monkeypatch.setattr(source, 'source_basis', lambda: basis)
    row, clock = sample()
    row['material_director'].update(version=source.PARAMETER_VERSION, sources=basis,
        raw_native_parameters={'contact_radius_m': .005, 'time_s': .025},
        native_parameters={'time_s': .025, 'fields': {'contact_radius_m': {'value': .005, 'source': 'native_host_metadata', 'valid': True}}})
    snapshot = capture_snapshot(row, source_instance_id=123, read_clock=lambda: clock, backend_profile=profile)
    events = []
    display = LiveDeformxDisplay(lambda snap: events.append('publish'), lambda v: events.append(v), radius=.005, source_instance_id=123, backend_profile=profile)
    events.clear()
    display.publish_snapshot(snapshot, read_clock=lambda: clock)
    assert events == [False, 'publish', True]
    clock['reset_id'] += 1
    with pytest.raises(ValueError):
        display.publish_snapshot(snapshot, read_clock=lambda: clock)
    assert events[-1] is False
    display_bad = LiveDeformxDisplay(lambda snap: events.append('publish'), lambda v: events.append(v), radius=.006, source_instance_id=123, backend_profile=profile)
    with pytest.raises(ValueError):
        display_bad.publish_snapshot(snapshot, read_clock=lambda: snapshot['identity'])


@pytest.mark.parametrize("changed", ["backend", "tree", "path", "raw_radius", "radius_source", "raw_time", "native_time"])
def test_source_and_radius_mismatch_fail_closed(changed):
    row, clock = sample()
    packet = row['material_director']
    if changed == 'backend': packet['sources']['backend_sha256'] = 'wrong'
    if changed == 'tree': packet['sources']['source_tree_sha256'] = 'wrong'
    if changed == 'path': packet['sources']['backend_path'] = 'wrong'
    if changed == 'raw_radius': packet['raw_native_parameters']['contact_radius_m'] = .006
    if changed == 'raw_time': packet['raw_native_parameters']['time_s'] = 99
    if changed == 'native_time': packet['native_parameters']['time_s'] = 99
    if changed == 'radius_source': packet['native_parameters']['fields']['contact_radius_m']['source'] = 'fixture'
    with pytest.raises(ValueError):
        capture_snapshot(row, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)


def test_owned_snapshot_rejects_other_rod():
    row, clock = sample()
    snapshot = capture_snapshot(row, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)
    events = []
    display = LiveDeformxDisplay(lambda s: events.append('publish'), lambda v: events.append(v), radius=.005, source_instance_id=124, backend_profile=PROFILE)
    events.clear()
    with pytest.raises(ValueError):
        display.publish_snapshot(snapshot, read_clock=lambda: clock)
    assert events == [False]


def test_native_float32_radius_matches_same_python_radius():
    row, clock = sample()
    native_radius = float(np.float32(.005))
    row['material_director']['raw_native_parameters']['contact_radius_m'] = native_radius
    row['material_director']['native_parameters']['fields']['contact_radius_m']['value'] = native_radius
    snapshot = capture_snapshot(row, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)
    events = []
    display = LiveDeformxDisplay(lambda s: events.append('publish'), lambda v: events.append(v), radius=.005, source_instance_id=123, backend_profile=PROFILE)
    events.clear()
    display.publish_snapshot(snapshot, read_clock=lambda: clock)
    assert events == [False, 'publish', True]
    display_wrong = LiveDeformxDisplay(lambda s: events.append('publish'), lambda v: events.append(v), radius=.0051, source_instance_id=123, backend_profile=PROFILE)
    events.clear()
    with pytest.raises(ValueError):
        display_wrong.publish_snapshot(snapshot, read_clock=lambda: clock)
    assert events == [False]


def reset_sample():
    row, clock = sample()
    clock.update(rod_tick=0, physics_step=144, timestamp_s=144/480, solver_time_s=.3)
    rod = row['rod']
    rod.update(clock, contact_valid=False)
    packet = row['material_director']
    packet['identity'] = copy.deepcopy(clock)
    packet['boundary'] = 'completed_reset_endpoint'
    packet['raw_native_parameters']['time_s'] = .3
    packet['native_parameters']['time_s'] = .3
    return rod, packet, clock


def test_reset_snapshot_uses_real_reset_receipt_and_packet():
    from go2_pvcnn.control.dh116.live_deformx_display import capture_reset_snapshot
    rod, packet, clock = reset_sample()
    snap = capture_reset_snapshot(rod, packet, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)
    assert snap['phase'] == 'reset'
    assert snap['identity'] == clock
    assert 'contact' not in snap and 'contact_valid' not in snap
    packet['node_positions_m'][:] = 9
    assert snap['rod_position'][0, 1] == pytest.approx(.01)
    events = []
    display = LiveDeformxDisplay(lambda s: events.append('publish'), lambda v: events.append(v), radius=.005, source_instance_id=123, backend_profile=PROFILE)
    events.clear()
    display.publish_snapshot(snap, read_clock=lambda: clock)
    assert events == [False, 'publish', True]
    clock['rod_tick'] += 12
    with pytest.raises(ValueError):
        display.publish_snapshot(snap, read_clock=lambda: clock)
    assert events[-1] is False


@pytest.mark.parametrize('field', ['contact', 'tick', 'source', 'episode', 'reset', 'nodes', 'boundary', 'clock'])
def test_reset_snapshot_rejects_wrong_reset_source(field):
    from go2_pvcnn.control.dh116.live_deformx_display import capture_reset_snapshot
    rod, packet, clock = reset_sample()
    if field == 'contact': rod['contact_valid'] = True
    if field == 'tick': rod['rod_tick'] = packet['identity']['rod_tick'] = clock['rod_tick'] = 12
    if field == 'source': packet['source_instance_id'] = 124
    if field == 'episode': packet['identity']['episode_uid'] = 'other'
    if field == 'reset': packet['identity']['reset_id'] += 1
    if field == 'nodes': rod['node_positions'][0, 1, 0] = 9
    if field == 'boundary': packet['boundary'] = 'completed_physics_endpoint'
    if field == 'clock': clock['reset_id'] += 1
    with pytest.raises(ValueError):
        capture_reset_snapshot(rod, packet, source_instance_id=123, read_clock=lambda: clock, backend_profile=PROFILE)
