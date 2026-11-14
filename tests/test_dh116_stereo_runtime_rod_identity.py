"""Independent rigid/deformable task source contracts. CPU fixtures only."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_stereo_native as native
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from test_dh116_stereo_runtime_teacher_identity import bridge as teacher_bridge, run_collection


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/dh116_training/stereo_native_task_v1.json'


def test_default_profile_records_the_actual_solid_cylinder():
    spec = json.loads(PROFILE.read_text(encoding='utf-8'))
    assert spec.get('task_kind') == 'rigid_proxy'
    assert spec['physical_object_kind'] == 'solid_cylinder'
    i = spec['arguments'].index('--object-kind')
    assert spec['arguments'][i + 1] == 'solid_cylinder'
    identity = native.native_task_identity(spec)
    assert identity['task_kind'] == 'rigid_proxy'
    assert identity['physical_object_kind'] == 'solid_cylinder'
    assert identity['rod_state_available'] is False


def test_deformable_request_blocks_before_rigid_output_or_isaac(tmp_path, monkeypatch):
    spec = json.loads(PROFILE.read_text(encoding='utf-8'))
    spec.update(task_kind='deformable_rod', physical_object_kind='cosserat_rod', arguments=[])
    path = tmp_path / 'deformable.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    output = tmp_path / 'output'
    entered = []

    @contextmanager
    def record_native_lock():
        entered.append('native_lock')
        raise AssertionError('deformable request must never reach the rigid launcher')
        yield

    monkeypatch.setattr(native, 'native_instance_lock', record_native_lock)
    with pytest.raises(RuntimeError, match='deformable_rod.*Cosserat'):
        native.launch_native(lambda _: {}, native_arguments=path, output=output, enabled=True)
    assert not entered and not output.exists()


def test_profile_label_cannot_override_rigid_launcher_argv(tmp_path):
    spec = json.loads(PROFILE.read_text(encoding='utf-8'))
    spec.update(task_kind='rigid_proxy', physical_object_kind='hollow_straw')
    path = tmp_path / 'wrong.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='physical_object_kind.*--object-kind'):
        native.launch_native(lambda _: {}, native_arguments=path, output=output, enabled=True)
    assert not output.exists()


def test_task_identity_differentiates_authored_rigid_and_rod_profiles():
    spec = json.loads(PROFILE.read_text(encoding='utf-8'))
    a = native.native_task_identity(spec)
    b = native.native_task_identity(dict(spec, task_kind='deformable_rod',
        physical_object_kind='cosserat_rod', arguments=[]))
    assert a['task_identity_sha256'] != b['task_identity_sha256']
    assert a['state_source'] == 'rigid_body_sdk_pose_velocity'
    assert b['state_source'] == 'cosserat_solver_node_readback'


def test_legacy_unlabeled_profile_resolves_only_as_rigid_proxy():
    spec = dict(version='dh116-stereo-native-bootstrap-v1', arguments=['--object-kind=hollow_straw'])
    assert native.native_task_identity(spec)['physical_object_kind'] == 'hollow_straw'
    spec['arguments'] += ['--object-kind', 'solid_cylinder']
    with pytest.raises(ValueError, match='duplicate --object-kind'):
        native.native_task_identity(spec)


def test_actual_teacher_camera_and_raw_rows_share_frozen_rigid_binding(teacher_bridge, tmp_path, monkeypatch):
    spec = json.loads(PROFILE.read_text(encoding='utf-8'))
    identity = native.native_task_identity(spec)
    teacher_bridge._task_identity = identity
    descriptor = teacher_bridge.task_identity
    descriptor['physical_object_kind'] = 'cosserat_rod'
    assert teacher_bridge.task_identity['physical_object_kind'] == 'solid_cylinder'

    run_collection(teacher_bridge, tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (tmp_path/'native_physics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['kind'] for row in rows] == ['episode_start'] + ['sample'] * 16
    for row in rows:
        assert row['task_identity'] == identity
        assert row['task_identity_sha256'] == identity['task_identity_sha256']
        assert row['pre_action_identity']['task_identity'] == identity
        assert row['pre_action_identity']['task_identity_sha256'] == identity['task_identity_sha256']


def provider_class():
    from go2_pvcnn.control.dh116 import tabletop_stereo_rod_state as rod
    return rod.DeformableRodStateProvider


def rod_packet():
    positions = torch.tensor([[[0., 0., .4], [.1, 0., .4], [.21, 0., .4]]])
    return dict(version='dh116-deformable-rod-readback-v1', coordinate_frame='world',
        units=dict(node_positions='m', velocities='m/s', centerline='m', tangent='1',
            curvature='1/m', distributed_contact='N', deformation='1', rest_lengths_m='m'),
        env_ids=[0], episode_uids=['episode0'], reset_ids=[0], physics_step=16, timestamp_s=16/480,
        node_positions=positions, velocities=torch.zeros_like(positions), centerline=positions.clone(),
        tangent=torch.tensor([[[1., 0., 0.]]*3]), curvature=torch.zeros(1, 3),
        distributed_contact=torch.tensor([[[0., 0., 1.], [float('nan')]*3, [0., 0., 0.]]]),
        contact_valid=torch.tensor([[True, False, True]]), rest_lengths_m=torch.full((1, 2), .1),
        deformation=torch.tensor([[0., .1]]))


def bound_provider(tmp_path, packet, *, sha=None):
    backend = tmp_path / 'backend_fixture.py'
    backend.write_text('# test backend file used for byte identity only\n', encoding='utf-8')
    solver = SimpleNamespace(read_rod_snapshot=lambda: packet)
    return provider_class()(solver, backend='DeformX', backend_artifact=backend,
        backend_sha256=file_sha256(backend) if sha is None else sha)


def read(provider, **changes):
    kwargs = dict(env_ids=[0], episode_uids=['episode0'], reset_ids=[0], physics_step=16, timestamp_s=16/480)
    kwargs.update(changes)
    return provider.snapshot(**kwargs)


def test_solver_packet_retains_node_geometry_and_has_owned_readback(tmp_path):
    packet = rod_packet()
    provider = bound_provider(tmp_path, packet)
    row = read(provider)
    assert row['task_kind'] == 'deformable_rod'
    assert row['source_scope'] == 'solver_readback_unqualified'
    assert row['coordinate_frame'] == 'world'
    assert row['backend'] == 'DeformX'
    assert row['units']['distributed_contact'] == 'N'
    assert row['contact_valid'].tolist() == [[True, False, True]]
    assert torch.isnan(row['distributed_contact'][0, 1]).all()
    torch.testing.assert_close(row['deformation'], torch.tensor([[0., .1]]))
    row['node_positions'][0, 0].zero_()
    torch.testing.assert_close(read(provider)['node_positions'], packet['node_positions'])
    assert packet['node_positions'][0, 0, 2] > 0


@pytest.mark.parametrize('missing', ['node_positions', 'velocities', 'centerline', 'tangent',
    'curvature', 'distributed_contact', 'deformation', 'contact_valid', 'rest_lengths_m'])
def test_root_pose_cannot_substitute_missing_node_fields(tmp_path, missing):
    packet = rod_packet()
    packet.pop(missing)
    packet['root_pose'] = torch.zeros(1, 7)
    with pytest.raises(ValueError, match=missing):
        read(bound_provider(tmp_path, packet))


@pytest.mark.parametrize(('field','value'), [('episode_uids',['other']), ('reset_ids',[1]),
    ('env_ids',[2]), ('physics_step', 15), ('timestamp_s', .1)])
def test_readback_rejects_different_time_or_episode(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        read(bound_provider(tmp_path, rod_packet()), **{field:value})


@pytest.mark.parametrize(('field','value'), [
    ('node_positions',torch.zeros(1, 7)), ('velocities',torch.zeros(1, 4, 3)),
    ('centerline',torch.zeros(1, 3, 3)), ('tangent',torch.ones(1, 3, 3)),
    ('curvature',-torch.ones(1, 3)), ('distributed_contact',torch.zeros(1, 3, 3)),
    ('deformation',torch.zeros(1, 2)), ('contact_valid',torch.zeros(1, 3)),
    ('coordinate_frame','unknown'), ('units',{'node_positions':'cm'})])
def test_rejects_invalid_rod_contract_values(tmp_path, field, value):
    packet = rod_packet()
    packet[field] = value
    with pytest.raises(ValueError, match=field):
        read(bound_provider(tmp_path, packet))


def test_backend_sha_is_bound_to_actual_file_and_cannot_change(tmp_path):
    with pytest.raises(ValueError, match='SHA'):
        bound_provider(tmp_path, rod_packet(), sha='0'*64)
    with pytest.raises(ValueError, match='SHA'):
        bound_provider(tmp_path, rod_packet(), sha='a'*64)
    provider = bound_provider(tmp_path, rod_packet())
    (tmp_path / 'backend_fixture.py').write_text('modified', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA'):
        read(provider)
