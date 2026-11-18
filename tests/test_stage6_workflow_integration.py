"""Synthetic CPU fileflow regressions; no native trial or grasp proof is produced.

Qualification flags below exercise the existing schema gates only. The source
checkpoint, controller records, camera and assets are synthetic test fixtures.
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from test_bimanual_checkpoint_contract import derive, setup as adapter_setup
from test_vision_experiment_checkpoint import same
from test_vision_sequence_dataset import fixture_context, fixture_data
from test_vision_sequence_training import command_dataset

from go2_pvcnn.control.dh116 import tabletop_bimanual_checkpoint_contract as checkpoint_api
from go2_pvcnn.control.dh116 import tabletop_vision_stage6_workflow as workflow
from go2_pvcnn.control.dh116.tabletop_surface_contract import (
    ACTION_NAME, LEGACY_OBSERVATION_VERSION, bind_contract,
)
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import (
    LEGACY_VERSION, _ADDED, upgrade_action_contract,
)
from go2_pvcnn.control.dh116.tabletop_vision_quaternion import HEMISPHERE_WRIST_QUATERNIONS


@pytest.fixture(autouse=True)
def isolated_cpu_rng_and_threads():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(2)
        try:
            yield
        finally:
            torch.set_num_threads(threads)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def _bound_inputs(tmp_path, *, episode_count=4):
    adapter = adapter_setup.__wrapped__(tmp_path)
    actor, recorded, controller_contract = command_dataset()
    source = copy.deepcopy(adapter.before)
    source['policy'].update({'actor.' + name: value.clone() for name, value in actor.items()})
    camera = tmp_path / 'camera.json'
    heldout = tmp_path / 'frozen_heldout.json'
    _write_json(camera, {'synthetic_cpu_fixture': 'camera'})
    _write_json(heldout, {'synthetic_cpu_fixture': 'heldout_manifest'})

    # Extend the existing complete-PPO fixture with the sensor/surface contract
    # required by the file workflow; retain all original non-Actor state.
    base = copy.deepcopy(source['contract'])
    for name in _ADDED:
        base.pop(name)
    base['version'] = LEGACY_VERSION
    base['network'].pop('actor_output')
    base.update(actor_wrist_quaternion_preprocessing=HEMISPHERE_WRIST_QUATERNIONS,
                camera_config_sha256=workflow.file_identity(camera)['sha256'],
                primary_reset_sha256='2' * 64, secondary_reset_sha256='3' * 64)
    critic = base['privileged_critic_contract']
    critic.pop('surface_contract_sha256')
    critic.update(action_size=24, action=ACTION_NAME, observation_size=351,
                  observation_version=LEGACY_OBSERVATION_VERSION)
    base['privileged_critic_contract'] = bind_contract(critic)
    source['contract'] = upgrade_action_contract(base)
    torch.save(source, adapter.source)
    derive(adapter)
    source = torch.load(adapter.output, map_location='cpu', weights_only=False)
    core = checkpoint_api.unwrap_physical_adapter_checkpoint(source, allow_explicit=True)

    data = fixture_data((16,) * episode_count,
                        groups=[f'CPU_SYNTHETIC_SOURCE_{i}' for i in range(episode_count)])
    data['observations'] = recorded.data['observations'].repeat(episode_count, 1)
    data['actions'].zero_()
    data['reference_actions'].zero_()
    for row in data['sample_metadata']:
        row['previous_action'] = [0.] * 24
    data['visual_contract'] = copy.deepcopy(core['contract'])
    context = fixture_context(data)
    for name in ('controller_start', 'physics_context', 'teacher_targets', 'events'):
        context[name] = {
            key: torch.cat([value.clone() for _ in range(episode_count)])
            for key, value in recorded.context[name].items()
        }
    context['controller_source_sha256'] = controller_contract['controller_source_sha256']
    context['controller_contract'] = controller_contract
    context['teacher_shadow_parity'] = dict(
        passed=True, native_sdk_readback_required=True,
        native_sdk_readback={name: dict(passed=True) for name in ('backend', 'command_cache')})
    dataset_path, context_path = tmp_path / 'teacher.pt', tmp_path / 'context.pt'
    torch.save(data, dataset_path)
    context['dataset_sha256'] = workflow.file_identity(dataset_path)['sha256']
    torch.save(context, context_path)

    report_path = tmp_path / 'teacher_report.json'
    _write_json(report_path, dict(
        synthetic_cpu_fixture=True, native_runtime_verified=True, successful_episodes=4,
        ppo_updates=0, teacher_success_is_student_success=False,
        successful_dataset=dict(**workflow.file_identity(dataset_path), samples=len(data['observations'])),
        successful_sequence_context=workflow.file_identity(context_path),
        episodes=copy.deepcopy(data['episodes'])))
    assets = {side: tmp_path / (side + '.fixture.usd') for side in ('left', 'right')}
    inputs_path = tmp_path / 'inputs.json'
    _write_json(inputs_path, dict(
        version=workflow.VERSION, actor_dim=3844, critic_dim=371, action_dim=24,
        source_checkpoint=workflow.file_identity(adapter.output),
        dataset=workflow.file_identity(dataset_path), sequence_context=workflow.file_identity(context_path),
        teacher_collection_report=workflow.file_identity(report_path),
        assets={side: workflow.file_identity(path) for side, path in assets.items()},
        camera=workflow.file_identity(camera), final_heldout_manifest=workflow.file_identity(heldout),
        preprocessing=dict(wrist_quaternion=HEMISPHERE_WRIST_QUATERNIONS,
                           point_activation='relu', mirror='m0_raw'), model_version=source['version']))
    return SimpleNamespace(inputs=inputs_path, report=report_path, source_path=adapter.output,
                           source=source, assets=assets)


def _prepare(fixture, output):
    workflow.prepare_sequences(fixture.inputs, fixture.report, output)
    return output / 'prepared.json'


def test_input_resolution_binds_actual_model_asset_camera_and_new_collection(tmp_path):
    fixture = _bound_inputs(tmp_path)
    complete = json.loads(fixture.inputs.read_text())
    launch = tmp_path / 'source_launch.json'
    _write_json(launch, {'command':['bash','scripts/python.sh','train.py',
                                  '--depth-camera-config',complete['camera']['path']]})
    discovery = tmp_path / 'discovery.json'
    _write_json(discovery,dict(version=workflow.VERSION,actor_dim=3844,critic_dim=371,action_dim=24,
                              source_checkpoint=complete['source_checkpoint'],
                              source_launch=workflow.file_identity(launch)))
    resolved = workflow.resolve_stage6_inputs(discovery,fixture.report,
                    complete['final_heldout_manifest']['path'],tmp_path/'resolved.json')
    for name in ('source_checkpoint','dataset','sequence_context','teacher_collection_report',
                 'assets','camera','preprocessing','model_version','final_heldout_manifest'):
        assert resolved[name] == complete[name]


@pytest.mark.parametrize('profile', ('B0', 'B1', 'B2'))
def test_v4_sequence_fileflow_preserves_every_non_actor_source_state(tmp_path, profile):
    fixture = _bound_inputs(tmp_path)
    source_identity = workflow.file_identity(fixture.source_path)
    prepared = _prepare(fixture, tmp_path / 'prepared')
    output = tmp_path / profile
    report = workflow.train_sequences(prepared, output, profile=profile, updates=2, checkpoint_interval=1)

    assert report['valid_exposures'] == 128 and report['padding_exposures'] == 0
    assert [item['step'] for item in report['checkpoints']] == [1, 2]
    assert report['native_success_claimed'] is False and report['ppo_updates'] == 0
    assert report['actor_wrist_quaternion_preprocessing'] == HEMISPHERE_WRIST_QUATERNIONS
    assert report['output_tanh_count'] == 1
    assert json.loads((output / 'sequence_training_report.json').read_text()) == report
    for item in report['checkpoints']:
        assert workflow.file_identity(item['path']) == {name: item[name] for name in ('path', 'sha256')}
        after = torch.load(item['path'], map_location='cpu', weights_only=False)
        checkpoint_api.unwrap_physical_adapter_checkpoint(after, allow_explicit=True)
        for name, value in fixture.source.items():
            if name != 'policy':
                same(value, after[name])
        for name, value in fixture.source['policy'].items():
            if not name.startswith('actor.'):
                same(value, after['policy'][name])
        assert after['stage6_sequence_bc']['actor_point_activation'] == 'relu'
        assert after['stage6_pointnet_probe']['step'] <= item['step']
        assert 'events' in after['stage6_pointnet_probe']
        assert after['stage6_diagnostic_input']['observations'].shape == (4, 16, 3844)
    assert any(not torch.equal(value, after['policy'][name])
               for name, value in fixture.source['policy'].items() if name.startswith('actor.'))
    assert workflow.file_identity(fixture.source_path) == source_identity


def test_prepare_rejects_four_reported_successes_with_only_one_pool_episode(tmp_path):
    fixture = _bound_inputs(tmp_path, episode_count=1)
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='four|episode|count'):
        _prepare(fixture, output)
    assert not output.exists()


def test_prepare_rejects_report_path_replacement_even_with_identical_bytes(tmp_path):
    fixture = _bound_inputs(tmp_path)
    replacement = tmp_path / 'unbound_report.json'
    replacement.write_bytes(fixture.report.read_bytes())
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='bound|manifest|identity'):
        workflow.prepare_sequences(fixture.inputs, replacement, output)
    assert not output.exists()


def test_training_rechecks_asset_bytes_after_data_preparation(tmp_path):
    fixture = _bound_inputs(tmp_path)
    prepared = _prepare(fixture, tmp_path / 'prepared')
    fixture.assets['left'].write_bytes(b'externally changed synthetic fixture')
    output = tmp_path / 'training'
    with pytest.raises(ValueError, match='SHA|asset'):
        workflow.train_sequences(prepared, output, profile='B0', updates=1)
    assert not output.exists()


def test_training_rejects_replacement_of_the_pinned_input_manifest(tmp_path):
    fixture = _bound_inputs(tmp_path)
    prepared = _prepare(fixture, tmp_path / 'prepared')
    manifest = json.loads(fixture.inputs.read_text())
    manifest['synthetic_cpu_fixture'] = 'externally replaced manifest'
    _write_json(fixture.inputs, manifest)
    output = tmp_path / 'training'
    with pytest.raises(ValueError, match='SHA|manifest|bound'):
        workflow.train_sequences(prepared, output, profile='B0', updates=1)
    assert not output.exists()


def test_freeze_rejects_nested_target_before_creation_or_recursive_walk(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'entry.py').write_text('pass\n', encoding='utf-8')
    target = source / 'stage6_eval_source'

    def unexpected_walk(*args, **kwargs):
        pytest.fail('nested source freeze must stop before walking its own output tree')

    monkeypatch.setattr(workflow.os, 'walk', unexpected_walk)
    with pytest.raises(ValueError, match='outside|source|nested'):
        workflow.freeze_source(source, target)
    assert not target.exists()
