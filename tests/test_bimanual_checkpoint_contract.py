"""Synthetic CPU contract fixtures, excluded from native qualification counts."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_vision_experiment_checkpoint import checkpoint, same
from go2_pvcnn.control.dh116 import tabletop_bimanual_checkpoint_contract as api
from go2_pvcnn.control.dh116.tabletop_bimanual_physical_action_adapter import BimanualPhysicalActionAdapter
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_squashed_checkpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rehash(value):
    value.pop('sha256', None)
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                              allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def setup(tmp_path):
    source, before = checkpoint(tmp_path)
    models, assets = {}, {}
    for side in ('left', 'right'):
        asset = tmp_path / (side + '.fixture.usd')
        asset.write_bytes(('synthetic_' + side).encode())
        assets[side] = str(asset)
        models[side] = SimpleNamespace(side=side, source_sha256=sha(asset))
    adapter = BimanualPhysicalActionAdapter(models).contract()
    teacher = dict(status='completed', mode='teacher_reference_mpc', native_runtime_verified=True,
                   student_policy_trained=False, teacher_success_is_student_success=False,
                   ppo_updates=0, physical_sdk_adapter_max_executed_action_difference=0.,
                   successful_episodes=4, native_successful_episodes=4, observed_terminal_successes=4,
                   episodes={str(i): dict(env_id=i, first_episode=True, native_physics=True,
                             success=True, recovered=False, grasp_style='pinch_thumb_index',
                             terminal=dict(success=True),
                             native_contact_style_audit=dict(style='pinch_thumb_index')) for i in range(4)})
    teacher_path = tmp_path / 'teacher_collection_report.json'
    mapping_path = tmp_path / 'mapping_probe_summary.json'
    write(teacher_path, teacher)
    write(mapping_path, dict(status='completed', cases=53, failures=[], source_assets=assets))
    write(tmp_path / 'physical_action_adapter_contract.json', {'physical_action_adapter': adapter})
    return SimpleNamespace(source=source, before=before, output=tmp_path/'derived.pt',
                           adapter=adapter, teacher=teacher, teacher_path=teacher_path,
                           mapping_path=mapping_path)


def derive(context, **changes):
    options = dict(expected_source_sha256=sha(context.source), adapter_contract=context.adapter,
                   teacher_report_path=context.teacher_path, mapping_report_path=context.mapping_path)
    options.update(changes)
    return api.derive_physical_adapter_checkpoint(context.source, context.output, **options)


def test_explicit_derivation_preserves_every_original_state_and_source_bytes(setup):
    source_bytes = setup.source.read_bytes()
    rng = torch.get_rng_state().clone()
    report = derive(setup)
    saved = torch.load(setup.output, map_location='cpu', weights_only=False)
    assert saved['version'] == saved['contract']['version'] == api.VERSION
    same(saved['contract'][api.ADAPTER_KEY], setup.adapter)
    core = api.unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    same(core, setup.before)
    assert setup.source.read_bytes() == source_bytes
    assert torch.equal(rng, torch.get_rng_state())
    assert report['checkpoint_sha256'] == sha(setup.output)
    assert report['source_sha256'] == sha(setup.source)
    assert report['teacher_success_is_student_success'] is False
    assert json.loads(setup.output.with_suffix('.json').read_text()) == report
    assert api.validate_physical_adapter_runtime(saved, setup.adapter) == setup.adapter
    with pytest.raises(ValueError, match='explicit opt-in'):
        api.unwrap_physical_adapter_checkpoint(saved)
    with pytest.raises(ValueError, match='requires v3'):
        validate_squashed_checkpoint(saved)


def test_original_v3_validates_without_adapter_or_input_mutation(setup):
    before = copy.deepcopy(setup.before)
    same(api.unwrap_physical_adapter_checkpoint(setup.before), before)
    same(setup.before, before)
    assert api.ADAPTER_KEY not in setup.before['contract']
    with pytest.raises(ValueError, match='explicit V4'):
        api.validate_physical_adapter_runtime(setup.before, setup.adapter)


@pytest.mark.parametrize('case', ['source_hash', 'source_version', 'physics', 'weight_shape',
                                  'iteration_zero', 'adam_empty', 'adapter_hash', 'adapter_dimensions',
                                  'adapter_physical_rate', 'adapter_asset', 'already_bound'])
def test_invalid_source_state_or_declared_adapter_rejects_before_writes(setup, case):
    options = {}
    if case == 'source_hash':
        options['expected_source_sha256'] = '0' * 64
    elif case.startswith('adapter_'):
        adapter = copy.deepcopy(setup.adapter)
        if case == 'adapter_hash': adapter['sha256'] = '0' * 64
        elif case == 'adapter_dimensions': adapter['actor_size'] = 3845
        elif case == 'adapter_physical_rate': adapter['action']['joint_rate_rad_s'] = .31
        elif case == 'adapter_asset': adapter['source_asset_sha256']['right'] = '0' * 64
        if case != 'adapter_hash': rehash(adapter)
        options['adapter_contract'] = adapter
    else:
        saved = copy.deepcopy(setup.before)
        if case == 'source_version': saved['version'] = 'dh116-vision-student-ppo-v2-bounded'
        elif case == 'physics': saved['contract']['privileged_critic_contract']['object_contact_material']['profile'] = 'changed'
        elif case == 'weight_shape': saved['policy']['critic.4.bias'] = torch.zeros(2)
        elif case == 'iteration_zero': saved['iteration'] = 0
        elif case == 'adam_empty': saved['optimizer']['state'].clear()
        elif case == 'already_bound': saved['contract'][api.ADAPTER_KEY] = setup.adapter
        torch.save(saved, setup.source)
    with pytest.raises(ValueError): derive(setup, **options)
    assert not setup.output.exists() and not setup.output.with_suffix('.json').exists()


@pytest.mark.parametrize('case', ['not_native', 'student_claim', 'student_trained', 'ppo_update',
    'nonzero_execution_difference', 'missing_four', 'duplicate_env', 'recovered', 'not_first',
    'fixture_episode', 'terminal_failure', 'missing_style_audit', 'count_mismatch', 'native_count',
    'teacher_adapter_mismatch', 'mapping_failed', 'mapping_cases', 'mapping_asset_mismatch'])
def test_native_gates_require_consistent_teacher_and_mapping_evidence(setup, case):
    report = copy.deepcopy(setup.teacher)
    if case == 'not_native': report['native_runtime_verified'] = False
    elif case == 'student_claim': report['teacher_success_is_student_success'] = True
    elif case == 'student_trained': report['student_policy_trained'] = True
    elif case == 'ppo_update': report['ppo_updates'] = 1
    elif case == 'nonzero_execution_difference': report['physical_sdk_adapter_max_executed_action_difference'] = 1e-16
    elif case == 'missing_four': del report['episodes']['3']
    elif case == 'duplicate_env': report['episodes']['3']['env_id'] = 0
    elif case == 'recovered': report['episodes']['0']['recovered'] = True
    elif case == 'not_first': report['episodes']['0']['first_episode'] = False
    elif case == 'fixture_episode': report['episodes']['0']['native_physics'] = False
    elif case == 'terminal_failure': report['episodes']['0']['terminal']['success'] = False
    elif case == 'missing_style_audit': del report['episodes']['0']['native_contact_style_audit']
    elif case == 'count_mismatch': report['successful_episodes'] = 3
    elif case == 'native_count': report['native_successful_episodes'] = 3
    elif case == 'teacher_adapter_mismatch':
        adapter = copy.deepcopy(setup.adapter)
        adapter['adapter_source_sha256'] = '0' * 64
        rehash(adapter)
        write(setup.teacher_path.parent/'physical_action_adapter_contract.json', {'physical_action_adapter': adapter})
    elif case.startswith('mapping_'):
        mapping = json.loads(setup.mapping_path.read_text())
        if case == 'mapping_failed': mapping['failures'] = [{'case': 'joint_limit'}]
        elif case == 'mapping_cases': mapping['cases'] = 52
        else: Path(mapping['source_assets']['right']).write_bytes(b'changed fixture geometry')
        write(setup.mapping_path, mapping)
    write(setup.teacher_path, report)
    with pytest.raises(ValueError): derive(setup)
    assert not setup.output.exists() and not setup.output.with_suffix('.json').exists()


@pytest.mark.parametrize('destination', ['checkpoint', 'sidecar', 'race_checkpoint', 'race_sidecar'])
def test_exclusive_outputs_and_rollback_preserve_competing_files(setup, monkeypatch, destination):
    blocked = setup.output if destination.endswith('checkpoint') else setup.output.with_suffix('.json')
    original = Path.open
    raced = []
    if destination.startswith('race_'):
        def open_with_race(path, mode='r', *args, **kwargs):
            if path == blocked and mode in ('xb', 'x') and not raced:
                raced.append(True)
                with original(path, 'wb') as stream: stream.write(b'competing')
            return original(path, mode, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', open_with_race)
    else:
        blocked.write_bytes(b'competing')
    with pytest.raises(FileExistsError): derive(setup)
    assert blocked.read_bytes() == b'competing'
    other = setup.output.with_suffix('.json') if blocked == setup.output else setup.output
    assert not other.exists()


@pytest.mark.parametrize('changed_path', ['source', 'teacher', 'mapping', 'asset'])
def test_readonly_source_and_native_report_changes_abort_and_cleanup_owned_files(setup, monkeypatch, changed_path):
    original = api._unchanged_files
    checks = []
    def changed(identities):
        checks.append(True)
        if len(checks) == 2:
            path = dict(source=setup.source, teacher=setup.teacher_path, mapping=setup.mapping_path,
                        asset=Path(json.loads(setup.mapping_path.read_text())['source_assets']['left']))[changed_path]
            path.write_bytes(path.read_bytes() + b'external change')
        original(identities)
    monkeypatch.setattr(api, '_unchanged_files', changed)
    with pytest.raises(RuntimeError, match='evidence changed'): derive(setup)
    assert not setup.output.exists() and not setup.output.with_suffix('.json').exists()


@pytest.mark.skipif(os.name == 'nt', reason='Windows prevents replacing an open exclusive file')
def test_rollback_never_deletes_competitor_replacing_an_owned_path(setup, monkeypatch):
    original = api._unchanged_files
    checks = []
    def replaced(identities):
        checks.append(True)
        if len(checks) == 2:
            setup.output.rename(setup.output.with_suffix('.moved'))
            setup.output.write_bytes(b'competing replacement')
            raise RuntimeError('external replacement')
        original(identities)
    monkeypatch.setattr(api, '_unchanged_files', replaced)
    with pytest.raises(RuntimeError, match='external replacement'): derive(setup)
    assert setup.output.read_bytes() == b'competing replacement'
    assert not setup.output.with_suffix('.json').exists()


@pytest.mark.parametrize('case', ['weight_shape_changed', 'physics_changed', 'qualification_hash',
                                  'qualification_gate', 'runtime_adapter_changed', 'unversioned'])
def test_unwrap_validates_bound_state_provenance_and_runtime_adapter(setup, case):
    derive(setup)
    saved = torch.load(setup.output, map_location='cpu', weights_only=False)
    options = {'allow_explicit': True}
    if case == 'weight_shape_changed': saved['policy']['critic.4.bias'] = torch.zeros(2)
    elif case == 'physics_changed': saved['contract']['privileged_critic_contract']['object_contact_material']['profile'] = 'changed'
    elif case == 'qualification_hash': saved['contract'][api.QUALIFICATION_KEY]['sha256'] = '0' * 64
    elif case == 'qualification_gate':
        saved['contract'][api.QUALIFICATION_KEY]['teacher']['pinch_first_episode_successes'] = 3
        rehash(saved['contract'][api.QUALIFICATION_KEY])
    elif case == 'runtime_adapter_changed':
        expected = copy.deepcopy(setup.adapter)
        expected['adapter_source_sha256'] = '0' * 64
        rehash(expected)
        options['expected_adapter_contract'] = expected
    elif case == 'unversioned': saved['version'] = saved['contract']['version'] = api.SQUASHED_VERSION
    with pytest.raises(ValueError): api.unwrap_physical_adapter_checkpoint(saved, **options)


def test_native_adapter_qualification_does_not_freeze_future_valid_optimizer_updates(setup):
    derive(setup)
    saved = torch.load(setup.output, map_location='cpu', weights_only=False)
    qualification = copy.deepcopy(saved['contract'][api.QUALIFICATION_KEY])
    saved['policy']['critic.4.bias'][0] += .01
    saved['iteration'] += 1
    saved['timesteps'] += saved['contract']['num_envs'] * saved['contract']['rollout_steps']
    for state in saved['optimizer']['state'].values(): state['step'] += 1
    core = api.unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    assert core['iteration'] == setup.before['iteration'] + 1
    assert not torch.equal(core['policy']['critic.4.bias'], setup.before['policy']['critic.4.bias'])
    same(saved['contract'][api.QUALIFICATION_KEY], qualification)


def test_portable_bound_checkpoint_needs_no_external_reports_at_inference(setup):
    derive(setup)
    saved = torch.load(setup.output, map_location='cpu', weights_only=False)
    mapping = json.loads(setup.mapping_path.read_text())
    for path in [setup.teacher_path, setup.mapping_path,
                 setup.teacher_path.parent/'physical_action_adapter_contract.json',
                 *map(Path, mapping['source_assets'].values())]:
        path.unlink()
    core = api.unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    same(core, setup.before)
    expected = saved['contract'][api.ADAPTER_KEY]
    assert api.require_physical_adapter_runtime_match(expected, setup.adapter) == setup.adapter
    wrong = copy.deepcopy(setup.adapter)
    wrong['source_asset_sha256']['left'] = '0' * 64
    rehash(wrong)
    with pytest.raises(ValueError, match='actual runtime'):
        api.require_physical_adapter_runtime_match(expected, wrong)
