"""CPU cache/provenance fixtures never reuse or certify native physics."""
import copy
import gzip
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def api():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_comparison_cache')


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(tmp_path, *, physics_error=None):
    inputs = {}
    for name in ('checkpoint', 'teacher_dataset', 'sequence_context'):
        path = tmp_path / (name + '.bin'); path.write_bytes(name.encode())
        inputs[name] = identity(path)
    native = tmp_path / 'native'; native.mkdir()
    stream = native / 'same_state_teacher_observation_comparisons.jsonl.gz'
    with gzip.open(stream, 'xt', encoding='utf-8') as output:
        for i in range(8): output.write(json.dumps(dict(comparison='same_exact_saved_Teacher_3844D_observation', fixture_row=i)) + '\n')
    from go2_pvcnn.control.dh116.tabletop_vision_branching_eval import comparison_dependencies
    dependencies = comparison_dependencies()
    implementation = dependencies['tabletop_vision_branching_eval']
    summary = dict(**identity(stream), policy_rows=8,
        comparison_implementation_sha256=implementation, comparison_inputs=copy.deepcopy(inputs),
        comparison_dependencies=dependencies,
        reused_offline_comparisons=False, first_numeric_difference={'0': {'tick': 0}},
        worst_action_difference={}, worst_conditional_target_difference={}, physical_failure_causality_claimed=False)
    report = dict(native_runtime_verified=True, status='error' if physics_error else 'completed',
        error=physics_error, mode='source', native_success_count=0,
        inputs=copy.deepcopy(inputs), same_state_comparisons=summary)
    path = native / 'stage6_native_report.json'; path.write_text(json.dumps(report))
    return path, report, inputs, implementation, stream


def test_cache_copies_complete_offline_stream_and_reuses_only_bound_comparisons(tmp_path):
    source, report, config, implementation, stream = fixture(tmp_path)
    before = {path: path.read_bytes() for path in [source, stream, *(Path(x['path']) for x in config.values())]}
    cache = api().build_comparison_cache(source, tmp_path / 'cache')
    assert cache == identity(Path(cache['path']))
    loaded = api().load_comparison_cache(cache, dict(config, mode='branch', branch_steps=60), implementation_sha256=implementation)
    assert loaded['reused_offline_comparisons'] is True and loaded['native_trials_reused'] == 0
    assert loaded['comparison_inputs'] == config
    assert loaded['first_numeric_difference'] == report['same_state_comparisons']['first_numeric_difference']
    assert Path(loaded['path']).parent == tmp_path / 'cache'
    assert Path(loaded['path']).read_bytes() == stream.read_bytes()
    assert loaded['cache_provenance']['scope'] == 'offline_saved_teacher_observation_comparisons_only'
    assert loaded['cache_provenance']['native_physics_reused'] is False
    assert loaded['cache_provenance']['native_reset_reused'] is False
    assert loaded['cache_provenance']['native_terminal_reused'] is False
    assert all(path.read_bytes() == value for path, value in before.items())


def test_physical_failure_or_incomplete_episode_is_retained_and_never_upgraded(tmp_path):
    source, report, config, implementation, _ = fixture(tmp_path, physics_error='native physics failed after complete offline comparison')
    report['status'] = 'not_completed'; source.write_text(json.dumps(report))
    cache = api().build_comparison_cache(source, tmp_path / 'cache')
    loaded = api().load_comparison_cache(cache, config, implementation_sha256=implementation)
    assert loaded['cache_provenance']['source_status'] == 'not_completed'
    assert loaded['cache_provenance']['source_error'] == report['error']
    assert 'native_success_count' not in loaded and loaded['native_trials_reused'] == 0


@pytest.mark.parametrize('bad', ['not_native', 'missing_summary', 'missing_code', 'wrong_inputs', 'already_cached', 'wrong_row_count'])
def test_builder_rejects_missing_or_inconsistent_native_offline_provenance(tmp_path, bad):
    source, report, _, _, _ = fixture(tmp_path)
    if bad == 'not_native': report['native_runtime_verified'] = False
    elif bad == 'missing_summary': report['same_state_comparisons'] = None
    elif bad == 'missing_code': report['same_state_comparisons'].pop('comparison_implementation_sha256')
    elif bad == 'wrong_inputs': report['same_state_comparisons']['comparison_inputs']['checkpoint']['sha256'] = '0' * 64
    elif bad == 'already_cached': report['same_state_comparisons']['reused_offline_comparisons'] = True
    elif bad == 'wrong_row_count': report['same_state_comparisons']['policy_rows'] = 9
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError): api().build_comparison_cache(source, tmp_path / 'cache')


@pytest.mark.parametrize('changed', ['checkpoint', 'teacher_dataset', 'sequence_context', 'source_report', 'source_stream', 'copy_stream', 'manifest'])
def test_any_changed_source_or_cache_bytes_refuse_without_fallback(tmp_path, changed):
    source, _, config, implementation, stream = fixture(tmp_path)
    cache = api().build_comparison_cache(source, tmp_path / 'cache')
    metadata = json.loads(Path(cache['path']).read_text())
    if changed in config: target = Path(config[changed]['path'])
    elif changed == 'source_report': target = source
    elif changed == 'source_stream': target = stream
    elif changed == 'copy_stream': target = Path(metadata['cached_stream']['path'])
    else: target = Path(cache['path'])
    target.write_bytes(target.read_bytes() + b'changed')
    with pytest.raises(ValueError): api().load_comparison_cache(cache, config, implementation_sha256=implementation)


def test_same_bytes_at_different_input_path_and_changed_code_are_not_equivalent(tmp_path):
    source, _, config, implementation, _ = fixture(tmp_path)
    cache = api().build_comparison_cache(source, tmp_path / 'cache')
    moved = tmp_path / 'copy-checkpoint.bin'; moved.write_bytes(Path(config['checkpoint']['path']).read_bytes())
    wrong = dict(config, checkpoint=identity(moved))
    with pytest.raises(ValueError): api().load_comparison_cache(cache, wrong, implementation_sha256=implementation)
    with pytest.raises(ValueError): api().load_comparison_cache(cache, config, implementation_sha256='a' * 64)


def test_new_reset_and_terminal_configuration_does_not_reuse_any_physics(tmp_path):
    source, _, config, implementation, _ = fixture(tmp_path)
    cache = api().build_comparison_cache(source, tmp_path / 'cache')
    expected = dict(config, mode='branch', event='HOLD', branch_steps=1,
        source_initial_state={'path':'/new/actual-reset', 'sha256':'f' * 64})
    loaded = api().load_comparison_cache(cache, expected, implementation_sha256=implementation)
    assert loaded['native_trials_reused'] == 0


def test_truncated_gzip_or_nonfinite_json_cannot_be_declared_complete(tmp_path):
    source, report, _, _, stream = fixture(tmp_path)
    stream.write_bytes(stream.read_bytes()[:-8])
    report['same_state_comparisons'].update(identity(stream)); source.write_text(json.dumps(report))
    with pytest.raises(ValueError): api().build_comparison_cache(source, tmp_path / 'cache-truncated')
    with gzip.open(stream, 'wt') as output: output.write('{"not_finite":1e999}\n')
    report['same_state_comparisons'].update(identity(stream)); report['same_state_comparisons']['policy_rows'] = 1
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError): api().build_comparison_cache(source, tmp_path / 'cache-nonfinite')


def test_output_is_exclusive_and_modified_metadata_cannot_override_native_source_summary(tmp_path):
    source, _, config, implementation, _ = fixture(tmp_path)
    output = tmp_path / 'cache'; cache = api().build_comparison_cache(source, output)
    with pytest.raises(FileExistsError): api().build_comparison_cache(source, output)
    manifest = Path(cache['path']); metadata = json.loads(manifest.read_text())
    metadata['source_summary']['first_numeric_difference'] = {'invented': 'value'}
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        api().load_comparison_cache(identity(manifest), config, implementation_sha256=implementation)


def test_changed_numeric_dependency_refuses_even_when_branch_and_three_inputs_are_unchanged(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_vision_branching_eval as branch
    source, _, config, implementation, _ = fixture(tmp_path)
    cache = api().build_comparison_cache(source, tmp_path / 'cache')
    changed = dict(branch.comparison_dependencies())
    changed['tabletop_vision_inference'] = '0' * 64
    monkeypatch.setattr(branch, 'comparison_dependencies', lambda: changed)
    with pytest.raises(ValueError, match='dependenc|implementation'):
        api().load_comparison_cache(cache, config, implementation_sha256=implementation)
