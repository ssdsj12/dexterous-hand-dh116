"""CPU synthetic V4 qualification fixtures; no native or Student success."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_vision_physical_adapter_inference import qualified
from test_bimanual_checkpoint_contract import derive
from test_vision_evaluation_trials import manifest
from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_physics_sha256


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/run_dh116_stage5_native.py'


def api():
    spec = importlib.util.spec_from_file_location('stage5_launcher', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(qualified, *, split='validation'):
    value = manifest(100, split)
    source = qualified.before['contract']
    value['metadata']['camera_sha256'] = source['camera_config_sha256']
    value['metadata']['physics_sha256'] = evaluation_physics_sha256(source['privileged_critic_contract'])
    value['metadata']['seed'] = 116507
    path = qualified.output.parent / 'independent_manifest.json'
    path.write_text(json.dumps(value))
    return dict(checkpoint_path=qualified.output,
        expected_checkpoint_sha256=hashlib.sha256(qualified.output.read_bytes()).hexdigest(),
        manifest_path=path)


def template():
    return ['bash', '/source/scripts/python.sh', '/source/scripts/train_dh116_tabletop_gpu_ppo.py',
        '--output', '/old', '--num-envs', '8', '--seed', '123',
        '--vision-student', '--tensor-pipeline', '--headless',
        '--vision-camera-mode', 'fixed', '--vision-width', '224', '--vision-height', '96',
        '--vision-evaluation-checkpoint', '/old.pt', '--vision-evaluation-manifest', '/old.json']


@pytest.mark.parametrize('split', ['test', 'validation'])
def test_bound_eval_flags_contract_and_source_command_unchanged(qualified, split):
    options = inputs(qualified, split=split)
    command = template()
    original = command.copy()
    source_bytes = qualified.output.read_bytes()
    rng = torch.get_rng_state().clone()
    result = api().build_bound_evaluation_command(command, **options)
    assert command == original
    assert qualified.output.read_bytes() == source_bytes
    assert torch.equal(rng, torch.get_rng_state())
    for flag, expected in (
        ('--vision-evaluation-checkpoint', str(qualified.output.resolve())),
        ('--vision-evaluation-manifest', str(options['manifest_path'].resolve())),
        ('--seed', '116507'), ('--num-envs', '200'),
        ('--vision-action-distribution', 'tanh_gaussian'),
        ('--vision-wrist-quaternion-preprocessing', 'raw')):
        assert result.count(flag) == 1
        assert result[result.index(flag)+1] == expected
    assert result.count('--bimanual-checkpoint-adapter') == 1
    assert '--paired-bias-arm' not in result
    assert not (qualified.output.parent / 'new_eval').exists()


@pytest.mark.parametrize('bad', ['hash', 'v3', 'camera', 'physics', 'count', 'count_extra'])
def test_invalid_checkpoint_or_manifest_rejected_without_outputs(qualified, bad):
    options = inputs(qualified)
    if bad == 'hash':
        options['expected_checkpoint_sha256'] = '0' * 64
    elif bad == 'v3':
        options['checkpoint_path'] = qualified.source
        options['expected_checkpoint_sha256'] = hashlib.sha256(qualified.source.read_bytes()).hexdigest()
    else:
        value = json.loads(options['manifest_path'].read_text())
        if bad in ('camera', 'physics'):
            value['metadata'][bad + '_sha256'] = '0' * 64
        elif bad == 'count':
            value['trials'].pop()
        else:
            extra = manifest(101, 'validation')
            extra['metadata'] = value['metadata']
            value = extra
        options['manifest_path'].write_text(json.dumps(value))
    with pytest.raises(ValueError):
        api().build_bound_evaluation_command(template(), **options)
    assert not (qualified.output.parent / 'new_eval').exists()


def test_hemisphere_preprocessing_comes_from_qualified_model_contract(qualified):
    saved = torch.load(qualified.source, map_location='cpu', weights_only=False)
    saved['contract']['actor_wrist_quaternion_preprocessing'] = 'wrist_quaternion_hemisphere_v1'
    torch.save(saved, qualified.source)
    qualified.output = qualified.output.parent / 'hemisphere_v4.pt'
    derive(qualified)
    result = api().build_bound_evaluation_command(template() +
        ['--vision-wrist-quaternion-preprocessing=raw'], **inputs(qualified))
    assert result.count('--vision-wrist-quaternion-preprocessing') == 1
    assert result[result.index('--vision-wrist-quaternion-preprocessing')+1] == 'wrist_quaternion_hemisphere_v1'
    assert '--vision-wrist-quaternion-preprocessing=raw' not in result


@pytest.mark.parametrize('extras', [
    ['--paired-bias-arm', 'm0'], ['--paired-bias-arm', 'm2'],
    ['--vision-mirror-preprocessing', 'm2_paired_canonical_v1'],
    ['--surface-teacher-replay', '/teacher'], ['--vision-m2-replay', '/replay'],
    ['--bimanual-explicit-sdk-adapter'], ['--bimanual-mapping-probe'],
])
def test_bound_eval_cannot_mix_paired_teacher_or_replay(qualified, extras):
    with pytest.raises(ValueError, match='independent plain M0'):
        api().build_bound_evaluation_command(template() + extras, **inputs(qualified))


@pytest.mark.parametrize('mode', ['replay', 'probe', 'z0', 'z3', 'teacher', 'teacher_adapter'])
def test_new_arguments_are_exclusive_to_bound_mode(tmp_path, mode):
    result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(tmp_path),
        '--name', 'never_created', '--mode', mode, '--vision-checkpoint', 'source.pt',
        '--checkpoint-sha256', '0'*64, '--vision-manifest', 'source.json'],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'bound_student_eval' in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('missing', ['--vision-checkpoint', '--checkpoint-sha256', '--vision-manifest'])
def test_bound_mode_requires_all_three_arguments_before_freezing(tmp_path, missing):
    arguments = ['--vision-checkpoint', 'source.pt', '--checkpoint-sha256', '0'*64,
                 '--vision-manifest', 'source.json']
    index = arguments.index(missing)
    del arguments[index:index+2]
    result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(tmp_path),
        '--name', 'never_created', '--mode', 'bound_student_eval'] + arguments,
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'requires' in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('placement', ['cwd_prefix', 'substring_collision', 'external',
                                     'modified_checkpoint_after_lock', 'modified_manifest_after_lock'])
def test_main_child_argv_uses_the_exact_sha_verified_inputs(qualified, monkeypatch, placement):
    """Run the real launcher factory with subprocess/lock mocked, never Isaac."""
    module = api()
    original = inputs(qualified)
    root = qualified.output.parent / 'launcher_root'
    root.mkdir()
    old_source = root / 'old_template'
    old_source.mkdir()
    source_tree = root / 'source'
    source_tree.mkdir()
    if placement == 'cwd_prefix':
        selected = old_source
        redirected = root / 'bound_eval_source'
    elif placement == 'substring_collision':
        selected = root / 'old_template_extra'
        selected.mkdir()
        redirected = root / 'bound_eval_source_extra'
        redirected.mkdir()
    else:
        selected = root / 'external'
        selected.mkdir()
        redirected = None
    checkpoint, frozen = selected/'model.pt', selected/'manifest.json'
    checkpoint.write_bytes(qualified.output.read_bytes())
    frozen.write_bytes(original['manifest_path'].read_bytes())
    # Distinct B content models the source-tree copy the old all-token replace
    # incorrectly selected after only checking the original A inputs.
    if placement == 'cwd_prefix':
        (source_tree/'model.pt').write_bytes(b'WRONG copied model B')
        (source_tree/'manifest.json').write_bytes(b'WRONG copied manifest B')
    elif redirected is not None:
        (redirected/'model.pt').write_bytes(b'WRONG substring model B')
        (redirected/'manifest.json').write_bytes(b'WRONG substring manifest B')
    (root/'evaluation_template.json').write_text(json.dumps(dict(
        cwd=str(old_source), command=template(), env={})))
    captured = []
    def fake_run(command, **_kwargs):
        if command[0] == 'pgrep':
            if placement == 'modified_checkpoint_after_lock':
                checkpoint.write_bytes(b'changed while waiting')
            elif placement == 'modified_manifest_after_lock':
                frozen.write_bytes(b'changed while waiting')
            return SimpleNamespace(stdout='', returncode=1)
        captured.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    monkeypatch.setattr(module.fcntl, 'flock', lambda *_: None)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--root', str(root), '--name', 'bound_eval',
        '--mode', 'bound_student_eval', '--vision-checkpoint', str(checkpoint),
        '--checkpoint-sha256', hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        '--vision-manifest', str(frozen)])
    if placement.startswith('modified_'):
        with pytest.raises(ValueError, match='changed while awaiting native lock'):
            module.main()
        assert captured == []
        return
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0 and len(captured) == 1
    child = captured[0]
    actual_checkpoint = Path(child[child.index('--vision-evaluation-checkpoint')+1])
    actual_manifest = Path(child[child.index('--vision-evaluation-manifest')+1])
    assert actual_checkpoint == checkpoint.resolve()
    assert actual_manifest == frozen.resolve()
    assert actual_checkpoint.read_bytes() == qualified.output.read_bytes()
    assert actual_manifest.read_bytes() == original['manifest_path'].read_bytes()


@pytest.mark.parametrize('changed_flag', ['--vision-evaluation-checkpoint', '--vision-evaluation-manifest'])
def test_final_child_argv_must_match_pinned_path_even_if_other_file_bytes_match(tmp_path, changed_flag):
    module = api()
    arguments, hashes, command = {}, {}, []
    for flag, name in (('--vision-evaluation-checkpoint', 'model.pt'),
                       ('--vision-evaluation-manifest', 'manifest.json')):
        path = tmp_path / name
        path.write_bytes(name.encode())
        expected = str(path.resolve())
        arguments[flag] = expected
        hashes[expected] = hashlib.sha256(path.read_bytes()).hexdigest()
        command.extend((flag, expected))
    module._verify_bound_evaluation_inputs(command, arguments, hashes)
    replacement = tmp_path / 'same_bytes_different_identity'
    replacement.write_bytes(Path(arguments[changed_flag]).read_bytes())
    command[command.index(changed_flag)+1] = str(replacement)
    with pytest.raises(ValueError, match='child argv differs'):
        module._verify_bound_evaluation_inputs(command, arguments, hashes)
