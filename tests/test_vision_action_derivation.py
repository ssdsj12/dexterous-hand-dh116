"""Offline distribution derivation preserves trusted visual training state."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tensordict import TensorDict

from test_vision_training import fixture
from go2_pvcnn.control.dh116.tabletop_vision_training import _contract, _save
from go2_pvcnn.control.dh116.tabletop_vision_student import make_vision_algorithm


def derivation_module():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_action_derivation'
    assert importlib.util.find_spec(name), 'explicit offline visual action derivation is missing'
    return importlib.import_module(name)


def observations():
    policy = torch.randn(2, 3844)*.2
    policy.reshape(2, 4, 961)[..., 768:896] = 1.
    return TensorDict(dict(policy=policy, critic=torch.randn(2, 371)), [2])


def source_checkpoint(tmp_path):
    torch.set_num_threads(2)
    dw, task, env, args = fixture(tmp_path)
    algorithm = make_vision_algorithm(observations(), 2)
    # A real Adam step makes nonzero moments for every parameter, without physics.
    sum(parameter.square().sum() for parameter in algorithm.policy.parameters()).backward()
    algorithm.optimizer.step()
    algorithm.learning_rate = .000417
    algorithm.optimizer.param_groups[0]['lr'] = algorithm.learning_rate
    env.episodes[:] = torch.tensor([55, 76])
    env.successes[:] = torch.tensor([3, 9])
    source = tmp_path/'source_1439.pt'
    _save(source, algorithm, _contract(args, {'observation_size': 371}), 1439, 11341824, env)
    saved = torch.load(source, weights_only=False)
    saved['cuda_rng'] = [torch.arange(64, dtype=torch.uint8)]
    saved['resume_migration'] = {'version': 'dh116-visual-downsize-resume-v1', 'source_iteration': 5}
    saved['historical_extra'] = {'unmodified': [7, 'source']}
    torch.save(saved, source)
    return source, saved


def assert_same(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.device == right.device
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_same(a, b)
    else:
        assert left == right


def test_explicit_derivation_preserves_full_training_state_and_provenance(tmp_path, monkeypatch):
    module = derivation_module()
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import upgrade_action_contract
    source, before = source_checkpoint(tmp_path)
    source_bytes = source.read_bytes()
    output = tmp_path/'new'/'derived.pt'
    original_load = module.torch.load
    calls = []
    def cpu_load(*args, **kwargs):
        calls.append(kwargs.get('map_location'))
        return original_load(*args, **kwargs)
    monkeypatch.setattr(module.torch, 'load', cpu_load)
    report = module.derive_action_checkpoint(source, output)
    assert calls == ['cpu']
    assert source.read_bytes() == source_bytes
    after = original_load(output, weights_only=False)
    assert after['version'] == 'dh116-vision-student-ppo-v3-squashed-gaussian'
    assert after['contract'] == upgrade_action_contract(before['contract'])
    for key, value in before.items():
        if key not in ('version', 'contract'):
            assert_same(value, after[key])
    meta = after['action_derivation']
    assert meta['source_sha256'] == hashlib.sha256(source_bytes).hexdigest()
    assert meta['source_version'] == before['version'] and meta['source_iteration'] == 1439
    assert meta['status'] == 'warm_start_not_exact_resume'
    assert meta['source_distribution'] == 'gaussian_clip'
    assert meta['target_distribution'] == 'tanh_gaussian'
    assert meta['rng_coverage']['preserved'] == ['torch_cpu', 'torch_cuda_saved_states']
    assert meta['rng_coverage']['unavailable_in_source'] == ['python', 'numpy']
    assert report['action_derivation'] == meta
    assert json.loads(output.with_suffix('.json').read_text()) == report


def test_derived_policy_reproduces_independent_legacy_deterministic_actions(tmp_path):
    module = derivation_module()
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import make_squashed_vision_algorithm
    source, before = source_checkpoint(tmp_path)
    output = tmp_path/'derived.pt'
    module.derive_action_checkpoint(source, output)
    derived = torch.load(output, weights_only=False)
    packet = observations()
    old = make_vision_algorithm(packet, 1).policy
    new = make_squashed_vision_algorithm(packet, 1).policy
    old.load_state_dict(before['policy'])
    new.load_state_dict(derived['policy'])
    with torch.no_grad():
        expected = old.act_inference(packet)
        actual = new.act_inference(packet)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert old.actor is not new.actor


@pytest.mark.parametrize('collision', ['checkpoint', 'sidecar'])
def test_existing_destination_is_never_overwritten(tmp_path, collision):
    module = derivation_module()
    source, _ = source_checkpoint(tmp_path)
    source_bytes = source.read_bytes()
    output = tmp_path/'derived.pt'
    blocked = output if collision == 'checkpoint' else output.with_suffix('.json')
    blocked.write_bytes(b'existing artifact must survive')
    with pytest.raises(FileExistsError): module.derive_action_checkpoint(source, output)
    assert blocked.read_bytes() == b'existing artifact must survive'
    assert source.read_bytes() == source_bytes
    other = output.with_suffix('.json') if collision == 'checkpoint' else output
    assert not other.exists()


@pytest.mark.parametrize('collision', ['checkpoint', 'sidecar'])
def test_competing_destination_writer_survives_without_partial_derivation(tmp_path, monkeypatch, collision):
    module = derivation_module()
    source, _ = source_checkpoint(tmp_path)
    output = tmp_path/'derived.pt'
    blocked = output if collision == 'checkpoint' else output.with_suffix('.json')
    original_open = Path.open
    raced = []
    def racing_open(path, mode='r', *args, **kwargs):
        if path == blocked and mode in ('xb', 'x') and not raced:
            raced.append(True)
            with original_open(path, 'wb') as stream:
                stream.write(b'competing writer owns this')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', racing_open)
    with pytest.raises(FileExistsError): module.derive_action_checkpoint(source, output)
    assert raced and blocked.read_bytes() == b'competing writer owns this'
    other = output.with_suffix('.json') if collision == 'checkpoint' else output
    assert not other.exists()


@pytest.mark.parametrize('bad_state', ['policy', 'optimizer', 'learning_rate', 'wrong_version', 'shape', 'maximize'])
def test_derivation_rejects_nonfinite_or_incompatible_state_before_outputs(tmp_path, bad_state):
    module = derivation_module()
    source, saved = source_checkpoint(tmp_path)
    if bad_state == 'policy': saved['policy']['log_std'][0] = float('nan')
    elif bad_state == 'optimizer': saved['optimizer']['state'][0]['exp_avg'][0] = float('inf')
    elif bad_state == 'learning_rate': saved['learning_rate'] = float('inf')
    elif bad_state == 'wrong_version': saved['version'] = 'unknown-same-shaped-policy'
    elif bad_state == 'shape': saved['policy']['actor.action_net.0.weight'] = torch.zeros(128, 792)
    elif bad_state == 'maximize': saved['optimizer']['param_groups'][0]['maximize'] = True
    torch.save(saved, source)
    source_bytes = source.read_bytes()
    output = tmp_path/'derived.pt'
    with pytest.raises(ValueError): module.derive_action_checkpoint(source, output)
    assert not output.exists() and not output.with_suffix('.json').exists()
    assert source.read_bytes() == source_bytes


def test_derivation_never_reuses_source_path(tmp_path):
    module = derivation_module()
    source, _ = source_checkpoint(tmp_path)
    original = source.read_bytes()
    with pytest.raises((ValueError, FileExistsError)): module.derive_action_checkpoint(source, source)
    assert source.read_bytes() == original


@pytest.mark.parametrize('case', [
    'empty_adam', 'missing_adam_parameter', 'negative_second_moment',
    'negative_max_moment', 'step_negative', 'step_vector',
    'noise_below_min', 'noise_above_max', 'rng_one_byte',
])
def test_derivation_rejects_incomplete_unloadable_state_before_output(tmp_path, case):
    from test_vision_checkpoint_state import corrupt_state
    module = derivation_module()
    source, saved = source_checkpoint(tmp_path)
    corrupt_state(saved, case)
    torch.save(saved, source)
    before = source.read_bytes()
    output = tmp_path/'invalid'/'derived.pt'
    with pytest.raises(ValueError): module.derive_action_checkpoint(source, output)
    assert source.read_bytes() == before
    assert not output.exists() and not output.with_suffix('.json').exists()
    assert not output.parent.exists()


def test_derivation_accepts_reordered_complete_weight_dictionary(tmp_path):
    module = derivation_module()
    source, saved = source_checkpoint(tmp_path)
    saved['policy'] = dict(reversed(list(saved['policy'].items())))
    torch.save(saved, source)
    output = tmp_path/'derived.pt'
    module.derive_action_checkpoint(source, output)
    actual = torch.load(output, weights_only=False)
    assert_same(actual['policy'], saved['policy'])


def test_derivation_cli_help_does_not_import_torch_or_require_site_packages():
    root = Path(__file__).resolve().parents[1]
    script = root/'scripts'/'derive_dh116_vision_action_checkpoint.py'
    assert script.is_file(), 'offline derivation CLI is missing'
    result = subprocess.run([sys.executable, '-S', str(script), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--input' in result.stdout and '--output' in result.stdout
