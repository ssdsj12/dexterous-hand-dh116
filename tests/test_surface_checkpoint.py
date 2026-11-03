"""Full-state, CPU-only tests of the immutable 351 -> 371 derivation."""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODULE = 'go2_pvcnn.control.dh116.tabletop_surface_checkpoint'
SCRIPT = ROOT / 'scripts/derive_dh116_surface_checkpoint.py'
MOMENTS = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')
EXPANDED = ('actor.0.weight', 'critic.0.weight')


def migration():
    assert importlib.util.find_spec(MODULE), 'surface checkpoint migration module is missing'
    return importlib.import_module(MODULE)


class Policy(nn.Module):
    def __init__(self, width=351):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((24,), -1.2))
        self.actor = nn.Sequential(nn.Linear(width, 128), nn.Tanh(), nn.Linear(128, 128),
                                   nn.Tanh(), nn.Linear(128, 24), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(width, 128), nn.Tanh(), nn.Linear(128, 128),
                                    nn.Tanh(), nn.Linear(128, 1))


def same(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            same(left[key], right[key])
        if hasattr(left, '__dict__'):
            same(vars(left), vars(right))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def step(optimizer, parameters):
    for i, parameter in enumerate(parameters):
        parameter.grad = ((torch.arange(parameter.numel()).reshape(parameter.shape) % 31)
                          .to(parameter.dtype) + i + 1) / 1000
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def fixture(directory, *, amsgrad=True, auxiliary='populated', named=False):
    from go2_pvcnn.control.dh116.tabletop_policy_noise import CONTRACT
    from go2_pvcnn.control.dh116.tabletop_surface_exploration import STABLE_HOLD_REWARD_CONTRACT
    directory.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1301)
        policy = Policy()
        ppo = torch.optim.Adam(policy.parameters(), lr=2e-4, amsgrad=amsgrad)
        step(ppo, list(policy.parameters()))
        aux = torch.optim.Adam(policy.actor.parameters(), lr=.001, amsgrad=amsgrad)
        if auxiliary == 'populated':
            step(aux, list(policy.actor.parameters()))
        contract = dict(observation_size=351,
            observation_version='dh116-surface-sharing-lift-objective-v2',
            action='24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1',
            action_size=24, policy_command_size=24, num_envs=2, physics_hz=480, control_hz=60,
            policy_noise=copy.deepcopy(CONTRACT), surface_training_objective=copy.deepcopy(STABLE_HOLD_REWARD_CONTRACT),
            success_dwell_s=5., success_slip_m=.0005, success_bottom_m=[.12, .12],
            surface_force_sharing=dict(version='opposed_surface_sharing_v1', action_size=24,
                                       lift_reference_bottom_m=[.09, .09]))
        saved = dict(version='dh116-rsl-tensor-ppo-v1', contract=contract,
            policy=policy.state_dict(), optimizer=ppo.state_dict(), iteration=17, timesteps=123456,
            learning_rate=2e-4, episodes=torch.tensor([100, 200]), successes=torch.tensor([1, 0]),
            recovered=torch.tensor([3, 4]), reached_counts=torch.arange(4), transition_counts=torch.arange(16),
            torch_rng=torch.get_rng_state(), cuda_rng=[torch.arange(12, dtype=torch.uint8)],
            numpy_rng=np.random.RandomState(17).get_state(), python_rng=random.Random(17).getstate(),
            combined_failure_resets=torch.tensor([7, 8]), numerical_resets=torch.tensor(13),
            close_completion_event_index=23, require_native_student_validation=True,
            surface_student_diagnostic_updates=9,
            surface_imitation=dict(dataset_sha256='a' * 64, initial_weight=.1, decay=1.,
                batch_size=256, update='independent_actor_adam_stable_hold_v2',
                source='qualified_native_pre_action_observation_action_pairs',
                holding_weight=20., wrist_weight=4., auxiliary_learning_rate=.001),
            surface_repair_migration=dict(version='previous_repair', invariant_checks={'old': True}),
            extra_ppo_state=dict(schedule='adaptive', returns=torch.tensor([1., 2.]),
                nested=[np.array([1., 2.]), {'extra': (torch.tensor(3.), 'retained')}]))
        if auxiliary != 'absent':
            saved['surface_imitation_optimizer'] = aux.state_dict()
        if named:
            saved['optimizer']['param_groups'][0]['param_names'] = list(dict(policy.named_parameters()))
            if auxiliary != 'absent':
                saved['surface_imitation_optimizer']['param_groups'][0]['param_names'] = list(dict(policy.actor.named_parameters()))
        for key in ('optimizer', 'surface_imitation_optimizer'):
            if key in saved:
                saved[key]['extra_metadata'] = dict(source='retained_adam', counters=torch.tensor([17, 8]))
                saved[key]['param_groups'][0]['audit_tag'] = 'retained_group'
                for entry in saved[key]['state'].values():
                    entry['extra_state'] = dict(counter=23, history=torch.tensor([4., 7.]))
    source, output = directory / 'source.pt', directory / 'derived.pt'
    torch.save(saved, source)
    return source, output, saved


def test_reward_stage_derivation_preserves_full_state_and_source_files(tmp_path, monkeypatch):
    from scripts import derive_dh116_surface_stage as stage
    from go2_pvcnn.control.dh116.tabletop_surface_contract import validate_contract

    old, source, _ = fixture(tmp_path)
    old_bytes = old.read_bytes()
    migration().derive_surface_checkpoint(old, source)
    demonstrations = tmp_path / 'demo.pt'
    torch.save({'synthetic': True}, demonstrations)
    saved = torch.load(source, weights_only=False)
    saved['surface_imitation']['dataset_sha256'] = hashlib.sha256(demonstrations.read_bytes()).hexdigest()
    torch.save(saved, source)
    source_bytes = source.read_bytes()
    out = tmp_path / 'lift.pt'
    monkeypatch.setattr(stage, 'validate_dataset', lambda data, contract: validate_contract(contract))
    monkeypatch.setattr(sys, 'argv', ['derive', '--input', str(source),
        '--demonstrations', str(demonstrations), '--reward-profile', 'v3_lift_only', '--output', str(out)])
    stage.main()
    after = torch.load(out, weights_only=False)
    for key in saved.keys() - {'contract'}:
        same(saved[key], after[key])
    assert old.read_bytes() == old_bytes and source.read_bytes() == source_bytes
    assert after['contract']['surface_reward_ablation']['disabled_legacy_terms'] == ['dex_lift', 'lift_milestone']
    assert after['contract']['surface_contract_sha256'] != saved['contract']['surface_contract_sha256']
    assert after['contract']['surface_contract'] == saved['contract']['surface_contract']
    with pytest.raises(FileExistsError):
        stage.main()


def expected_state(saved):
    from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract
    expected = copy.deepcopy(saved)
    expected['contract'] = bind_contract(saved['contract'])
    names = list(saved['policy'])
    for name in EXPANDED:
        old = saved['policy'][name]
        expected['policy'][name] = torch.cat((old, old.new_zeros((128, 20))), dim=1)
        for key, identifier in (('optimizer', names.index(name)), ('surface_imitation_optimizer', 0)):
            if key not in saved or (key == 'surface_imitation_optimizer' and name.startswith('critic.')):
                continue
            for field in MOMENTS:
                entry = expected[key]['state'].get(identifier, {})
                if field in entry:
                    entry[field] = torch.cat((entry[field], entry[field].new_zeros((128, 20))), dim=1)
    return expected


@pytest.mark.parametrize('amsgrad', [False, True])
@pytest.mark.parametrize('auxiliary', ['absent', 'empty', 'populated'])
@pytest.mark.parametrize('named', [False, True])
def test_full_state_and_exact_prefix_moments_preserved(tmp_path, amsgrad, auxiliary, named):
    trainer_module = 'go2_pvcnn.control.dh116.tabletop_tensor_training'
    trainer_before = sys.modules.get(trainer_module)
    source, output, saved = fixture(tmp_path, amsgrad=amsgrad, auxiliary=auxiliary, named=named)
    original = source.read_bytes()
    rng = torch.get_rng_state().clone()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    manifest = migration().derive_surface_checkpoint(source, output)
    after = torch.load(output, map_location='cpu', weights_only=False)
    same(expected_state(saved), after)  # Every field, including unknown extra state.
    assert source.read_bytes() == original
    same(rng, torch.get_rng_state())
    same(numpy_rng, np.random.get_state())
    same(python_rng, random.getstate())
    from go2_pvcnn.control.dh116.tabletop_surface_contract import semantic_sha256, validate_contract
    validate_contract(after['contract'])
    source_sha = hashlib.sha256(original).hexdigest()
    assert manifest['source_sha256_before'] == manifest['source_sha256_after'] == source_sha
    assert manifest['derived_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest['old_contract_sha256'] == semantic_sha256(saved['contract'])
    assert manifest['new_contract_sha256'] == semantic_sha256(after['contract'])
    assert manifest == json.loads(output.with_suffix('.json').read_text())
    assert all(manifest['invariant_checks'].values())
    assert sys.modules.get(trainer_module) is trainer_before


def test_actor_critic_equivalence_and_real_adam_resume(tmp_path):
    source, output, saved = fixture(tmp_path)
    migration().derive_surface_checkpoint(source, output)
    after = torch.load(output, weights_only=False)
    with torch.random.fork_rng(devices=[]):
        old, new = Policy(), Policy(371)
        old.load_state_dict(saved['policy']); new.load_state_dict(after['policy'])
        inputs = torch.randn(11, 351)
        for suffix in (torch.zeros(11, 20), torch.randn(11, 20)):
            extended = torch.cat((inputs, suffix), dim=1)
            torch.testing.assert_close(old.actor(inputs), new.actor(extended), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(old.critic(inputs), new.critic(extended), rtol=1e-6, atol=1e-7)
        # Both real Adam instances accept expanded state and update all parameters.
        ppo = torch.optim.Adam(new.parameters(), amsgrad=True)
        ppo.load_state_dict(after['optimizer'])
        aux = torch.optim.Adam(new.actor.parameters(), amsgrad=True)
        aux.load_state_dict(after['surface_imitation_optimizer'])
        step(ppo, list(new.parameters())); step(aux, list(new.actor.parameters()))
        assert all(torch.isfinite(value).all() for value in new.parameters())
        assert ppo.state[new.actor[0].weight]['step'].item() == 2
        assert aux.state[new.actor[0].weight]['step'].item() == 2


@pytest.mark.parametrize('fault', ['policy_order', 'ppo_order', 'aux_order', 'ppo_names', 'aux_names',
    'ppo_same_shape_order', 'aux_same_shape_order', 'ppo_noncanonical_named_ids', 'aux_noncanonical_named_ids',
    'ppo_duplicate', 'aux_duplicate', 'ppo_id', 'aux_id', 'ppo_extra_group', 'aux_extra_group',
    'ppo_unknown_state', 'aux_unknown_state', 'extra_parameter', 'missing_parameter',
    'actor_shape', 'critic_shape', 'policy_dtype', 'ppo_shape', 'aux_shape', 'ppo_dtype',
    'ppo_missing_moment', 'aux_missing_moment', 'ppo_amsgrad', 'aux_amsgrad', 'ppo_step', 'aux_step'])
def test_rejects_ambiguous_topology_and_optimizer_mapping_before_output(tmp_path, fault):
    source, output, saved = fixture(tmp_path)
    key = 'surface_imitation_optimizer' if fault.startswith('aux_') else 'optimizer'
    optimizer = saved[key]; group = optimizer['param_groups'][0]
    if fault == 'policy_order': saved['policy'].move_to_end('actor.0.bias')
    if fault in ('ppo_order', 'aux_order'): group['params'][2:4] = group['params'][2:4][::-1]
    if fault.endswith('_same_shape_order'):
        a, b = (1, 3) if key == 'surface_imitation_optimizer' else (2, 4)
        assert optimizer['state'][a]['exp_avg'].shape == optimizer['state'][b]['exp_avg'].shape
        group['params'][a], group['params'][b] = b, a
    if fault.endswith('_noncanonical_named_ids'):
        names = list(saved['policy']) if key == 'optimizer' else [name.removeprefix('actor.') for name in saved['policy'] if name.startswith('actor.')]
        group['param_names'] = names
        group['params'] = [identifier + 100 for identifier in group['params']]
        optimizer['state'] = {identifier + 100: value for identifier, value in optimizer['state'].items()}
    if fault.endswith('_names'): group['param_names'] = ['wrong'] * len(group['params'])
    if fault.endswith('_duplicate'): group['params'][1] = group['params'][0]
    if fault.endswith('_id'): group['params'][0] = 42
    if fault.endswith('_extra_group'): optimizer['param_groups'].append(copy.deepcopy(group))
    if fault.endswith('_unknown_state'): optimizer['state'][99] = copy.deepcopy(optimizer['state'][0])
    if fault == 'extra_parameter': saved['policy']['other'] = torch.zeros(24)
    if fault == 'missing_parameter': del saved['policy']['actor.2.bias']
    if fault == 'actor_shape': saved['policy']['actor.0.weight'] = torch.zeros(128, 401)
    if fault == 'critic_shape': saved['policy']['critic.4.weight'] = torch.zeros(2, 128)
    if fault == 'policy_dtype': saved['policy']['actor.0.weight'] = saved['policy']['actor.0.weight'].double()
    if fault.endswith('_shape') and fault.startswith(('ppo_', 'aux_')): optimizer['state'][0]['exp_avg'] = torch.zeros(1)
    if fault == 'ppo_dtype': optimizer['state'][0]['exp_avg'] = optimizer['state'][0]['exp_avg'].double()
    if fault.endswith('_missing_moment'): del optimizer['state'][0]['exp_avg_sq']
    if fault.endswith('_amsgrad'): del optimizer['state'][0]['max_exp_avg_sq']
    if fault.endswith('_step'): optimizer['state'][0]['step'] = torch.ones(2)
    torch.save(saved, source)
    with pytest.raises(ValueError): migration().derive_surface_checkpoint(source, output)
    assert not output.exists() and not output.with_suffix('.json').exists()


@pytest.mark.parametrize('state_kind', ['empty', 'partial'])
def test_preserves_lazy_ppo_state_and_does_not_invent_missing_moments(tmp_path, state_kind):
    source, output, saved = fixture(tmp_path)
    if state_kind == 'empty':
        saved['optimizer']['state'].clear()
    else:
        del saved['optimizer']['state'][1]  # Actor input moments not initialized yet.
        saved['surface_imitation_optimizer']['state'][0] = {}
    torch.save(saved, source)
    migration().derive_surface_checkpoint(source, output)
    same(expected_state(saved), torch.load(output, weights_only=False))


@pytest.mark.parametrize('fault', ['version', 'counter', 'observation401', 'already371', 'observation_version',
    'action26', 'conflicting_action_size', 'action_name', 'surface_schema', 'surface_sha',
    'explicit_action_schema', 'nested_schema', 'sharing_size', 'malformed_nested',
    'nested_order', 'nested_scale', 'infinite_policy', 'nan_ppo', 'nan_aux',
    'infinite_counter', 'nan_cuda_rng', 'infinite_extra', 'nan_numpy', 'nan_numpy_scalar', 'nan_metadata'])
def test_rejects_incomplete_nonfinite_or_spoofed_source_before_output(tmp_path, fault):
    source, output, saved = fixture(tmp_path)
    contract = saved['contract']
    if fault == 'version': saved['version'] = 'unknown'
    if fault == 'counter': del saved['timesteps']
    if fault == 'observation401': contract['observation_size'] = 401
    if fault == 'already371': contract['observation_size'] = 371
    if fault == 'observation_version': contract['observation_version'] = 'unknown'
    if fault == 'action26': contract['policy_command_size'] = 26
    if fault == 'conflicting_action_size': contract['action_size'] = 26
    if fault == 'action_name': contract['action'] = 'unknown'
    if fault == 'surface_schema': contract['surface_contract'] = {'action': {'size': 26}}
    if fault == 'surface_sha': contract['surface_contract_sha256'] = '0' * 64
    if fault == 'explicit_action_schema': contract['action_contract'] = {'size': 24, 'axis_order': ['Z', 'Y', 'X']}
    if fault == 'nested_schema': contract['policy_control'] = dict(action_contract={'size': 24, 'axis_order': ['Z', 'Y', 'X']})
    if fault == 'sharing_size': contract['surface_force_sharing']['action_size'] = 26
    if fault == 'malformed_nested': contract['free_wrist_grasp'] = None
    if fault == 'nested_order': contract['free_wrist_grasp'] = dict(action_order='reversed')
    if fault == 'nested_scale': contract['policy_control'] = dict(wrist_rate_m_s=.02)
    if fault == 'infinite_policy': saved['policy']['actor.0.weight'][0, 0] = float('inf')
    if fault == 'nan_ppo': saved['optimizer']['state'][1]['exp_avg'][0, 0] = float('nan')
    if fault == 'nan_aux': saved['surface_imitation_optimizer']['state'][0]['exp_avg'][0, 0] = float('nan')
    if fault == 'infinite_counter': saved['episodes'] = torch.tensor([float('inf')])
    if fault == 'nan_cuda_rng': saved['cuda_rng'].append(torch.tensor([float('nan')]))
    if fault == 'infinite_extra': saved['extra_ppo_state']['bad'] = float('inf')
    if fault == 'nan_numpy': saved['extra_ppo_state']['bad'] = np.array([np.nan])
    if fault == 'nan_numpy_scalar': saved['extra_ppo_state']['bad'] = np.float32('nan')
    if fault == 'nan_metadata': saved['surface_imitation']['initial_weight'] = float('nan')
    torch.save(saved, source)
    with pytest.raises(ValueError): migration().derive_surface_checkpoint(source, output)
    assert not output.exists() and not output.with_suffix('.json').exists()


@pytest.mark.parametrize('destination', ['checkpoint', 'sidecar', 'source', 'json_source', 'same_outputs', 'symlink'])
def test_never_overwrites_or_aliases_input_or_existing_outputs(tmp_path, destination):
    source, output, _ = fixture(tmp_path)
    if destination == 'checkpoint': output.write_bytes(b'checkpoint sentinel')
    if destination == 'sidecar': output.with_suffix('.json').write_bytes(b'sidecar sentinel')
    if destination == 'source': output = source
    if destination == 'json_source':
        source = tmp_path / 'derived.json'; source.write_bytes((tmp_path / 'source.pt').read_bytes())
    if destination == 'same_outputs': output = tmp_path / 'derived.json'
    if destination == 'symlink':
        output.symlink_to(source)
    files = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises((ValueError, FileExistsError)): migration().derive_surface_checkpoint(source, output)
    for path, content in files.items(): assert path.read_bytes() == content
    assert set(tmp_path.iterdir()) == set(files)


def test_detects_source_change_and_removes_its_partial_outputs(tmp_path, monkeypatch):
    source, output, _ = fixture(tmp_path)
    module = migration(); original_save = torch.save
    def changed_during_write(value, stream, *args, **kwargs):
        original_save(value, stream, *args, **kwargs)
        with source.open('ab') as source_stream: source_stream.write(b'changed')
    monkeypatch.setattr(module.torch, 'save', changed_during_write)
    with pytest.raises(RuntimeError, match='source.*changed'):
        module.derive_surface_checkpoint(source, output)
    assert not output.exists() and not output.with_suffix('.json').exists()


def test_late_sidecar_race_preserves_competing_file(tmp_path, monkeypatch):
    source, output, _ = fixture(tmp_path)
    module = migration(); original_open = Path.open
    def competing_open(path, mode='r', *args, **kwargs):
        if path == output.with_suffix('.json') and mode.startswith('x'):
            with original_open(path, 'wb') as stream: stream.write(b'competing writer')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', competing_open)
    with pytest.raises(FileExistsError): module.derive_surface_checkpoint(source, output)
    assert not output.exists()
    assert output.with_suffix('.json').read_bytes() == b'competing writer'


def test_cli_help_without_imports_and_real_full_state_derivation(tmp_path):
    # -S excludes site-packages: --help must work even without torch/numpy.
    result = subprocess.run([sys.executable, '-B', '-S', str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--input' in result.stdout and '--output' in result.stdout
    source, output, saved = fixture(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--input', str(source), '--output', str(output)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    same(expected_state(saved), torch.load(output, weights_only=False))
    assert json.loads(result.stdout) == json.loads(output.with_suffix('.json').read_text())
