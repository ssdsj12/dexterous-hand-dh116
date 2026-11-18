"""CPU tests for real-transition, success-only teacher recording."""
import importlib
import importlib.util

import pytest
import torch


def recorder(num_envs=2, **kwargs):
    name = 'go2_pvcnn.control.dh116.tabletop_demonstrations'
    assert importlib.util.find_spec(name) is not None, 'success recorder is not implemented'
    return importlib.import_module(name).SuccessEpisodeRecorder(num_envs, **kwargs)


def transition(n=2, value=0., done=None, success=None, failure=None, phase=True):
    obs = torch.full((n, 156), value, dtype=torch.float64)
    actions = torch.full((n, 20), value / 10., dtype=torch.float64)
    rewards = torch.arange(n, dtype=torch.float64) + value
    dones = torch.tensor(done if done is not None else [False] * n)
    extras = {
        'terminal_observation': obs + .5,
        'success': torch.tensor(success if success is not None else [False] * n),
        'failure': torch.tensor(failure if failure is not None else [0] * n),
    }
    if phase:
        extras['phase'] = torch.full((n,), 3, dtype=torch.long)
    return obs, actions, rewards, dones, extras


def test_keeps_only_explicit_success_and_pairs_terminal_observations():
    r = recorder()
    first = transition(value=1.)
    r.append(*first)
    # TensorEnv's returned reset observation is deliberately not passed here.
    last = transition(value=2., done=[True, True], success=[True, False], failure=[0, 8])
    r.append(*last)
    assert r.episode_counts.tolist() == [1, 1]
    assert len(r.episodes) == 1
    ep = r.episodes[0]
    assert (ep['env_index'], ep['episode_index'], ep['success']) == (0, 0, True)
    torch.testing.assert_close(ep['observations'][:, 0], torch.tensor([1., 2.]))
    torch.testing.assert_close(ep['actions'][:, 0], torch.tensor([.1, .2]))
    torch.testing.assert_close(ep['next_observations'][:, 0], torch.tensor([1.5, 2.5]))
    torch.testing.assert_close(ep['rewards'], torch.tensor([1., 2.]))
    assert ep['dones'].tolist() == [False, True]
    assert ep['success_mask'].tolist() == [False, True]
    assert ep['failure'].tolist() == [0, 0]
    assert ep['phase'].tolist() == [3, 3]
    for key in ('observations', 'actions', 'next_observations', 'rewards'):
        assert ep[key].device.type == 'cpu' and ep[key].dtype == torch.float32
    assert r.summary()['failed_episodes'] == 1


def test_independent_reset_boundaries_do_not_mix_episode_samples():
    r = recorder()
    r.append(*transition(value=1., done=[True, False], failure=[3, 0]))
    r.append(*transition(value=2., done=[True, True], success=[True, True]))
    by_env = {ep['env_index']: ep for ep in r.episodes}
    assert by_env[0]['episode_index'] == 1
    assert by_env[0]['observations'][:, 0].tolist() == [2.]
    assert by_env[1]['episode_index'] == 0
    assert by_env[1]['observations'][:, 0].tolist() == [1., 2.]
    assert r.episode_counts.tolist() == [2, 1]


def test_done_and_failure_are_not_success_labels():
    r = recorder()
    r.append(*transition(done=[True, True], success=[False, True], failure=[0, 2]))
    assert r.episodes == []
    assert r.summary()['failed_episodes'] == 2


def test_overlong_episode_is_dropped_until_its_reset_boundary():
    r = recorder(1, max_steps=2)
    for value in (1., 2., 3.):
        r.append(*transition(1, value=value))
    r.append(*transition(1, value=4., done=[True], success=[True]))
    assert r.episodes == []
    assert r.summary()['overflow_episodes'] == 1
    r.append(*transition(1, value=5., done=[True], success=[True]))
    assert r.episodes[0]['observations'][:, 0].tolist() == [5.]
    assert r.episodes[0]['episode_index'] == 1


def test_recorder_copies_inputs_before_the_caller_reuses_tensor_storage():
    r = recorder(1)
    values = transition(1, value=1.)
    r.append(*values)
    for tensor in values[:3]:
        tensor.fill_(99.)
    values[4]['terminal_observation'].fill_(99.)
    r.append(*transition(1, value=2., done=[True], success=[True]))
    assert r.episodes[0]['observations'][0, 0] == 1.
    assert r.episodes[0]['next_observations'][0, 0] == 1.5


@pytest.mark.parametrize('field', ['observations', 'actions', 'rewards', 'terminal_observation'])
@pytest.mark.parametrize('invalid', ['shape', 'nonfinite'])
def test_invalid_transition_is_rejected_without_mutating_counters(field, invalid):
    r = recorder()
    obs, actions, rewards, dones, extras = transition(done=[True, True], success=[True, True])
    tensors = {'observations': obs, 'actions': actions, 'rewards': rewards,
               'terminal_observation': extras['terminal_observation']}
    bad = tensors[field]
    bad = bad[:, :-1] if invalid == 'shape' and bad.ndim == 2 else bad[:-1] if invalid == 'shape' else bad.clone()
    if invalid == 'nonfinite':
        bad.flatten()[0] = float('nan')
    tensors[field] = bad
    extras['terminal_observation'] = tensors['terminal_observation']
    with pytest.raises(ValueError, match='shape|finite'):
        r.append(tensors['observations'], tensors['actions'], tensors['rewards'], dones, extras)
    assert r.episodes == [] and r.episode_counts.tolist() == [0, 0]


@pytest.mark.parametrize('key', ['success', 'terminal_observation'])
def test_required_real_environment_evidence_cannot_be_missing(key):
    r = recorder()
    values = transition()
    del values[4][key]
    with pytest.raises(ValueError, match=key):
        r.append(*values)


def test_nonboolean_success_is_rejected_instead_of_coerced_to_true():
    r = recorder()
    values = transition(done=[True, True])
    values[4]['success'] = torch.tensor([.5, float('nan')])
    with pytest.raises(ValueError, match='success'):
        r.append(*values)


def test_optional_phase_is_omitted_when_environment_does_not_supply_it():
    r = recorder(1)
    r.append(*transition(1, done=[True], success=[True], phase=False))
    assert 'phase' not in r.episodes[0]


def test_save_preserves_contract_and_pending_qualification(tmp_path):
    r = recorder(1)
    r.append(*transition(1, done=[True], success=[True]))
    path = tmp_path / 'teacher.pt'
    contract = {'source_hashes': {'task': 'abc'}, 'physics': {'dt': 1 / 480}}
    qualification = {'qualified': False, 'repeat_successes': 1}
    r.save(path, contract, qualification)
    data = torch.load(path, weights_only=True)
    assert data['version'] == 'dh116-tabletop-demonstrations-v1'
    assert (data['observation_dim'], data['action_dim']) == (156, 20)
    assert data['contract'] == contract and data['qualification'] == qualification
    assert len(data['episodes']) == 1 and data['episodes'][0]['success'] is True


def test_empty_success_dataset_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match='success'):
        recorder().save(tmp_path / 'empty.pt', {}, {'qualified': False})
    assert not (tmp_path / 'empty.pt').exists()


@pytest.mark.parametrize('bad', ['shape', 'nonfinite'])
def test_save_revalidates_public_episode_tensors(tmp_path, bad):
    r = recorder(1)
    r.append(*transition(1, done=[True], success=[True]))
    if bad == 'shape':
        r.episodes[0]['actions'] = torch.zeros((2, 20))
    else:
        r.episodes[0]['rewards'][0] = float('inf')
    with pytest.raises(ValueError, match='shape|finite'):
        r.save(tmp_path / 'invalid.pt', {}, {'qualified': True})


def physical_evidence(n=1, value=0.):
    return {
        'contact_center_relative_m': torch.full((n, 2, 3), value, dtype=torch.float64),
        'contact_normal': torch.full((n, 2, 3), value, dtype=torch.float64),
        'contact_valid': torch.zeros((n, 2), dtype=torch.bool),
        'normal_vertical_force_n': torch.full((n,), value, dtype=torch.float64),
        'lift_vertical_force_n': torch.full((n,), value, dtype=torch.float64),
        'object_bottom_m': torch.full((n,), .3 + value, dtype=torch.float64),
    }


def test_physical_evidence_survives_recording_reused_inputs_and_serialization(tmp_path):
    r = recorder(1)
    first = transition(1)
    first[4]['physical_evidence'] = physical_evidence()
    r.append(*first)
    first[4]['physical_evidence']['contact_center_relative_m'].fill_(99.)
    last = transition(1, done=[True], success=[True])
    last[4]['physical_evidence'] = physical_evidence(value=.03)
    last[4]['physical_evidence']['contact_valid'].fill_(True)
    r.append(*last)
    evidence = r.episodes[0]['physical_evidence']
    assert evidence['contact_center_relative_m'].shape == (2, 2, 3)
    assert evidence['contact_center_relative_m'][0].eq(0.).all()
    assert evidence['contact_valid'].dtype == torch.bool
    assert evidence['contact_valid'].tolist() == [[False, False], [True, True]]
    assert evidence['object_bottom_m'].dtype == torch.float32
    assert all(value.device.type == 'cpu' for value in evidence.values())
    path = tmp_path / 'evidence.pt'
    r.save(path, {}, {'qualified': False})
    saved = torch.load(path, weights_only=True)['episodes'][0]['physical_evidence']
    assert saved.keys() == evidence.keys()
    for key in evidence:
        torch.testing.assert_close(saved[key], evidence[key])


@pytest.mark.parametrize('change', ['missing', 'keys', 'shape', 'nonfinite'])
def test_inconsistent_physical_evidence_is_rejected_before_recording(change):
    r = recorder(1)
    first = transition(1)
    first[4]['physical_evidence'] = physical_evidence()
    r.append(*first)
    last = transition(1, done=[True], success=[True])
    evidence = physical_evidence()
    if change == 'keys':
        del evidence['contact_normal']
    elif change == 'shape':
        evidence['contact_normal'] = torch.zeros((1, 3, 3))
    elif change == 'nonfinite':
        # A false validity mask does not authorize silent NaN sanitation.
        evidence['contact_center_relative_m'][0, 0, 0] = float('nan')
    if change != 'missing':
        last[4]['physical_evidence'] = evidence
    with pytest.raises(ValueError, match='physical_evidence'):
        r.append(*last)
    assert r.episode_counts.tolist() == [0]
    assert r.summary()['buffered_steps'] == [1]


@pytest.mark.parametrize('change', ['missing', 'shape', 'nonfinite'])
def test_save_revalidates_physical_evidence(change, tmp_path):
    r = recorder(1)
    values = transition(1, done=[True], success=[True])
    values[4]['physical_evidence'] = physical_evidence()
    r.append(*values)
    if change == 'missing':
        del r.episodes[0]['physical_evidence']
    elif change == 'shape':
        r.episodes[0]['physical_evidence']['contact_valid'] = torch.zeros((1, 3), dtype=torch.bool)
    else:
        r.episodes[0]['physical_evidence']['object_bottom_m'][0] = float('inf')
    with pytest.raises(ValueError, match='physical_evidence'):
        r.save(tmp_path / 'invalid-evidence.pt', {}, {'qualified': False})
