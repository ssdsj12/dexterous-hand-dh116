"""CPU-only library provenance and narrow initialization-family tests."""
import copy
import hashlib
import importlib
import importlib.util
import json

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_surface_imitation as bc


def reset_contract(index, n=2):
    return dict(observation_size=351, policy_command_size=24,
                observation_version=bc.OBSERVATION_VERSION, num_envs=n,
                rollout_steps=24, minibatches=2, physics_hz=480, control_hz=60,
                reward='surface_v2', cad_sha256='d' * 64,
                controller=dict(version='surface', effort_nm=.2),
                close_seed_sha256=hashlib.sha256(f'seed{index}'.encode()).hexdigest(),
                grasp_initialization_sha256=hashlib.sha256(f'init{index}'.encode()).hexdigest(),
                success_bottom_m=[.39] * n,
                surface_force_sharing=dict(lift_reference_bottom_m=[.36] * n))


def native_data(index, n=4):
    generator = torch.Generator().manual_seed(index)
    obs = torch.randn(n, 351, generator=generator) * .1
    return dict(version=bc.DATASET_VERSION, contract=reset_contract(index),
                observations=obs, actions=torch.tanh(obs[:, :24]),
                next_observations=obs + .01, rewards=torch.arange(n).float(),
                dones=torch.arange(n) == n - 1, episode_ids=torch.full((n,), 7, dtype=torch.long),
                episode_grasp_styles={7: f'native_style_{index}'},
                qualification=dict(qualified=True, successful_replays=1,
                                   proof_path=f'proof/replay{index}.json'))


def declared_family(count=2):
    return [dict(label=f'style{i}', contract=reset_contract(i),
                 source_dataset_sha256=hashlib.sha256(f'data{i}'.encode()).hexdigest(),
                 proof_path=f'proof/{i}.json') for i in range(count)]


def family_data(count=2):
    data = native_data(0)
    data['reset_family'] = declared_family(count)
    return data


@pytest.fixture
def library():
    # Missing API is reported as a test failure, not a collection error.
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_surface_demo_library')


def test_library_module_is_available():
    assert importlib.util.find_spec('go2_pvcnn.control.dh116.tabletop_surface_demo_library') is not None


@pytest.mark.parametrize('count', [2, 3])
def test_declared_family_accepts_each_exact_member_and_batch_change(count):
    data = family_data(count)
    for i in range(count):
        bc.validate_dataset(data, reset_contract(i, n=64))
        bc.validate_reset_family(data['reset_family'], reset_contract(0), reset_contract(i, n=16))


def test_declared_family_rejects_undeclared_initialization():
    with pytest.raises(ValueError, match='family|member'):
        bc.validate_dataset(family_data(), reset_contract(9))


@pytest.mark.parametrize('saved,current', [(9, 0), (0, 9)])
def test_explicit_block_switch_requires_both_contracts_to_be_members(saved, current):
    with pytest.raises(ValueError, match='family|member'):
        bc.validate_reset_family(declared_family(), reset_contract(saved), reset_contract(current))


@pytest.mark.parametrize('bad', [[], None, {}, [dict(label='fake')]])
def test_explicit_family_cannot_be_empty_or_malformed(bad):
    data = native_data(0)
    data['reset_family'] = bad
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


@pytest.mark.parametrize('change', [
    lambda f: f[1].update(contract=reset_contract(0)),
    lambda f: f[1]['contract'].update(physics_hz=240),
    lambda f: f[1]['contract'].update(policy_command_size=20),
    lambda f: f[1]['contract'].update(cad_sha256='e' * 64),
    lambda f: f[1]['contract'].update(reward='other'),
    lambda f: f[1]['contract']['controller'].update(effort_nm=.3),
    lambda f: f[1]['contract'].pop('grasp_initialization_sha256'),
    lambda f: f[1]['contract'].update(close_seed_sha256='fake'),
    lambda f: f[1].update(label=' '),
    lambda f: f[1].update(proof_path=''),
    lambda f: f[1].update(source_dataset_sha256='not-sha256'),
    lambda f: f[1].update(qualification=dict(qualified=False, successful_replays=1, proof_path='proof/1.json')),
    lambda f: f[1].update(qualification=dict(qualified=True, successful_replays=0, proof_path='proof/1.json')),
])
def test_family_rejects_fake_members_and_physics_changes(change):
    data = family_data()
    change(data['reset_family'])
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


def test_family_does_not_strip_nested_initialization_metadata():
    data = family_data()
    data['reset_family'][1]['contract']['controller']['close_seed_sha256'] = 'x' * 64
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


def test_base_contract_must_itself_be_a_declared_member():
    data = family_data()
    data['contract'] = reset_contract(8)
    with pytest.raises(ValueError):
        bc.validate_dataset(data, reset_contract(0))


def test_family_does_not_treat_numeric_zero_as_false_controller_metadata():
    data = family_data()
    data['contract']['controller']['override'] = False
    for member in data['reset_family']:
        member['contract']['controller']['override'] = False
    data['reset_family'][1]['contract']['controller']['override'] = 0
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


def test_sha_case_changes_do_not_invent_a_second_initialization():
    data = family_data()
    data['reset_family'][1]['contract'] = reset_contract(0)
    for field in ('close_seed_sha256', 'grasp_initialization_sha256'):
        data['reset_family'][1]['contract'][field] = data['reset_family'][1]['contract'][field].upper()
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


def test_family_rejects_an_undeclared_combination_of_declared_reset_hashes():
    data = family_data()
    current = reset_contract(0)
    current['close_seed_sha256'] = reset_contract(1)['close_seed_sha256']
    with pytest.raises(ValueError):
        bc.validate_dataset(data, current)


def write_components(tmp_path, count=2):
    components = []
    for i in range(count):
        path = tmp_path / f'native{i}.pt'
        torch.save(native_data(i, n=4 * (i + 1)), path)
        components.append((f'style{i}', path))
    return components


@pytest.mark.parametrize('count', [2, 3])
def test_merge_preserves_columns_episodes_styles_and_provenance(library, tmp_path, count):
    components = write_components(tmp_path, count)
    originals = [p.read_bytes() for _, p in components]
    output = tmp_path / 'library.pt'
    data = library.merge_datasets(components, output)
    saved = torch.load(output, weights_only=True)
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    assert data['version'] == bc.DATASET_VERSION
    assert saved['contract'] == reset_contract(0)
    assert manifest['qualified'] is True and manifest['successful_replays'] == count
    assert len(data['reset_family']) == count
    offset = 0
    for i, ((label, path), raw) in enumerate(zip(components, originals)):
        source = torch.load(path, weights_only=True)
        n = len(source['actions'])
        for key in ('observations', 'actions', 'next_observations', 'rewards', 'dones'):
            assert torch.equal(data[key][offset:offset + n], source[key])
        assert data['episode_ids'][offset:offset + n].tolist() == [i] * n
        assert data['episode_grasp_styles'][i] == f'native_style_{i}'
        assert data['episode_style_labels'][i] == label
        assert data['sample_weights'][offset:offset + n].sum() == pytest.approx(1.)
        digest = hashlib.sha256(raw).hexdigest()
        member = data['reset_family'][i]
        assert member['source_dataset_sha256'] == digest
        assert member['contract'] == source['contract']
        assert member['proof_path'] == source['qualification']['proof_path']
        assert manifest['components'][i]['path'] == str(path.resolve())
        assert manifest['components'][i]['sha256'] == digest
        assert manifest['components'][i]['qualification'] == source['qualification']
        assert path.read_bytes() == raw
        bc.validate_dataset(data, reset_contract(i, n=64))
        offset += n


def test_repeated_style_labels_balance_the_combined_style_sample_count(library, tmp_path):
    components = write_components(tmp_path, 3)
    components[1] = ('style0', components[1][1])
    data = library.merge_datasets(components, tmp_path / 'same_style.pt')
    # style0 has 4+8 rows; style2 has 12 rows, hence equal per-row weights.
    assert torch.allclose(data['sample_weights'], torch.full_like(data['sample_weights'], 1 / 12))


def test_merge_remaps_multiple_noncontiguous_episode_ids_in_original_order(library, tmp_path):
    components = write_components(tmp_path)
    source = native_data(0, n=6)
    source['episode_ids'] = torch.tensor([9, 9, 9, 3, 3, 3])
    source['episode_grasp_styles'] = {9: 'tripod', 3: 'pinch'}
    source['qualification']['successful_replays'] = 2
    torch.save(source, components[0][1])
    data = library.merge_datasets(components, tmp_path / 'episodes.pt')
    assert data['episode_ids'].tolist() == [0, 0, 0, 1, 1, 1] + [2] * 8
    assert data['episode_grasp_styles'] == {0: 'tripod', 1: 'pinch', 2: 'native_style_1'}
    assert data['episode_style_labels'] == {0: 'style0', 1: 'style0', 2: 'style1'}
    assert data['qualification']['successful_replays'] == 3


@pytest.mark.parametrize('field', ['next_observations', 'rewards', 'dones'])
def test_merge_rejects_partially_missing_optional_columns(library, tmp_path, field):
    components = write_components(tmp_path)
    source = torch.load(components[1][1], weights_only=True)
    del source[field]
    torch.save(source, components[1][1])
    output = tmp_path / 'rejected.pt'
    with pytest.raises(ValueError):
        library.merge_datasets(components, output)
    assert not output.exists() and not output.with_suffix('.manifest.json').exists()


def test_merge_allows_optional_columns_absent_from_every_component(library, tmp_path):
    components = write_components(tmp_path)
    for _, path in components:
        source = torch.load(path, weights_only=True)
        for key in ('next_observations', 'rewards', 'dones'):
            del source[key]
        torch.save(source, path)
    merged = library.merge_datasets(components, tmp_path / 'minimal.pt')
    assert all(key not in merged for key in ('next_observations', 'rewards', 'dones'))


@pytest.mark.parametrize('change', [
    lambda d: d['contract'].update(physics_hz=240),
    lambda d: d['contract'].update(policy_command_size=26),
    lambda d: d['contract'].update(reward='other'),
    lambda d: d['contract'].update(cad_sha256='e' * 64),
    lambda d: d['contract']['controller'].update(effort_nm=.3),
    lambda d: d['contract'].update(success_bottom_m=[.39, .4]),
    lambda d: d['qualification'].update(qualified=False),
    lambda d: d['qualification'].update(proof_path=''),
    lambda d: d.update(rewards=torch.zeros(1)),
    lambda d: d['next_observations'].__setitem__((0, 0), float('nan')),
    lambda d: d.update(dones=torch.full_like(d['rewards'], .5)),
    lambda d: d.update(actions=d['actions'].double()),
    lambda d: d['qualification'].update(successful_replays=2),
    lambda d: d['contract']['surface_force_sharing'].update(lift_reference_bottom_m=[.36, .37]),
])
def test_merge_fails_closed_before_writing_outputs(library, tmp_path, change):
    components = write_components(tmp_path)
    data = torch.load(components[1][1], weights_only=True)
    change(data)
    torch.save(data, components[1][1])
    output = tmp_path / 'invalid.pt'
    with pytest.raises(ValueError):
        library.merge_datasets(components, output)
    assert not output.exists() and not output.with_suffix('.manifest.json').exists()


@pytest.mark.parametrize('bad', [[], [(' ', 'missing.pt')], [('style', 'missing.pt')]])
def test_bad_components_are_rejected(library, tmp_path, bad):
    with pytest.raises(ValueError):
        library.merge_datasets(bad, tmp_path / 'invalid.pt')


def test_merge_must_not_overwrite_an_original_component(library, tmp_path):
    components = write_components(tmp_path)
    path = components[0][1]
    before = path.read_bytes()
    with pytest.raises(ValueError):
        library.merge_datasets(components, path)
    assert path.read_bytes() == before
