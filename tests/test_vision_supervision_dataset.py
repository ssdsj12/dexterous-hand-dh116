"""CPU schema tests; fixtures below are not native grasp verification."""
import copy
import importlib

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT


@pytest.fixture
def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset')


def visual_contract():
    return dict(actor_observation_size=3844, critic_observation_size=371,
                actor_uses_privileged_state=False, action_contract=copy.deepcopy(ACTION_CONTRACT),
                camera_config_sha256='1' * 64, primary_reset_sha256='2' * 64,
                secondary_reset_sha256='3' * 64, num_envs=2)


def dataset(module, *, successful=True):
    meta = []
    episodes = {}
    for ep, (style, group) in enumerate((('pinch_thumb_index', 'a' * 64),
                                        ('tripod_thumb_index_middle', 'b' * 64))):
        episodes[str(ep)] = dict(env_id=ep, first_episode=True, success=successful,
            recovered=False, native_physics=True, grasp_style=style,
            source_trajectory_id=group, source_episode_id=ep * 2,
            proof_path=f'native/episode_{ep}.json')
        for tick in range(3):
            meta.append(dict(env_id=ep, episode_id=ep, tick=tick, frame_id=tick + 1,
                physics_step=tick * 8, simulation_time_s=tick / 60,
                grasp_style=style, source_trajectory_id=group, source_episode_id=ep * 2,
                previous_action=[0.] * 24, frame_raw_chunk=f'rgbd/{tick}.pt'))
    return module.build_dataset(torch.zeros(6, 3844), torch.zeros(6, 24),
        torch.zeros(6, 24), meta, episodes, visual_contract=visual_contract(),
        teacher_source_sha256='c' * 64, require_successful=successful)


def test_build_validation_is_pure_and_marks_schema_scope(module):
    data = dataset(module)
    before = copy.deepcopy(data)
    assert module.validate_dataset(data) is data
    assert torch.equal(data['observations'], before['observations'])
    assert data['sample_metadata'] == before['sample_metadata']
    assert data['validation_scope'] == 'schema_only_not_native_verification'
    assert data['source_trajectory_ids'] == ['a' * 64] * 3 + ['b' * 64] * 3


@pytest.mark.parametrize('dimension', [351, 371, 401, 3843])
def test_privileged_or_wrong_observation_shape_rejected(module, dimension):
    data = dataset(module)
    data['observations'] = torch.zeros(6, dimension)
    with pytest.raises(ValueError, match='3844'):
        module.validate_dataset(data)


@pytest.mark.parametrize('key', ['observations', 'actions', 'reference_actions'])
def test_nonfinite_and_integer_tensors_rejected(module, key):
    data = dataset(module)
    data[key][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        module.validate_dataset(data)
    data[key] = torch.zeros_like(data[key], dtype=torch.long)
    with pytest.raises(ValueError, match='floating'):
        module.validate_dataset(data)


@pytest.mark.parametrize('key', ['actions', 'reference_actions'])
def test_bounded_actions_and_endpoint_labels(module, key):
    data = dataset(module)
    data[key][0, 0] = 1.
    data[key][0, 1] = -1.
    # Remove optional previous-action columns so endpoint sample has no false history.
    for row in data['sample_metadata']:
        row.pop('previous_action')
    module.validate_dataset(data)
    data[key][0, 0] = 1.001
    with pytest.raises(ValueError, match='range'):
        module.validate_dataset(data)


@pytest.mark.parametrize('change', [
    lambda d: d['visual_contract'].update(actor_uses_privileged_state=True),
    lambda d: d['visual_contract'].update(critic_observation_size=3844),
    lambda d: d['visual_contract']['action_contract'].update(joint_rate_rad_s=1.),
    lambda d: d.update(teacher_source_sha256='not-a-digest'),
    lambda d: d['visual_contract'].update(camera_config_sha256='missing'),
])
def test_contract_and_source_provenance_rejected(module, change):
    data = dataset(module)
    change(data)
    with pytest.raises(ValueError):
        module.validate_dataset(data)


@pytest.mark.parametrize('field', ['first_episode', 'success', 'native_physics'])
def test_failed_or_non_native_rows_never_enter_success_pool(module, field):
    data = dataset(module)
    data['episodes']['0'][field] = False
    with pytest.raises(ValueError, match='successful'):
        module.validate_dataset(data)
    module.validate_dataset(data, require_successful=False)


def test_recovery_excluded_from_success_pool(module):
    data = dataset(module)
    data['episodes']['0']['recovered'] = True
    with pytest.raises(ValueError, match='successful'):
        module.validate_dataset(data)
    module.validate_dataset(data, require_successful=False)


@pytest.mark.parametrize('field,bad', [('tick', 0), ('frame_id', 1),
                                    ('simulation_time_s', 0.), ('physics_step', 0)])
def test_pre_action_rows_must_be_ordered_and_unique(module, field, bad):
    data = dataset(module)
    data['sample_metadata'][1][field] = bad
    with pytest.raises(ValueError, match='order|tick|unique'):
        module.validate_dataset(data)


def test_wrong_episode_binding_or_source_group_rejected(module):
    data = dataset(module)
    data['sample_metadata'][0]['source_episode_id'] = 42
    with pytest.raises(ValueError, match='episode'):
        module.validate_dataset(data)
    data = dataset(module)
    data['source_trajectory_ids'][0] = 'separate-source'
    with pytest.raises(ValueError, match='source_trajectory'):
        module.validate_dataset(data)


def test_previous_action_is_the_actual_prior_issued_action(module):
    data = dataset(module)
    data['sample_metadata'][1]['previous_action'][0] = .2
    with pytest.raises(ValueError, match='previous_action'):
        module.validate_dataset(data)


def test_grouped_split_keeps_replay_copies_together_and_reports_fit_only(module):
    data = dataset(module)
    split = module.split_by_source(data, ['b' * 64])
    assert split['train_indices'].tolist() == [0, 1, 2]
    assert split['heldout_indices'].tolist() == [3, 4, 5]
    assert split['source_trajectory_count'] == 2
    assert split['heldout_independent_source_count'] == 1
    assert not split['fit_only']
    assert module.split_by_source(data)['fit_only']
    with pytest.raises(ValueError, match='train'):
        module.split_by_source(data, ['a' * 64, 'b' * 64])
    with pytest.raises(ValueError, match='unknown'):
        module.split_by_source(data, ['unknown'])


def test_save_is_exclusive_and_source_cannot_be_overwritten(module, tmp_path):
    data = dataset(module)
    path = tmp_path / 'visual.pt'
    proof = module.save_dataset(data, path)
    assert proof['samples'] == 6 and len(proof['sha256']) == 64
    loaded = torch.load(path, weights_only=True)
    module.validate_dataset(loaded)
    with pytest.raises(FileExistsError):
        module.save_dataset(data, path)
    data['teacher_source_path'] = str(tmp_path / 'source.pt')
    with pytest.raises(ValueError, match='source'):
        module.save_dataset(data, data['teacher_source_path'])


def test_visual_contract_batch_changes_only(module):
    data = dataset(module)
    current = dict(visual_contract(), num_envs=128)
    module.validate_dataset(data, current)
    current['camera_config_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='visual contract'):
        module.validate_dataset(data, current)


def test_teacher_loader_uses_bound_bytes_and_groups_replayed_source(module, monkeypatch, tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_surface_imitation as old
    # Test only dispatch/provenance here; full 371D validity covered by real corpus audit.
    monkeypatch.setattr(old, 'validate_dataset', lambda data, contract=None: data)
    source = dict(contract={'observation_size': 371,
        'surface_contract': {'action': ACTION_CONTRACT}},
        observations=torch.zeros(10, 371), actions=torch.zeros(10, 24),
        episode_ids=torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3]),
        episode_grasp_styles={0: 'pinch_thumb_index', 1: 'pinch_thumb_index',
                              2: 'tripod_thumb_index_middle', 3: 'tripod_thumb_index_middle'},
        reset_family=[dict(label='pinch_thumb_index', source_dataset_sha256='a' * 64),
                      dict(label='tripod_thumb_index_middle', source_dataset_sha256='b' * 64)])
    path = tmp_path / 'source.pt'
    torch.save(source, path)
    result = module.load_teacher_reference_dataset(path)
    assert len(result['source_sha256']) == 64
    assert result['references']['pinch_thumb_index']['source_episode_id'] == 0
    assert result['references']['tripod_thumb_index_middle']['source_episode_id'] == 2
    assert result['references']['tripod_thumb_index_middle']['actions'].shape == (3, 24)
    assert result['independent_source_trajectory_count'] == 2
    assert result['episodes'][0]['source_trajectory_id'] == result['episodes'][1]['source_trajectory_id']
    assert result['native_success_revalidated'] is False
    source['contract']['surface_contract']['action'] = dict(ACTION_CONTRACT, size=26)
    torch.save(source, path)
    with pytest.raises(ValueError, match='24D'):
        module.load_teacher_reference_dataset(path)


def with_terminal_records(module):
    data = dataset(module)
    for ep in data['episodes'].values():
        ep['terminal'] = dict(success=True, recovered=False, failure=0, control_steps=3)
    return data


def test_success_pool_requires_complete_labels_through_real_terminal_tick(module):
    data = with_terminal_records(module)
    module.validate_dataset(data)
    # The teacher reference stopped after tick 2; physics ran zero actions for
    # two more ticks and only then reported success. Its final history/actions
    # are absent, so the existing prefix is not a complete successful episode.
    data['episodes']['0']['terminal']['control_steps'] = 5
    with pytest.raises(ValueError, match='terminal.*sample|sample.*terminal'):
        module.validate_dataset(data)
    module.validate_dataset(data, require_successful=False)


@pytest.mark.parametrize('field,value', [
    ('success', False), ('success', 1), ('recovered', True), ('recovered', 0),
    ('failure', 1), ('failure', False), ('control_steps', 0),
    ('control_steps', True), ('control_steps', 3.), ('control_steps', 2),
])
def test_success_terminal_evidence_must_match_first_episode(module, field, value):
    data = with_terminal_records(module)
    data['episodes']['0']['terminal'][field] = value
    with pytest.raises(ValueError, match='terminal'):
        module.validate_dataset(data)
    # Failed/incomplete audit data stays retainable, never admitted to BC.
    module.validate_dataset(data, require_successful=False)


def test_terminal_record_cannot_be_bound_to_other_environment_samples(module):
    data = with_terminal_records(module)
    data['sample_metadata'][2]['env_id'] = 1
    with pytest.raises(ValueError, match='episode'):
        module.validate_dataset(data)


def test_missing_terminal_is_only_schema_validation_not_native_proof(module):
    data = dataset(module)
    module.validate_dataset(data)
    assert all('terminal' not in episode for episode in data['episodes'].values())
    assert data['validation_scope'] == 'schema_only_not_native_verification'


def test_bad_terminal_type_does_not_bypass_success_guard(module):
    data = with_terminal_records(module)
    data['episodes']['0']['terminal'] = 'not-a-terminal-record'
    with pytest.raises(ValueError, match='terminal'):
        module.validate_dataset(data)
