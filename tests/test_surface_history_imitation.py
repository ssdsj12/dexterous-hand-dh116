import copy

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_surface_imitation as bc
from test_surface_imitation import dataset


def history_data():
    from go2_pvcnn.control.dh116.tabletop_surface_contact_history import CONTRACT
    data = dataset()
    data['observations'] = torch.cat((data['observations'], torch.zeros(len(data['observations']), 50)), 1)
    data['contract'].update(observation_size=401, observation_version='dh116-surface-sharing-history-v3',
                            surface_contact_history=copy.deepcopy(CONTRACT))
    return data


def test_history_is_an_explicit_input_version_not_silent_padding():
    data = history_data()
    bc.validate_dataset(data)
    bad = copy.deepcopy(data)
    bad['contract']['observation_version'] = bc.OBSERVATION_VERSION
    with pytest.raises(ValueError):bc.validate_dataset(bad)
    bad = copy.deepcopy(data)
    bad['contract']['surface_contact_history']['ema_alpha'] = .5
    with pytest.raises(ValueError):bc.validate_dataset(bad)


def test_history_actor_retains_24_bounded_commands():
    actor = bc.make_actor(401)
    assert actor[0].in_features == 401
    assert actor(torch.zeros(2, 401)).shape == (2, 24)
    assert actor(torch.randn(2, 401)).abs().max() <= 1


def test_input_extension_is_explicit_and_initially_preserves_old_predictions(tmp_path):
    old = dataset()
    source = tmp_path / 'source.pt'
    bc.pretrain(old, source, steps=2)
    before = source.read_bytes()
    state = torch.load(source, weights_only=True)
    actor = bc.make_actor();actor.load_state_dict(state['actor'])
    data = history_data()
    expected = float((actor(old['observations']) - old['actions']).square().mean())
    with pytest.raises(ValueError):bc.pretrain(data, tmp_path / 'bad.pt', steps=1, warm_start=source)
    metrics = bc.pretrain(data, tmp_path / 'history.pt', steps=1, warm_start=source, extend_contact_history=True)
    assert metrics['initial_mse'] == pytest.approx(expected, abs=1e-7)
    assert metrics['input_extension'] == dict(old_observation_size=351,new_observation_size=401,new_columns='zero')
    assert source.read_bytes() == before
    saved = torch.load(tmp_path / 'history.pt', weights_only=True)
    assert saved['actor']['0.weight'].shape == (128, 401)


def test_actual_task_history_appends_without_changing_existing_observations():
    from test_surface_exploration import surface_task
    from go2_pvcnn.control.dh116.tabletop_surface_contact_history import enable_contact_history
    task, world = surface_task()
    # Match the production source contract's scoring feature (351 vs the
    # fixture's otherwise valid 350D no-scoring-clock diagnostic profile).
    task.force_reward_start='downward_after_close'
    before = task.observe().clone()
    enable_contact_history(task)
    after = task.observe()
    assert after.shape == (2, 401)
    assert torch.equal(after[:, :351], before)
    assert not after[:, 351:].any()
    assert task.last_observation.shape == after.shape
