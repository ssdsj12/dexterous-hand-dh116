import numpy as np
import pytest

from test_dh116_env_expansion import source, vector, assert_state_equal
from go2_pvcnn.control.dh116.tabletop_playback import load_playback_policy


def test_playback_single_env_predicts_without_changing_saved_model(source):
    path, contract, original = source
    before = (path.read_bytes(), path.with_suffix('.json').read_bytes())
    env = vector(1)
    model, metadata = load_playback_policy(path, env, dict(contract, num_envs=1))
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1, 2) and np.isfinite(action).all()
    assert model.num_timesteps == original.num_timesteps
    assert model._n_updates == original._n_updates
    assert_state_equal(original.policy.state_dict(), model.policy.state_dict())
    assert_state_equal(original.policy.optimizer.state_dict(), model.policy.optimizer.state_dict())
    assert before == (path.read_bytes(), path.with_suffix('.json').read_bytes())


@pytest.mark.parametrize('change', [dict(physics_hz=240), dict(reward='different'),
                                   dict(rod={'dt': 2e-5}), dict(num_envs=2)])
def test_playback_rejects_changes_other_than_display_count(source, change):
    path, contract, _ = source
    current = dict(contract, num_envs=1)
    current.update(change)
    with pytest.raises(ValueError, match='contract|one environment'):
        load_playback_policy(path, vector(1), current)
