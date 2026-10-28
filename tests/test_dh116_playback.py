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


def test_material_comparison_is_explicit_inference_only_and_preserves_checkpoint(source):
    from go2_pvcnn.control.dh116.tabletop_material import material_contract
    path,contract,original=source
    current=dict(contract,num_envs=1,**material_contract('high_friction'))
    with pytest.raises(ValueError,match='contract'):
        load_playback_policy(path,vector(1),current)
    before=path.with_suffix('.json').read_bytes()
    model,metadata=load_playback_policy(path,vector(1),current,allow_material_change=True)
    assert_state_equal(original.policy.state_dict(),model.policy.state_dict())
    assert metadata['playback_material_override']['current']==current['object_contact_material']
    assert before==path.with_suffix('.json').read_bytes()
    for change in [dict(physics_hz=1),dict(reward='different'),dict(hand_contact_material={'static_friction':99})]:
        with pytest.raises(ValueError,match='contract'):
            load_playback_policy(path,vector(1),dict(current,**change),allow_material_change=True)
