import json

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_joint_limit_replay import load_actions, replay_action


def test_replay_rejects_missing_environment_and_policy_step(tmp_path):
    path = tmp_path / 'policy.jsonl'
    row = dict(env_id=0, policy_step=8, action24=[0.] * 24, episode_id=0)
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='environment'):
        load_actions(path, 2)
    path.write_text(json.dumps(row) + '\n' + json.dumps(dict(row, policy_step=10)) + '\n')
    with pytest.raises(ValueError, match='consecutive'):
        load_actions(path, 1)


def test_interventions_change_only_selected_physical_channels():
    raw = torch.linspace(-1.2, 1.2, 24).reshape(1, 24)
    baseline = replay_action(raw, 'recorded')
    assert baseline.abs().max() == 1
    for mode, region in [('neutral_fingers', slice(0, 12)), ('neutral_rotation', slice(18, 24)),
                         ('neutral_translation', slice(12, 18))]:
        reference = baseline.clone()
        reference[:, region] = 0
        assert torch.equal(replay_action(raw, mode), reference)
    assert torch.equal(replay_action(raw, 'omit_inverse'), raw.clamp(-1, 1))
