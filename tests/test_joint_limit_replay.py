import json

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_joint_limit_replay import load_actions, replay_action
from go2_pvcnn.control.dh116.tabletop_joint_limit_replay import late_replay_action


@pytest.mark.parametrize('arm,section', [('right_rotation_after240',slice(21,24)),
                                       ('right_translation_after240',slice(15,18))])
def test_late_intervention_preserves_prefix_and_all_other_channels(arm,section):
    action=torch.linspace(-.9,.9,48).reshape(2,24)
    original=action.clone()
    assert torch.equal(late_replay_action(action,arm,239),original)
    expected=action.clone()
    expected[:,section]=0
    assert torch.equal(late_replay_action(action,arm,240),expected)
    assert torch.equal(action,original)


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
