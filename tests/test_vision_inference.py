"""CPU inference oracles, with real historical actor weights and no simulator."""
import copy
import hashlib
import importlib
import importlib.util

import pytest
import torch

from test_vision_action_derivation import source_checkpoint, observations
from go2_pvcnn.control.dh116.tabletop_vision_action_derivation import derive_action_checkpoint


def api():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_inference'
    assert importlib.util.find_spec(name), 'inference-only visual loader is missing'
    return importlib.import_module(name)


@pytest.mark.parametrize('squashed', [False, True])
def test_loader_preserves_rng_source_and_matches_actor_without_ppo(tmp_path, monkeypatch, squashed):
    source, saved = source_checkpoint(tmp_path)
    if squashed:
        target = tmp_path/'derived.pt'
        derive_action_checkpoint(source, target)
        source = target
        saved = torch.load(source, weights_only=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    import go2_pvcnn.control.dh116.tabletop_vision_student as student
    actor = student.VisionStudentActor()
    actor.load_state_dict({k[6:]: v for k, v in saved['policy'].items() if k.startswith('actor.')})
    obs = observations()
    expected = actor(obs['policy']).detach()
    for constructor in ('make_vision_algorithm',):
        monkeypatch.setattr(student, constructor, lambda *a, **k: pytest.fail('PPO constructed'))
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('Adam constructed'))
    before = torch.get_rng_state().clone()
    loaded = api().load_vision_inference(source, saved['contract'])
    assert torch.equal(before, torch.get_rng_state())
    result = loaded.act(obs)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    altered = obs.clone(); altered['critic'].fill_(float('nan'))
    torch.testing.assert_close(loaded.act(altered), result, atol=0, rtol=0)
    assert not result.requires_grad and not loaded.actor.training
    assert not any(p.requires_grad for p in loaded.actor.parameters())
    assert not hasattr(loaded, 'optimizer') and not hasattr(loaded, 'storage')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest == loaded.checkpoint_sha256
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match='3844'):
        loaded.act(torch.zeros(2, 371))


def test_contract_allows_explicit_evaluation_batch_only_and_rejects_provenance_change(tmp_path):
    source, saved = source_checkpoint(tmp_path)
    module = api()
    current = module.evaluation_contract(saved['contract'], 4)
    assert current['num_envs'] == 4 and current['groups'][0]['num_envs'] == 2
    loaded = module.load_vision_inference(source, current)
    assert loaded.provenance['mode'] == 'evaluation_only'
    for path in [('network', 'action_mlp'), ('camera_config_sha256',),
                 ('action_contract', 'translation_rate_m_s'), ('privileged_critic_contract', 'physics_hz')]:
        bad = copy.deepcopy(current); parent = bad
        for key in path[:-1]: parent = parent[key]
        parent[path[-1]] = 'changed'
        with pytest.raises(ValueError, match='contract'):
            module.load_vision_inference(source, bad)


def test_source_contract_requires_real_hash_provenance_even_when_current_matches(tmp_path):
    source,saved=source_checkpoint(tmp_path)
    saved['contract']['camera_config_sha256']='authored_without_hash'
    torch.save(saved,source)
    with pytest.raises(ValueError,match='provenance'):
        api().load_vision_inference(source,saved['contract'])
