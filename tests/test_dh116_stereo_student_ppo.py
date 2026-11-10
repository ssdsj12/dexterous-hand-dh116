"""CPU-only asymmetric Student optimization/export; no native qualification."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_dh116_stereo_estimator import history
from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract

torch.set_num_threads(1)
NETWORK = dict(embed_dim=32, history=4, num_heads=4, num_layers=1, dropout=0.)


def module(name):
    target = 'go2_pvcnn.control.dh116.tabletop_stereo_' + name
    assert importlib.util.find_spec(target), 'Phase6 module missing: ' + target
    return importlib.import_module(target)


def config():
    return dict(network=NETWORK.copy(), ppo=dict(epochs=1, minibatches=1), rollout_steps=1,
        auxiliary_bc=dict(enabled=False))


def setup_trainer():
    torch.manual_seed(52)
    m = module('student_ppo')
    return m.make_student_trainer(config(), 1), m.history_to_tree(history(1))


def student_contract():
    return make_contract('student', config(), calibration_sha256='a' * 64)


def test_asymmetric_student_real_encoder_gradients_and_critic_isolation():
    engine, actor_tree = setup_trainer()
    critic = torch.ones(1, 371)
    before = engine.model.latent_mean(actor_tree).detach().clone()
    for value in (critic, critic * 10):
        assert torch.equal(before, engine.model.latent_mean(actor_tree))
        assert engine.model.value(value).shape == (1,)
    assert not torch.equal(engine.model.value(critic), engine.model.value(critic*10))
    sampled = engine.collect_step(actor_tree, critic)
    engine.record_step(torch.ones(1), torch.tensor([False]), torch.zeros(1), executed_actions=sampled['actions'])
    batch = engine.rollout.batch()
    assert batch['actor_obs']['depth_valid'].dtype == torch.bool
    assert batch['actor_obs']['history_valid'].dtype == torch.bool
    assert 'privileged_state' not in batch['actor_obs']
    backbone_before = next(engine.model.actor.estimator.encoder.backbone.parameters()).detach().clone()
    stats = engine.update()
    assert stats['samples'] == 1
    parameter = next(engine.model.actor.estimator.encoder.backbone.parameters())
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert not torch.equal(parameter, backbone_before)


def test_history_tree_neutralizes_only_masked_na_and_preserves_input():
    h = history(1)
    h.depth_m[0, -1, 0, 0] = float('nan')
    h.depth_confidence[0, -1, 0, 0] = float('nan')
    h.depth_valid[0, -1, 0, 0] = False
    h.points_world[0, -1, 0] = float('nan')
    h.point_valid[0, -1, 0] = False
    tree = module('student_ppo').history_to_tree(h)
    assert all(torch.isfinite(v).all() for v in tree.values())
    assert torch.isnan(h.depth_m[0, -1, 0, 0])
    assert torch.equal(tree['history_valid'], h.history_valid)
    assert not tree['depth_valid'][0, -1, 0, 0]
    h.depth_valid[0, -1, 0, 0] = True
    with pytest.raises(ValueError):
        module('student_ppo').history_to_tree(h)


@pytest.mark.parametrize('key', ['privileged_state', 'teacher_action', 'critic_obs', 'wire_nodes', 'success'])
def test_actor_tree_rejects_truth_keys(key):
    m = module('student_ppo')
    tree = m.history_to_tree(history(1))
    tree[key] = torch.zeros(1, 371)
    with pytest.raises((ValueError, TypeError), match='unknown|privileged|sensor|unexpected'):
        m.StereoAsymmetricPolicy(NETWORK).latent_mean(tree)


def test_critic_loss_cannot_backpropagate_into_actor():
    engine, tree = setup_trainer()
    engine.model.zero_grad(set_to_none=True)
    engine.model.value(torch.ones(1, 371)).square().mean().backward()
    assert all(p.grad is None for p in engine.model.actor.parameters())
    assert any(p.grad is not None for p in engine.model.critic.parameters())
    assert all(getattr(m, 'p', 0.) == 0. for m in engine.model.actor.modules() if isinstance(m, torch.nn.Dropout))


def test_checkpoint_and_actor_only_export_replay_identical(tmp_path):
    engine, tree = setup_trainer()
    cp = module('checkpoint')
    contract = student_contract()
    native = tmp_path / 'student.pt'
    cp.save_student_checkpoint(native, engine, contract, network_config=NETWORK, history_reset='clear')
    expected = engine.model.deterministic(tree).detach()
    restored, _ = setup_trainer()
    cp.load_student_checkpoint(native, restored, contract, history_reset='clear')
    torch.testing.assert_close(expected, restored.model.deterministic(tree), rtol=0, atol=0)
    output = tmp_path / 'actor.pt'
    cp.export_actor(native, output)
    payload = torch.load(output, map_location='cpu', weights_only=False)
    assert 'optimizer' not in payload and 'critic' not in payload and 'teacher' not in payload
    assert all(not key.startswith('critic.') for key in payload['actor'])
    deployed = cp.load_deployment_actor(output, expected_calibration_sha256='a' * 64)
    assert type(deployed).__name__ == 'StereoStudentActor' and not hasattr(deployed, 'critic')
    from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistoryBatch
    torch.testing.assert_close(expected, deployed.deterministic(StereoHistoryBatch(**tree)), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        cp.export_actor(native, output)
    with pytest.raises(ValueError, match='calibration'):
        cp.load_deployment_actor(output, expected_calibration_sha256='b' * 64)
    # Fresh interpreter proves deployment loading imports no Teacher or PPO runtime.
    code = ("import sys;from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import load_deployment_actor;"
        f"load_deployment_actor({str(output)!r});"
        "assert not any('privileged_teacher' in n or 'privileged_ppo' in n for n in sys.modules)")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('corruption', ['legacy', 'calibration', 'input_keys', 'actor_sha', 'network'])
def test_export_rejects_legacy_or_changed_contract(tmp_path, corruption):
    engine, _ = setup_trainer()
    cp = module('checkpoint')
    path, exported = tmp_path / 'student.pt', tmp_path / 'actor.pt'
    cp.save_student_checkpoint(path, engine, student_contract(), network_config=NETWORK, history_reset='clear')
    cp.export_actor(path, exported)
    payload = torch.load(exported, map_location='cpu', weights_only=False)
    if corruption == 'legacy': payload['version'] = 'dh116-visual-3844-v1'
    elif corruption == 'calibration': payload['calibration_sha256'] = 'b' * 64
    elif corruption == 'input_keys': payload['actor_contract']['input_keys'].append('privileged_state')
    elif corruption == 'actor_sha': payload['actor'][next(iter(payload['actor']))].add_(.1)
    elif corruption == 'network': payload['network_config']['history'] = 3
    torch.save(payload, exported)
    with pytest.raises(ValueError):
        cp.load_deployment_actor(exported)


def test_auxiliary_bc_requires_independent_qualified_interface_and_clean_boundary():
    engine, tree = setup_trainer()
    m = module('student_ppo')
    with pytest.raises(ValueError, match='qualification|qualified|metadata'):
        m.attach_auxiliary_bc(engine, object(), metadata={})
    engine.collect_step(tree, torch.zeros(1, 371))
    with pytest.raises(RuntimeError, match='pending|PPO|boundary'):
        m.attach_auxiliary_bc(engine, object(), metadata={'qualified': True})


def test_default_config_uses_stereo_inputs_and_disables_implicit_bc():
    path = Path(__file__).resolve().parents[1] / 'configs/dh116_training/stereo_student_ppo_v1.json'
    assert path.is_file(), 'Phase6 default config missing'
    value = json.loads(path.read_text())
    assert value['critic_observation_size'] == 371 and value['action_size'] == 24
    assert value['actor_input'] == 'StereoHistoryBatch_only'
    assert value['network']['dropout'] == 0.
    assert value['auxiliary_bc']['enabled'] is False


def test_trained_checkpoint_restores_adam_rng_and_rejects_bad_moment_atomically(tmp_path):
    engine, tree = setup_trainer()
    cp = module('checkpoint')
    sample = engine.collect_step(tree, torch.ones(1, 371))
    engine.record_step(torch.ones(1), torch.tensor([False]), torch.zeros(1), executed_actions=sample['actions'])
    engine.update()
    path = tmp_path / 'trained.pt'
    cp.save_student_checkpoint(path, engine, student_contract(), network_config=NETWORK, history_reset='clear')
    expected_random = torch.rand(4)
    restored, _ = setup_trainer()
    cp.load_student_checkpoint(path, restored, student_contract(), history_reset='clear')
    assert restored.optimizer.state_dict()['state']
    assert restored.counters == engine.counters
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    assert all(torch.equal(v, restored.model.state_dict()[k]) for k, v in engine.model.state_dict().items())
    payload = torch.load(path, map_location='cpu', weights_only=False)
    first = next(iter(payload['optimizer']['state']))
    payload['optimizer']['state'][first]['exp_avg'] = torch.zeros(9)
    broken = tmp_path / 'broken.pt'; torch.save(payload, broken)
    before = copy.deepcopy(restored.model.state_dict()); rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='Adam|moment'):
        cp.load_student_checkpoint(broken, restored, student_contract(), history_reset='clear')
    assert all(torch.equal(v, restored.model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state())


def test_training_checkpoint_requires_clean_boundary_and_exact_experiment_contract(tmp_path):
    engine, tree = setup_trainer()
    cp = module('checkpoint'); contract = student_contract()
    path = tmp_path / 'clean.pt'
    cp.save_student_checkpoint(path, engine, contract, network_config=NETWORK, history_reset='clear')
    other = make_contract('student', config(), calibration_sha256='b' * 64)
    with pytest.raises(ValueError, match='contract|calibration'):
        cp.load_student_checkpoint(path, engine, other, history_reset='clear')
    engine.collect_step(tree, torch.ones(1, 371))
    with pytest.raises(RuntimeError, match='boundary|pending'):
        cp.save_student_checkpoint(tmp_path / 'dirty.pt', engine, contract, network_config=NETWORK, history_reset='clear')


def test_declared_resolution_and_backbone_are_real_constraints():
    m = module('student_ppo')
    with pytest.raises(ValueError, match='backbone|ResNet'):
        m.StereoAsymmetricPolicy(dict(NETWORK, backbone={'type': 'PointNet'}))
    model = m.StereoAsymmetricPolicy(dict(NETWORK, image_resolution_wh=[16, 16]))
    with pytest.raises(ValueError, match='resolution|RGB'):
        model.deterministic(m.history_to_tree(history(1)))  # fixture is 32x32.


def test_bc_role_can_save_actor_package_without_ppo_or_native_claim(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistoryBatch
    cp=module('checkpoint')
    assert callable(getattr(cp,'save_actor_package',None)), 'public BC actor package API missing'
    engine,tree=setup_trainer()
    contract=make_contract('bc',config(),calibration_sha256='a'*64)
    output=tmp_path/'bc_actor.pt'
    cp.save_actor_package(output,engine.model.actor,contract,network_config=NETWORK)
    deployed=cp.load_deployment_actor(output,expected_calibration_sha256='a'*64)
    torch.testing.assert_close(deployed.deterministic(StereoHistoryBatch(**tree)),
        engine.model.actor.deterministic(StereoHistoryBatch(**tree)),rtol=0,atol=0)
    payload=torch.load(output,map_location='cpu',weights_only=False)
    assert payload['actor_contract']['training_kind']=='bc'
    assert payload['actor_contract']['native_qualified'] is False
    with pytest.raises(FileExistsError):
        cp.save_actor_package(output,engine.model.actor,contract,network_config=NETWORK)
    with pytest.raises(ValueError,match='version|contract'):
        cp.save_actor_package(tmp_path/'legacy.pt',engine.model.actor,{'version':'visual3844'},network_config=NETWORK)
