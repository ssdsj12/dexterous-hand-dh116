"""CPU fixtures exercise Teacher BC wiring and grant no native evidence."""
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_privileged_cli as cli
from go2_pvcnn.control.dh116 import tabletop_privileged_warm_start as warm
from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import load_checkpoint


CONFIG = Path(__file__).resolve().parents[1] / 'configs/dh116_training/privileged_teacher_ppo_guarded_v2.json'


def fixture(path, monkeypatch, identity=None):
    from test_dh116_teacher_expert_assembly import artifacts, CPU_only_gate
    from go2_pvcnn.control.dh116 import tabletop_privileged_expert_assembly as build
    source_identity, manifest = artifacts(path.parent)
    CPU_only_gate(monkeypatch)
    output = path.parent / 'assembled'
    build.assemble_experts(manifest, source_identity if identity is None else identity, output)
    return torch.load(output / 'privileged_native_expert.pt', weights_only=True)


@pytest.mark.parametrize('fault', ['privileged', 'action', 'order', 'state', 'empty_mask'])
def test_warm_start_reopens_aligned_source_and_rejects_invalid_rows(tmp_path, monkeypatch, fault):
    source = tmp_path / 'source.pt'
    payload = fixture(source, monkeypatch)
    if fault == 'privileged':
        content = torch.load(source, weights_only=True)
        content['privileged_valid'][0, 24] = False
        torch.save(content, source)
        payload['episodes'][0]['source_sha256'] = cli.file_sha256(source)
    elif fault == 'action':
        payload['actions'][0, 0] = -.3
    elif fault == 'state':
        payload['observations'][0, 300] = .3
    elif fault == 'order':
        content = torch.load(source, weights_only=True)
        content['sample_metadata'].reverse()
        torch.save(content, source)
        payload['episodes'][0]['source_sha256'] = cli.file_sha256(source)
    else:
        payload['valid_mask'][0].zero_()
    with pytest.raises(ValueError):
        warm.verify_supervised_rows(payload)


def test_offline_actor_bc_checkpoint_loads_into_r0_without_critic_noise_or_adam_transfer(tmp_path, monkeypatch):
    config, ppo = cli.load_config(CONFIG)
    source = tmp_path / 'source.pt'
    identity = cli.training_identity(ppo, dict(test_only=True))
    payload = fixture(source, monkeypatch, identity)
    expert = tmp_path / 'expert.pt'
    torch.save(payload, expert)
    verification = []
    replay = warm.verify_supervised_rows
    def counted(value):
        verification.append(1)
        return replay(value)
    monkeypatch.setattr(warm, 'verify_supervised_rows', counted)
    torch.manual_seed(116)
    reference = cli.PPOTrainer(cli.PrivilegedTeacher(**config['model']), ppo,
        rollout_steps=config['rollout_steps'], num_envs=2)
    output = tmp_path / 'warm'
    result = warm.warm_start_teacher(config, ppo, identity, expert,
        expert_sha256=cli.file_sha256(expert), output=output, steps=2, num_envs=2, seed=116)
    load_checkpoint(output / result['checkpoint_file'], reference, identity, history_reset='clear')
    assert result['native_grasp_qualification'] is False
    assert result['ppo_optimizer_state_empty'] and result['critic_and_noise_unchanged']
    assert len(verification) == 1 and reference.policy_version == 2
    assert not reference.optimizer.state and reference.counters['updates'] == 0
    assert (output / 'bc_iterations.jsonl').is_file()
    assert (output / 'warm_start.json').is_file()
    assert (output / 'expert_bc_optimizer.pt').is_file()


def test_wrong_expert_sha_and_unqualified_source_create_no_run(tmp_path):
    config, ppo = cli.load_config(CONFIG)
    identity = cli.training_identity(ppo, dict(test_only=True))
    expert = tmp_path / 'expert.pt'
    torch.save(dict(observations=torch.zeros(2, 371)), expert)
    output = tmp_path / 'warm'
    with pytest.raises(ValueError, match='SHA'):
        warm.warm_start_teacher(config, ppo, identity, expert,
            expert_sha256='a' * 64, output=output, steps=1, num_envs=2, seed=116)
    assert not output.exists()
    with pytest.raises(ValueError, match='versioned'):
        warm.warm_start_teacher(config, ppo, identity, expert,
            expert_sha256=cli.file_sha256(expert), output=output, steps=1, num_envs=2, seed=116)
    assert not output.exists()


def test_frozen_ppo_preflight_rejects_actor_bc_or_fabricated_counters(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_privileged_evaluation as evaluation
    from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import save_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_physics_sha256

    config, ppo = cli.load_config(CONFIG)
    contract = dict(num_envs=2, physics_hz=480, control_hz=60)
    identity = cli.training_identity(ppo, contract)
    engine = cli.PPOTrainer(cli.PrivilegedTeacher(**config['model']), ppo,
        rollout_steps=config['rollout_steps'], num_envs=2)
    bc_only = tmp_path / 'bc.pt'
    engine.policy_version = 8
    save_checkpoint(bc_only, engine, identity, history_reset='clear')
    with pytest.raises(ValueError, match='PPO updates'):
        evaluation.inspect_trained_teacher_checkpoint(bc_only, cli.file_sha256(bc_only), config, ppo, contract=contract)
    invalid = torch.load(bc_only, weights_only=False)
    invalid['counters']['updates'] = 1
    invalid['counters']['environment_steps'] = 256
    invalid_path = tmp_path / 'invalid.pt'
    torch.save(invalid, invalid_path)
    with pytest.raises(ValueError, match='Adam'):
        evaluation.inspect_trained_teacher_checkpoint(invalid_path, cli.file_sha256(invalid_path), config, ppo, contract=contract)

    r0 = tmp_path / 'run'
    r0.mkdir()
    from test_dh116_privileged_cli import CPUFixtureBridge
    bridge = CPUFixtureBridge()
    short = dict(config, rollout_steps=2)
    fast = cli.PPOConfig(epochs=1, minibatches=1, lr=1e-6, target_kl=.02)
    result = cli.train_teacher(bridge, short, fast, output=r0, updates=1)
    assert result['updates'] == 1
    real = r0 / 'teacher_update_000001.pt'
    proof = evaluation.inspect_trained_teacher_checkpoint(real, cli.file_sha256(real), short, fast,
        contract=bridge.contract)
    assert proof['ppo_updates'] == 1
    assert proof['physics_sha256'] == evaluation_physics_sha256(bridge.contract)
    assert proof['native_grasp_qualification'] is False
