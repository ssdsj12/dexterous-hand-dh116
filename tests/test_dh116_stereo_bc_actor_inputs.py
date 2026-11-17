"""CPU composition regressions; no native or successful-data qualification."""
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_bc import StereoBCTrainer
from go2_pvcnn.control.dh116.tabletop_stereo_dagger_trainer import StereoDAggerTrainer
from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import (
    batch_dataset, build_dataset,
)
from go2_pvcnn.control.dh116.tabletop_stereo_contract import state_contract
from test_dh116_stereo_collection import record
from test_dh116_stereo_estimator import history

torch.set_num_threads(1)

MODES = ('latent', 'estimated_task_state', 'hybrid')


def actor_for(mode):
    row = state_contract()['dimensions'][116]
    return make_stereo_actor(dict(
        embed_dim=32,
        history=4,
        num_heads=4,
        num_layers=1,
        actor_input_mode=mode,
        task_state_config=dict(
            allowlist=[116],
            source_qualification={
                key: True for key in row['required_validity_keys']
            },
        ),
    ))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('trainer_type', [StereoBCTrainer, StereoDAggerTrainer])
def test_supervised_step_uses_selected_actor_inputs_once(mode, trainer_type):
    torch.manual_seed(116)
    data = build_dataset(
        [record()], source_sha256='f' * 64, schema_only=True,
    )
    batch = batch_dataset(data, [0])
    actor = actor_for(mode).eval()
    actor.estimator.encoder.backbone.train()
    original_modes = [module.training for module in actor.modules()]
    before = actor.action_net[-1].weight.detach().clone()
    log_std = actor.log_std.detach().clone()
    calls = []
    hook = actor.estimator.register_forward_hook(
        lambda module, arguments, output: calls.append(output)
    )
    try:
        report = trainer_type(
            actor, {'lambda_state': 0.}, schema_only=True,
        ).step(batch)
    finally:
        hook.remove()

    assert len(calls) == 1
    assert report['steps'] == 1
    assert report['gradient_norms']['action'] > 0
    assert report['native_qualified'] is False
    assert report['ppo_storage_modified'] is False
    assert not torch.equal(before, actor.action_net[-1].weight)
    assert torch.equal(actor.log_std, log_std)
    assert actor.log_std.grad is None
    assert [module.training for module in actor.modules()] == original_modes

    if mode != 'latent':
        gradient = actor.estimator.heads['privileged_state'].weight.grad
        assert gradient is not None
        assert gradient[116].abs().sum() > 0
        assert gradient[180:200].abs().sum() == 0


def test_latent_feature_refactor_preserves_exact_action_math():
    torch.manual_seed(116)
    actor = actor_for('latent').eval()
    sensors = history(1)
    with torch.no_grad():
        estimates = actor.estimator(sensors)
        expected = actor.action_net(estimates['latent'])
        features = actor.features_from_estimates(estimates, sensors)
        actual = actor.latent_mean(sensors)

    assert torch.equal(features, estimates['latent'])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
