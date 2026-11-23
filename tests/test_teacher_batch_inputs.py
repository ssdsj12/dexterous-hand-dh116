"""Exercise batched native-source validation and PPO with explicit CPU fixtures."""

from dataclasses import replace

import pytest
import torch

from test_native_teacher_context import backend_profile
from test_native_teacher_model import cpu_model, raw_cpu_packets


def lanes(profile, grasp, n):
    for env in range(n):
        pre, rod, parameters, context = raw_cpu_packets(profile, env_id=env)
        yield pre, rod, parameters, replace(context, metadata=replace(context.metadata, grasp=grasp))


@pytest.mark.parametrize("grasp", ["pinch", "tripod", "power"])
def test_each_grasp_consumes_32_source_specific_observations_in_the_existing_ppo(backend_profile, grasp):
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_teacher_batch_inputs import pack_teacher_group

    model = cpu_model(backend_profile, grasp=grasp)[0]
    observations = pack_teacher_group(model, lanes(backend_profile, grasp, 32),
                                      backend_profile=backend_profile)
    assert observations["rod_nodes"].shape == (32, 41, 13)
    assert observations["common_valid"].sum().item() == 32 * 66
    assert torch.isnan(observations["common371"][~observations["common_valid"]]).all()
    trainer = DeformablePPOTrainer(model, PPOConfig(lr=1e-5, epochs=1, minibatches=1),
                                  rollout_steps=2, num_envs=32)
    for _ in range(2):
        issued = trainer.collect_step(observations, observations)
        assert issued["actions"].shape == (32, 24)
        trainer.record_step(torch.linspace(-1., 1., 32), torch.ones(32, dtype=torch.bool),
            torch.zeros(32), executed_actions=issued["actions"], collection_token=issued["collection_token"])
    result = trainer.update()
    assert result["samples"] == 64 and result["minibatch_updates"] == 1


def test_batch_rejects_repeating_same_lane(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_teacher_batch_inputs import pack_teacher_group

    model = cpu_model(backend_profile)[0]
    lane = next(lanes(backend_profile, "pinch", 1))
    with pytest.raises(ValueError, match="distinct.*environment"):
        pack_teacher_group(model, (lane, lane), backend_profile=backend_profile)
