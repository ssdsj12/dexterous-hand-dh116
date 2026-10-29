import copy
import pytest
import torch
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('resize_ppo_checkpoint',Path(__file__).resolve().parents[1]/'scripts/resize_ppo_checkpoint.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
resize=module.resize


def test_resize_keeps_weights_optimizer_and_training_progress():
    source=dict(contract=dict(num_envs=10,minibatches=8,rollout_steps=24),
        policy={'weight':torch.arange(12).reshape(3,4)},optimizer={'state':{0:{'exp_avg':torch.ones(4)}}},
        iteration=3,timesteps=720,episodes=torch.arange(10),successes=torch.zeros(10),
        recovered=torch.ones(10),reached_counts=torch.ones(4),transition_counts=torch.ones(16))
    original=copy.deepcopy(source)
    out,report=resize(source,2)
    assert out['contract']['num_envs']==2 and source['contract']['num_envs']==10
    assert out['iteration']==3 and out['timesteps']==720
    torch.testing.assert_close(out['policy']['weight'],original['policy']['weight'])
    torch.testing.assert_close(out['optimizer']['state'][0]['exp_avg'],torch.ones(4))
    torch.testing.assert_close(out['episodes'],torch.tensor([0,1]))
    assert out['successes'].shape==(2,) and out['reached_counts'].shape==(4,)
    assert report['per_env_counters']=='first_n_slots_retained; physical_scenes_reset'


def test_resize_rejects_growth_and_invalid_counter_shapes():
    with pytest.raises(ValueError):resize(dict(contract=dict(num_envs=2)),3)
    with pytest.raises(ValueError):resize(dict(contract=dict(num_envs=4),episodes=torch.zeros(3)),2)
