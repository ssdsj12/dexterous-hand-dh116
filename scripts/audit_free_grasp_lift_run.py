"""Verify continuous-mode contract, reward sum and finite PPO state."""
import json
import sys
from pathlib import Path
import torch

run=Path(sys.argv[1])
contract=json.loads((run/'tensor_contract.json').read_text())
assert contract['free_grasp_lift']['stage_flow']=='none'
assert contract['free_grasp_lift']['pose_latch'] is False
assert contract['observation_size']==160
for key in ('grasp_readiness','policy_lift_hold','constant_force_hold','close_completion_force_gate'):
    assert key not in contract,key
rows=[json.loads(line) for line in (run/'iterations.jsonl').read_text().splitlines()]
errors=[]
for row in rows:
    assert row['exploration_mode']=='continuous_free_grasp_lift'
    assert 'phase_occupancy' not in row
    assert row['force_reward_enabled_env_fraction']==1.
    assert row['downward_penalty_enabled_env_fraction']==1.
    for key in ('stage','contact_dwell','grasp_ready','close_downward'):
        assert row['reward_terms'][key]==0.,key
    error=abs(sum(row['reward_terms'].values())-row['mean_step_reward'])
    errors.append(error)
    assert error<1e-5,error
saved=torch.load(run/'tensor_ppo.pt',map_location='cpu',weights_only=False)
def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite(item)
finite(saved['policy']);finite(saved['optimizer'])
print(json.dumps(dict(iteration=saved['iteration'],timesteps=saved['timesteps'],
    observation_size=160,maximum_reward_sum_error=max(errors),
    policy_optimizer_finite=True,stage_flow='none',upward_downward_scoring_all_episode=True)))
