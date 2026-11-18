"""Audit the active PPO reward configuration and saved rollout summaries."""
import json
import math
import sys
from pathlib import Path
import torch

run=Path(sys.argv[1])
expected={'grasp_acquisition','grasp_ready','lift','lift_milestone','stable_lift_hold',
          'force_progress','contact','downward_force','failure','success'}
checkpoint=torch.load(run/'tensor_ppo.pt',map_location='cpu',weights_only=False)
contract=checkpoint['contract']
for key in ('grasp_acquisition','grasp_shaping','stable_lift_reward','vertical_force_reward'):
    assert key in contract, f'Missing active reward contract: {key}'
assert contract['observation_size']==(198 if 'force_region_control' in contract else 188)
if 'force_region_control' in contract:
    expected|={'force_tracking','contact_region'}
if 'phase_force_reward' in contract:
    expected|={'grasp_upward_force','lift_upward_force'}
count=0
max_error=0.
for line in (run/'iterations.jsonl').read_text().splitlines():
    row=json.loads(line)
    terms=row['reward_terms']
    assert expected<=terms.keys(),expected-terms.keys()
    total=sum(terms.values())
    assert math.isfinite(total)
    error=abs(total-row['mean_step_reward'])
    assert math.isclose(total,row['mean_step_reward'],abs_tol=2e-6,rel_tol=2e-6), (row['iteration'],error)
    max_error=max(max_error,error)
    count+=1
assert count
for name,value in checkpoint['policy'].items():
    if torch.is_tensor(value):assert torch.isfinite(value).all(),name
print(json.dumps(dict(completed_updates=checkpoint['iteration'],audited_rollouts=count,
    all_reward_terms_present=True,max_reward_sum_error=max_error,
    observation_size=contract['observation_size'],reward_terms=sorted(expected))))
