import json
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
monitor = json.loads((root / 'monitor.json').read_text())
contract = json.loads((root / 'tensor_contract.json').read_text())
checkpoint = torch.load(root / 'tensor_ppo.pt', map_location='cpu', weights_only=False)
expected_size=198 if 'force_region_control' in contract else (188 if 'stable_lift_reward' in contract else 186)
assert contract['observation_size'] == expected_size, contract['observation_size']
assert contract['policy_lift_hold']['version'] == 'policy_lift_hold_v1'
assert monitor['iteration'] >= 1
assert checkpoint['optimizer']['state'], 'No optimizer update saved'
for name, value in checkpoint['policy'].items():
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), name
for state in checkpoint['optimizer']['state'].values():
    for name, value in state.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), name
print(json.dumps({key: monitor.get(key) for key in (
    'status', 'iteration', 'timesteps', 'successes', 'phase_reached_episodes')}))
if 'stable_lift_reward' in contract:
    assert contract['stable_lift_reward']['version']=='measured_stable_lift_reward_v1'
    if 'reward_terms' in monitor:
        assert 'stable_lift_hold' in monitor['reward_terms']
if 'force_region_control' in contract:
    assert contract['force_action_range_n']==[2.,12.]
    assert contract['constant_force_hold']['reference_after_close']=='immutable_including_recovery'
    assert contract['force_region_control']['contact_region_weights']==dict(below=2.,above=.5,invalid=0.)
    assert 2.<=monitor['force_control']['requested_min_n']<=monitor['force_control']['requested_max_n']<=12.
if 'phase_force_reward' in contract:
    assert 'grasp_upward_force' in monitor['reward_terms']
    assert 'lift_upward_force' in monitor['reward_terms']
if 'initial_contact_validation' in contract:
    initial=json.loads((root/'initial_contact_validation.json').read_text())
    assert initial['all_pad_references_below_axis']
    assert initial['minimum_measured_clearance_m']>=.0003
print(f'Observation contract {expected_size}; policy and optimizer finite; PPO update saved')
