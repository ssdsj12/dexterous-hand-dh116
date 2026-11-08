import json
import numpy as np
import pytest
import torch
from tools.audit_vision_paired_bias import Arm, check_replay, compare, mean_physical_z

from go2_pvcnn.control.dh116.tabletop_vision_paired_bias import (
    neutralize_right_z, paired_identity, require_identical_pair, summarize_trials,
)


def test_replay_changes_only_right_translation_z_and_preserves_input():
    actions = torch.linspace(-1, 1, 48).reshape(2, 24)
    baseline = actions.clone()
    result = neutralize_right_z(actions)
    keep = torch.arange(24) != 17
    assert torch.equal(result[:, keep], baseline[:, keep])
    assert torch.equal(result[:, 17], torch.zeros(2))
    assert torch.equal(actions, baseline)


def test_rejects_reordered_actual_or_authored_initial_states():
    rows = [{'trial_id': str(i), 'authored_state_sha256': f'authored{i}',
             'readback_state_sha256': f'actual{i}'} for i in range(2)]
    a = paired_identity('checkpoint', 'manifest', 9, {'actual_states': rows})
    b = paired_identity('checkpoint', 'manifest', 9, {'actual_states': rows[::-1]})
    with pytest.raises(ValueError, match='paired identity'):
        require_identical_pair(a, b)
    require_identical_pair(a, dict(a))


def test_summary_excludes_unfinished_and_has_conditional_slip_denominator():
    rows = [{'grasp_style':'g','terminal':terminal,'valid':True,'lift_peak_m':lift,
             'hold_max_s':0.,'slip_max_m':None,'opposed_samples':1,'contact_valid_samples':2,
             'first_terminal_success':False,'failure_reason':'joint_limit',
             'task_return':reward,'right_palm_final_dz_m':0.01}
            for terminal,lift,reward in [(True,.007,2.),(False,.1,90.)]]
    value = summarize_trials(rows)['g']
    assert value['completed']==1 and value['expected']==2
    assert value['lift_mm']['p95']==pytest.approx(7)
    assert value['joint_limit']==1 and value['native_success_rate']==0
    assert value['opposed_contact_fraction']==.5 and value['slip_mm']['count']==0
    assert value['mean_episode_return']==2.


def test_intervention_checks_all_other_channels_and_zero_tail():
    baseline = np.full((2,24),.1,dtype=np.float32)
    observed = np.vstack([baseline,np.zeros((1,24),dtype=np.float32)])
    observed[:,17] = 0
    assert check_replay(baseline, observed, 19, zero_z=True)==3
    observed[-1,12] = .001
    with pytest.raises(ValueError,match='fixed-channel'):
        check_replay(baseline, observed, 19, zero_z=True)
    baseline[0,17] = -.25
    assert check_replay(baseline,baseline.copy(),12,zero_z=False)==2


def test_common_z_uses_physical_frame_and_partial_tick_weight():
    actions=np.zeros((2,24),np.float32)
    actions[0,14]=1.
    actions[1,17]=-1.
    assert mean_physical_z(actions,10)==pytest.approx([.8,-.2])


def _write_pair_fixture(path, arm, end):
    path.mkdir()
    state = {'trial_id':'a','authored_state_sha256':'w','readback_state_sha256':'r'}
    (path/'evaluation_reset_readback.json').write_text(json.dumps({'actual_states':[state]}))
    fields = ['live','done','native_success','measurement_valid','lift_m','qualified_hold_s',
              'slip_m','slip_valid','bilateral_opposed_contact','contact_valid','failure',
              'right_palm_dz_m','physics_step','timestamp_s']
    samples = np.zeros((16,1,len(fields)),np.float64)
    samples[:,:,0]=samples[:,:,3]=samples[:,:,9]=1
    samples[:,:,4]=.002; samples[:,:,6]=np.nan
    samples[end-1,0,1]=1
    samples[end:,0,4]=2.  # Post-reset observation must never enter the trial peak.
    samples[:,0,12]=np.arange(16)+12
    samples[:,0,13]=np.arange(16)/480
    np.savez(path/'native_samples.npz',samples=samples,fields=fields)
    actions=np.ones((2,1,24),np.float32)/2
    actions[1]=0  # M0 ends on tick 0, so its defined tail is all zero.
    if arm=='right_z_zero': actions[:,:,17]=0
    np.save(path/'issued_actions.npy',actions)
    row=dict(env_id=0,trial_id='a',terminal=True,physical_samples=end,terminal_step=end+11,
             first_terminal_success=False,failure_reason='timeout',grasp_style='pinch',task_return=0.)
    result=dict(complete=True,identity=dict(states=[['a','w','r']]),arm=arm,trials=[row])
    (path/'paired_results.json').write_text(json.dumps(result))
    return Arm(path)


def test_first_terminal_pair_and_control_tail(tmp_path):
    baseline=_write_pair_fixture(tmp_path/'baseline','m0',6)
    perturbed=_write_pair_fixture(tmp_path/'perturbation','right_z_zero',14)
    control=_write_pair_fixture(tmp_path/'replay','replay',8)
    result=compare(baseline,perturbed,schedule=control)
    row=result['trials'][0]
    assert row['common_physics_steps']==8 and row['extension_beyond_m0_control_steps']==1
    assert row['full'][1]['lift_mm']==2
    assert result['groups']['pinch']['common'][0]['slip_mm']['count']==0
