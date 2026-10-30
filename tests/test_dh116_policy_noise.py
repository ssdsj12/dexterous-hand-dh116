import math
import torch
import pytest
from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm, observations, checkpoint_contract_changes


def test_entropy_only_updates_cannot_expand_noise_beyond_action_scale():
    alg=make_algorithm(observations(torch.zeros(4,150)),2)
    with torch.no_grad(): alg.policy.log_std.fill_(math.log(.99))
    for group in alg.optimizer.param_groups: group['lr']=.1
    for _ in range(12):
        alg.optimizer.zero_grad()
        alg.policy.act(observations(torch.zeros(4,150)))
        (-alg.policy.entropy.mean()).backward()
        alg.optimizer.step()
        assert alg.policy.log_std.exp().max()<=1.+1e-6


def test_resume_projection_preserves_network_and_unaffected_adam_coordinates():
    from go2_pvcnn.control.dh116.tabletop_policy_noise import project_policy_noise
    alg=make_algorithm(observations(torch.zeros(4,150)),2)
    p=alg.policy.log_std
    with torch.no_grad(): p.copy_(torch.tensor([math.log(16.),math.log(.001)]+[math.log(.3)]*18))
    state=alg.optimizer.state[p]
    state.update(step=torch.tensor(5.),exp_avg=torch.ones_like(p),exp_avg_sq=torch.ones_like(p)*2)
    before={k:v.clone() for k,v in alg.policy.state_dict().items()}
    changed=project_policy_noise(alg)
    assert changed.tolist()==[True,True]+[False]*18
    assert torch.allclose(p[:2].exp(),torch.tensor([1.,.05]))
    assert not state['exp_avg'][:2].any() and not state['exp_avg_sq'][:2].any()
    assert torch.equal(state['exp_avg'][2:],torch.ones(18)) and state['step']==5
    assert all(torch.equal(v,alg.policy.state_dict()[k]) for k,v in before.items() if k!='log_std')
    assert not project_policy_noise(alg).any()


def test_nonfinite_noise_is_rejected_not_hidden_by_projection():
    from go2_pvcnn.control.dh116.tabletop_policy_noise import project_policy_noise
    alg=make_algorithm(observations(torch.zeros(4,150)),2)
    for value in (float('inf'),float('-inf'),float('nan')):
        with torch.no_grad():alg.policy.log_std.fill_(value)
        with pytest.raises(RuntimeError,match='nonfinite'):project_policy_noise(alg)


def test_noise_contract_migration_is_exact_and_one_way():
    from go2_pvcnn.control.dh116.tabletop_policy_noise import CONTRACT
    old=dict(tensor_training='dh116-rsl-tensor-ppo-v1',policy='bounded_gaussian_mean_v1',
             action='20D_piecewise_joint_envelope_palm_load_v1')
    new=dict(old,policy_noise=CONTRACT)
    assert checkpoint_contract_changes(old,new)==['bounded_policy_noise_v1']
    assert checkpoint_contract_changes(new,new)==[]
    for source,target in [(new,old),(old,dict(new,policy_noise=dict(CONTRACT,max_std=2.))),
                          (old,dict(new,action='different')),
                          (dict(old,policy='other'),dict(new,policy='other'))]:
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(source,target)


def test_real_ppo_update_projects_every_minibatch_and_preserves_raw_likelihood_samples():
    obs=observations(torch.zeros(8,150));alg=make_algorithm(obs,4)
    with torch.no_grad():alg.policy.log_std.fill_(0.)
    raw=[]
    with torch.inference_mode():
        for _ in range(4):
            a=alg.act(obs);raw.append(a.clone())
            # Environment evaluates clipped actions; likelihood uses raw samples.
            alg.process_env_step(obs,-a.clamp(-1,1).square().sum(1),torch.zeros(8,dtype=torch.bool),{})
        alg.compute_returns(obs)
    assert torch.equal(alg.storage.actions,torch.stack(raw))
    assert (alg.storage.actions.abs()>1).any()
    observed=[]
    def audit(*_):
        std=alg.policy.log_std.exp()
        assert std.min()>=.05-1e-6 and std.max()<=1.+1e-6
        observed.append(True)
    alg.optimizer.register_step_post_hook(audit)
    result=alg.update()
    assert len(observed)==40
    assert all(math.isfinite(v) for v in result.values())


def test_projected_noise_can_learn_away_from_upper_boundary():
    alg=make_algorithm(observations(torch.zeros(4,150)),2)
    with torch.no_grad():alg.policy.log_std.fill_(0.)
    alg.optimizer.zero_grad();alg.policy.log_std.sum().backward();alg.optimizer.step()
    assert alg.policy.log_std.exp().max()<1.
