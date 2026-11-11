import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig, PPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_teacher import PrivilegedTeacher
from go2_pvcnn.control.dh116 import tabletop_privileged_cli as cli


def engine():
    torch.manual_seed(116)
    e = PPOTrainer(PrivilegedTeacher(), PPOConfig(epochs=2, minibatches=2,
        target_kl=.02, kl_backtrack_steps=3), rollout_steps=2, num_envs=2)
    for _ in range(2):
        e.collect_step(torch.ones(2,371), torch.ones(2,371))
        e.record_step(torch.ones(2), torch.zeros(2,dtype=torch.bool), torch.zeros(2))
    return e


@pytest.mark.parametrize('crash', [False, True])
def test_guard_restores_rng_at_each_rejected_candidate_and_exception(crash):
    e = engine()
    for p in e.model.parameters(): p.grad = torch.ones_like(p)
    before = copy.deepcopy(e.model.state_dict())
    adam = copy.deepcopy(e.optimizer.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    seen = []
    def rejected(batch, indices):
        seen.append((torch.rand(3).tolist(), random.random(), float(np.random.rand())))
        if crash: raise RuntimeError('explicit audit failure')
        return 10.
    e._minibatch_exact_kl = rejected
    if crash:
        with pytest.raises(RuntimeError,match='audit failure'): e._optimizer_step(e.rollout.batch())
    else:
        lr, attempts = e._optimizer_step(e.rollout.batch())
        assert lr is None and len(attempts)==4 and all(x==seen[0] for x in seen)
    assert torch.equal(torch.get_rng_state(),cpu_rng)
    assert random.getstate()==py_rng
    assert np.array_equal(np.random.get_state()[1],np_rng[1])
    assert e.optimizer.state_dict()==adam
    assert all(torch.equal(v,e.model.state_dict()[k]) for k,v in before.items())


def test_epoch_trace_reports_accepted_full_rollout_policy_and_three_action_groups():
    e = engine()
    batch=e.rollout.batch()
    result=e.update()
    trace=result['epoch_trace']
    assert trace and len(trace)<=2
    assert sum(x['attempted_minibatches'] for x in trace)==len(result['minibatch_trace'])
    assert sum(x['accepted_minibatches'] for x in trace)==result['minibatch_updates']
    assert all(x['exact_full_rollout_kl']<=.02 for x in trace)
    assert all(math.isfinite(x['sample_weighted_pre_step_entropy']) for x in trace)
    assert trace[-1]['exact_full_rollout_kl']==pytest.approx(e._minibatch_exact_kl(batch,torch.arange(4)))
    groups=trace[-1]['action_groups']
    assert set(groups)=={'fingers12','translation6','rotation6'}
    assert [len(groups[k]['latent_std']) for k in groups]==[12,6,6]
    assert trace[-1]['policy_epoch_index']==len(trace)-1


def test_formal_trainer_guard_config_is_opt_in_and_bounded128_capture_is_supported(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'configs/dh116_training/privileged_teacher_ppo_guarded_v2.json'
    assert path.exists(), 'formal opt-in guarded trainer config missing'
    cfg,ppo=cli.load_config(path)
    assert cfg['rollout_steps']==128 and ppo.target_kl==.02 and ppo.kl_backtrack_steps==3
    assert cfg['reward_profile']=='teacher_r0_existing'
    assert cli.main('train',['--config',str(path),'--num-envs','2','--updates','1',
        '--record-diagnostic-rollout','--check-config'])==0


def test_128_diagnostic_saves_owned_rollout_and_aggregated_three_groups(tmp_path):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract, ACTION_NAME, LEGACY_OBSERVATION_VERSION
    class Bridge:
        n=2
        device=torch.device('cpu')
        contract=bind_contract(dict(action_size=24,action=ACTION_NAME,observation_size=351,observation_version=LEGACY_OBSERVATION_VERSION))
        critic=torch.zeros(2,371)
        env=SimpleNamespace(successes=torch.zeros(2,dtype=torch.long),episodes=torch.zeros(2,dtype=torch.long))
        def step(self, a):
            reward=-a.square().mean(-1)
            return self.critic.clone(),reward,torch.zeros(2,dtype=torch.bool),{'terms':{'fixture':reward}}
    cfg,_=cli.load_config(Path(__file__).resolve().parents[1]/'configs/dh116_training/privileged_teacher_ppo_v1.json')
    cfg['rollout_steps']=128
    result=cli.train_teacher(Bridge(),cfg,PPOConfig(epochs=1,minibatches=2,target_kl=.02,kl_backtrack_steps=3),
                            output=tmp_path,updates=1,record_diagnostic_rollout=True)
    b=torch.load(tmp_path/'diagnostic_rollout.pt',weights_only=False)
    assert b['batch']['actions'].shape==(256,24)
    row=json.loads((tmp_path/'iterations.jsonl').read_text())
    groups=row['action_diagnostics']['semantic_groups']
    assert set(groups)=={'fingers12','translation6','rotation6'}
    assert sum(len(x['indices']) for x in groups.values())==24
    assert row['gae_diagnostics']['terminal_count']==0
    assert row['checkpoint_roundtrip']['optimizer_equal']


def test_legacy_mpc_audit_distinguishes_matching_actions_from_missing_privileged_states():
    from go2_pvcnn.control.dh116 import tabletop_privileged_dataset as data
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import canonical_sha256
    audit=getattr(data,'audit_expert_compatibility',None)
    assert callable(audit), 'reproducible source compatibility audit required'
    payload={'version':'dh116-vision-supervised-action-dataset-v1',
        'observations':torch.zeros(2,3844),'actions':torch.zeros(2,24),
        'visual_contract':{'action_contract':ACTION_CONTRACT},
        'episodes':{'0':{'native_success':True}}}
    result=audit(payload,expected_identity={'action_contract_sha256':canonical_sha256(ACTION_CONTRACT),
        'state_contract_sha256':'a'*64})
    assert result['action_contract_matches']
    assert result['observations_shape']==[2,3844]
    assert not result['bc_admitted']
    assert 'missing_pre_action_privileged_371' in result['blockers']
    assert result['requires_recollection']


def test_matching_width_alone_cannot_admit_mpc_expert():
    from go2_pvcnn.control.dh116 import tabletop_privileged_dataset as data
    audit=getattr(data,'audit_expert_compatibility',None)
    assert callable(audit)
    result=audit({'observations':torch.zeros(2,371),'actions':torch.zeros(2,24)},expected_identity={})
    assert not result['bc_admitted'] and not result['action_contract_matches']
    assert 'missing_or_mismatched_action_contract' in result['blockers']
    assert 'missing_or_mismatched_state_contract' in result['blockers']


def test_guarded_resume_repeats_policy_adam_rng_and_epoch_traces(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_privileged_checkpoint as cp
    e=engine();e.update()
    identity=cp.build_identity(e.config,state_contract_sha256='a'*64,source_sha256={})
    p=tmp_path/'guarded.pt';cp.save_checkpoint(p,e,identity,history_reset='clear')
    def run(model):
        for _ in range(2):
            model.collect_step(torch.ones(2,371),torch.ones(2,371))
            model.record_step(torch.ones(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2))
        return model.update()
    first=run(e)
    recipient=engine()
    # engine() has pending rollout data; use a clean same-configuration trainer.
    recipient=PPOTrainer(PrivilegedTeacher(),e.config,rollout_steps=2,num_envs=2)
    cp.load_checkpoint(p,recipient,identity,history_reset='clear')
    second=run(recipient)
    assert first==second
    assert all(torch.equal(x,recipient.model.state_dict()[k]) for k,x in e.model.state_dict().items())
    for k,v in e.optimizer.state_dict()['state'].items():
        assert all(torch.equal(x,recipient.optimizer.state_dict()['state'][k][name]) for name,x in v.items())
