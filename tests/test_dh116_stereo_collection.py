"""CPU-only executed collection fixtures; never native success qualification."""
import copy
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from test_dh116_stereo_estimator import sensor
from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistory


PREFIX='go2_pvcnn.control.dh116.tabletop_stereo_'


def module(name):
    name=PREFIX+name
    assert importlib.util.find_spec(name),'independent stereo collection module missing: '+name
    return importlib.import_module(name)


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(24)*.1);self.calls=[]
    def latent_mean(self,history):
        assert not isinstance(history,dict)
        self.calls.append(history.proprio.detach().clone())
        return self.weight[None].expand(len(history),-1)+history.proprio[:,-1,:1]
    def deterministic(self,history): return self.latent_mean(history).tanh()


class TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(24)*.2)
        self.log_std=torch.nn.Parameter(torch.full((24,),-2.3));self.calls=[]
    def latent_mean(self,privileged): return self.weight[None].expand(len(privileged),-1)+privileged[:,:1]
    def deterministic(self,privileged):
        self.calls.append(privileged.detach().clone());return self.latent_mean(privileged).tanh()


class FakeEnv:
    evidence_scope='cpu_fixture'
    def __init__(self,batch=2,terminal_tick=None):
        self.n=batch;self.tick=0;self.reset_id=0;self.terminal_tick=terminal_tick
        self.store=StereoHistory(batch);self.actions=[]
        self._refresh(torch.zeros(batch,24))
    def _refresh(self,issued):
        frame=sensor(self.n);frame.proprio[:,0]=self.tick*.01
        frame.proprio[:,38:]=issued
        self.history=self.store.update(frame)
    def identities(self):
        return [dict(run_uuid='fixture-run',env_id=i,episode_uid=f'episode-{i}-{self.reset_id}',
            reset_id=self.reset_id,policy_tick=self.tick,physics_tick_start=8*self.tick,
            physics_tick_end=8*self.tick,simulation_timestamp=self.tick/60,frame_id=100+self.tick,
            calibration_sha256='c'*64,physical_initial_condition_sha256=('a' if i==0 else 'b')*64,
            source_episode_group=f'fixture-source-{i}-{self.reset_id}') for i in range(self.n)]
    def snapshot(self):
        m=module('teacher_collector');privileged=torch.zeros(self.n,371)
        privileged[:,0]=self.tick*.05+torch.arange(self.n)*.1
        return m.PreActionState(self.history.clone(),privileged,
            torch.zeros(self.n,371,dtype=torch.bool),self.identities())
    def step(self,issued):
        m=module('teacher_collector');before=self.tick
        self.actions.append(issued.detach().clone());self.tick+=1
        done=torch.full((self.n,),self.terminal_tick==self.tick,dtype=torch.bool)
        if bool(done.any()):
            self.reset_id+=1;self.store.reset(done);issued_next=torch.zeros_like(issued)
        else: issued_next=issued
        self._refresh(issued_next)
        info=dict(first_episode=torch.full((self.n,),self.reset_id<=1,dtype=torch.bool),
            recovered=torch.zeros(self.n,dtype=torch.bool),restored=torch.zeros(self.n,dtype=torch.bool),
            success=torch.zeros(self.n,dtype=torch.bool),evidence_scope='cpu_fixture',
            execution=[dict(physics_tick_start=before*8,physics_tick_end=self.tick*8,
                simulation_timestamp_start=before/60,simulation_timestamp_end=self.tick/60) for _ in range(self.n)])
        return m.StepResult(self.snapshot(),torch.ones(self.n),done,info,issued.clone(),
            {'joint_position_rad':torch.ones(self.n,12)*self.tick})


def record(env=None):
    env=FakeEnv() if env is None else env
    return module('teacher_collector').collect_step(env,TinyStudent(),TinyTeacher(),schema_only=True)


def test_required_collection_modules_have_explicit_red():
    for name in ('teacher_collector','supervision_dataset','bc'): module(name)


def test_teacher_collection_owns_real_execution_and_preserves_pre_action_state():
    env=FakeEnv();teacher=TinyTeacher();student=TinyStudent();before=env.snapshot().clone()
    row=module('teacher_collector').collect_step(env,student,teacher,schema_only=True,teacher_probability=1.)
    assert len(env.actions)==1
    torch.testing.assert_close(row['teacher_mean'],teacher.deterministic(before.privileged))
    torch.testing.assert_close(row['issued_actions'],env.actions[0])
    assert not row['teacher_mean'].requires_grad and row['teacher_sample'] is None
    assert row['action_owner']==['teacher','teacher'] and row['mode']=='teacher_reference'
    assert row['identities'][0]['physics_tick_start']==0
    assert row['execution'][0]['physics_tick_end']==8
    assert row['applied_targets']['joint_position_rad'].shape==(2,12)
    assert row['schema_only'] is True and row['native_qualified'] is False
    assert row['sensor_history']['proprio'][:,-1,0].eq(0.).all()


def test_collection_rejects_stale_or_privileged_identity_mismatch_before_step():
    env=FakeEnv();original=env.snapshot
    def stale():
        state=original();state.sensor_history.image_age[:,-1]=1.;return state
    env.snapshot=stale
    with pytest.raises(ValueError,match='stale|age'): record(env)
    assert len(env.actions)==0
    def mismatch():
        state=original();state.privileged_identities=copy.deepcopy(state.identities)
        state.privileged_identities[0]['frame_id']+=1;return state
    env.snapshot=mismatch
    with pytest.raises(ValueError,match='align|identity'): record(env)
    assert len(env.actions)==0


def test_schema_dataset_is_exclusive_and_cannot_be_loaded_as_native_success(tmp_path):
    m=module('supervision_dataset');env=FakeEnv();rows=[record(env),record(env)]
    data=m.build_dataset(rows,source_sha256='f'*64,schema_only=True)
    assert data['schema_only'] is True and len(data['rows'])==4
    path=tmp_path/'pool.pt';m.save_dataset(data,path)
    with pytest.raises(FileExistsError):m.save_dataset(data,path)
    with pytest.raises(ValueError,match='schema|native|fixture'):m.load_dataset(path)
    loaded=m.load_dataset(path,allow_schema_only=True)
    batch=m.batch_dataset(loaded,[0,1,2])
    assert batch['history'].proprio.shape==(3,4,62)
    assert batch['teacher_actions'].shape==(3,24)
    assert batch['teacher_actions'].requires_grad is False
    split=m.split_by_initial_condition(data,heldout_initial_condition_sha256={'b'*64})
    assert len(split['train_indices'])==len(split['heldout_indices'])==2


@pytest.mark.parametrize('field', ['action_contract_sha256','state_schema_sha256'])
def test_dataset_rejects_contract_mismatch(field):
    m=module('supervision_dataset');data=m.build_dataset([record()],source_sha256='f'*64,schema_only=True)
    data['contract'][field]='0'*64
    with pytest.raises(ValueError,match='contract|schema|SHA'):m.validate_dataset(data,allow_schema_only=True)


def test_dataset_rejects_duplicate_clock_and_success_without_native_proof():
    m=module('supervision_dataset');row=record()
    with pytest.raises(ValueError,match='duplicate|identity|ordered'):
        m.build_dataset([row,row],source_sha256='f'*64,schema_only=True)
    with pytest.raises(ValueError,match='qualified|native|proof|schema'):
        m.build_dataset([row],source_sha256='f'*64,schema_only=False)


def test_bc_uses_detached_labels_and_separate_optimizer_at_clean_ppo_boundary():
    m=module('bc');data_module=module('supervision_dataset')
    data=data_module.build_dataset([record()],source_sha256='f'*64,schema_only=True)
    batch=data_module.batch_dataset(data,[0,1]);actor=TinyStudent()
    ppo=SimpleNamespace(clean_boundary=False,storage=object())
    trainer=m.StereoBCTrainer(actor,ppo_trainer=ppo,schema_only=True)
    before=actor.weight.detach().clone();storage=ppo.storage
    with pytest.raises(RuntimeError,match='PPO|boundary'):trainer.step(batch)
    ppo.clean_boundary=True;report=trainer.step(batch)
    assert not torch.equal(actor.weight,before) and report['steps']==1
    assert ppo.storage is storage and report['native_qualified'] is False
    assert report['gradient_norms']['action']>0


def test_bc_actual_stereo_estimator_updates_from_minibatch():
    m=module('bc');data_module=module('supervision_dataset')
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStudentActor
    data=data_module.build_dataset([record()],source_sha256='f'*64,schema_only=True)
    actor=StereoStudentActor(embed_dim=32);trainer=m.StereoBCTrainer(actor,schema_only=True)
    report=trainer.step(data_module.batch_dataset(data,[0,1]))
    assert report['losses']['action']>=0 and report['gradient_norms']['action']>0


def test_schema_success_flag_and_bc_batch_tampering_are_rejected():
    m=module('supervision_dataset');data=m.build_dataset([record()],source_sha256='f'*64,schema_only=True)
    poisoned=copy.deepcopy(data);poisoned['successful_demonstrations']=True
    with pytest.raises(ValueError,match='schema|success'):m.validate_dataset(poisoned,allow_schema_only=True)
    batch=m.batch_dataset(data,[0,1]);batch['teacher_actions'][0,0]=-.9
    with pytest.raises(ValueError,match='admitted|batch'):
        module('bc').StereoBCTrainer(TinyStudent(),schema_only=True).step(batch)


def test_actual_temporal_aux_requires_correspondence_and_backpropagates_when_present():
    m=module('supervision_dataset');env=FakeEnv();record(env);row=record(env)
    row['extra_labels']['velocity']=torch.zeros(2,128,3)
    row['label_validity']['velocity']=torch.ones(2,128,3,dtype=torch.bool)
    row['label_validity']['temporal_correspondence']=torch.ones(2,128,dtype=torch.bool)
    data=m.build_dataset([row],source_sha256='f'*64,schema_only=True)
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStudentActor
    trainer=module('bc').StereoBCTrainer(StereoStudentActor(embed_dim=32),
        config={'lambda_temporal':.01},schema_only=True)
    report=trainer.step(m.batch_dataset(data,[0,1]))
    assert report['gradient_norms']['temporal']>0 and report['losses']['temporal']>=0
