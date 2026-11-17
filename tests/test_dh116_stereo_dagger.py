"""Online visited-state DAgger contract; every environment here is CPU-only."""
import copy

import pytest
import torch

from test_dh116_stereo_collection import module,FakeEnv,TinyStudent,TinyTeacher


def test_required_dagger_modules_have_explicit_red():
    for name in ('dagger','dagger_dataset','dagger_trainer'):module(name)


def test_student_executes_whole_action_and_teacher_labels_exact_visited_state():
    m=module('dagger');env=FakeEnv();student=TinyStudent();teacher=TinyTeacher()
    records=[]
    for _ in range(3):
        before=env.snapshot().clone()
        row=m.collect_step(env,student,teacher,teacher_probability=0.,schema_only=True)
        torch.testing.assert_close(teacher.calls[-1],before.privileged)
        torch.testing.assert_close(row['teacher_mean'],teacher.latent_mean(before.privileged).tanh())
        torch.testing.assert_close(row['issued_actions'],row['student_nominal'])
        assert row['action_owner']==['student']*2 and row['mode']=='dagger_student'
        records.append(row)
    assert len(env.actions)==3
    assert records[0]['identities'][0]['policy_tick']==0 and records[2]['identities'][0]['policy_tick']==2
    assert records[0]['schema_only'] is True and records[0]['native_qualified'] is False


def test_mixing_switches_whole24d_action_per_environment_without_interpolation():
    m=module('dagger');env=FakeEnv(batch=8);student=TinyStudent();teacher=TinyTeacher()
    owners=[];env.set_action_owners=lambda selected:owners.append(selected)
    row=m.collect_step(env,student,teacher,teacher_probability=.5,schema_only=True,generator=torch.Generator().manual_seed(116))
    assert set(row['action_owner'])=={'student','teacher'}
    assert owners==[row['action_owner']]
    for index,owner in enumerate(row['action_owner']):
        expected=row['teacher_mean'][index] if owner=='teacher' else row['student_nominal'][index]
        torch.testing.assert_close(row['issued_actions'][index],expected,rtol=0,atol=0)


def test_teacher_only_d0_is_reference_and_formal_dagger_requires_qualification():
    m=module('dagger');env=FakeEnv();student=TinyStudent();teacher=TinyTeacher()
    row=m.collect_step(env,student,teacher,teacher_probability=1.,schema_only=True)
    assert row['mode']=='teacher_reference' and row['dagger_student_samples']==0
    fresh=FakeEnv()
    with pytest.raises((ValueError,FileNotFoundError),match='Teacher|teacher|qualif|proof'):
        m.collect_step(fresh,student,teacher,schema_only=False)
    assert not fresh.actions


def test_dagger_aggregation_is_appendonly_unique_and_failure_states_remain_non_success():
    m=module('dagger_dataset');collector=module('dagger')
    env=FakeEnv(terminal_tick=1);student=TinyStudent();teacher=TinyTeacher()
    first=collector.collect_step(env,student,teacher,schema_only=True)
    second=collector.collect_step(env,student,teacher,schema_only=True)
    pool=m.DAggerDataset(source_sha256='f'*64,schema_only=True)
    pool.append(first);pool.append(second)
    assert len(pool)==4 and pool.payload()['successful_demonstrations'] is False
    assert pool.payload()['rows'][2]['history']['history_valid'].sum()==1
    with pytest.raises(ValueError,match='duplicate|identity'):pool.append(first)
    assert len(pool)==4


def test_dagger_rejects_reset_boundary_history_and_nonmonotonic_timestamps():
    m=module('dagger_dataset');collector=module('dagger');env=FakeEnv()
    first=collector.collect_step(env,TinyStudent(),TinyTeacher(),schema_only=True)
    second=collector.collect_step(env,TinyStudent(),TinyTeacher(),schema_only=True)
    second['identities'][0]['simulation_timestamp']=0.
    second['privileged_identities'][0]['simulation_timestamp']=0.
    second['execution'][0]['simulation_timestamp_start']=0.
    pool=m.DAggerDataset(source_sha256='f'*64,schema_only=True);pool.append(first)
    with pytest.raises(ValueError,match='time|order|clock'):pool.append(second)


def test_dagger_trainer_never_inserts_auxiliary_rows_into_ppo_storage():
    m=module('dagger_trainer');dataset_module=module('dagger_dataset');collector=module('dagger')
    env=FakeEnv();student=TinyStudent();teacher=TinyTeacher()
    pool=dataset_module.DAggerDataset(source_sha256='f'*64,schema_only=True)
    pool.append(collector.collect_step(env,student,teacher,schema_only=True))
    trainer=m.StereoDAggerTrainer(student,schema_only=True)
    before=teacher.weight.detach().clone()
    reports=trainer.train(pool,batch_size=1,epochs=1)
    assert len(reports)==2 and all(r['native_qualified'] is False for r in reports)
    torch.testing.assert_close(teacher.weight,before)


def test_reset_history_contamination_is_rejected_at_actual_step_boundary():
    env=FakeEnv(terminal_tick=1);original=env.step
    def contaminated(action):
        result=original(action)
        result.next_state.sensor_history.history_valid[:,0]=True
        result.next_state.sensor_history.dt[:,0]=1/60
        result.next_state.sensor_history.proprio[:,0]=result.next_state.sensor_history.proprio[:,-1]
        return result
    env.step=contaminated
    with pytest.raises(ValueError,match='reset history|earlier episode'):
        module('dagger').collect_step(env,TinyStudent(),TinyTeacher(),schema_only=True)
