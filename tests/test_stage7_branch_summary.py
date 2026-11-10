"""Synthetic stdlib CPU fixtures, never evidence of native success."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_vision_stage7_branch_summary.py'


def module():
    assert MODULE.is_file(), 'read-only Stage7 Student-segment summary is missing'
    spec = importlib.util.spec_from_file_location('stage7_segment_summary', MODULE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return identity(path)


def jsonl(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return identity(path)


@pytest.fixture
def files(tmp_path):
    batch = tmp_path/'batch'; batch.mkdir()
    common = {name: write(tmp_path/(name+'.json'), {'synthetic_cpu_fixture': name})
              for name in ('teacher_dataset', 'sequence_context', 'source_initial_state', 'source_collection_report')}
    models = {arm: dict(checkpoint=write(tmp_path/(arm+'.pt'), {'synthetic_cpu_fixture': arm}))
              for arm in ('E0', 'B0', 'B1', 'B2')}
    refs = [dict(source=dict(env_id=i, episode_id=i, episode_uid=f'teacher:{i}:first', run_uuid='teacher'),
                 events={event: dict(available=True, tick=2) for event in ('PRE_CONTACT', 'CONTACT', 'LIFT_ONSET', 'HOLD')})
            for i in range(4)]
    ref_id = write(tmp_path/'event_reference.json', dict(cases=refs))
    jobs = []
    for arm in models:
        for event in ('PRE_CONTACT', 'CONTACT', 'LIFT_ONSET', 'HOLD'):
            for horizon in (1, 4, 16, 60):
                name = f'branch_{arm}_{event}_{horizon:02d}'
                cfg = dict(version='dh116-stage6-native-v1', mode='branch', initial_state_tolerance=0.,
                           checkpoint=models[arm]['checkpoint'], event=event, branch_steps=horizon,
                           event_reference=ref_id, **common)
                jobs.append(dict(job_id=name, arm=arm, mode='branch', event=event, branch_steps=horizon,
                                 config=write(batch/'configs'/(name+'.json'), cfg), output=str(batch/'jobs'/name),
                                 cold_reset=True, evaluated_env_ids=[0,1,2,3], neutral_env_ids=[4,5,6,7]))
    job = jobs[0]; folder = Path(job['output']); folder.mkdir(parents=True)
    raw = []; policies = []; source_rows = []; cases = []
    for env in range(4):
        uid = f'run:{env}:0'
        for tick in range(3):
            owner = 'teacher_prefix' if tick < 2 else 'student'
            policies.append(dict(env_id=env, policy_step=tick, run_uuid='run', episode_id=0, episode_uid=uid,
                                 recorded_teacher_episode_uid=f'teacher:{env}:first', owner=owner,
                                 clock_before_action=dict(physics_step=tick*8, simulation_step_index=18+tick*8,
                                                          timestamp_s=(18+tick*8)/480)))
            for sub in range(1,9):
                step = tick*8+sub
                raw.append(dict(env_id=env, source_episode_id=str(env), episode_uid=uid, policy_step=tick,
                                physics_step=step, timestamp_s=(18+step)/480, action_owner=owner, failure_code=0,
                                payload=dict(dt_s=1/480, live=True, done=False, native_success=False,
                                             measurement_valid=True, lift_valid=True,
                                             lift_m=.09 if step < 16 else .03 if step == 16 else .01+sub*.001,
                                             qualified_hold_s=4+step/480, qualified_hold_valid=True,
                                             contact_valid=True, bilateral_opposed_contact=True,
                                             slip_m=0., slip_valid=False, prior_anchor_latched=False, slip_applicable=False)))
        source_rows.append(dict(env_id=env, episode_id=str(env), execution_episode_id=0, execution_episode_uid=uid,
                                terminal=None, observed_physics_samples=24, teacher_prefix_policy_steps_executed=2,
                                student_policy_steps_executed=1, teacher_replay_policy_steps_executed=0,
                                native_success=False, max_net_lift_m=.09, max_qualified_hold_s=4.05))
        cases.append(dict(env_id=env, episode_id=str(env), source_episode_uid=f'teacher:{env}:first',
                          source_run_uuid='teacher', event_entry_ticks={k:2 for k in refs[env]['events']}))
    raw_path=folder/'source_physics_480hz.jsonl.gz'; policy_path=folder/'live_policy_60hz.jsonl.gz'
    raw.sort(key=lambda r:(r['physics_step'],r['env_id'])); policies.sort(key=lambda r:(r['policy_step'],r['env_id']))
    report=dict(version='dh116-stage6-native-v1', mode='branch', status='bounded_branch_completed', error=None,
                native_runtime_verified=True, execution='native_isaac_tensor_world', physics_hz=480, policy_hz=60,
                ppo_updates=0, event=job['event'], branch_steps=1, execution_run_uuid='run',
                reset_clock=dict(physics_step=0,simulation_step_index=18,timestamp_s=18/480),
                evaluated_source_count=4, completed_count=0, source_rows=source_rows, source_case_definitions=cases,
                inputs=dict(checkpoint=models['E0']['checkpoint'],event_reference=ref_id,**common),
                raw_physics=jsonl(raw_path,raw))
    jsonl(policy_path,policies)
    report_path=folder/'stage6_native_report.json'
    outcome=dict(job_id=job['job_id'],arm=job['arm'],report=write(report_path,report),
                 effective_native_config=job['config'],summary=dict(source_rows=source_rows),
                 source_reset_comparison=write(folder/'source_reset_comparison.json',dict(matched=True,tolerance=0.)),
                 source_camera_warmup_comparison=write(folder/'source_camera_warmup_comparison.json',
                                                      dict(matched=True,tolerance=0.,before_clock={'x':0},after_clock={'x':0})))
    outcome_path=batch/(job['job_id']+'_outcome.json');write(outcome_path,outcome)
    plan=dict(version='dh116-stage7-finite-branch-batch-v1',arm='all',models=models,common_inputs=common,
              event_reference=ref_id,event_reference_cases=refs,jobs=jobs,bounded_job_count=64,maximum_job_count=64)
    plan_path=batch/'batch_plan.json';write(plan_path,plan)
    return dict(root=tmp_path, plan=plan, plan_path=plan_path, job=job, raw=raw, raw_path=raw_path,
                policies=policies,policy_path=policy_path,report=report,report_path=report_path,
                outcome=outcome,outcome_path=outcome_path)


def refresh(f):
    f['report']['raw_physics']=jsonl(f['raw_path'],f['raw'])
    jsonl(f['policy_path'],f['policies'])
    f['outcome']['summary']['source_rows']=copy.deepcopy(f['report']['source_rows'])
    f['outcome']['report']=write(f['report_path'],f['report'])
    write(f['outcome_path'],f['outcome'])


def summarize(f, **kwargs):
    return module().summarize_batch(identity(f['plan_path']), f['root']/'analysis', **kwargs)


def test_student_lift_peak_cannot_inherit_teacher_prefix_peak(files):
    result=summarize(files, progress=True); row=result['jobs'][0]['environments'][0]
    assert row['student_net_lift_m']['first']==pytest.approx(.011)
    assert row['student_net_lift_m']['last']==pytest.approx(.018)
    assert row['student_net_lift_m']['peak']['value']==pytest.approx(.018)
    peak=row['student_net_lift_m']['peak'];assert peak['env_id']==0 and peak['execution_episode_uid']=='run:0:0'
    assert (peak['policy_step'],peak['world_physics_step'],peak['timestamp_s'])==(2,24,42/480)
    assert peak['raw_pointer']['line_1based']==93 and peak['raw_pointer']['file']==files['report']['raw_physics']
    assert row['handoff_prefix_baseline']['value']==pytest.approx(.03)
    assert row['student_delta_from_handoff_m']['peak']==pytest.approx(-.012)
    assert row['teacher_prefix_policy_steps']==2 and row['student_policy_steps']==1
    assert row['teacher_prefix_physics_samples']==16 and row['student_physics_samples']==8
    assert row['switch_reached'] and row['ending']=='bounded_censored' and row['terminal'] is None
    assert row['per_pad_contact']['value'] is None and row['per_digit_force']['value'] is None
    assert 'pure_student_success_rate' not in result


def test_inherited_qualified_hold_counter_is_not_student_hold(files):
    row=summarize(files,progress=True)['jobs'][0]['environments'][0]
    hold=row['qualified_hold_counter_s']
    assert hold['first']==pytest.approx(4+17/480) and hold['last']==pytest.approx(4.05)
    assert hold['max']==pytest.approx(4.05) and hold['includes_teacher_prefix_inheritance'] is True
    assert row['student_observed_duration_s']==pytest.approx(8/480)
    assert row['five_second_student_hold_qualification_issued'] is False


def test_invalid_contact_missing_measurement_and_loss_break_continuity(files):
    student=[r for r in files['raw'] if r['env_id']==0 and r['action_owner']=='student']
    student[2]['payload']['contact_valid']=False
    student[4]['payload']['measurement_valid']=False
    student[6]['payload']['bilateral_opposed_contact']=False
    refresh(files)
    contact=summarize(files,progress=True)['jobs'][0]['environments'][0]['bilateral_opposed_contact']
    assert contact['valid_samples']==6 and contact['missing_samples']==2 and contact['true_samples']==5
    assert contact['contact_loss_count']==1
    assert contact['total_valid_contact_duration_s']==pytest.approx(5/480)
    assert contact['longest_continuous_duration_s']==pytest.approx(2/480)
    assert contact['valid_sample_coverage']==pytest.approx(.75)


def test_missing_raw_sample_breaks_continuity_without_synthesizing_false(files):
    files['raw']=[r for r in files['raw'] if not(r['env_id']==0 and r['physics_step']==20)]
    files['report']['source_rows'][0]['observed_physics_samples']-=1;refresh(files)
    row=summarize(files,progress=True)['jobs'][0]['environments'][0]
    contact=row['bilateral_opposed_contact']
    assert row['student_expected_physics_samples']==8 and row['student_missing_physics_samples']==1
    assert contact['false_samples']==0 and contact['missing_samples']==1
    assert contact['total_valid_contact_duration_s']==pytest.approx(7/480)
    assert contact['longest_continuous_duration_s']==pytest.approx(4/480)


def test_no_valid_fixed_anchor_slip_is_na_not_zero(files):
    row=summarize(files,progress=True)['jobs'][0]['environments'][0]
    assert row['fixed_anchor_slip_m']==dict(max=None,valid_samples=0,missing_samples=8,na_reason='no_valid_fixed_anchor_slip_samples')


def test_slip_requires_all_four_independent_validity_masks(files):
    student=[r for r in files['raw'] if r['env_id']==0 and r['action_owner']=='student']
    for r in student:r['payload'].update(slip_m=.001,slip_valid=True,prior_anchor_latched=True,slip_applicable=True)
    for r,field in zip(student[:4],('measurement_valid','slip_valid','prior_anchor_latched','slip_applicable')):
        r['payload'][field]=False;r['payload']['slip_m']=9.
    refresh(files)
    slip=summarize(files,progress=True)['jobs'][0]['environments'][0]['fixed_anchor_slip_m']
    assert slip['max']==.001 and slip['valid_samples']==4


@pytest.mark.parametrize('bad', ['uid','raw_time','owner','policy_time','source_uid','raw_after_terminal','terminal_time'])
def test_identity_time_and_first_terminal_are_strict(files,bad):
    if bad=='uid':files['raw'][-1]['episode_uid']='reset:3:1'
    elif bad=='raw_time':files['raw'][-1]['timestamp_s']+=.001
    elif bad=='owner':files['raw'][-1]['action_owner']='teacher_prefix'
    elif bad=='policy_time':files['policies'][-1]['clock_before_action']['timestamp_s']+=.001
    elif bad=='source_uid':files['policies'][-1]['recorded_teacher_episode_uid']='another_source'
    else:
        end=next(r for r in files['raw'] if r['env_id']==0 and r['physics_step']==23)
        end['payload']['done']=True
        files['report']['source_rows'][0]['terminal']=dict(policy_step=2,physics_step=23,timestamp_s=end['timestamp_s'],failure_code=0,action_owner='student')
        if bad=='terminal_time':
            files['raw']=[r for r in files['raw'] if not(r['env_id']==0 and r['physics_step']==24)]
            files['report']['source_rows'][0]['observed_physics_samples']=23
            files['report']['source_rows'][0]['terminal']['timestamp_s']+=.001
    refresh(files)
    with pytest.raises(ValueError):summarize(files,progress=True)
    assert not (files['root']/'analysis'/'summary.json').exists()


def test_first_terminal_stops_before_remaining_horizon(files):
    end=next(r for r in files['raw'] if r['env_id']==0 and r['physics_step']==19)
    end['payload']['done']=True;end['failure_code']=3
    files['raw']=[r for r in files['raw'] if r['env_id']!=0 or r['physics_step']<=19]
    sr=files['report']['source_rows'][0];sr['observed_physics_samples']=19
    sr['terminal']=dict(policy_step=2,physics_step=19,timestamp_s=end['timestamp_s'],failure_code=3,action_owner='student')
    refresh(files)
    row=summarize(files,progress=True)['jobs'][0]['environments'][0]
    assert row['ending']=='first_terminal' and row['student_expected_physics_samples']==3
    assert row['student_physics_samples']==3 and row['student_missing_physics_samples']==0


def test_partial_batch_requires_explicit_progress_and_pending_is_not_failure(files):
    with pytest.raises(ValueError,match='64|complete'):summarize(files)
    assert not (files['root']/'analysis').exists()
    result=summarize(files,progress=True)
    assert result['status']=='progress_snapshot' and result['is_final'] is False
    assert result['completed_job_count']==1 and result['pending_job_count']==63
    assert all(r['status']=='pending' for r in result['pending_jobs'])
    assert result['failed_job_count']==0


def test_report_without_outcome_is_pending(files):
    files['outcome_path'].unlink();result=summarize(files,progress=True)
    assert result['completed_job_count']==0 and result['pending_job_count']==64


def test_changed_bound_input_sha_is_rejected(files):
    files['raw_path'].write_bytes(files['raw_path'].read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA|sha|identity'):summarize(files,progress=True)


def test_input_changed_during_analysis_is_rejected(files,monkeypatch):
    m=module();original=m._read_jsonl
    def mutate_after_read(path):
        yield from original(path)
        if Path(path)==files['raw_path']:
            files['raw_path'].write_bytes(files['raw_path'].read_bytes()+b'changed')
    monkeypatch.setattr(m,'_read_jsonl',mutate_after_read)
    with pytest.raises(ValueError,match='SHA|sha|identity'):
        m.summarize_batch(identity(files['plan_path']),files['root']/'analysis',progress=True)
    assert not (files['root']/'analysis'/'summary.json').exists()


def test_refuses_output_overwrite_and_preserves_input_bytes(files):
    before={p:identity(p) for p in (files['plan_path'],files['raw_path'],files['report_path'],files['policy_path'])}
    result=summarize(files,progress=True)
    assert result['input_sha256_before']==result['input_sha256_after']
    assert result['implementation']['sha256']==identity(MODULE)['sha256']
    assert all(identity(p)==v for p,v in before.items())
    with pytest.raises(FileExistsError):summarize(files,progress=True)


def completed_jobs(f, count=64, *, reuse_run=False):
    """Synthetic terminals allow every horizon to close on its first tick."""
    for job in f['plan']['jobs'][:count]:
        folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
        run='reused' if reuse_run else job['job_id'];report=copy.deepcopy(f['report'])
        raw=copy.deepcopy(f['raw']);policies=copy.deepcopy(f['policies']);outcome=copy.deepcopy(f['outcome'])
        report.update(execution_run_uuid=run,event=job['event'],branch_steps=job['branch_steps'],status='completed',completed_count=4)
        report['inputs']['checkpoint']=f['plan']['models'][job['arm']]['checkpoint']
        for row in policies:
            row.update(run_uuid=run,episode_uid=f"{run}:{row['env_id']}:0")
        for row in raw:
            row['episode_uid']=f"{run}:{row['env_id']}:0"
            if row['physics_step']==24:row['payload']['done']=True;row['failure_code']=3
        for row in report['source_rows']:
            row['execution_episode_uid']=f"{run}:{row['env_id']}:0"
            row['terminal']=dict(policy_step=2,physics_step=24,timestamp_s=42/480,failure_code=3,action_owner='student')
        report['raw_physics']=jsonl(folder/'source_physics_480hz.jsonl.gz',raw)
        jsonl(folder/'live_policy_60hz.jsonl.gz',policies)
        outcome.update(job_id=job['job_id'],arm=job['arm'],report=write(folder/'stage6_native_report.json',report),
                       effective_native_config=job['config'],summary=dict(source_rows=report['source_rows']),
                       source_reset_comparison=write(folder/'source_reset_comparison.json',dict(matched=True,tolerance=0.)),
                       source_camera_warmup_comparison=write(folder/'source_camera_warmup_comparison.json',
                           dict(matched=True,tolerance=0.,before_clock={'x':0},after_clock={'x':0})))
        write(f['plan_path'].parent/(job['job_id']+'_outcome.json'),outcome)


def test_all64_postchecked_outcomes_allow_final_without_pure_student_success(files):
    completed_jobs(files)
    result=summarize(files)
    assert result['is_final'] and result['status']=='final'
    assert result['completed_job_count']==64 and result['pending_job_count']==0
    assert all(row['ending']=='first_terminal' for job in result['jobs'] for row in job['environments'])
    assert result['student_task_success_qualification_issued'] is False


def test_jobs_cannot_reuse_one_execution_uid_as_fresh_trials(files):
    completed_jobs(files,2,reuse_run=True)
    with pytest.raises(ValueError,match='reuse|execution|UID'):summarize(files,progress=True)


def test_output_parent_alias_cannot_escape_independent_directory_rule(files):
    alias=files['root']/'alias';alias.symlink_to(files['plan_path'].parent,target_is_directory=True)
    with pytest.raises(ValueError,match='independent'):
        module().summarize_batch(identity(files['plan_path']),alias/'new_analysis',progress=True)


def test_per_environment_handoff_has_no_unowned_global_tail(files):
    # CONTACT env0 switches one tick earlier. Its source raw/policy legitimately
    # stop one tick before the global loop finishes the other three envs.
    job=files['plan']['jobs'][4];files['plan']['event_reference_cases'][0]['events']['CONTACT']['tick']=1
    new_ref=write(Path(files['plan']['event_reference']['path']),dict(cases=files['plan']['event_reference_cases']))
    files['plan']['event_reference']=new_ref
    for j in files['plan']['jobs']:
        cfg=json.loads(Path(j['config']['path']).read_text());cfg['event_reference']=new_ref
        j['config']=write(Path(j['config']['path']),cfg)
    files['outcome_path'].unlink()
    files['job']=job;folder=Path(job['output']);folder.mkdir(parents=True)
    files['raw_path']=folder/'source_physics_480hz.jsonl.gz';files['policy_path']=folder/'live_policy_60hz.jsonl.gz'
    files['report_path']=folder/'stage6_native_report.json'
    files['outcome_path']=files['plan_path'].parent/(job['job_id']+'_outcome.json')
    files['report']['inputs']['event_reference']=new_ref;files['report']['event']='CONTACT'
    files['report']['source_case_definitions'][0]['event_entry_ticks']['CONTACT']=1
    files['report']['source_rows'][0].update(teacher_prefix_policy_steps_executed=1,observed_physics_samples=16)
    files['raw']=[r for r in files['raw'] if r['env_id']!=0 or r['policy_step']<2]
    files['policies']=[r for r in files['policies'] if r['env_id']!=0 or r['policy_step']<2]
    for row in files['raw']:
        if row['env_id']==0 and row['policy_step']==1:row['action_owner']='student'
    for row in files['policies']:
        if row['env_id']==0 and row['policy_step']==1:row['owner']='student'
    files['outcome'].update(job_id=job['job_id'],effective_native_config=job['config'])
    for name in ('source_reset_comparison','source_camera_warmup_comparison'):
        files['outcome'][name]=write(folder/(name+'.json'),json.loads(Path(files['outcome'][name]['path']).read_text()))
    refresh(files);write(files['plan_path'],files['plan'])
    result=summarize(files,progress=True)
    row=result['jobs'][0]['environments'][0]
    assert row['handoff_tick']==1 and row['student_expected_physics_samples']==8
    assert row['student_physics_samples']==8 and row['student_missing_physics_samples']==0
    assert row['teacher_prefix_policy_steps']==1 and row['ending']=='bounded_censored'


def test_prefix_first_terminal_never_reaches_student_switch(files):
    end=next(r for r in files['raw'] if r['env_id']==0 and r['physics_step']==10)
    end['payload']['done']=True;end['failure_code']=3
    files['raw']=[r for r in files['raw'] if r['env_id']!=0 or r['physics_step']<=10]
    files['policies']=[r for r in files['policies'] if r['env_id']!=0 or r['policy_step']<2]
    files['report']['source_rows'][0].update(observed_physics_samples=10,student_policy_steps_executed=0,
        terminal=dict(policy_step=1,physics_step=10,timestamp_s=end['timestamp_s'],failure_code=3,action_owner='teacher_prefix'))
    refresh(files)
    row=summarize(files,progress=True)['jobs'][0]['environments'][0]
    assert not row['switch_reached'] and row['student_policy_steps']==0
    assert row['student_physics_samples']==0 and row['student_net_lift_m']['peak'] is None
    assert row['handoff_prefix_baseline'] is None and row['ending']=='first_terminal'


@pytest.mark.parametrize('prefix_state,expected_losses', [('valid',1),('invalid',0),('absent',0)])
def test_handoff_contact_loss_uses_only_adjacent_valid_prefix_sample(files,prefix_state,expected_losses):
    prefix=next(r for r in files['raw'] if r['env_id']==0 and r['physics_step']==16)
    if prefix_state=='invalid':prefix['payload']['measurement_valid']=False
    elif prefix_state=='absent':
        files['raw'].remove(prefix);files['report']['source_rows'][0]['observed_physics_samples']-=1
    for row in files['raw']:
        if row['env_id']==0 and row['action_owner']=='student':
            row['payload']['bilateral_opposed_contact']=False
    refresh(files)
    contact=summarize(files,progress=True)['jobs'][0]['environments'][0]['bilateral_opposed_contact']
    assert contact['contact_loss_count']==expected_losses
    assert contact['total_valid_contact_duration_s']==0 and contact['longest_continuous_duration_s']==0
    assert contact['true_samples']==0 and contact['valid_samples']==8


def test_student_contact_continuity_duration_does_not_include_prefix(files):
    contact=summarize(files,progress=True)['jobs'][0]['environments'][0]['bilateral_opposed_contact']
    assert contact['longest_continuous_duration_s']==pytest.approx(8/480)
    assert contact['longest_interval_start']['action_owner']=='student'
    assert contact['longest_interval_start']['world_physics_step']==17


def test_reported_first_terminal_after_own_horizon_is_rejected(files):
    env=0;policy=copy.deepcopy(next(p for p in files['policies'] if p['env_id']==env and p['policy_step']==2))
    policy.update(policy_step=3,clock_before_action=dict(physics_step=24,simulation_step_index=42,timestamp_s=42/480))
    files['policies'].append(policy)
    for sub in range(1,9):
        row=copy.deepcopy(next(r for r in files['raw'] if r['env_id']==env and r['physics_step']==24))
        row.update(policy_step=3,physics_step=24+sub,timestamp_s=(42+sub)/480)
        row['payload']['done']=sub==8;row['failure_code']=3 if sub==8 else 0
        files['raw'].append(row)
    files['report']['source_rows'][0].update(student_policy_steps_executed=2,observed_physics_samples=32,
        terminal=dict(policy_step=3,physics_step=32,timestamp_s=50/480,failure_code=3,action_owner='student'))
    refresh(files)
    with pytest.raises(ValueError,match='horizon|window'):summarize(files,progress=True)


def test_postchecked_job_failure_count_does_not_claim_batch_or_episode_failure_counts(files):
    result=summarize(files,progress=True)
    assert result['failure_count_scope']=='postchecked_job_infrastructure_only;not_episode_outcome_failure'
    assert result['batch_infrastructure_failure_status']=='not_inspected'
