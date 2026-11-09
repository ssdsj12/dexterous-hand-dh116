"""Read-only, SHA-bound summaries of completed finite Stage7 branch segments.

Only a driver-written outcome makes a job eligible. Teacher-prefix extrema in
the native report are deliberately not Student segment measurements. This
stdlib-only observer never imports a runtime, launches native work, or trains.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


VERSION = 'dh116-stage7-student-segment-summary-v1'
ARMS = ('E0', 'B0', 'B1', 'B2')
EVENTS = ('PRE_CONTACT', 'CONTACT', 'LIFT_ONSET', 'HOLD')
HORIZONS = (1, 4, 16, 60)
DT = 1 / 480


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


class _Inputs:
    def __init__(self):
        self.before = {}

    def check(self, identity):
        if not isinstance(identity, dict) or not {'path', 'sha256'} <= identity.keys():
            raise ValueError('explicit path/SHA256 identity required')
        actual = file_identity(identity['path'])
        if actual != {k: identity[k] for k in ('path', 'sha256')}:
            raise ValueError('immutable input SHA256 identity changed: ' + str(identity['path']))
        old = self.before.setdefault(actual['path'], actual['sha256'])
        if old != actual['sha256']:
            raise ValueError('conflicting immutable SHA256 identity')
        return actual

    def observed(self, path):
        return self.check(file_identity(path))

    def after(self):
        for path, digest in self.before.items():
            if file_identity(path)['sha256'] != digest:
                raise ValueError('input/code SHA256 changed during analysis: ' + path)
        return dict(self.before)


def _json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def _read_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('JSONL object required')
            yield number, row


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _four(rows, label):
    if (not isinstance(rows, list) or len(rows) != 4
            or sorted(row.get('env_id', -1) for row in rows) != [0, 1, 2, 3]):
        raise ValueError('exactly four actual environments required: ' + label)
    return {row['env_id']: row for row in rows}


def _plan_cases(plan, ledger):
    reference = ledger.check(plan['event_reference'])
    cases = _json(reference['path']).get('cases')
    if cases != plan.get('event_reference_cases') or not isinstance(cases, list) or len(cases) != 4:
        raise ValueError('plan/event-reference case identities differ')
    by_env = {}
    for case in cases:
        source = case['source']; env = source['env_id']
        if env in by_env or env not in range(4) or not source.get('episode_uid') or not source.get('run_uuid'):
            raise ValueError('four unique source episode UID/model/event identities required')
        for event in EVENTS:
            fact = case.get('events', {}).get(event, {})
            if fact.get('available') is not True or type(fact.get('tick')) is not int or fact['tick'] < 0:
                raise ValueError('all actual source event ticks must be available')
        by_env[env] = case
    if set(by_env) != set(range(4)):
        raise ValueError('four original source cases required')
    return by_env


def _clock(policies, reset):
    """Qualify recorded SDK affine clock; keep original payload dt separate."""
    anchors = [reset] + [row['clock_before_action'] for row in policies.values()]
    first = anchors[0]
    for anchor in anchors:
        if (any(type(anchor.get(k)) is not int or anchor[k] < 0 for k in ('physics_step', 'simulation_step_index'))
                or not _finite(anchor.get('timestamp_s'))):
            raise ValueError('finite actual World/SDK policy clock required')
    other = next((a for a in anchors if a['physics_step'] != first['physics_step']), None)
    if other is None:
        raise ValueError('two distinct actual policy anchors required to qualify recorded cadence')
    cadence = (other['timestamp_s'] - first['timestamp_s']) / (other['physics_step'] - first['physics_step'])
    if not _finite(cadence) or not math.isclose(cadence, DT, rel_tol=1e-5, abs_tol=0.):
        raise ValueError('recorded SDK cadence differs from original 480Hz scope')
    offset = first['simulation_step_index'] - first['physics_step']
    origin = first['timestamp_s'] - first['simulation_step_index'] * cadence

    def verify(world, when, anchor=None):
        if type(world) is not int or not _finite(when):
            raise ValueError('actual World/timestamp required')
        if anchor is not None and not 1 <= world - anchor['physics_step'] <= 8:
            raise ValueError('raw sample outside actual policy eight-substep interval')
        expected = origin + (world + offset) * cadence
        if abs(when - expected) > 16 * max(math.ulp(float(when)), math.ulp(float(expected))):
            raise ValueError('raw/policy timestamp does not match recorded SDK clock')

    for anchor in anchors:
        if anchor['simulation_step_index'] - anchor['physics_step'] != offset:
            raise ValueError('World/SDK offset changed')
        verify(anchor['physics_step'], anchor['timestamp_s'])
    return verify, dict(reported_sdk_cadence_s=cadence, reported_sdk_origin_s=origin,
                        sdk_minus_world_physics_step=offset, payload_duration_dt_s=DT,
                        independent_getter_post_clock_available=False,
                        clock_scope='affine_qualification_from_actual_policy_anchors;post_SDK_index_derived')


def _point(row, value, raw_id, model, case):
    return dict(value=float(value), env_id=row['env_id'], source_episode_uid=case['source']['episode_uid'],
                execution_episode_uid=row['episode_uid'], model=model, action_owner=row['action_owner'],
                policy_step=row['policy_step'], world_physics_step=row['physics_step'],
                timestamp_s=row['timestamp_s'], raw_pointer=dict(file=raw_id, line_1based=row['_line']))


def _environment(env, rows, policies, source_row, case, job, raw_id, model):
    student = [r for r in rows if r['action_owner'] == 'student']
    prefix = [r for r in rows if r['action_owner'] == 'teacher_prefix']
    switch = case['events'][job['event']]['tick']
    terminal = source_row.get('terminal')
    stop_tick = terminal['policy_step'] if terminal else switch + job['branch_steps'] - 1
    expected_student = 0
    for tick in range(switch, stop_tick + 1):
        anchor = policies[env, tick]['clock_before_action']['physics_step']
        expected_student += terminal['physics_step'] - anchor if terminal and tick == stop_tick else 8
    valid_lift = [r for r in student if r['payload'].get('measurement_valid') is True
                  and r['payload'].get('lift_valid') is True and _finite(r['payload'].get('lift_m'))]
    point = lambda r, value: _point(r, value, raw_id, model, case)
    peak_row = max(valid_lift, key=lambda r:r['payload']['lift_m']) if valid_lift else None
    peak = point(peak_row, peak_row['payload']['lift_m']) if peak_row else None
    baseline_row = prefix[-1] if prefix and prefix[-1]['policy_step'] == switch - 1 and prefix[-1]['physics_step'] == policies[env, switch-1]['clock_before_action']['physics_step'] + 8 else None
    baseline = (point(baseline_row, baseline_row['payload']['lift_m'])
                if baseline_row and baseline_row['payload'].get('measurement_valid') is True
                and baseline_row['payload'].get('lift_valid') is True and _finite(baseline_row['payload'].get('lift_m')) else None)
    lifts = dict(first=valid_lift[0]['payload']['lift_m'] if valid_lift else None,
                 last=valid_lift[-1]['payload']['lift_m'] if valid_lift else None,
                 peak=peak, valid_samples=len(valid_lift), missing_samples=expected_student-len(valid_lift))
    delta = {key: (lifts[key]-baseline['value'] if lifts[key] is not None and baseline else None) for key in ('first','last')}
    delta.update(peak=peak['value']-baseline['value'] if peak and baseline else None,
                 peak_sample=peak, scope='same_env_Student_net_lift_minus_last_real_prefix_post_sample')
    true_count = false_count = valid_count = losses = 0
    duration = longest = current = 0.; previous = None; start = best_start = best_end = None
    # Prefix contributes only the immediately preceding observed transition
    # state, never its duration or contact-run length to this Student segment.
    if (prefix and student and prefix[-1]['episode_uid']==student[0]['episode_uid']
            and prefix[-1]['physics_step']+1==student[0]['physics_step']
            and prefix[-1]['payload'].get('measurement_valid') is True
            and prefix[-1]['payload'].get('contact_valid') is True):
        previous = prefix[-1]
    for row in student:
        payload = row['payload']; consecutive = previous is not None and row['physics_step'] == previous['physics_step'] + 1
        valid = (payload.get('measurement_valid') is True and payload.get('contact_valid') is True
                 and type(payload.get('bilateral_opposed_contact')) is bool)
        contact = payload.get('bilateral_opposed_contact') if valid else None
        if not consecutive or contact is not True:
            current = 0.; start = None
        if valid:
            valid_count += 1
            if contact:
                true_count += 1; duration += payload['dt_s']; current += payload['dt_s']
                if start is None: start = row
                if current > longest: longest = current; best_start = start; best_end = row
            else:
                false_count += 1
                if (consecutive and previous['payload'].get('measurement_valid') is True
                        and previous['payload'].get('contact_valid') is True
                        and previous['payload'].get('bilateral_opposed_contact') is True): losses += 1
        previous = row
    contact_summary = dict(valid_samples=valid_count, true_samples=true_count, false_samples=false_count,
        missing_samples=expected_student-valid_count, contact_loss_count=losses,
        valid_sample_coverage=valid_count/expected_student if expected_student else None,
        total_valid_contact_duration_s=duration, longest_continuous_duration_s=longest,
        longest_interval_start=point(best_start, 1.) if best_start else None,
        longest_interval_end=point(best_end, 1.) if best_end else None,
        missingness='invalid_or_absent_samples_break_continuity;never_imputed_false',
        scope='aggregate_bilateral_opposed_only;not_individual_pad_contact')
    slip_rows = [r for r in student if all(r['payload'].get(k) is True for k in
                 ('measurement_valid','slip_valid','prior_anchor_latched','slip_applicable'))
                 and _finite(r['payload'].get('slip_m'))]
    slip = dict(max=None, valid_samples=len(slip_rows), missing_samples=expected_student-len(slip_rows),
                na_reason=None if slip_rows else 'no_valid_fixed_anchor_slip_samples')
    if slip_rows:
        r=max(slip_rows,key=lambda r:r['payload']['slip_m']);slip.update(max=r['payload']['slip_m'],max_sample=point(r,r['payload']['slip_m']))
    hold_rows = [r for r in student if r['payload'].get('measurement_valid') is True
                 and r['payload'].get('qualified_hold_valid') is True and _finite(r['payload'].get('qualified_hold_s'))]
    hold_values = [r['payload']['qualified_hold_s'] for r in hold_rows]
    hold = dict(first=hold_values[0] if hold_values else None,last=hold_values[-1] if hold_values else None,
                max=max(hold_values) if hold_values else None,valid_samples=len(hold_values),
                includes_teacher_prefix_inheritance=True,scope='native_counter;not_Student_only_hold_duration',
                max_sample=point(max(hold_rows,key=lambda r:r['payload']['qualified_hold_s']),max(hold_values)) if hold_values else None)
    return dict(env_id=env, source=case['source'], execution_episode_uid=source_row['execution_episode_uid'],
        event=job['event'],handoff_tick=switch,requested_student_policy_steps=job['branch_steps'],
        teacher_prefix_policy_steps=len({r['policy_step'] for r in prefix}), student_policy_steps=len({r['policy_step'] for r in student}),
        teacher_prefix_physics_samples=len(prefix),student_physics_samples=len(student),
        student_expected_physics_samples=expected_student,student_missing_physics_samples=expected_student-len(student),
        student_observed_duration_s=sum(r['payload']['dt_s'] for r in student),
        switch_reached=bool(student),ending='first_terminal' if terminal else 'bounded_censored',terminal=terminal,
        student_net_lift_m=lifts,handoff_prefix_baseline=baseline,student_delta_from_handoff_m=delta,
        bilateral_opposed_contact=contact_summary,fixed_anchor_slip_m=slip,qualified_hold_counter_s=hold,
        five_second_student_hold_qualification_issued=False,causal_claim_issued=False,
        per_pad_contact=dict(value=None,na_reason='per_pad_NPZ_not_read_by_this_raw_only_observer'),
        per_digit_force=dict(value=None,na_reason='per_digit_NPZ_not_read;aggregate_is_not_per_digit_force'))


def _summarize_job(job, plan, cases, outcome_path, ledger):
    outcome_id=ledger.observed(outcome_path);outcome=_json(outcome_id['path'])
    if (outcome.get('job_id'),outcome.get('arm')) != (job['job_id'],job['arm']):
        raise ValueError('outcome/job identity differs')
    original=ledger.check(job['config']);base=_json(original['path'])
    expected=dict(version='dh116-stage6-native-v1',mode='branch',initial_state_tolerance=0.,event=job['event'],
                  branch_steps=job['branch_steps'],checkpoint=plan['models'][job['arm']]['checkpoint'],
                  event_reference=plan['event_reference'],**plan['common_inputs'])
    if base != expected:raise ValueError('planned config/model/source/event/horizon identity differs')
    effective=ledger.check(outcome['effective_native_config']);config=_json(effective['path'])
    if {k:v for k,v in config.items() if k!='same_state_comparison_cache'} != expected:
        raise ValueError('effective native config differs from fixed job')
    if 'same_state_comparison_cache' in config:ledger.check(config['same_state_comparison_cache'])
    report_id=ledger.check(outcome['report']);folder=Path(job['output'])
    if Path(report_id['path']) != folder/'stage6_native_report.json':raise ValueError('report belongs to another job directory')
    report=_json(report_id['path']);inputs=dict(checkpoint=expected['checkpoint'],event_reference=expected['event_reference'],**plan['common_inputs'])
    if (report.get('mode')!='branch' or report.get('version')!='dh116-stage6-native-v1'
            or report.get('status') not in ('bounded_branch_completed','completed') or report.get('error') is not None
            or report.get('native_runtime_verified') is not True or report.get('synthetic_cpu_fixture')
            or report.get('execution')!='native_isaac_tensor_world' or report.get('ppo_updates')!=0
            or (report.get('physics_hz'),report.get('policy_hz'))!=(480,60)
            or (report.get('event'),report.get('branch_steps'))!=(job['event'],job['branch_steps'])
            or report.get('inputs')!=inputs or report.get('evaluated_source_count')!=4):
        raise ValueError('actual completed native report/model/source/event/horizon differs')
    for value in inputs.values():ledger.check(value)
    for name in ('source_reset_comparison','source_camera_warmup_comparison'):
        proof_id=ledger.check(outcome[name]);proof=_json(proof_id['path'])
        if (Path(proof_id['path']) != folder/(name+'.json') or proof.get('matched') is not True or proof.get('tolerance')!=0.
                or (name=='source_camera_warmup_comparison' and (not isinstance(proof.get('before_clock'),dict) or proof['before_clock']!=proof.get('after_clock')))):
            raise ValueError('actual exact source reset/camera proof required')
    definitions=_four(report.get('source_case_definitions'),'case definitions');outcomes=_four(report.get('source_rows'),'source rows')
    if outcome.get('summary',{}).get('source_rows')!=report['source_rows']:
        raise ValueError('driver postchecked outcome/report source rows differ')
    run=report.get('execution_run_uuid')
    if not isinstance(run,str) or not run:raise ValueError('actual execution UUID required')
    for env,case in cases.items():
        source=case['source'];definition=definitions[env];result=outcomes[env]
        if (str(definition.get('episode_id'))!=str(source['episode_id']) or definition.get('source_episode_uid')!=source['episode_uid']
                or definition.get('source_run_uuid')!=source['run_uuid']
                or any(definition.get('event_entry_ticks',{}).get(event)!=case['events'][event]['tick'] for event in EVENTS)
                or str(result.get('episode_id'))!=str(source['episode_id']) or type(result.get('execution_episode_id')) is not int
                or result.get('execution_episode_uid')!=f"{run}:{env}:{result['execution_episode_id']}"):
            raise ValueError('four source/execution UID or handoff event identities differ')
        terminal=result.get('terminal');switch=case['events'][job['event']]['tick']
        if terminal is not None:
            if (not isinstance(terminal,dict) or type(terminal.get('policy_step')) is not int
                    or not 0<=terminal['policy_step']<switch+job['branch_steps']):
                raise ValueError('first terminal must remain inside this environment planned handoff/horizon window')
            expected_owner='teacher_prefix' if terminal['policy_step']<switch else 'student'
            if terminal.get('action_owner')!=expected_owner:
                raise ValueError('first terminal owner differs from planned handoff window')
    policy_id=ledger.observed(folder/'live_policy_60hz.jsonl.gz');policies={}
    for line,row in _read_jsonl(policy_id['path']):
        env=row.get('env_id');tick=row.get('policy_step')
        if env not in cases or type(tick) is not int or tick<0 or (env,tick) in policies:
            raise ValueError('invalid or duplicate live policy env/tick')
        result=outcomes[env];switch=cases[env]['events'][job['event']]['tick'];terminal=result.get('terminal')
        end=terminal['policy_step']+1 if terminal else switch+job['branch_steps']
        owner='teacher_prefix' if tick<switch else 'student'
        if (tick>=end or row.get('owner')!=owner or row.get('run_uuid')!=run
                or row.get('episode_uid')!=result['execution_episode_uid'] or row.get('episode_id')!=result['execution_episode_id']
                or row.get('recorded_teacher_episode_uid')!=cases[env]['source']['episode_uid']):
            raise ValueError('live policy UID/owner/source/window differs')
        policies[env,tick]=row
    for env,result in outcomes.items():
        switch=cases[env]['events'][job['event']]['tick'];terminal=result.get('terminal')
        end=terminal['policy_step']+1 if terminal else switch+job['branch_steps']
        if {tick for e,tick in policies if e==env}!=set(range(end)):
            raise ValueError('missing actual policy window before first terminal/bounded end')
    verify_clock,clock=_clock(policies,report.get('reset_clock',{}))
    raw_id=ledger.check(report['raw_physics'])
    if Path(raw_id['path']).parent!=folder:raise ValueError('raw physics identity belongs to another job')
    by_env={i:[] for i in cases};closed=set()
    for line,row in _read_jsonl(raw_id['path']):
        env=row.get('env_id');tick=row.get('policy_step');policy=policies.get((env,tick));payload=row.get('payload',{})
        if (env not in cases or env in closed or policy is None or row.get('episode_uid')!=outcomes[env]['execution_episode_uid']
                or str(row.get('source_episode_id'))!=str(cases[env]['source']['episode_id'])
                or row.get('action_owner')!=policy['owner'] or payload.get('live') is not True):
            raise ValueError('raw UID/owner/window differs or sample follows first terminal')
        world=row.get('physics_step');when=row.get('timestamp_s');verify_clock(world,when,policy['clock_before_action'])
        if (not _finite(payload.get('dt_s')) or not math.isclose(payload['dt_s'],DT,rel_tol=0.,abs_tol=1e-12)
                or (by_env[env] and world<=by_env[env][-1]['physics_step'])):
            raise ValueError('raw order/duplicate/original physical dt differs')
        row['_line']=line;by_env[env].append(row)
        if payload.get('done') is True:
            terminal=outcomes[env].get('terminal')
            if (not isinstance(terminal,dict) or any(terminal.get(k)!=row.get(k) for k in
                    ('policy_step','physics_step','timestamp_s','failure_code','action_owner'))):
                raise ValueError('actual raw first terminal/report clock differs')
            closed.add(env)
    for env,result in outcomes.items():
        rows=by_env[env]
        if (bool(result.get('terminal'))!=(env in closed) or len(rows)!=result.get('observed_physics_samples')
                or len({r['policy_step'] for r in rows if r['action_owner']=='student'})!=result.get('student_policy_steps_executed')
                or len({r['policy_step'] for r in rows if r['action_owner']=='teacher_prefix'})!=result.get('teacher_prefix_policy_steps_executed')
                or result.get('teacher_replay_policy_steps_executed')!=0):
            raise ValueError('report first terminal or actual owned sample/step counts differ')
    return dict(job_id=job['job_id'],arm=job['arm'],event=job['event'],branch_steps=job['branch_steps'],status='postchecked_outcome',
        outcome=outcome_id,report=report_id,original_config=original,effective_config=effective,model=expected['checkpoint'],
        raw_physics=raw_id,policy_trace=policy_id,clock=clock,
        environments=[_environment(env,by_env[env],policies,outcomes[env],cases[env],job,raw_id,expected['checkpoint']) for env in range(4)])


def summarize_batch(plan_identity, output, *, progress=False):
    """Write a new independent directory; final requires all64 outcome files."""
    ledger=_Inputs();implementation=ledger.observed(__file__);plan_id=ledger.check(plan_identity);plan=_json(plan_id['path'])
    output=Path(output).resolve();parent=Path(plan_id['path']).parent
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    source=Path(plan['source']).resolve() if 'source' in plan else None
    if output==parent or parent in output.parents or (source is not None and (output==source or source in output.parents)):
        raise ValueError('new independent analysis directory outside native batch and source required')
    specs=[(arm,event,horizon) for arm in ARMS for event in EVENTS for horizon in HORIZONS]
    jobs=plan.get('jobs',[])
    if (plan.get('version')!='dh116-stage7-finite-branch-batch-v1' or plan.get('arm')!='all'
            or plan.get('bounded_job_count')!=64 or len(jobs)!=64):
        raise ValueError('fixed complete64 Stage7 plan required')
    for job,(arm,event,horizon) in zip(jobs,specs):
        name=f'branch_{arm}_{event}_{horizon:02d}'
        if (any(job.get(k)!=v for k,v in dict(job_id=name,arm=arm,event=event,branch_steps=horizon,mode='branch',cold_reset=True).items())
                or job.get('evaluated_env_ids')!=[0,1,2,3] or job.get('neutral_env_ids')!=[4,5,6,7]
                or Path(job['output'])!=parent/'jobs'/name or Path(job['config']['path'])!=parent/'configs'/(name+'.json')):
            raise ValueError('fixed job schedule/model/event/horizon/output identity differs')
    eligible=[job for job in jobs if (parent/(job['job_id']+'_outcome.json')).is_file()]
    if not progress and len(eligible)!=64:raise ValueError('require_complete: all64 driver postchecked outcomes required; use --progress for snapshot')
    cases=_plan_cases(plan,ledger)
    for model in plan['models'].values():ledger.check(model['checkpoint'])
    for value in plan['common_inputs'].values():ledger.check(value)
    for name in ('driver','legacy_helper'):
        if name in plan:ledger.check(plan[name])
    summaries=[_summarize_job(job,plan,cases,parent/(job['job_id']+'_outcome.json'),ledger) for job in eligible]
    execution_uids=[row['execution_episode_uid'] for summary in summaries for row in summary['environments']]
    if len(execution_uids)!=len(set(execution_uids)):
        raise ValueError('different jobs cannot reuse actual execution episode UIDs as fresh trials')
    pending=[dict(job_id=j['job_id'],arm=j['arm'],event=j['event'],branch_steps=j['branch_steps'],status='pending') for j in jobs if j not in eligible]
    after=ledger.after()
    result=dict(version=VERSION,status='progress_snapshot' if progress else 'final',is_final=not progress,
        plan=plan_id,implementation=implementation,completed_job_count=len(summaries),pending_job_count=len(pending),
        failed_job_count=0,failure_count_scope='postchecked_job_infrastructure_only;not_episode_outcome_failure',
        batch_infrastructure_failure_status='not_inspected',
        jobs=summaries,pending_jobs=pending,input_sha256_before=dict(ledger.before),input_sha256_after=after,
        native_trials_launched=0,ppo_updates=0,causal_claim_issued=False,student_task_success_qualification_issued=False,
        interpretation='short_Student_owned_censored_segments;maximum1s_does_not_replace_original5s_hold;no_pure_Student_success_rate',
        identity_scope='actual_plan_inputs_and_analysis_code;does_not_rehash_mutable_posthoc_source_tree')
    output.mkdir(parents=True,exist_ok=False);target=output/'summary.json'
    try:
        with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        ledger.after()
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--progress',action='store_true',help='Explicit snapshot; absent outcomes are pending, never failed')
    args=parser.parse_args(argv)
    result=summarize_batch(dict(path=str(args.plan.resolve(strict=True)),sha256=args.plan_sha256),args.output,progress=args.progress)
    print(json.dumps({k:result[k] for k in ('status','is_final','completed_job_count','pending_job_count','implementation')}))


if __name__=='__main__':
    main()
