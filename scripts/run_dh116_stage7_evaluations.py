"""Finite Stage7 cold-reset Teacher-prefix branches, explicit execution only.

All four genuine pure-Student source outcomes must be complete before a plan
is created. Saved-observation caches never replace a native reset or trial.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess


VERSION = 'dh116-stage7-finite-branch-batch-v1'
ARMS = ('E0', 'B0', 'B1', 'B2')
EVENTS = ('PRE_CONTACT', 'CONTACT', 'LIFT_ONSET', 'HOLD')
HORIZONS = (1, 4, 16, 60)
SOURCE_C58_SHA = 'c58f9cf0718d36cd9feec181c9180a9657e97b54d6fa8e99ce1b3f16244f6d97'
TEACHER02_SHA = '1a5faead16451a7dd0b523cdc6a1e236a6882c80c26a2a2bd295d2a2dfe2e2d9'
LOCK_PATH = '/tmp/xk-dh116-stage4-native.lock'
EXCLUDED = {'.git','artifacts','runs','archive','__pycache__','.pytest_cache','.venv'}


def job_specs(arm='all'):
    if arm not in ('all', *ARMS):
        raise ValueError('explicit all/E0/B0/B1/B2 arm required; no model ranking')
    return [dict(job_id=f'branch_{name}_{event}_{steps:02d}', arm=name, mode='branch',
                 event=event, branch_steps=steps)
            for name in (ARMS if arm == 'all' else (arm,))
            for event in EVENTS for steps in HORIZONS]


def _json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise ValueError('explicit JSON object required')
    return value


def _legacy(source):
    path=Path(source).resolve(strict=True)/'scripts/run_dh116_stage6_evaluations.py'
    spec=importlib.util.spec_from_file_location('stage7_original_stage6_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _source_api(source):
    # Invoke the original helper unchanged; do not replace its job_specs,
    # globals or provenance checks to make a legacy plan accept new events.
    return _legacy(source)._source_api(source)


def _identity(value,wf):
    return wf._checked_identity(value,Path.cwd()) if isinstance(value,dict) else wf.file_identity(value)


def _identity_leaves(value,prefix=''):
    """Retain original file paths as well as hashes, including nested assets."""
    if isinstance(value,dict) and 'path' in value and 'sha256' in value:
        return {prefix:dict(path=value['path'],sha256=value['sha256'])}
    if not isinstance(value,dict):return {}
    result={}
    for key,item in value.items():result.update(_identity_leaves(item,prefix+'/'+str(key)))
    return result


def _reset_proofs(report_path,wf,branch):
    parent=Path(report_path).parent;result={}
    for name in ('source_reset_comparison','source_camera_warmup_comparison'):
        identity=wf.file_identity(parent/(name+'.json'));value=_json(identity['path'])
        if (value.get('matched') is not True or value.get('tolerance')!=0.
                or any(value.get('fields',{}).get(field,{}).get(flag) is not True
                    for field in branch.SOURCE_STATE_FIELDS for flag in ('matched','dtype_equal','shape_equal','finite'))):
            raise ValueError('genuine source exact reset/readback proof required: '+name)
        if name=='source_camera_warmup_comparison' and (
                not isinstance(value.get('before_clock'),dict) or value['before_clock']!=value.get('after_clock')):
            raise ValueError('source camera warmup changed the native clock')
        result[name]=identity
    return result


def require_pure_source_report(report,*,arm,checkpoint,common_inputs,cases):
    """Require four first terminals; physical success is an outcome, not a gate."""
    if (report.get('mode')!='source' or report.get('status')!='completed' or report.get('error') is not None
            or report.get('native_runtime_verified') is not True or report.get('synthetic_cpu_fixture')
            or report.get('ppo_updates')!=0 or report.get('evaluated_source_count')!=4
            or report.get('completed_count')!=4 or report.get('inputs')!=dict(checkpoint=checkpoint,**common_inputs)):
        raise ValueError(arm+' requires completed genuine pure-source report with four first terminals and exact inputs')
    rows=report.get('source_rows',[]);run=report.get('execution_run_uuid')
    if (not isinstance(run,str) or not run or len(rows)!=4
            or sorted(row.get('env_id',-1) for row in rows)!=list(range(4))):
        raise ValueError('actual four source environments and execution UUID required')
    by_env={row['env_id']:row for row in rows};uids=set()
    for env in range(4):
        row=by_env[env];terminal=row.get('terminal');episode=row.get('execution_episode_id')
        if (type(episode) is not int or episode<0 or row.get('execution_episode_uid')!=f'{run}:{env}:{episode}'
                or row['execution_episode_uid'] in uids or type(row.get('native_success')) is not bool
                or row.get('teacher_prefix_policy_steps_executed')!=0 or row.get('teacher_replay_policy_steps_executed')!=0
                or type(row.get('student_policy_steps_executed')) is not int or row['student_policy_steps_executed']<1
                or not isinstance(terminal,dict) or terminal.get('action_owner')!='student'
                or any(type(terminal.get(k)) is not int or terminal[k]<0 for k in ('policy_step','physics_step','failure_code'))
                or type(terminal.get('timestamp_s')) not in (float,int) or not math.isfinite(terminal['timestamp_s'])
                or row['student_policy_steps_executed']!=terminal['policy_step']+1):
            raise ValueError('actual Student-only execution UID and owned first terminal required')
        uids.add(row['execution_episode_uid'])
    expected={case['env_id']:case for case in cases};definitions=report.get('source_case_definitions',[])
    if len(definitions)!=4 or {row.get('env_id') for row in definitions}!=set(expected):
        raise ValueError('source report must retain all four actual Teacher case identities')
    for row in definitions:
        case=expected[row['env_id']]
        if (str(row.get('episode_id'))!=str(case['episode_id']) or row.get('source_episode_uid')!=case['episode_uid']
                or row.get('source_run_uuid')!=case['run_uuid'] or row.get('recorded_teacher_tape_length')!=len(case['actions'])
                or row.get('source_trajectory_id')!=case['source_trajectory_id']):
            raise ValueError('source report case lineage differs from the original successful Teacher pool')
    successes=sum(row['native_success'] for row in rows)
    if report.get('native_success_count')!=successes or report.get('pure_student_native_success_count')!=successes:
        raise ValueError('source outcome counters disagree with the four actual first terminals')
    neutral=report.get('unevaluated_environments',[])
    if (len(neutral)!=4 or sorted(row.get('env_id',-1) for row in neutral)!=[4,5,6,7]
            or any(row.get('evaluated') is not False or row.get('action')!='neutral_zero24' for row in neutral)):
        raise ValueError('original neutral rows4..7 cannot enter evaluation counts')
    return dict(arm=arm,completed_first_terminal_count=4,pure_student_native_success_count=successes,
        physical_success_is_eligibility_gate=False,source_rows=deepcopy(rows),new_native_trials=0)


def _resolve(source,prepared_path,initial_state_path,*,training_reports,source_outcomes,replay_report,event_reference,native_template):
    source,wf,branch=_source_api(source);old=_legacy(source)
    prepared_id=_identity(prepared_path,wf);prepared,_=old._load_bound_prepared(prepared_id['path'])
    initial_id=_identity(initial_state_path,wf)
    if Path(initial_id['path']).parent!=Path(prepared['teacher_report']['path']).parent:
        raise ValueError('actual source collection cold-reset artifact required')
    old._check_initial_state(initial_id['path'])
    source_model=prepared['inputs']['source_checkpoint']
    if source_model['sha256']!=SOURCE_C58_SHA:raise ValueError('exact original c58 source checkpoint required')
    if set(training_reports or {})!=set(ARMS[1:]):raise ValueError('all B0/B1/B2 actual final3000 training reports required')
    before=old._validate_model(source_model['path'],None,arm='E0',prepared=prepared)
    models=dict(E0=dict(checkpoint=source_model,selection='actual_source_checkpoint_from_bound_inputs'))
    for arm in ARMS[1:]:
        identity=_identity(training_reports[arm],wf)
        models[arm]=old.select_completed_checkpoint(identity['path'],profile=arm,
            prepared_identity=prepared_id,source_identity=source_model)
        old._validate_model(models[arm]['checkpoint']['path'],before,arm=arm,prepared=prepared)
    common=dict(teacher_dataset=prepared['dataset'],sequence_context=prepared['context'],
        source_initial_state=initial_id,source_collection_report=prepared['teacher_report'])
    import torch
    data=torch.load(prepared['dataset']['path'],map_location='cpu',weights_only=False)
    context=torch.load(prepared['context']['path'],map_location='cpu',weights_only=False)
    cases=branch.source_cases(data,context,_json(prepared['teacher_report']['path']),dataset_sha256=prepared['dataset']['sha256'])
    if set(source_outcomes or {})!=set(ARMS):raise ValueError('all four completed pure-source outcomes required; no wait or native launch')
    outcomes={};execution_runs=set()
    for arm in ARMS:
        identity=_identity(source_outcomes[arm],wf);report=_json(identity['path'])
        summary=require_pure_source_report(report,arm=arm,checkpoint=models[arm]['checkpoint'],common_inputs=common,cases=cases)
        if report['execution_run_uuid'] in execution_runs:raise ValueError('source arms cannot reuse one actual execution run')
        execution_runs.add(report['execution_run_uuid'])
        outcomes[arm]=dict(report=identity,summary=summary,**_reset_proofs(identity['path'],wf,branch))
    replay_id=_identity(replay_report,wf);replay_proofs=_reset_proofs(replay_id['path'],wf,branch)
    replay=old.require_teacher_replay_baseline(_json(replay_id['path']),source_identity=source_model,common_inputs=common,
        reset_comparison=_json(replay_proofs['source_reset_comparison']['path']))
    from go2_pvcnn.control.dh116.tabletop_vision_stage7_event_reference import load_event_reference
    event_id=_identity(event_reference,wf)
    reference=load_event_reference(event_id,dict(source_checkpoint=source_model,native_replay_report=replay_id,**common))
    if (len(reference.get('cases',[]))!=4 or any(case.get('events',{}).get(event,{}).get('available') is not True
            or type(case['events'][event].get('tick')) is not int for case in reference['cases'] for event in EVENTS)):
        raise ValueError('all four events require actual available reference rows for every original source')
    template=_identity(native_template,wf)
    if template['sha256']!=TEACHER02_SHA:raise ValueError('exact actual Teacher02 template SHA required; physical/material parameters stay original')
    template_data=_json(template['path']);native_env=old.native_template_environment(template_data)
    for name,wanted in (('--seed','42'),('--num-envs','8'),('--physics-hz','480'),('--vision-width','224'),('--vision-height','96')):
        if old._single_option(template_data['command'],name)!=wanted:raise ValueError('original native template differs: '+name)
    if Path(old._single_option(template_data['command'],'--output')).resolve()!=Path(prepared['teacher_report']['path']).parent:
        raise ValueError('native template belongs to another actual Teacher collection')
    physical_inputs={name:wf.file_identity(old._single_option(template_data['command'],'--'+name)) for name in
        ('pose-summary','grasp-initialization','mixed-grasp-initialization','load-profile','depth-camera-config','vision-teacher-source')}
    original_files=_identity_leaves(prepared['inputs'],'prepared_inputs')
    original_files.update(_identity_leaves({k:reference[k] for k in ('inputs','artifacts','implementation')},'event_reference'))
    dependencies=branch.comparison_dependencies()
    if 'tabletop_vision_stage7_event_reference' not in dependencies:
        raise ValueError('fresh Stage7 comparison dependencies required; legacy13-dependency cache aliases refused')
    for name,digest in dependencies.items():
        if wf.file_identity(source/'src/go2_pvcnn/control/dh116'/(name+'.py'))['sha256']!=digest:
            raise ValueError('comparison dependency must be actual current source bytes: '+name)
    return source,wf,branch,dict(prepared=prepared_id,models=models,common_inputs=common,source_outcomes=outcomes,
        replay_report=replay_id,replay_proofs=replay_proofs,replay_baseline=replay,event_reference=event_id,
        template=template,native_environment_whitelist=native_env,comparison_dependencies=dependencies,
        physical_inputs=physical_inputs,original_input_files=original_files,event_reference_cases=deepcopy(reference['cases']),
        legacy_helper=wf.file_identity(source/'scripts/run_dh116_stage6_evaluations.py'))


def source_manifest(source,wf):
    source=Path(source).resolve(strict=True);result={}
    for root,dirs,files in os.walk(source):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED)
        for name in sorted(files):
            if name.endswith(('.pyc','.pyo')):continue
            path=Path(root)/name;result[str(path.relative_to(source))]=wf.file_identity(path)['sha256']
    return result


def _config(spec,bound,branch):
    return dict(version=branch.VERSION,mode='branch',checkpoint=bound['models'][spec['arm']]['checkpoint'],
        initial_state_tolerance=0.,event=spec['event'],branch_steps=spec['branch_steps'],
        event_reference=bound['event_reference'],**bound['common_inputs'])


def prepare_batch(source,prepared_path,initial_state_path,output,*,arm='all',training_reports=None,
                  source_outcomes=None,replay_report=None,event_reference=None,native_template=None):
    specs=job_specs(arm)
    source,wf,branch,bound=_resolve(source,prepared_path,initial_state_path,training_reports=training_reports,
        source_outcomes=source_outcomes,replay_report=replay_report,event_reference=event_reference,native_template=native_template)
    output=Path(output).absolute()
    if output==source or source in output.resolve().parents:raise ValueError('new independent output must be outside selected source')
    hashes=source_manifest(source,wf)
    output.mkdir(parents=True,exist_ok=False);(output/'configs').mkdir();jobs=[]
    for spec in specs:
        config_path=output/'configs'/(spec['job_id']+'.json');wf._json_new(config_path,_config(spec,bound,branch))
        # _resolve already re-derived all four event facts from original raw
        # evidence once. These configs contain only the fixed finite spec and
        # that same immutable reference identity. Each native process still
        # performs its full load_native_config and a fresh cold reset.
        branch.verify_inputs(_config(spec,bound,branch))
        target=output/'jobs'/spec['job_id']
        command=wf.build_stage6_native_command(_json(bound['template']['path']),source,target,mode='branch',native_config=config_path)
        jobs.append(dict(**spec,config=wf.file_identity(config_path),output=str(target),cold_reset=True,
            evaluated_env_ids=[0,1,2,3],neutral_env_ids=[4,5,6,7],projected_native_command=command))
    plan=dict(version=VERSION,arm=arm,source=str(source),**bound,source_sha256=hashes,driver=wf.file_identity(__file__),
        jobs=jobs,bounded_job_count=len(jobs),maximum_job_count=64,native_jobs_executed=0,concurrent_native_instances=1,
        comparison_caches={},automatic_offline_cache_arms=list(ARMS),
        cache_scope='saved_observation_metadata_only;never_physics_reset_contact_or_terminal',
        native_trials_reused=0,physical_state_reused=False,automatic_retry=False,pending_input_wait=False,
        ppo_updates=0,automatic_PPO=False,teacher_prefix_is_pure_student_success=False,
        checkpoint_selection='prespecified_final_update3000;no_MSE_or_peak_selection',native_lock=LOCK_PATH,
        lock_policy='per_job_nonblocking;busy_refuses;no_queue_no_retry',
        source_execution_gate='all_four_genuine_pure_source_first_terminal_reports_and_actual_replay_reset')
    wf._json_new(output/'batch_plan.json',plan);return plan


def validate_comparison_caches(caches,models,common_inputs,implementation_sha256):
    from go2_pvcnn.control.dh116.tabletop_vision_comparison_cache import load_comparison_cache
    wf=importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_stage6_workflow');result={}
    for arm,value in caches.items():
        if arm not in ARMS or arm not in models:raise ValueError('explicit selected E0/B0/B1/B2 cache arm required')
        identity=_identity(value,wf)
        load_comparison_cache(identity,dict(common_inputs,checkpoint=models[arm]['checkpoint']),implementation_sha256=implementation_sha256)
        result[arm]=identity
    return result


@contextmanager
def native_lock(path=LOCK_PATH):
    import fcntl
    with open(path,'a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('native global lock busy; no waiting or queued instance') from error
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def _bound_checks(plan,wf):
    for identity in (plan['prepared'],plan['template'],plan['driver'],plan['legacy_helper'],plan['event_reference'],
                     plan['replay_report'],*plan['replay_proofs'].values(),*plan['common_inputs'].values(),
                     *plan['physical_inputs'].values(),*plan['original_input_files'].values()):
        wf._checked_identity(identity,Path.cwd())
    for model in plan['models'].values():
        wf._checked_identity(model['checkpoint'],Path.cwd())
        if 'training_report' in model:wf._checked_identity(model['training_report'],Path.cwd())
    for value in plan['source_outcomes'].values():
        for name in ('report','source_reset_comparison','source_camera_warmup_comparison'):
            wf._checked_identity(value[name],Path.cwd())
    if source_manifest(plan['source'],wf)!=plan['source_sha256']:raise ValueError('current source bytes changed; prepare a new Stage7 freeze')


def launch_job(plan,job,config):
    """Own the same nonblocking lock fd through one complete child process."""
    source,wf,_=_source_api(plan['source']);target=Path(job['output']);frozen=target.with_name(target.name+'_source')
    with native_lock():
        busy=subprocess.run(['pgrep','-af','[s]cripts/train_dh116_tabletop_gpu_ppo.py'],capture_output=True,text=True)
        if busy.returncode not in (0,1) or busy.stdout.strip():raise RuntimeError('another native owner is active; no queue or second Isaac')
        _bound_checks(plan,wf);wf._checked_identity(config,Path.cwd())
        for p in (target,frozen,target.with_name(target.name+'.log'),target.with_name(target.name+'_launch.json'),target.with_name(target.name+'_exit.json')):
            if p.exists() or p.is_symlink():raise FileExistsError(p)
        hashes=wf.freeze_source(source,frozen)
        if hashes!=plan['source_sha256']:raise ValueError('new job freeze differs from prepared exact source bytes')
        template=_json(plan['template']['path'])
        command=wf.build_stage6_native_command(template,frozen,target,mode='branch',native_config=config['path'])
        from go2_pvcnn.control.dh116.tabletop_bimanual_checkpoint_contract import unwrap_physical_adapter_checkpoint
        import torch
        native_cfg=_json(config['path']);saved=torch.load(native_cfg['checkpoint']['path'],map_location='cpu',weights_only=False)
        core=unwrap_physical_adapter_checkpoint(saved,allow_explicit=True)
        wf._set_option(command,'--vision-wrist-quaternion-preprocessing',core['contract'].get('actor_wrist_quaternion_preprocessing','raw'))
        wf._json_new(target.with_name(target.name+'_launch.json'),dict(mode='branch',command=command,cwd=str(frozen),
            source_sha256=hashes,template=plan['template'],env=template.get('env',{}),
            bound_inputs=dict(native_config=config,checkpoint=native_cfg['checkpoint'],event_reference=plan['event_reference']),
            fresh_native_reset_required=True,native_trials_reused=0))
        with target.with_name(target.name+'.log').open('x') as stream:
            result=subprocess.run(command,cwd=frozen,env=dict(os.environ,**template.get('env',{})),stdout=stream,stderr=subprocess.STDOUT)
        wf._json_new(target.with_name(target.name+'_exit.json'),dict(exit_code=result.returncode))
        _bound_checks(plan,wf);wf._checked_identity(config,Path.cwd())
        for relative,digest in hashes.items():
            if wf.file_identity(frozen/relative)['sha256']!=digest:raise ValueError('original frozen job file changed: '+relative)
        if result.returncode:raise RuntimeError('native Stage7 job failed; log/freeze retained; no retry')
    return target


def execute_batch(plan_path):
    plan_path=Path(plan_path).resolve(strict=True);plan=_json(plan_path)
    if plan.get('version')!=VERSION or plan.get('native_lock')!=LOCK_PATH:raise ValueError('explicit finite Stage7 plan required')
    source,wf,branch=_source_api(plan['source']);plan_identity=wf.file_identity(plan_path)
    if _json(plan_path)!=plan:raise ValueError('prepared plan changed while reading')
    _bound_checks(plan,wf)
    _,_,_,bound=_resolve(source,plan['prepared'],plan['common_inputs']['source_initial_state'],
        training_reports={a:plan['models'][a]['training_report'] for a in ARMS[1:]},
        source_outcomes={a:plan['source_outcomes'][a]['report'] for a in ARMS},replay_report=plan['replay_report'],
        event_reference=plan['event_reference'],native_template=plan['template'])
    if any(plan.get(k)!=v for k,v in bound.items()) or wf.file_identity(__file__)!=plan['driver']:
        raise ValueError('prepared Stage7 bindings or driver bytes changed')
    specs=job_specs(plan['arm'])
    if len(plan.get('jobs',[]))!=len(specs):raise ValueError('finite Stage7 job schedule changed')
    for job,spec in zip(plan['jobs'],specs):
        if any(job.get(k)!=v for k,v in spec.items()):raise ValueError('job differs from fixed arm/event/horizon schedule')
        if Path(job['output'])!=plan_path.parent/'jobs'/spec['job_id']:raise ValueError('job output escaped prepared Stage7 directory')
        wf._checked_identity(job['config'],plan_path.parent)
        if Path(job['config']['path'])!=plan_path.parent/'configs'/(spec['job_id']+'.json') or _json(job['config']['path'])!=_config(spec,bound,branch):
            raise ValueError('native config differs from exact model/source/event reference')
        branch.verify_inputs(_config(spec,bound,branch))
        if any(p.exists() or p.is_symlink() for p in (Path(job['output']),Path(job['output']+'_source'))):
            raise FileExistsError(job['output'])
    wf._checked_identity(plan_identity,plan_path.parent)
    wf._json_new(plan_path.parent/'execution_started.json',dict(plan=plan_identity,bounded_jobs=len(specs),no_retry=True,sequential=True))
    available={};results=[];old=_legacy(source)
    common=dict(bound['common_inputs'],event_reference=bound['event_reference'])
    implementation=bound['comparison_dependencies']['tabletop_vision_branching_eval']
    try:
        for job in plan['jobs']:
            wf._checked_identity(plan_identity,plan_path.parent)
            _bound_checks(plan,wf)
            validate_comparison_caches(available,plan['models'],common,implementation)
            effective=old.effective_native_config(job,available)
            output=launch_job(plan,job,effective);report_path=output/'stage6_native_report.json';report=_json(report_path)
            wf._checked_identity(plan_identity,plan_path.parent)
            expected=dict(_config(job,bound,branch));expected.pop('version');expected.pop('mode');expected.pop('event');expected.pop('branch_steps');expected.pop('initial_state_tolerance')
            if (report.get('mode')!='branch' or report.get('event')!=job['event'] or report.get('branch_steps')!=job['branch_steps']
                    or report.get('error') is not None or report.get('inputs')!=expected):
                raise ValueError('actual branch report differs from requested fresh model/event/reference inputs')
            summary=old.summarize_native_report(report)
            proofs=_reset_proofs(report_path,wf,branch)
            value=dict(job_id=job['job_id'],arm=job['arm'],report=wf.file_identity(report_path),summary=summary,
                effective_native_config=effective,offline_comparison_cache=available.get(job['arm']),**proofs)
            if job['arm'] not in available:
                from go2_pvcnn.control.dh116.tabletop_vision_comparison_cache import build_comparison_cache
                cache=build_comparison_cache(report_path,plan_path.parent/'comparison_caches'/job['arm'])
                available.update(validate_comparison_caches({job['arm']:cache},plan['models'],common,implementation))
                value['created_offline_comparison_cache']=cache
            wf._json_new(plan_path.parent/(job['job_id']+'_outcome.json'),value);results.append(value)
    except BaseException as error:
        wf._json_new(plan_path.parent/'batch_execution_error.json',dict(error=repr(error),completed_jobs=results,no_retry=True,native_trials_reused=0));raise
    wf._checked_identity(plan_identity,plan_path.parent)
    summary=dict(version=VERSION,plan=plan_identity,results=results,comparison_caches=available,
        executed_jobs=len(results),native_trials_reused=0,physical_state_reused=False,ppo_updates=0,
        teacher_prefix_is_pure_student_success=False,heldout_generalization_claimed=False)
    wf._json_new(plan_path.parent/'batch_execution_report.json',summary);return summary


def _assignments(values,arms,parser,name):
    result={}
    for value in values:
        arm,sep,path=value.partition('=')
        if not sep or not path or arm not in arms or arm in result:parser.error(name+' requires unique '+ '|'.join(arms)+'=PATH')
        result[arm]=Path(path)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','prepared','source-initial-state','native-template','output','event-reference','replay-report'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--arm',choices=('all',*ARMS),default='all')
    parser.add_argument('--training-report',action='append',default=[],metavar='ARM=PATH')
    parser.add_argument('--source-outcome',action='append',default=[],metavar='ARM=PATH')
    parser.add_argument('--list-only',action='store_true');parser.add_argument('--execute',type=Path,metavar='BATCH_PLAN_JSON')
    args=parser.parse_args(argv)
    if args.list_only:
        print(json.dumps(dict(jobs=job_specs(args.arm),native_jobs_executed=0),indent=2));return 0
    if args.execute is not None:
        if any(getattr(args,n) is not None for n in ('source','prepared','source_initial_state','native_template','output','event_reference','replay_report')) or args.training_report or args.source_outcome:
            parser.error('--execute accepts only an already prepared Stage7 plan')
        result=execute_batch(args.execute)
    else:
        if any(getattr(args,n) is None for n in ('source','prepared','source_initial_state','native_template','output','event_reference','replay_report')):
            parser.error('explicit source/prepared/reset/template/output/event-reference/replay-report required; default is prepare only')
        result=prepare_batch(args.source,args.prepared,args.source_initial_state,args.output,arm=args.arm,
            training_reports=_assignments(args.training_report,ARMS[1:],parser,'--training-report'),
            source_outcomes=_assignments(args.source_outcome,ARMS,parser,'--source-outcome'),replay_report=args.replay_report,
            event_reference=args.event_reference,native_template=args.native_template)
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
