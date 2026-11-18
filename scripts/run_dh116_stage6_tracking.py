"""Prepare five finite native jobs; execute only through explicit --execute.

Tracking original/unilateral/common and fixed Stage A/right, B/both each get
their own source freeze, process, output and nonblocking global native lock.
Preparation is read-only CPU work. Busy ownership never waits, queues or
starts a second Isaac instance. Physical outcomes remain the runner's reports.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


VERSION='dh116-finite-recovery-tracking-launch-v1'
LOCK_PATH='/tmp/xk-dh116-stage4-native.lock'
TEACHER02_SHA='1a5faead16451a7dd0b523cdc6a1e236a6882c80c26a2a2bd295d2a2dfe2e2d9'
CALIBRATION_V02_SHA='bbbca0656b94bfac0eeeddd1f7152fc2e41a092cb0381a9a8231a915c7933adc'
SOURCE_C58_SHA='c58f9cf0718d36cd9feec181c9180a9657e97b54d6fa8e99ce1b3f16244f6d97'
EXCLUDED={'.git','artifacts','runs','archive','__pycache__','.pytest_cache','.venv'}
TEACHER_OPTIONS={'--vision-teacher-source':True,'--vision-teacher-mpc':False,
    '--vision-teacher-mpc-profile':True,'--vision-teacher-max-control-steps':True,
    '--vision-teacher-hold-tail-control-steps':True,'--vision-teacher-raw-chunk-frames':True,
    '--vision-sequence-context-sidecar':False,'--bimanual-teacher-physics-audit':False,
    '--bimanual-tripod-teacher-coordination-proof':True,'--bimanual-explicit-sdk-adapter':False,
    '--vision-teacher-perturbation-manifest':True}
CONFLICTS={'--resume','--vision-evaluation-checkpoint','--vision-evaluation-manifest',
    '--vision-stage6-native-config','--vision-tracking-experiment-config','--vision-m2-replay',
    '--tripod-recovery-stage','--paired-bias-arm'}


class NativeOwnerUnavailable(RuntimeError):pass


def identity(path):
    path=Path(path).resolve(strict=True);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(path=str(path),sha256=digest.hexdigest())


def checked(value):
    if not isinstance(value,dict) or not isinstance(value.get('path'),str):raise ValueError('explicit input identity required')
    actual=identity(value['path'])
    if actual.get('sha256')!=value.get('sha256'):raise ValueError('immutable SHA256 changed: '+value['path'])
    return actual


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def option(command,name):
    result=[]
    for i,token in enumerate(command):
        if token==name:
            if i+1>=len(command) or command[i+1].startswith('--'):raise ValueError('missing option value: '+name)
            result.append(command[i+1])
        elif token.startswith(name+'='):result.append(token.split('=',1)[1])
    if len(result)!=1:raise ValueError('one explicit option required: '+name)
    return result[0]


def _drop(command,options):
    result=[];i=0
    while i<len(command):
        token=command[i];name=token.split('=',1)[0]
        if name not in options:result.append(token);i+=1;continue
        if options[name] and '=' not in token:
            if i+1>=len(command) or command[i+1].startswith('--'):raise ValueError('missing removed option value: '+name)
            i+=1
        i+=1
    return result


def _set(command,name,value):return _drop(command,{name:True})+[name,str(value)]


def job_specs(phase='all'):
    tracking=[dict(job_id='tracking_'+arm,kind='tracking',arm=arm) for arm in ('original','unilateral','common')]
    recovery=[dict(job_id='stage_A_right',kind='recovery',stage='A',hands='right'),
        dict(job_id='stage_B_both',kind='recovery',stage='B',hands='both')]
    if phase=='all':return tracking+recovery
    if phase=='tracking':return tracking
    if phase=='recovery':return recovery
    raise ValueError('prespecified finite all/tracking/recovery phase required')


def build_command(template,frozen,output,job,config_path=None):
    original=template.get('command');command=list(original) if isinstance(original,list) else []
    if len(command)<3 or command[0]!='bash' or not command[1].endswith('/scripts/python.sh') or not command[2].endswith('/scripts/train_dh116_tabletop_gpu_ppo.py'):
        raise ValueError('actual native Teacher02 Python/script template required')
    if any(token.split('=',1)[0] in CONFLICTS for token in command):raise ValueError('native template contains conflicting runtime/old evaluation mode')
    for flag,wanted in [('--num-envs','8'),('--seed','42'),('--physics-hz','480'),('--vision-width','224'),('--vision-height','96'),
            ('--vision-camera-mode','fixed'),('--vision-action-distribution','tanh_gaussian')]:
        if option(command,flag)!=wanted:raise ValueError('original Teacher02 template differs: '+flag)
    if '--vision-teacher-mpc' not in command or option(command,'--vision-teacher-mpc-profile')!='pressure_geometry_constrained':raise ValueError('actual original Teacher02 ownership/profile required')
    option(command,'--vision-teacher-source')
    command[1]=str(Path(frozen)/'scripts/python.sh');command[2]=str(Path(frozen)/'scripts/train_dh116_tabletop_gpu_ppo.py')
    command=_set(command,'--output',output)
    if job.get('kind')=='tracking' and job.get('arm') in ('original','unilateral','common'):
        if config_path is None:raise ValueError('bound independent tracking config required')
        command=_drop(command,TEACHER_OPTIONS)
        command=_set(command,'--vision-wrist-quaternion-preprocessing','wrist_quaternion_hemisphere_v1')
        command=_set(command,'--vision-mirror-preprocessing','m0_raw')
        if '--bimanual-checkpoint-adapter' not in command:command.append('--bimanual-checkpoint-adapter')
        command=_set(command,'--vision-tracking-experiment-config',config_path)
    elif job.get('kind')=='recovery' and (job.get('stage'),job.get('hands')) in (('A','right'),('B','both')):
        command=_drop(command,{'--vision-sequence-context-sidecar':False})
        command=_set(command,'--vision-teacher-mpc-profile','pressure_reference')
        command=_set(command,'--vision-teacher-max-control-steps',560)
        command=_set(command,'--tripod-recovery-stage',job['stage'])
        command=_set(command,'--tripod-recovery-hands',job['hands'])
    else:raise ValueError('only explicit original/unilateral/common or fixed A/right,B/both is implemented')
    return command


def source_manifest(source):
    source=Path(source).resolve(strict=True);result={}
    for root,dirs,files in os.walk(source):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED)
        for name in sorted(files):
            if name.endswith(('.pyc','.pyo')):continue
            path=Path(root)/name;result[str(path.relative_to(source))]=identity(path)['sha256']
    return result


def verify_frozen_original_files(frozen,expected):
    for relative,digest in expected.items():
        path=Path(frozen)/relative
        if not path.is_file() or identity(path)['sha256']!=digest:raise ValueError('frozen original source changed: '+str(path))


@contextmanager
def native_lock(path=LOCK_PATH):
    import fcntl
    with open(path,'a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise NativeOwnerUnavailable('native global lock already held;no waiting or queued instance') from error
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def ensure_no_native_owner():
    result=subprocess.run(['pgrep','-af','[s]cripts/train_dh116_tabletop_gpu_ppo.py'],capture_output=True,text=True)
    if result.returncode not in (0,1) or result.stdout.strip():raise NativeOwnerUnavailable('another native owner is active;no wait or second instance')


def _source_api(source):
    source=Path(source).resolve(strict=True);sys.path.insert(0,str(source/'src'))
    names=('tabletop_vision_stage6_workflow','tabletop_full_episode_tracking_experiment','tabletop_vision_command_shadow')
    result={name:importlib.import_module('go2_pvcnn.control.dh116.'+name) for name in names}
    if any(source not in Path(module.__file__).resolve().parents for module in result.values()):raise ValueError('launcher imported another source tree')
    return source,result


def validate_calibration_binding(report,prepared,audit):
    if (report.get('version')!='dh116-teacher-tracking-calibration-v1' or report.get('qualified') is not True
            or report.get('target_source')!='physx_backend_get_dof_position_targets'
            or report.get('feedback_source')!='original_advance_returned_refresh_before_TensorEnv_terminal_reset'):
        raise ValueError('actual qualified02 PhysX/post-feedback calibration required')
    for report_key,prepared_key in [('dataset','dataset'),('context','context')]:
        if dict(path=report.get(report_key+'_path'),sha256=report.get(report_key+'_sha256'))!=prepared[prepared_key]:raise ValueError('calibration belongs to another actual02 pool')
    for key in ('controller_source_sha256','action_contract_sha256'):
        if report.get(key)!=audit[key]:raise ValueError('calibration controller/action source changed')
    p,trigger=report.get('teacher_error_p999_rad'),report.get('trigger_rad')
    if not isinstance(p,list) or len(p)!=12 or not isinstance(trigger,list) or len(trigger)!=12:raise ValueError('actual12joint calibration required')
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in p) or any(t!=max(.01,1.5*x) for x,t in zip(p,trigger)):raise ValueError('calibration perjoint trigger formula changed')


def tracking_config(arm,bound):
    if arm not in ('original','unilateral','common'):raise ValueError('prespecified tracking arm required')
    config=dict(version='dh116-full-episode-tracking-experiment-v1',arm=arm,num_envs=8,scene_seed=42,initial_state_tolerance=0.,
        checkpoint=bound['source_checkpoint'],source_initial_state=bound['source_initial_state'])
    if arm=='common':config['calibration']=dict(bound['calibration'],source_run='02')
    return config


def validate_job_config(job,spec,bound):
    if spec['kind']!='tracking':
        if 'config' in job:raise ValueError('fixed recovery cannot inherit a tracking configuration')
        return None
    path=checked(job['config'])['path']
    if read_json(path)!=tracking_config(spec['arm'],bound):raise ValueError('tracking configuration differs from requested arm/source/calibration')
    return path


def prepare_batch(source,teacher_template,prepared_path,source_initial_state,calibration_path,output,*,phase='all'):
    source,modules=_source_api(source);wf=modules['tabletop_vision_stage6_workflow'];runtime=modules['tabletop_full_episode_tracking_experiment']
    prepared_id=identity(prepared_path);prepared=read_json(prepared_path);inputs=wf.load_stage6_inputs(prepared['inputs']['input_manifest']['path'])
    if prepared.get('version')!='dh116-stage6-prepared-sequences-v1' or inputs!=prepared['inputs']:raise ValueError('actual02 bound prepared manifest required')
    for key,input_key in [('dataset','dataset'),('context','sequence_context'),('teacher_report','teacher_collection_report')]:
        if prepared[key]!=inputs[input_key]:raise ValueError('prepared pool binding changed')
        checked(prepared[key])
    template_id=identity(teacher_template);calibration_id=identity(calibration_path)
    if template_id['sha256']!=TEACHER02_SHA:raise ValueError('exact actualTeacher02 launch SHA required')
    if calibration_id['sha256']!=CALIBRATION_V02_SHA:raise ValueError('exact actualqualified02v02 calibration SHA required')
    template=read_json(teacher_template)
    if Path(option(template['command'],'--output')).resolve()!=Path(prepared['teacher_report']['path']).parent:raise ValueError('template output belongs to another collection')
    shadow=modules['tabletop_vision_command_shadow'];audit=dict(controller_source_sha256=shadow.controller_source_sha256(),action_contract_sha256=shadow.action_contract_sha256())
    calibration=read_json(calibration_path);validate_calibration_binding(calibration,prepared,audit)
    runtime.load_qualified_calibration(dict(calibration_id,source_run='02'),audit)
    checkpoint=checked(inputs['source_checkpoint'])
    if checkpoint['sha256']!=SOURCE_C58_SHA:raise ValueError('complete originalc58 source SHA required')
    initial_id=identity(source_initial_state)
    if Path(initial_id['path']).parent!=Path(prepared['teacher_report']['path']).parent:raise ValueError('actual02 native reset artifact required')
    import torch
    initial=torch.load(initial_id['path'],map_location='cpu',weights_only=False)
    if initial.get('seed')!=42 or initial.get('source')!='actual_native_cold_reset_before_camera':raise ValueError('actual original seed42 reset required')
    for name,shape in [('command',(8,12)),('roots',(8,2,7)),('object_position',(8,3)),('object_quaternion',(8,4)),('reference_bottom',(8,))]:
        value=initial.get(name)
        if not isinstance(value,torch.Tensor) or value.shape!=shape or not torch.isfinite(value).all():raise ValueError('actual reset shape/nonfinite: '+name)
    for name in ('q','qd'):
        value=initial.get(name)
        if not isinstance(value,torch.Tensor) or value.ndim!=3 or value.shape[:2]!=(8,2) or not torch.isfinite(value).all():raise ValueError('actual full q/qd reset required')
    bound=dict(prepared=prepared_id,teacher_template=template_id,source_checkpoint=checkpoint,source_initial_state=initial_id,
        calibration=calibration_id,teacher_dataset=prepared['dataset'],sequence_context=prepared['context'],teacher_report=prepared['teacher_report'])
    # Preserve external asset/scene/camera bytes named by the actual template.
    for flag in ('--pose-summary','--grasp-initialization','--mixed-grasp-initialization','--load-profile','--depth-camera-config','--vision-teacher-source'):
        bound[flag.lstrip('-').replace('-','_')]=identity(option(template['command'],flag))
    output=Path(output).absolute()
    if output==source or source in output.resolve().parents:raise ValueError('private output must be outside source')
    output.mkdir(parents=True,exist_ok=False);(output/'configs').mkdir();(output/'jobs').mkdir()
    jobs=[]
    for spec in job_specs(phase):
        target=output/'jobs'/spec['job_id'];frozen=target.with_name(target.name+'_source');row=dict(spec,output=str(target),frozen_source=str(frozen))
        config_path=None
        if spec['kind']=='tracking':
            config=tracking_config(spec['arm'],bound)
            config_path=output/'configs'/(spec['job_id']+'.json');new_json(config_path,config);runtime.load_tracking_config(config_path)
            row['config']=identity(config_path)
        row['command']=build_command(template,frozen,target,spec,config_path);jobs.append(row)
    hashes=source_manifest(source)
    for item in bound.values():checked(item)
    plan=dict(version=VERSION,phase=phase,source=str(source),source_sha256=hashes,launcher=identity(__file__),bound_inputs=bound,
        template_env=template.get('env',{}),jobs=jobs,bounded_job_count=len(jobs),maximum_jobs=5,native_jobs_executed=0,
        native_lock=LOCK_PATH,lock_policy='perjob_nonblocking;busy_refuses_immediately;no_wait_no_queue_no_retry',
        pure_student_results='tracking_original only;hybrid counters separate',teacher_mpc_common_priority_qualified=False,
        moving_wrist_qualification_granted=False,historical_M2_env799_932_661_reproduction_claimed=False,
        reset_scope='actual original8env seed42 coldreset;not a restored PhysX solver/contact checkpoint')
    new_json(output/'batch_plan.json',plan)
    return dict(plan=identity(output/'batch_plan.json'),bounded_job_count=len(jobs),native_jobs_executed=0)


def execute_batch(plan_path):
    plan_path=Path(plan_path).resolve(strict=True);plan=read_json(plan_path)
    if plan.get('version')!=VERSION or plan.get('native_lock')!=LOCK_PATH:raise ValueError('explicit finite prepared plan required')
    specs=job_specs(plan['phase'])
    if len(plan.get('jobs',[]))!=len(specs):raise ValueError('prespecified finite job count changed')
    source,modules=_source_api(plan['source']);wf=modules['tabletop_vision_stage6_workflow'];template=read_json(checked(plan['bound_inputs']['teacher_template'])['path'])
    checked(plan['launcher'])
    if identity(__file__)!=plan['launcher']:raise ValueError('launcher changed after preparation')
    for item in plan['bound_inputs'].values():checked(item)
    if source_manifest(source)!=plan['source_sha256']:raise ValueError('selected source changed;prepare a new reviewed plan')
    for job,spec in zip(plan['jobs'],specs):
        if any(job.get(k)!=v for k,v in spec.items()):raise ValueError('job arm/stage/hands differ from finite schedule')
        target=plan_path.parent/'jobs'/spec['job_id'];frozen=target.with_name(target.name+'_source')
        if job['output']!=str(target) or job['frozen_source']!=str(frozen):raise ValueError('job output/freeze escaped finite batch')
        config_path=validate_job_config(job,spec,plan['bound_inputs'])
        if config_path is not None and Path(config_path)!=plan_path.parent/'configs'/(spec['job_id']+'.json'):raise ValueError('tracking config escaped finite batch')
        if job['command']!=build_command(template,frozen,target,spec,config_path):raise ValueError('prepared command changed')
        for path in (target,frozen,target.with_name(target.name+'.log'),target.with_name(target.name+'_launch.json'),target.with_name(target.name+'_exit.json')):
            if path.exists() or path.is_symlink():raise FileExistsError(path)
    new_json(plan_path.parent/'execution_started.json',dict(plan=identity(plan_path),bounded_jobs=len(specs),no_retry=True,perjob_lock=True))
    results=[]
    try:
        for job in plan['jobs']:
            with native_lock():
                ensure_no_native_owner()
                for item in plan['bound_inputs'].values():checked(item)
                if 'config' in job:checked(job['config'])
                if source_manifest(source)!=plan['source_sha256']:raise ValueError('source changed between native jobs')
                frozen=Path(job['frozen_source']);target=Path(job['output']);hashes=wf.freeze_source(source,frozen)
                if hashes!=plan['source_sha256']:raise ValueError('frozen source differs from prepared exact bytes')
                launch=dict(job=job,source_sha256=hashes,template=plan['bound_inputs']['teacher_template'],env=plan['template_env'],cwd=str(frozen),
                    bound_inputs=plan['bound_inputs'],production_evaluation_certificate_issued=False,moving_wrist_qualification_granted=False)
                new_json(target.with_name(target.name+'_launch.json'),launch)
                print(json.dumps(dict(event='starting',job_id=job['job_id'],output=str(target))),flush=True)
                with target.with_name(target.name+'.log').open('x') as stream:
                    process=subprocess.run(job['command'],cwd=frozen,env=dict(os.environ,**plan['template_env']),stdout=stream,stderr=subprocess.STDOUT)
                exit_id=target.with_name(target.name+'_exit.json');new_json(exit_id,dict(exit_code=process.returncode))
                for item in plan['bound_inputs'].values():checked(item)
                if 'config' in job:checked(job['config'])
                verify_frozen_original_files(frozen,hashes)
                if source_manifest(source)!=plan['source_sha256']:raise ValueError('selected mutable source changed during native job;original frozen evidence retained')
                report_path=target/('tracking_full_episode_report.json' if job['kind']=='tracking' else 'tripod_stage_report.json')
                result=dict(job_id=job['job_id'],process_exit_code=process.returncode,exit_record=identity(exit_id),
                    report=identity(report_path) if report_path.is_file() else None,no_retry=True,original_input_and_frozen_bytes_unchanged=True)
                # Preserve runner outcomes, including physical failure or fixed
                # geometry refusal, without issuing a cross-mode success count.
                if report_path.is_file():
                    native_report=read_json(report_path);result['reported_status']=native_report.get('status',native_report.get('decision'))
                results.append(result);new_json(plan_path.parent/(job['job_id']+'_outcome.json'),result)
                print(json.dumps(dict(event='finished',**result)),flush=True)
    except BaseException as error:
        new_json(plan_path.parent/'batch_execution_error.json',dict(error=repr(error),completed_processes=results,no_queued_instance=True));raise
    summary=dict(version=VERSION,plan=identity(plan_path),results=results,started_jobs=len(results),maximum_jobs=5,
        production_evaluation_certificate_issued=False,moving_wrist_qualification_granted=False,teacher_mpc_common_priority_qualified=False,
        historical_M2_reproduction_claimed=False,ppo_updates=0)
    new_json(plan_path.parent/'batch_execution_report.json',summary);return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--teacher-template',type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--source-initial-state',type=Path)
    parser.add_argument('--calibration',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--phase',choices=('all','tracking','recovery'),default='all')
    parser.add_argument('--plan',type=Path);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.plan is not None:
        if any(getattr(args,key) is not None for key in ('source','teacher_template','prepared','source_initial_state','calibration','output')):parser.error('--plan cannot be combined with preparation inputs')
        if not args.execute:print(json.dumps(dict(plan=identity(args.plan),native_jobs_executed=0,execution_requested=False),indent=2));return 0
        result=execute_batch(args.plan)
    else:
        if args.execute:parser.error('prepare first;execution requires an existing --plan and explicit --execute')
        if any(getattr(args,key) is None for key in ('source','teacher_template','prepared','source_initial_state','calibration','output')):parser.error('explicit actual source/template/prepared/reset/calibration/output required')
        result=prepare_batch(args.source,args.teacher_template,args.prepared,args.source_initial_state,args.calibration,args.output,phase=args.phase)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
