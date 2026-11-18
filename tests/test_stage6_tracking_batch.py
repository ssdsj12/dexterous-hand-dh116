"""CPU-only launcher contracts; no simulator subprocess is invoked."""
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    path=Path(__file__).resolve().parents[1]/'scripts/run_dh116_stage6_tracking.py'
    assert path.is_file(), 'independent finite recovery/tracking launcher is required'
    spec=importlib.util.spec_from_file_location('recovery_tracking_launcher_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def template():
    return dict(command=['bash','/actual/source/scripts/python.sh','/actual/source/scripts/train_dh116_tabletop_gpu_ppo.py',
        '--num-envs','8','--seed','42','--physics-hz','480','--vision-width','224','--vision-height','96',
        '--solver-type','0','--position-iterations','64','--velocity-iterations','16','--mimic-frequency','400',
        '--passive-velocity-limit-rad-s','2.97','--end-support-clearance-m','.06','--grasp-controller','policy',
        '--palm-transport','driven_carriage','--surface-force-sharing','--free-wrist-grasp','--continuous-motion',
        '--pressure-relief','--force-feedback','--full-hand-contact','--surface-feedback',
        '--vision-camera-mode','fixed','--vision-action-distribution','tanh_gaussian',
        '--vision-teacher-source','/actual/teacher.pt','--vision-teacher-mpc',
        '--vision-teacher-mpc-profile','pressure_geometry_constrained','--vision-teacher-max-control-steps','1900',
        '--vision-teacher-hold-tail-control-steps','600','--vision-teacher-raw-chunk-frames','16',
        '--vision-sequence-context-sidecar','--output','/actual/teacher02'],env={'CUDA_VISIBLE_DEVICES':'3'})


def test_schedule_is_finite_prespecified_and_stage_hands_are_explicit():
    module=api();jobs=module.job_specs('all')
    assert [j['job_id'] for j in jobs]==['tracking_original','tracking_unilateral','tracking_common','stage_A_right','stage_B_both']
    assert module.job_specs('tracking')==jobs[:3] and module.job_specs('recovery')==jobs[3:]
    assert jobs[3]['stage']=='A' and jobs[3]['hands']=='right'
    assert jobs[4]['stage']=='B' and jobs[4]['hands']=='both'
    with pytest.raises(ValueError):module.job_specs('repeat_until_success')


@pytest.mark.parametrize('arm',['original','unilateral','common'])
def test_tracking_argv_preserves_physics_and_removes_all_teacher_ownership(arm):
    module=api();source=template();job=dict(job_id='tracking_'+arm,kind='tracking',arm=arm)
    command=module.build_command(source,Path('/new/frozen'),Path('/new/output'),job,Path('/bound/config.json'))
    assert command[:3]==['bash','/new/frozen/scripts/python.sh','/new/frozen/scripts/train_dh116_tabletop_gpu_ppo.py']
    for name in module.TEACHER_OPTIONS:assert not any(x==name or x.startswith(name+'=') for x in command)
    assert module.option(command,'--vision-tracking-experiment-config')=='/bound/config.json'
    assert '--bimanual-checkpoint-adapter' in command
    for name in ['--physics-hz','--solver-type','--position-iterations','--velocity-iterations','--mimic-frequency',
            '--passive-velocity-limit-rad-s','--end-support-clearance-m','--num-envs','--seed']:
        assert module.option(command,name)==module.option(source['command'],name)
    for name in ['--pressure-relief','--surface-force-sharing','--force-feedback','--full-hand-contact','--surface-feedback']:
        assert name in command
    assert module.option(command,'--vision-wrist-quaternion-preprocessing')=='wrist_quaternion_hemisphere_v1'


@pytest.mark.parametrize('stage,hands',[('A','right'),('B','both')])
def test_fixed_stage_argv_changes_only_declared_teacher_recovery_scope(stage,hands):
    module=api();job=dict(job_id='stage_'+stage+'_'+hands,kind='recovery',stage=stage,hands=hands)
    command=module.build_command(template(),Path('/new/frozen'),Path('/new/output'),job)
    assert '--vision-teacher-mpc' in command and module.option(command,'--vision-teacher-source')=='/actual/teacher.pt'
    assert module.option(command,'--vision-teacher-mpc-profile')=='pressure_reference'
    assert module.option(command,'--vision-teacher-max-control-steps')=='560'
    assert module.option(command,'--tripod-recovery-stage')==stage and module.option(command,'--tripod-recovery-hands')==hands
    assert '--vision-sequence-context-sidecar' not in command and '--vision-tracking-experiment-config' not in command


@pytest.mark.parametrize('bad',['seed','envs','mode','stage_hands','unqualified_mode'])
def test_wrong_template_or_candidate_scope_is_rejected_before_launch(bad):
    module=api();source=template();job=dict(job_id='stage_A_right',kind='recovery',stage='A',hands='right')
    if bad=='seed':source['command'][source['command'].index('--seed')+1]='43'
    elif bad=='envs':source['command'][source['command'].index('--num-envs')+1]='4'
    elif bad=='mode':source['command']+=['--vision-evaluation-checkpoint','/not/allowed.pt']
    elif bad=='stage_hands':job['hands']='both'
    else:job=dict(job_id='bad',kind='moving_wrist_teacher')
    with pytest.raises(ValueError):module.build_command(source,Path('/new/frozen'),Path('/new/output'),job)


def test_prepare_is_default_and_execute_is_never_inferred(monkeypatch,tmp_path):
    module=api();calls=[]
    monkeypatch.setattr(module,'prepare_batch',lambda *a,**k:calls.append(('prepare',a,k)) or {'prepared':True})
    monkeypatch.setattr(module,'execute_batch',lambda *a,**k:calls.append(('execute',a,k)))
    assert module.main(['--source','/source','--teacher-template','/teacher.json','--prepared','/prepared.json',
        '--source-initial-state','/reset.pt','--calibration','/cal.json','--output',str(tmp_path/'new')])==0
    assert len(calls)==1 and calls[0][0]=='prepare'


def test_source_hash_manifest_matches_freeze_exclusions_and_detects_changed_bytes(tmp_path):
    module=api();source=tmp_path/'source';source.mkdir();(source/'code.py').write_text('actual code')
    (source/'artifacts').mkdir();(source/'artifacts/ignored.log').write_text('private')
    (source/'__pycache__').mkdir();(source/'__pycache__/ignored.pyc').write_bytes(b'bytecode')
    before=module.source_manifest(source)
    assert set(before)=={'code.py'}
    (source/'code.py').write_text('changed code')
    assert before!=module.source_manifest(source)


def test_global_native_lock_is_nonblocking_and_never_queues_a_second_owner(tmp_path):
    module=api();lock=tmp_path/'sole-owner.lock'
    with module.native_lock(lock):
        with pytest.raises(module.NativeOwnerUnavailable,match='already held'):
            with module.native_lock(lock):pytest.fail('second owner was admitted')


def test_busy_process_refuses_before_freezing_or_starting_a_native_job(monkeypatch):
    module=api();calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:type('Result',(),{'returncode':0,'stdout':'existing native process'})())
    with pytest.raises(module.NativeOwnerUnavailable,match='active'):module.ensure_no_native_owner()
    assert not calls


def test_config_cannot_silently_change_requested_arm_or_add_common_to_original(tmp_path):
    module=api();bound={k:{'path':'/actual/'+k,'sha256':'a'*64} for k in ('source_checkpoint','source_initial_state','calibration')}
    config=dict(version='dh116-full-episode-tracking-experiment-v1',arm='common',num_envs=8,scene_seed=42,
        initial_state_tolerance=0.,checkpoint=bound['source_checkpoint'],source_initial_state=bound['source_initial_state'],
        calibration=dict(bound['calibration'],source_run='02'))
    path=tmp_path/'original.json';path.write_text(json.dumps(config));job=dict(config=module.identity(path))
    with pytest.raises(ValueError,match='configuration differs'):module.validate_job_config(job,dict(kind='tracking',arm='original'),bound)
    config['arm']='original';config.pop('calibration');path.write_text(json.dumps(config));job['config']=module.identity(path)
    assert module.validate_job_config(job,dict(kind='tracking',arm='original'),bound)==str(path)


@pytest.mark.parametrize('bad',['unqualified','cache_target','other_pool','other_source','wrong_trigger'])
def test_missing_real_calibration_qualification_or_binding_never_becomes_launchable(bad):
    module=api();prepared={k:dict(path='/actual/'+k,sha256='a'*64) for k in ('dataset','context')}
    audit=dict(controller_source_sha256='b'*64,action_contract_sha256='c'*64)
    report=dict(version='dh116-teacher-tracking-calibration-v1',qualified=True,
        target_source='physx_backend_get_dof_position_targets',feedback_source='original_advance_returned_refresh_before_TensorEnv_terminal_reset',
        dataset_path='/actual/dataset',dataset_sha256='a'*64,context_path='/actual/context',context_sha256='a'*64,
        **audit,teacher_error_p999_rad=[.002]*12,trigger_rad=[.01]*12)
    module.validate_calibration_binding(report,prepared,audit)
    if bad=='unqualified':report['qualified']=False
    elif bad=='cache_target':report['target_source']='IsaacLab_command_cache'
    elif bad=='other_pool':report['dataset_sha256']='d'*64
    elif bad=='other_source':report['controller_source_sha256']='e'*64
    else:report['trigger_rad'][0]=.02
    with pytest.raises(ValueError):module.validate_calibration_binding(report,prepared,audit)


def test_execute_refuses_busy_owner_before_any_freeze_or_subprocess(monkeypatch,tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace as NS
    module=api();source=tmp_path/'source';source.mkdir();(source/'actual.py').write_text('stable code')
    batch=tmp_path/'batch';batch.mkdir();(batch/'jobs').mkdir();template_path=batch/'teacher.json';template_path.write_text(json.dumps(template()))
    calls=[]
    def freeze(*args):calls.append('freeze');pytest.fail('busy owner must refuse before freeze')
    monkeypatch.setattr(module,'_source_api',lambda *args:(source,{'tabletop_vision_stage6_workflow':NS(freeze_source=freeze)}))
    @contextmanager
    def available_test_lock():yield
    monkeypatch.setattr(module,'native_lock',available_test_lock)
    def busy():raise module.NativeOwnerUnavailable('existing native owner active')
    monkeypatch.setattr(module,'ensure_no_native_owner',busy)
    jobs=[]
    for spec in module.job_specs('recovery'):
        target=batch/'jobs'/spec['job_id'];frozen=target.with_name(target.name+'_source')
        jobs.append(dict(spec,output=str(target),frozen_source=str(frozen),command=module.build_command(template(),frozen,target,spec)))
    plan=dict(version=module.VERSION,native_lock=module.LOCK_PATH,phase='recovery',source=str(source),
        source_sha256=module.source_manifest(source),launcher=module.identity(module.__file__),
        bound_inputs={'teacher_template':module.identity(template_path)},jobs=jobs)
    path=batch/'batch_plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(module.NativeOwnerUnavailable):module.execute_batch(path)
    assert not calls and not list((batch/'jobs').iterdir())
    assert (batch/'batch_execution_error.json').is_file()


def test_after_job_original_frozen_bytes_are_rechecked_without_treating_new_runtime_logs_as_source(tmp_path):
    module=api();frozen=tmp_path/'frozen';frozen.mkdir();(frozen/'controller.py').write_text('original unchanged code')
    before=module.source_manifest(frozen)
    (frozen/'runtime_new.log').write_text('simulation log')
    module.verify_frozen_original_files(frozen,before)
    (frozen/'controller.py').write_text('changed code')
    with pytest.raises(ValueError,match='frozen original source changed'):module.verify_frozen_original_files(frozen,before)
