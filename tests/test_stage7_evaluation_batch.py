"""Synthetic CPU file fixtures; none is evidence of a new native trial."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_dh116_stage7_evaluations.py'
    assert path.is_file(), 'independent finite Stage7 branch driver is missing'
    spec = importlib.util.spec_from_file_location('stage7_finite_driver', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_schedule_has_all_four_arms_four_events_and_four_lengths():
    module = driver(); jobs = module.job_specs()
    assert len(jobs) == len({row['job_id'] for row in jobs}) == 64
    assert {row['arm'] for row in jobs} == {'E0','B0','B1','B2'}
    assert {row['event'] for row in jobs} == {'PRE_CONTACT','CONTACT','LIFT_ONSET','HOLD'}
    assert {row['branch_steps'] for row in jobs} == {1,4,16,60}
    assert all(row['mode'] == 'branch' for row in jobs)
    assert jobs[:4] == [dict(job_id=f'branch_E0_PRE_CONTACT_{n:02d}',arm='E0',mode='branch',
                            event='PRE_CONTACT',branch_steps=n) for n in (1,4,16,60)]
    for arm in ('E0','B0','B1','B2'):
        selected = module.job_specs(arm)
        assert len(selected) == 16 and {row['arm'] for row in selected} == {arm}
    with pytest.raises(ValueError): module.job_specs('BEST_MSE')


def test_exact_actual_teacher02_template_digest_is_pinned():
    assert getattr(driver(),'TEACHER02_SHA',None)=='1a5faead16451a7dd0b523cdc6a1e236a6882c80c26a2a2bd295d2a2dfe2e2d9'


@pytest.fixture
def files(tmp_path, monkeypatch):
    from test_stage6_workflow_integration import _bound_inputs, _prepare
    from go2_pvcnn.control.dh116 import tabletop_vision_stage6_workflow as wf
    from go2_pvcnn.control.dh116 import tabletop_vision_branching_eval as branch
    from go2_pvcnn.control.dh116.tabletop_vision_stage7_event_reference import build_event_reference
    module = driver(); fixture = _bound_inputs(tmp_path)
    data = torch.load(tmp_path/'teacher.pt',map_location='cpu',weights_only=False)
    context = torch.load(tmp_path/'context.pt',map_location='cpu',weights_only=False)
    for env in range(4):
        start = env*16
        context['events']['opposed_contact'][start:start+16] = False
        context['events']['opposed_contact'][start+2:start+16] = True
        context['events']['measured_palm_velocity_m_s'][start:start+16] = 0
        context['events']['measured_palm_velocity_m_s'][start+3:start+6,:,2] = .001
        context['events']['native_hold_qualified'][start:start+16] = False
        context['events']['native_hold_qualified'][start+6:start+16] = True
    torch.save(context,tmp_path/'context.pt')
    collection = json.loads(fixture.report.read_text()); collection.pop('synthetic_cpu_fixture',None)
    collection['successful_sequence_context'] = wf.file_identity(tmp_path/'context.pt')
    write_json(fixture.report,collection)
    inputs = json.loads(fixture.inputs.read_text())
    inputs['sequence_context'] = wf.file_identity(tmp_path/'context.pt')
    inputs['teacher_collection_report'] = wf.file_identity(fixture.report)
    write_json(fixture.inputs,inputs)
    prepared = _prepare(fixture,tmp_path/'prepared'); prepared_data = json.loads(prepared.read_text())
    source = wf.file_identity(fixture.source_path)
    # The fixture checkpoint has deliberately synthetic bytes. Production keeps
    # the fixed real c58 hash; only this CPU fixture's expected digest is replaced.
    monkeypatch.setattr(module,'SOURCE_C58_SHA',source['sha256'])
    initial = tmp_path/'initial_native_state.pt'
    roots=torch.zeros(8,2,7);roots[...,3]=1.;quat=torch.zeros(8,4);quat[:,0]=1.
    torch.save(dict(seed=42,source='actual_native_cold_reset_before_camera',q=torch.zeros(8,2,11),
        qd=torch.zeros(8,2,11),command=torch.zeros(8,12),roots=roots,object_position=torch.zeros(8,3),
        object_quaternion=quat,reference_bottom=torch.zeros(8)),initial)
    common = dict(teacher_dataset=prepared_data['dataset'],sequence_context=prepared_data['context'],
        source_collection_report=prepared_data['teacher_report'],source_initial_state=wf.file_identity(initial))
    external={}
    for name in ('pose-summary','grasp-initialization','mixed-grasp-initialization','load-profile','depth-camera-config','vision-teacher-source'):
        path=tmp_path/(name+'.json');write_json(path,{'synthetic_cpu_fixture':name});external[name]=path
    template=tmp_path/'teacher02_launch.json'
    write_json(template,dict(command=['bash','/actual/scripts/python.sh','/actual/scripts/train_dh116_tabletop_gpu_ppo.py',
        '--seed','42','--num-envs','8','--physics-hz','480','--vision-width','224','--vision-height','96',
        '--output',str(tmp_path),'--vision-teacher-mpc',
        '--vision-teacher-mpc-profile','pressure_geometry_constrained','--object-material','extreme_grip',
        '--kit_args=--allow-root --/renderer/activeGpu=3']+[item for name,path in external.items() for item in ('--'+name,str(path))],env=dict(CUDA_VISIBLE_DEVICES='3',
        DH116_PYTHON='/actual/python',OMP_NUM_THREADS='2')))
    monkeypatch.setattr(module,'TEACHER02_SHA',wf.file_identity(template)['sha256'])
    models={'E0':source}; training={}
    for index,arm in enumerate(('B0','B1','B2')):
        saved=copy.deepcopy(fixture.source)
        key=next(k for k in saved['policy'] if k.startswith('actor.'));saved['policy'][key].add_(.001*(index+1))
        saved['stage6_sequence_bc']=dict(profile=arm,updates=3000,status='completed',source_checkpoint=source,
            dataset=prepared_data['dataset'],context=prepared_data['context'])
        model=tmp_path/(arm+'_actual_final.pt');torch.save(saved,model);models[arm]=wf.file_identity(model)
        report=tmp_path/(arm+'_training.json')
        write_json(report,dict(profile=arm,status='completed',steps=3000,requested_steps=3000,
            valid_exposures=192000,source_immutable=True,ppo_updates=0,prepared=wf.file_identity(prepared),
            source_checkpoint=source,sequence_report=dict(sequence_length=16,min_valid_steps=16,
                max_valid_steps=16,padded_window_count=0),checkpoint_selection='prespecified_final_update_not_train_MSE',
            checkpoints=[dict(step=3000,status='completed',**models[arm])]))
        training[arm]=report
    fields={n:dict(matched=True,dtype_equal=True,shape_equal=True,finite=True) for n in branch.SOURCE_STATE_FIELDS}
    reset=dict(matched=True,tolerance=0.,fields=fields)
    cases=[]; policies=[]; raw=[]; outcomes=[]
    for env in range(4):
        meta=data['sample_metadata'][env*16];uid=f'replay:{env}:0'
        cases.append(dict(env_id=env,episode_id=str(env),source_episode_uid=meta['episode_uid'],
            source_run_uuid=meta['run_uuid'],recorded_teacher_tape_length=16,
            source_trajectory_id=meta['source_trajectory_id'],event_entry_ticks=dict(CONTACT=2,LIFT_ONSET=3,HOLD=6)))
        outcomes.append(dict(env_id=env,episode_id=str(env),execution_episode_id=0,execution_episode_uid=uid,
            terminal=dict(policy_step=15,physics_step=128,timestamp_s=148/480,failure_code=3,
                action_owner='teacher_replay'),native_success=False,student_policy_steps_executed=0,
            teacher_prefix_policy_steps_executed=0,teacher_replay_policy_steps_executed=16))
        for tick in range(16):
            i=env*16+tick;action=data['actions'][i].tolist();previous=data['sample_metadata'][i]['previous_action']
            policies.append(dict(env_id=env,policy_step=tick,run_uuid='replay',episode_id=0,episode_uid=uid,
                recorded_teacher_episode_uid=meta['episode_uid'],owner='teacher_replay',issued_action=action,
                recorded_teacher_action=action,actual_previous_issued_action=previous,
                clock_before_action=dict(physics_step=8*tick,simulation_step_index=20+8*tick,timestamp_s=(20+8*tick)/480)))
            for sub in range(1,9):
                step=tick*8+sub
                raw.append(dict(env_id=env,source_episode_id=str(env),episode_uid=uid,policy_step=tick,
                    physics_step=step,timestamp_s=(20+step)/480,action_owner='teacher_replay',failure_code=3 if step==128 else 0,
                    payload=dict(live=True,done=step==128,native_success=False,measurement_valid=True,
                        contact_valid=True,bilateral_opposed_contact=tick>1 or tick==1 and sub>=3)))
    replay_dir=tmp_path/'replay';replay_dir.mkdir()
    for name,rows in [('source_physics_480hz.jsonl.gz',sorted(raw,key=lambda r:(r['physics_step'],r['env_id']))),
                      ('live_policy_60hz.jsonl.gz',sorted(policies,key=lambda r:(r['policy_step'],r['env_id'])))]:
        with gzip.open(replay_dir/name,'wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    replay=dict(version=branch.VERSION,mode='teacher-replay',status='completed',error=None,native_runtime_verified=True,
        execution_run_uuid='replay',ppo_updates=0,evaluated_source_count=4,completed_count=4,native_success_count=0,
        pure_student_native_success_count=0,source_rows=outcomes,source_case_definitions=cases,
        inputs=dict(checkpoint=source,**common),raw_physics=wf.file_identity(replay_dir/'source_physics_480hz.jsonl.gz'))
    neutral=[dict(env_id=i,action='neutral_zero24',evaluated=False,expert=False) for i in range(4,8)]
    replay['unevaluated_environments']=neutral
    replay_path=replay_dir/'stage6_native_report.json';write_json(replay_path,replay)
    write_json(replay_dir/'source_reset_comparison.json',reset)
    write_json(replay_dir/'source_camera_warmup_comparison.json',dict(reset,before_clock={'step':0},after_clock={'step':0}))
    eventref=tmp_path/'event_reference.json'
    build_event_reference(dict(source_checkpoint=source,native_replay_report=wf.file_identity(replay_path),**common),eventref)
    sources={}
    for arm in models:
        folder=tmp_path/('source_'+arm);folder.mkdir();report=copy.deepcopy(replay)
        report.update(mode='source',execution_run_uuid=arm,inputs=dict(checkpoint=models[arm],**common))
        for env,row in enumerate(report['source_rows']):
            row.update(execution_episode_uid=f'{arm}:{env}:0',student_policy_steps_executed=16,
                teacher_replay_policy_steps_executed=0)
            row['terminal']['action_owner']='student'
        path=folder/'stage6_native_report.json';write_json(path,report)
        write_json(folder/'source_reset_comparison.json',reset)
        write_json(folder/'source_camera_warmup_comparison.json',dict(reset,before_clock={'step':0},after_clock={'step':0}))
        sources[arm]=path
    source_root=Path(wf.__file__).resolve().parents[4]
    return SimpleNamespace(module=module,wf=wf,source=source_root,prepared=prepared,initial=initial,template=template,
        training=training,models=models,common=common,replay=replay_path,eventref=eventref,sources=sources,root=tmp_path,external=external)


def prepare(f,output,arm='all'):
    return f.module.prepare_batch(f.source,f.prepared,f.initial,output,arm=arm,training_reports=f.training,
        source_outcomes=f.sources,replay_report=f.replay,event_reference=f.eventref,native_template=f.template)


def test_prepare_binds_all_baselines_event_reference_and_only_offline_reuse(files,monkeypatch):
    f=files;monkeypatch.setattr(f.wf,'launch_native',lambda *a,**k:pytest.fail('prepare cannot launch native'))
    plan=prepare(f,f.root/'batch')
    assert plan['version']=='dh116-stage7-finite-branch-batch-v1' and len(plan['jobs'])==64
    assert plan['native_jobs_executed']==0 and plan['concurrent_native_instances']==1
    assert plan['automatic_retry'] is False and plan['ppo_updates']==0
    assert set(plan['source_outcomes'])==set(f.models)
    assert plan['event_reference']==f.wf.file_identity(f.eventref)
    assert plan['native_trials_reused']==0 and plan['physical_state_reused'] is False
    assert set(plan['automatic_offline_cache_arms'])==set(f.models)
    assert 'tabletop_vision_stage7_event_reference' in plan['comparison_dependencies']
    assert plan['legacy_helper']==f.wf.file_identity(f.source/'scripts/run_dh116_stage6_evaluations.py')
    for job in plan['jobs']:
        config=json.loads(Path(job['config']['path']).read_text())
        assert config['event_reference']==plan['event_reference']
        assert config['checkpoint']==f.models[job['arm']] and job['cold_reset'] is True
        assert job['neutral_env_ids']==[4,5,6,7]
    assert len(prepare(f,f.root/'one_arm','B1')['jobs'])==16


def test_prepare_validates_same_immutable_event_facts_once_for_64_generated_configs(files,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_vision_stage7_event_reference as reference
    calls=[];original=reference.load_event_reference
    def counted(*args,**kwargs):
        calls.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(reference,'load_event_reference',counted)
    plan=prepare(files,files.root/'one_verified_reference')
    assert len(plan['jobs'])==64 and len(calls)==1


@pytest.mark.parametrize('bad',['missing_B2','pending_B2','wrong_owner','missing_terminal','wrong_model',
    'wrong_common','wrong_arm','reset','warmup_clock','duplicate_uid','teacher_steps','synthetic','count'])
def test_all_four_genuine_source_gate_closes_before_output_creation(files,bad):
    f=files;path=f.sources['B2'];value=json.loads(path.read_text())
    if bad=='missing_B2':f.sources.pop('B2')
    elif bad=='pending_B2':value['status']='in_progress'
    elif bad=='wrong_owner':value['source_rows'][0]['terminal']['action_owner']='teacher_prefix'
    elif bad=='missing_terminal':value['source_rows'][0]['terminal']=None
    elif bad=='wrong_model':value['inputs']['checkpoint']=f.models['B1']
    elif bad=='wrong_common':value['inputs']['source_initial_state']=f.models['E0']
    elif bad=='wrong_arm':value['mode']='branch'
    elif bad=='reset':
        p=path.parent/'source_reset_comparison.json';r=json.loads(p.read_text());r['fields']['q']['matched']=False;write_json(p,r)
    elif bad=='warmup_clock':
        p=path.parent/'source_camera_warmup_comparison.json';r=json.loads(p.read_text());r['after_clock']={'step':1};write_json(p,r)
    elif bad=='duplicate_uid':value['source_rows'][1]['execution_episode_uid']=value['source_rows'][0]['execution_episode_uid']
    elif bad=='teacher_steps':value['source_rows'][0]['teacher_prefix_policy_steps_executed']=1
    elif bad=='synthetic':value['synthetic_cpu_fixture']=True
    elif bad=='count':value['completed_count']=3
    write_json(path,value);out=f.root/'refused'
    with pytest.raises((ValueError,FileNotFoundError)):prepare(f,out,'E0')
    assert not out.exists()


def test_prepare_requires_real_c58_and_final3000_reports_even_for_filtered_arm(files):
    f=files;f.module.SOURCE_C58_SHA='0'*64
    with pytest.raises(ValueError,match='c58'):prepare(f,f.root/'wrong_c58','E0')
    f.module.SOURCE_C58_SHA=f.models['E0']['sha256']
    p=f.training['B2'];r=json.loads(p.read_text());r['status']='in_progress';write_json(p,r)
    with pytest.raises(ValueError,match='3000'):prepare(f,f.root/'pending_final','E0')


def test_changed_template_material_cannot_enter_even_with_same_clock_and_reset(files):
    f=files;r=json.loads(f.template.read_text());r['command'][r['command'].index('--object-material')+1]='different'
    write_json(f.template,r)
    with pytest.raises(ValueError,match='Teacher02|template.*SHA'):prepare(f,f.root/'wrong_material')


@pytest.mark.parametrize('bad',['reused_run','terminal_policy_count'])
def test_source_first_episode_execution_cannot_be_reused_or_truncated(files,bad):
    f=files;p=f.sources['B2'];r=json.loads(p.read_text())
    if bad=='reused_run':
        r['execution_run_uuid']='B1'
        for env,row in enumerate(r['source_rows']):row['execution_episode_uid']=f'B1:{env}:0'
    else:r['source_rows'][0]['student_policy_steps_executed']-=1
    write_json(p,r)
    with pytest.raises(ValueError,match='execution|terminal|policy'):prepare(f,f.root/'invalid_execution')


@pytest.mark.parametrize('bad',['config','schedule','driver_bytes','source_bytes','source_outcome','external_profile','plan_during_preflight'])
def test_execute_revalidates_plan_sources_and_schedule_before_native(files,monkeypatch,bad):
    f=files;out=f.root/'batch';plan=prepare(f,out,'E0');path=out/'batch_plan.json'
    monkeypatch.setattr(f.module,'launch_job',lambda *a,**k:pytest.fail('mutated input must not reach native'))
    if bad=='config':
        p=Path(plan['jobs'][0]['config']['path']);r=json.loads(p.read_text());r['branch_steps']=60;write_json(p,r)
    elif bad=='schedule':plan['jobs'][0]['event']='HOLD';write_json(path,plan)
    elif bad=='driver_bytes':plan['driver']['sha256']='0'*64;write_json(path,plan)
    elif bad=='source_bytes':plan['source_sha256']['scripts/run_dh116_stage7_evaluations.py']='0'*64;write_json(path,plan)
    elif bad=='source_outcome':
        p=f.sources['B1'];r=json.loads(p.read_text());r['status']='in_progress';write_json(p,r)
    elif bad=='external_profile':f.external['load-profile'].write_text('{"changed":true}')
    elif bad=='plan_during_preflight':
        original=f.module._bound_checks
        def change_plan(bound,wf):
            original(bound,wf);changed=json.loads(path.read_text());changed['concurrent_native_instances']=2
            write_json(path,changed);monkeypatch.setattr(f.module,'_bound_checks',original)
        monkeypatch.setattr(f.module,'_bound_checks',change_plan)
    with pytest.raises(ValueError):f.module.execute_batch(path)
    assert not (out/'execution_started.json').exists()


def test_busy_owner_refuses_before_freeze(files,monkeypatch):
    f=files;plan=prepare(f,f.root/'batch','E0')
    monkeypatch.setattr(f.module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='82267 native owner'))
    monkeypatch.setattr(f.wf,'freeze_source',lambda *a,**k:pytest.fail('busy owner must not freeze or queue'))
    with pytest.raises(RuntimeError,match='owner|busy'):f.module.launch_job(plan,plan['jobs'][0],plan['jobs'][0]['config'])


def test_lock_is_nonblocking_and_released_on_error(tmp_path):
    import fcntl
    module=driver();path=tmp_path/'native.lock'
    with path.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='busy|lock'):
            with module.native_lock(path):pytest.fail('no queued lock')
    with pytest.raises(RuntimeError,match='fixture'):
        with module.native_lock(path):raise RuntimeError('fixture')
    with module.native_lock(path):pass


@pytest.mark.parametrize('arm',['E0','B0','B1','B2'])
def test_cache_binding_accepts_each_arm_only_with_exact_event_reference(files,monkeypatch,arm):
    f=files;from go2_pvcnn.control.dh116 import tabletop_vision_comparison_cache as cache
    seen=[]
    def load(identity,expected_config,*,implementation_sha256):
        seen.append(copy.deepcopy(expected_config));return {}
    monkeypatch.setattr(cache,'load_comparison_cache',load)
    path=f.root/'cache.json';path.write_text('{}');cache_id=f.wf.file_identity(path)
    result=f.module.validate_comparison_caches({arm:cache_id},{a:dict(checkpoint=v) for a,v in f.models.items()},
        dict(f.common,event_reference=f.wf.file_identity(f.eventref)),f.module._source_api(f.source)[2].comparison_dependencies()['tabletop_vision_branching_eval'])
    assert result=={arm:cache_id} and seen[0]['event_reference']==f.wf.file_identity(f.eventref)
    assert seen[0]['checkpoint']==f.models[arm]


def test_cli_list_is_nonexecuting_and_flags_are_explicit(capsys):
    module=driver();assert module.main(['--list-only','--arm','B0'])==0
    value=json.loads(capsys.readouterr().out);assert len(value['jobs'])==16 and value['native_jobs_executed']==0
    with pytest.raises(SystemExit):module.main(['--execute','missing.json','--source-outcome','B2=other'])


def test_64_fresh_branch_calls_reuse_only_each_arms_first_offline_comparisons(files,monkeypatch):
    f=files
    from go2_pvcnn.control.dh116 import tabletop_vision_comparison_cache as cache
    out=f.root/'batch';plan=prepare(f,out);calls=[]
    def synthetic_child(bound,job,config_identity):
        # This replaces process launch only. Real immutable file/cache validators
        # and the entire finite driver execute; these are never native results.
        config=json.loads(Path(config_identity['path']).read_text());folder=Path(job['output']);folder.mkdir(parents=True)
        report=json.loads(f.sources[job['arm']].read_text())
        report.update(mode='branch',status='bounded_branch_completed',event=job['event'],branch_steps=job['branch_steps'],
            inputs=dict(checkpoint=f.models[job['arm']],event_reference=bound['event_reference'],**f.common),
            pure_student_native_success_count=0)
        implementation=bound['comparison_dependencies']['tabletop_vision_branching_eval']
        reused='same_state_comparison_cache' in config
        if reused:
            comparisons=cache.load_comparison_cache(config['same_state_comparison_cache'],config,
                implementation_sha256=implementation)
        else:
            stream=folder/'same_state_teacher_observation_comparisons.jsonl.gz'
            with gzip.open(stream,'wt') as output:
                for i in range(3):output.write(json.dumps({'synthetic_offline_row':i})+'\n')
            comparisons=dict(**f.wf.file_identity(stream),policy_rows=3,comparison_implementation_sha256=implementation,
                comparison_dependencies=bound['comparison_dependencies'],reused_offline_comparisons=False,
                comparison_inputs={k:config[k] for k in ('checkpoint','teacher_dataset','sequence_context','event_reference')},
                physical_failure_causality_claimed=False)
        report['same_state_comparisons']=comparisons
        write_json(folder/'stage6_native_report.json',report)
        for name in ('source_reset_comparison','source_camera_warmup_comparison'):
            (folder/(name+'.json')).write_bytes((f.sources[job['arm']].parent/(name+'.json')).read_bytes())
        calls.append((job['arm'],job['event'],job['branch_steps'],reused,folder));return folder
    monkeypatch.setattr(f.module,'launch_job',synthetic_child)
    result=f.module.execute_batch(out/'batch_plan.json')
    assert result['executed_jobs']==64 and result['native_trials_reused']==0 and result['physical_state_reused'] is False
    assert len({row[4] for row in calls})==64 and set(result['comparison_caches'])==set(f.models)
    for arm in f.models:
        selected=[row for row in calls if row[0]==arm]
        assert len(selected)==16 and selected[0][3] is False and all(row[3] is True for row in selected[1:])
        metadata=json.loads(Path(result['comparison_caches'][arm]['path']).read_text())
        assert metadata['comparison_inputs']['event_reference']==plan['event_reference']
        assert metadata['native_trials_reused']==0
