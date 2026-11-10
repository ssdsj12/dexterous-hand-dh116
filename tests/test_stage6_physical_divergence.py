"""Read-only repeat-range analysis, never a controller or success certificate."""
import copy
import importlib
import math

import numpy as np
import pytest


def api():
    name='go2_pvcnn.control.dh116.tabletop_vision_physical_divergence'
    assert importlib.util.find_spec(name) is not None, 'independent physical repeat-range analysis module required'
    return importlib.import_module(name)


def sample(tick=0,substep=1,*,q=0.,target=0.,owner='teacher_replay',uid='run:0:0',terminal=False,contact=None):
    step=tick*8+substep;when=(step+18)*float(np.float32(1/480))
    return dict(env_id=0,source_episode_uid='original02:0:first',execution_episode_uid=uid,
        policy_step=tick,substep=substep,world_physics_step=step,timestamp_s=when,action_owner=owner,
        phase='post_physics_before_terminal_reset',q_rad=np.full(12,q,dtype=np.float64),
        qd_rad_s=np.zeros(12),backend_target_rad=np.full(12,target,dtype=np.float64),
        base_link_position_w_m=np.zeros((2,3)),palm_link_position_w_m=None,
        tracking_pair_available=True,independent_getter_clock_pair_available=False,
        clock_quality='shared_postphysical_callback_sample_clock',
        contact_valid=None if contact is None else True,bilateral_opposed_contact=contact,
        slip_valid=False,slip_m=None,lift_valid=False,lift_m=None,
        failure_code=0,terminal=terminal,live=True,original_task_event=None)


def run(uid,rows):return dict(execution_run_uuid=uid,native_runtime_verified=True,samples=rows,provenance={'fixture_only':True})


def test_exact_two_recorded_repeats_preserve_zero_range_without_statistical_confidence_claim():
    module=api();first=run('Teacher02',[sample(uid='Teacher02:0:first')]);second=run('actual_tape_repeat',[sample()])
    envelope=module.build_repeat_envelope([first,second]);report=module.compare_to_repeat(run('candidate',[sample(owner='student')]),envelope,np.full(12,.01))
    assert envelope['reference_run_count']==2 and envelope['statistical_confidence_interval'] is False
    assert envelope['reference_exact_match']['q_rad']['all_equal'] is True
    assert envelope['reference_exact_match']['backend_target_rad']['all_equal'] is True
    row=report['per_env'][0]
    assert row['first_numeric_command_departure'] is None and row['first_numeric_physical_departure'] is None
    assert row['first_significant_tracking_departure'] is None
    assert report['causality_claimed'] is False and report['control_modified'] is False
    assert report['availability']['repeat_contact']['available'] is False
    assert report['availability']['palm_link_position_w_m']['available'] is False


def test_small_physical_numeric_difference_is_not_a_significant_tracking_or_task_event():
    module=api();envelope=module.build_repeat_envelope([run('a',[sample()]),run('b',[sample()])])
    candidate=sample(q=1e-12,target=1e-12,owner='student')
    report=module.compare_to_repeat(run('candidate',[candidate]),envelope,np.full(12,.01))
    row=report['per_env'][0]
    assert row['first_numeric_command_departure'] is not None
    assert row['first_numeric_physical_departure'] is not None
    assert row['first_significant_tracking_departure'] is None and row['first_absolute_task_event'] is None


@pytest.mark.parametrize('owner',['teacher_prefix','student'])
def test_tracking_departure_requires_both_outside_repeat_range_and_qualified_calibration_threshold(owner):
    module=api();refs=[sample(substep=1),sample(substep=2)]
    envelope=module.build_repeat_envelope([run('a',refs),run('b',copy.deepcopy(refs))])
    candidate=[sample(substep=1,q=.005,owner=owner),sample(substep=2,q=.02,owner=owner)]
    report=module.compare_to_repeat(run('candidate',candidate),envelope,np.full(12,.01))
    event=report['per_env'][0]['first_significant_tracking_departure']
    assert event['substep']==2 and event['action_owner']==owner and event['max_abs_tracking_error_rad']==.02
    assert event['cause_claimed'] is False
    assert event['owner_scope']==('prefix_event_not_Student_evidence' if owner=='teacher_prefix' else 'Student_owned_sample;chronology_not_causality')
    assert event['independent_getter_clock_pair_available'] is False


def test_large_teacher_tracking_repeat_error_is_not_relabelled_as_candidate_departure():
    module=api();envelope=module.build_repeat_envelope([run('a',[sample(q=.02)]),run('b',[sample(q=.03)])])
    report=module.compare_to_repeat(run('candidate',[sample(q=.025,owner='student')]),envelope,np.full(12,.01))
    assert report['per_env'][0]['first_significant_tracking_departure'] is None


def test_post_contact_missing_in_original_teacher_keeps_repeat_contact_na_and_absolute_loss_separate():
    module=api();teacher=[sample(substep=1),sample(substep=2)]
    replay=[sample(substep=1,contact=True),sample(substep=2,contact=False)]
    envelope=module.build_repeat_envelope([run('Teacher02',teacher),run('replay',replay)])
    report=module.compare_to_repeat(run('candidate',[sample(substep=1,contact=True,owner='teacher_prefix'),sample(substep=2,contact=False,owner='teacher_prefix')]),envelope,np.full(12,.01))
    row=report['per_env'][0]
    assert row['first_absolute_task_event']['kind']=='effective_opposed_contact_lost'
    assert row['first_absolute_task_event']['repeat_divergence_qualified'] is False
    assert row['first_repeat_contact_loss'] is None
    assert report['availability']['repeat_contact']['maximum_reference_samples']==1


def test_repeat_contact_departure_requires_two_same_postphysical_valid_opposed_references():
    module=api();refs=[sample(substep=1,contact=True),sample(substep=2,contact=True)]
    envelope=module.build_repeat_envelope([run('a',refs),run('b',copy.deepcopy(refs))])
    report=module.compare_to_repeat(run('candidate',[sample(substep=1,contact=True),sample(substep=2,contact=False)]),envelope,np.full(12,.01))
    assert report['per_env'][0]['first_repeat_contact_loss']['substep']==2
    assert report['per_env'][0]['first_repeat_contact_loss']['cause_claimed'] is False


def test_first_terminal_freezes_identity_and_postterminal_reset_data_cannot_replace_it():
    module=api();refs=[sample(substep=1),sample(substep=2)]
    envelope=module.build_repeat_envelope([run('a',refs),run('b',copy.deepcopy(refs))])
    terminal=sample(substep=1,terminal=True);after=sample(substep=2,q=.2,uid='next_episode')
    report=module.compare_to_repeat(run('candidate',[terminal,after]),envelope,np.full(12,.01))
    row=report['per_env'][0]
    assert row['first_terminal']['execution_episode_uid']=='run:0:0' and row['first_terminal']['substep']==1
    assert row['first_significant_tracking_departure'] is None and report['postterminal_samples_excluded']==1
    changed=sample(substep=2,uid='unreported_reset')
    with pytest.raises(ValueError,match='episode identity'):module.compare_to_repeat(run('candidate',[sample(),changed]),envelope,np.full(12,.01))


@pytest.mark.parametrize('bad',['same_run','not_native','pre_physics','duplicate'])
def test_repeat_qualification_rejects_same_run_copies_non_native_preq_and_duplicate_identity(bad):
    module=api();a=run('a',[sample()]);b=run('b',[sample()])
    if bad=='same_run':b['execution_run_uuid']='a'
    elif bad=='not_native':b['native_runtime_verified']=False
    elif bad=='pre_physics':b['samples'][0]['phase']='pre_control'
    else:b['samples'].append(sample())
    with pytest.raises(ValueError):module.build_repeat_envelope([a,b])


def test_clock_bridge_uses_recorded_policy_sdk_anchor_and_reports_roundoff_without_getter_invention():
    module=api();cadence=float(np.float32(1/480));bridge=module.ReportedClockBridge(cadence,0.)
    anchor={'physics_step':0,'simulation_step_index':18,'timestamp_s':18*cadence}
    bridge.add_policy_anchor(anchor)
    result=bridge.verify_post(1,19*cadence,anchor)
    assert result['substep']==1 and result['derived_sdk_physics_step']==19
    assert result['independent_getter_clock_pair_available'] is False
    assert bridge.summary()['maximum_timestamp_residual_s']==0.
    assert bridge.summary()['sdk_minus_world_physics_step']==18
    with pytest.raises(ValueError,match='mapping'):bridge.add_policy_anchor(dict(physics_step=8,simulation_step_index=27,timestamp_s=27*cadence))
    with pytest.raises(ValueError,match='timestamp'):bridge.verify_post(2,20*cadence+1e-7,anchor)


def test_active_fingers_use_names_and_native_units_instead_of_aggregating_all17_as_radians():
    module=api();names=['wrist_x','finger21','finger12','finger51','finger41','finger31','finger11']
    values=np.array([99.,3.,2.,6.,5.,4.,1.]);types=np.array([1,0,0,0,0,0,0],dtype=np.uint8)
    units={name:('m' if name=='wrist_x' else 'rad') for name in names}
    assert module.active_fingers(values,names,types,units).tolist()==[1.,2.,3.,4.,5.,6.]
    types[1]=1
    with pytest.raises(ValueError,match='units'):module.active_fingers(values,names,types,units)


def test_tracking_pair_missing_or_mismatched_clock_never_substitutes_cache_or_preq():
    module=api();row=sample(q=.2);row['tracking_pair_available']=False
    envelope=module.build_repeat_envelope([run('a',[sample()]),run('b',[sample()])])
    result=module.compare_to_repeat(run('candidate',[row]),envelope,np.full(12,.01))
    assert result['per_env'][0]['first_significant_tracking_departure'] is None
    assert result['unavailable_counts']['candidate_tracking_pair']==1
    row=sample();row['q_clock']=(1,.1);row['backend_clock']=(1,.2)
    with pytest.raises(ValueError,match='clock'):module.build_repeat_envelope([run('a',[row]),run('b',[sample()])])


def test_different_original_source_episode_lineages_cannot_be_treated_as_missing_repeat_rows():
    module=api();a=sample();b=sample();b['source_episode_uid']='another_original_source'
    with pytest.raises(ValueError,match='lineage'):module.build_repeat_envelope([run('a',[a]),run('b',[b])])


def test_teacher_loader_joins_exact_post_sdk_packets_and_does_not_fill_contact_from_precontrol():
    from test_stage6_tracking_calibration import evidence,calibrate
    module=api();data,context,_=evidence();calibration=calibrate(data,context)
    context['post_physics_feedback'].reverse();context['native_backend_targets'].reverse()
    result=module.load_teacher_context(data,context,calibration,provenance={'fixture_only':True})
    assert len(result['samples'])==48
    first=result['samples'][0]
    assert first['q_rad'][0]==pytest.approx(-.001)
    assert first['backend_target_rad'][0]==0. and first['q_rad'][0]!=500.
    assert first['contact_valid'] is None and first['bilateral_opposed_contact'] is None
    assert first['independent_getter_clock_pair_available'] is True
    assert first['world_physics_step'] is None
    context['post_physics_feedback'][0]['physics_substeps'][0]['source']='pre_control'
    with pytest.raises(ValueError,match='post-physics'):module.load_teacher_context(data,context,calibration,provenance={})


def native_files(tmp_path):
    import gzip,json
    module=api();cadence=float(np.float32(1/480));root=tmp_path/'native';root.mkdir()
    audit=root/'evaluation_action_diagnostics';audit.mkdir()
    names=['wrist_x','wrist_y','wrist_z','wrist_roll','wrist_pitch','wrist_yaw',
        'finger11','finger21','finger31','finger41','finger51','finger12','finger22','finger32','finger42','finger52','finger13']
    units={name:('m' if i<3 else 'rad') for i,name in enumerate(names)}
    shape=(8,4);arrays={}
    for name,value in dict(env_id=np.arange(4)[None].repeat(8,0),episode_id=np.zeros(shape,int),reset_epoch=np.zeros(shape,int),
            policy_step=np.zeros(shape,int),physics_step=np.arange(1,9)[:,None].repeat(4,1),
            timestamp_s=(np.arange(19,27)*cadence)[:,None].repeat(4,1),live_before_substep=np.ones(shape,bool),
            first_trial_eligible=np.ones(shape,bool),after_first_trial_terminal=np.zeros(shape,bool),
            terminal=np.zeros(shape,bool),episode_uid=np.array([[f'native:0:{env}:0' for env in range(4)]]*8)).items():arrays[name]=value
    arrays['terminal'][-1]=True
    arrays['actor_request24']=np.zeros((*shape,24))
    for flag in ('available','finite','source_valid'):arrays['actor_request24__'+flag]=np.ones(shape,bool)
    for side in ('left','right'):
        for name,value in [('q',np.zeros((*shape,17))),('qd',np.zeros((*shape,17))),
                ('backend_target',np.zeros((*shape,17),np.float32)),
                ('dof_types',np.broadcast_to(np.array([1,1,1]+[0]*14,np.uint8),(*shape,17)).copy())]:
            field=side+'_'+name;arrays[field]=value
            for flag in ('available','finite','source_valid'):arrays[field+'__'+flag]=np.ones(shape,bool)
    path=audit/'physics_00000.npz';np.savez_compressed(path,**arrays)
    checkpoint=root/'checkpoint.bin';checkpoint.write_bytes(b'fixture-only-model')
    checkpoint_id=module.file_identity(checkpoint)
    summary=dict(version='dh116-stage6-source-physics-audit-v1',status='recorded',error=None,checkpoint_sha256=checkpoint_id['sha256'],
        capture_timing='native_after_consume_before_terminal_reset',native_backend_target='root_physx_view.get_dof_position_targets;no_cache_fallback',
        source_events=[dict(kind='target_source_status',hand=side,physics_step=1,reset_epoch=0,evidence=dict(joint_names=names,
            units_by_name=units,backend=dict(available=True,source='physx_backend_get_dof_position_targets',error=None))) for side in ('left','right')],
        files=[module.file_identity(path)])
    summary_path=audit/'summary.json';summary_path.write_text(json.dumps(summary))
    cases=[dict(env_id=env,episode_id=str(env),source_episode_uid=f'original02:{env}:first',source_run_uuid='original02') for env in range(4)]
    rows=[];live=[];terminals=[]
    for env in range(4):
        uid=f'actual_execution:{env}:0'
        live.append(dict(env_id=env,policy_step=0,clock_before_action=dict(physics_step=0,simulation_step_index=18,timestamp_s=18*cadence),
            run_uuid='actual_execution',episode_id=0,episode_uid=uid,recorded_teacher_episode_uid=cases[env]['source_episode_uid'],owner='teacher_replay',issued_action=[0.]*24))
        for sub in range(1,9):rows.append(dict(env_id=env,source_episode_id=env,episode_uid=uid,policy_step=0,physics_step=sub,
            timestamp_s=(18+sub)*cadence,action_owner='teacher_replay',failure_code=0,
            payload=dict(live=True,done=sub==8,native_success=sub==8,measurement_valid=True,contact_valid=True,bilateral_opposed_contact=True,
                slip_valid=False,slip_m=None,lift_valid=True,lift_m=.001),diagnostics={}))
        terminals.append(dict(env_id=env,execution_episode_id=0,execution_episode_uid=uid,max_net_lift_m=.001,
            terminal=dict(policy_step=0,physics_step=8,timestamp_s=26*cadence,failure_code=0,action_owner='teacher_replay')))
    raw=root/'source_physics_480hz.jsonl.gz'
    with gzip.open(raw,'wt') as stream:
        for row in sorted(rows,key=lambda row:(row['physics_step'],row['env_id'])):stream.write(json.dumps(row)+'\n')
    policy=root/'live_policy_60hz.jsonl.gz'
    with gzip.open(policy,'wt') as stream:
        for row in live:stream.write(json.dumps(row)+'\n')
    data_id=dict(path='/bound/data.pt',sha256='a'*64);context_id=dict(path='/bound/context.pt',sha256='b'*64)
    report=dict(version='dh116-stage6-native-v1',mode='teacher-replay',status='completed',error=None,native_runtime_verified=True,
        execution_run_uuid='actual_execution',evaluated_source_count=4,completed_count=4,source_case_definitions=cases,source_rows=terminals,
        raw_physics=module.file_identity(raw),inputs=dict(teacher_dataset=data_id,sequence_context=context_id,checkpoint=checkpoint_id))
    report_path=root/'stage6_native_report.json';report_path.write_text(json.dumps(report))
    calibration=dict(qualified=True,clock_mapping=dict(reported_sdk_cadence_s=cadence,reported_sdk_time_origin_s=0.))
    reset=root/'source_reset_comparison.json';reset.write_text(json.dumps(dict(matched=True,scope='fixture_actual_cold_reset_comparison')))
    return dict(report=module.file_identity(report_path),audit_summary=module.file_identity(summary_path),policy_trace=module.file_identity(policy),
        reset_comparison=module.file_identity(reset)),data_id,context_id,calibration,arrays,path


def test_native_loader_bridges_different_uid_namespaces_by_actual_clock_episode_and_policy_anchor(tmp_path):
    module=api();files,dataset,context,calibration,arrays,path=native_files(tmp_path)
    result=module.load_native_run(files,dataset_id=dataset,context_id=context,calibration=calibration)
    assert len(result['samples'])==32
    row=result['samples'][0]
    assert row['execution_episode_uid']=='actual_execution:0:0' and row['source_episode_uid']=='original02:0:first'
    assert row['diagnostic_episode_uid']=='native:0:0:0'
    assert row['substep']==1 and row['world_physics_step']==1
    assert row['q_rad'].shape==(12,) and row['independent_getter_clock_pair_available'] is False
    assert row['palm_link_position_w_m'] is None
    assert result['provenance']['clock_bridge']['maximum_timestamp_residual_s']==0.


@pytest.mark.parametrize('changed',['timestamp','episode','cache_source','missing_q','stale_file','reset_mismatch'])
def test_native_loader_refuses_mixed_clock_or_episode_and_keeps_missing_q_unavailable(tmp_path,changed):
    import json
    module=api();files,dataset,context,calibration,arrays,path=native_files(tmp_path)
    if changed=='timestamp':arrays['timestamp_s'][0,0]+=1e-5
    elif changed=='episode':arrays['episode_id'][0,0]=9
    elif changed=='missing_q':arrays['left_q__available'][0,0]=False
    elif changed=='stale_file':path.write_bytes(path.read_bytes()+b'changed')
    elif changed=='reset_mismatch':
        open(files['reset_comparison']['path'],'w').write(json.dumps(dict(matched=False)))
        files['reset_comparison']=module.file_identity(files['reset_comparison']['path'])
    else:
        summary=json.loads(open(files['audit_summary']['path']).read());summary['native_backend_target']='IsaacLab_command_cache'
        open(files['audit_summary']['path'],'w').write(json.dumps(summary));files['audit_summary']=module.file_identity(files['audit_summary']['path'])
    if changed in ('timestamp','episode','missing_q'):
        np.savez_compressed(path,**arrays)
        summary=json.loads(open(files['audit_summary']['path']).read());summary['files']=[module.file_identity(path)]
        open(files['audit_summary']['path'],'w').write(json.dumps(summary));files['audit_summary']=module.file_identity(files['audit_summary']['path'])
    if changed=='missing_q':
        result=module.load_native_run(files,dataset_id=dataset,context_id=context,calibration=calibration)
        row=next(row for row in result['samples'] if row['env_id']==0 and row['substep']==1)
        assert row['q_rad'] is None and row['tracking_pair_available'] is False
    else:
        with pytest.raises(ValueError):module.load_native_run(files,dataset_id=dataset,context_id=context,calibration=calibration)


def full_analysis_config(tmp_path):
    import json,torch
    module=api();native,_,_,clock,arrays,npz_path=native_files(tmp_path);root=tmp_path/'inputs';root.mkdir()
    data=dict(sample_metadata=[],episodes={});context=dict(controller_source_sha256='c'*64,action_contract_sha256='d'*64,
        native_backend_targets=[],post_physics_feedback=[],events={'opposed_contact':'not postphysics evidence'})
    names=['wrist_x','wrist_y','wrist_z','wrist_roll','wrist_pitch','wrist_yaw',
        'finger11','finger21','finger31','finger41','finger51','finger12','finger22','finger32','finger42','finger52','finger13']
    cadence=clock['clock_mapping']['reported_sdk_cadence_s']
    for env in range(4):
        uid=f'original02:{env}:first';key=('original02',env,uid,0)
        data['sample_metadata'].append(dict(run_uuid='original02',env_id=env,episode_uid=uid,episode_id=env,tick=0,first_physics_step=18))
        data['episodes'][str(env)]=dict(first_episode=True,native_physics=True,success=True,recovered=False)
        back=[];post=[]
        for sub in range(1,9):
            when=(18+sub)*cadence
            back.append(dict(sdk_physics_step=18+sub,sdk_timestamp_s=when,clock_aligned=True,hands=[dict(joint_names=names,
                backend=dict(available=True,source='physx_backend_get_dof_position_targets',error=None,values=np.zeros((1,17),np.float32),
                    finite_env=np.ones(8,bool),shape=[8,17],dtype='float32')) for _ in range(2)]))
            post.append(dict(measured_q_rad=torch.zeros(1,12,dtype=torch.float64),measured_qd_rad_s=torch.zeros(1,12,dtype=torch.float64),
                measured_root_pose=torch.zeros(1,2,7,dtype=torch.float64),sdk_physics_step=18+sub,sdk_timestamp_s=when,
                backend_readback_sdk_physics_step=18+sub,backend_readback_sdk_timestamp_s=when,clock_aligned=True,
                source_valid=torch.tensor([True]),native_runtime_verified=True,source='original_advance_returned_refresh_before_TensorEnv_terminal_reset'))
        context['native_backend_targets'].append(dict(sample_key=key,physics_substeps=back))
        context['post_physics_feedback'].append(dict(sample_key=key,physics_substeps=post))
    context['samples']={key:[row[key] for row in data['sample_metadata']] for key in ('run_uuid','env_id','episode_uid','tick')}
    context['physics_context']=dict(live=torch.ones(4,8,dtype=torch.bool),source_valid=torch.ones(4,8,dtype=torch.bool))
    dataset_path=root/'dataset.pt';context_path=root/'context.pt';torch.save(data,dataset_path);torch.save(context,context_path)
    dataset_id=module.file_identity(dataset_path);context_id=module.file_identity(context_path)
    report=json.loads(open(native['report']['path']).read());checkpoint_id=report['inputs']['checkpoint']
    report['inputs']=dict(teacher_dataset=dataset_id,sequence_context=context_id,checkpoint=checkpoint_id)
    report['execution_controller_contract']=dict(controller_source_sha256='c'*64,action_contract_sha256='d'*64)
    for name in ('source_initial_state','source_collection_report'):
        path=root/(name+'.bin');path.write_bytes(name.encode());report['inputs'][name]=module.file_identity(path)
    open(native['report']['path'],'w').write(json.dumps(report));native['report']=module.file_identity(native['report']['path'])
    calibration=dict(version='dh116-teacher-tracking-calibration-v1',qualified=True,
        target_source='physx_backend_get_dof_position_targets',feedback_source='original_advance_returned_refresh_before_TensorEnv_terminal_reset',
        controller_source_sha256='c'*64,action_contract_sha256='d'*64,dataset_sha256=dataset_id['sha256'],context_sha256=context_id['sha256'],
        clock_mapping=dict(clock['clock_mapping'],sdk_minus_metadata_physics_step=0),teacher_error_p999_rad=[0.]*12,trigger_rad=[.01]*12)
    cal_path=root/'calibration.json';cal_path.write_text(json.dumps(calibration))
    config=dict(version='dh116-stage6-physical-repeat-analysis-inputs-v1',dataset=dataset_id,context=context_id,
        calibration=module.file_identity(cal_path),teacher_repeat=native,candidate=native)
    path=root/'config.json';path.write_text(json.dumps(config));return path,config


def test_full_read_only_analysis_saves_exact_repeat_proof_input_hashes_and_na_contact(tmp_path):
    import json
    module=api();config_path,config=full_analysis_config(tmp_path);output=tmp_path/'actual_analysis.json'
    report=module.save_physical_divergence(config_path,output)
    assert report['inputs_sha256_unchanged'] is True and report['native_jobs_executed']==0
    assert report['analysis']['reference_exact_match']['q_rad']['all_equal'] is True
    assert report['analysis']['availability']['repeat_contact']['available'] is False
    assert report['analysis']['availability']['repeat_contact']['maximum_reference_samples']==1
    assert all(row['first_significant_tracking_departure'] is None for row in report['analysis']['per_env'])
    assert json.loads(output.read_text())['analysis']['causality_claimed'] is False
    with pytest.raises(FileExistsError):module.save_physical_divergence(config_path,output)


@pytest.mark.parametrize('changed',['audit_model','diagnostic_namespace','issued_action','audit_other_repeat_same_basename'])
def test_other_execution_audit_cannot_be_bound_just_by_shared_clock_and_episode_numbers(tmp_path,changed):
    import json,shutil
    module=api();files,dataset,context,calibration,arrays,path=native_files(tmp_path)
    if changed=='audit_other_repeat_same_basename':
        other=tmp_path/'other_repeat'/path.parent.parent.name/'evaluation_action_diagnostics';shutil.copytree(path.parent,other)
        files['audit_summary']=module.file_identity(other/'summary.json')
    elif changed=='audit_model':
        summary=json.loads(open(files['audit_summary']['path']).read());summary['checkpoint_sha256']='f'*64
        open(files['audit_summary']['path'],'w').write(json.dumps(summary));files['audit_summary']=module.file_identity(files['audit_summary']['path'])
    else:
        if changed=='diagnostic_namespace':arrays['episode_uid'][:]=np.array([['other_job:0:'+str(env)+':0' for env in range(4)]]*8)
        else:arrays['actor_request24'][0,0,0]=.1
        np.savez_compressed(path,**arrays)
        summary=json.loads(open(files['audit_summary']['path']).read());summary['files']=[module.file_identity(path)]
        open(files['audit_summary']['path'],'w').write(json.dumps(summary));files['audit_summary']=module.file_identity(files['audit_summary']['path'])
    with pytest.raises(ValueError):module.load_native_run(files,dataset_id=dataset,context_id=context,calibration=calibration)


def test_raw_net_lift_peak_keeps_same_env_uid_clock_argmax_and_matches_original_report(tmp_path):
    module=api();files,dataset,context,calibration,arrays,path=native_files(tmp_path)
    result=module.load_native_run(files,dataset_id=dataset,context_id=context,calibration=calibration)
    peaks=result['provenance']['net_lift_peaks']
    assert len(peaks)==4
    for env,peak in enumerate(peaks):
        assert peak['env_id']==env and peak['execution_episode_uid']==f'actual_execution:{env}:0'
        assert peak['policy_step']==0 and peak['world_physics_step']==1 and peak['substep']==1
        assert peak['timestamp_s']==19*float(np.float32(1/480))
        assert peak['net_lift_m']==.001 and peak['matches_source_report_max_exactly'] is True
        assert peak['source_valid'] is True


def test_original_sdk_q_plus_two_distinct_native_replays_enables_contact_only_from_the_two_post_replays():
    module=api();original=sample(q=0.);a=sample(q=.001,contact=True);b=sample(q=-.001,contact=True)
    envelope=module.build_repeat_envelope([run('original02',[original]),run('replay01',[a]),run('replay02',[b])])
    assert envelope['reference_run_count']==3
    assert envelope['availability']['q_rad']['maximum_reference_samples']==3
    assert envelope['availability']['repeat_contact']['maximum_reference_samples']==2
    assert envelope['availability']['repeat_contact']['available'] is True
    candidate=sample(q=.02,contact=False,owner='student')
    report=module.compare_to_repeat(run('candidate',[candidate]),envelope,np.full(12,.01))
    event=report['per_env'][0]['first_significant_tracking_departure']
    assert event['observed_reference_samples']==3
    assert event['repeat_min_tracking_error_rad']==[0.]*12
    assert event['repeat_max_tracking_error_rad']==[.001]*12
    named=event['significant_named_joints'][2]
    assert named['hand']=='left' and named['joint']=='finger21'
    assert named['signed_backend_minus_postq_rad']==-.02
    assert report['per_env'][0]['first_repeat_contact_loss'] is not None
    assert report['reference_run_count']==3 and report['statistical_confidence_interval'] is False


def test_copy_of_one_native_replay_cannot_be_counted_as_the_second_contact_reference():
    module=api();references=[run('original02',[sample()]),run('replay01',[sample(contact=True)]),run('replay01',[sample(contact=True)])]
    with pytest.raises(ValueError,match='distinct'):module.build_repeat_envelope(references)


def test_original_teacher_inactive_substeps_are_excluded_instead_of_being_marked_live():
    from test_stage6_tracking_calibration import evidence,calibrate
    module=api();data,context,_=evidence();context['physics_context']['live'][0,-1]=False
    calibration=calibrate(data,context);assert calibration['qualified'] is True
    result=module.load_teacher_context(data,context,calibration,provenance={})
    assert len(result['samples'])==47
    assert result['provenance']['excluded_nonlive_or_source_invalid_substeps']==1


def test_candidate_unavailable_fields_are_counted_separately_from_no_departure():
    module=api();envelope=module.build_repeat_envelope([run('a',[sample()]),run('b',[sample()])])
    candidate=sample();candidate['qd_rad_s']=None;candidate['tracking_pair_available']=False
    report=module.compare_to_repeat(run('candidate',[candidate]),envelope,np.full(12,.01))
    assert report['unavailable_counts']['candidate_qd_rad_s']==1
    assert report['unavailable_counts']['candidate_contact']==1
    assert report['unavailable_counts']['candidate_tracking_pair']==1


def second_native_execution(tmp_path,first,report_inputs):
    import json,gzip
    directory=tmp_path/'independent_repeat02';directory.mkdir()
    files,_,_,_,_,_=native_files(directory)
    report=json.loads(open(files['report']['path']).read());report['execution_run_uuid']='actual_execution02';report['inputs']=copy.deepcopy(report_inputs)
    report['execution_controller_contract']=dict(controller_source_sha256='c'*64,action_contract_sha256='d'*64)
    for row in report['source_rows']:row['execution_episode_uid']=f'actual_execution02:{row["env_id"]}:0'
    for path_key in ('raw_physics','policy_trace'):
        path=report['raw_physics']['path'] if path_key=='raw_physics' else files['policy_trace']['path']
        with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
        for row in rows:
            row['episode_uid']=f'actual_execution02:{row["env_id"]}:0'
            if path_key=='policy_trace':row['run_uuid']='actual_execution02'
        with gzip.open(path,'wt') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        if path_key=='raw_physics':report['raw_physics']=api().file_identity(path)
        else:files['policy_trace']=api().file_identity(path)
    open(files['report']['path'],'w').write(json.dumps(report));files['report']=api().file_identity(files['report']['path'])
    return files


def test_native_teacher_repeats_list_keeps_one_contact_na_and_two_contact_eligible_without_new_initial_states(tmp_path):
    import json
    module=api();config_path,config=full_analysis_config(tmp_path)
    first=config.pop('teacher_repeat');report_inputs=json.loads(open(first['report']['path']).read())['inputs']
    config['native_teacher_repeats']=[first];config_path.write_text(json.dumps(config))
    one=module.save_physical_divergence(config_path,tmp_path/'one_repeat.json')
    assert one['native_teacher_repeat_count']==1 and one['postphysical_reference_contact_available'] is False
    second=second_native_execution(tmp_path,first,report_inputs)
    config['native_teacher_repeats'].append(second);second_config=config_path.with_name('two_repeat_inputs.json');second_config.write_text(json.dumps(config))
    two=module.save_physical_divergence(second_config,tmp_path/'two_repeats.json')
    assert two['native_teacher_repeat_count']==2 and two['reference_run_count']==3
    assert two['postphysical_reference_contact_available'] is True
    assert two['analysis']['availability']['repeat_contact']['maximum_reference_samples']==2
    assert two['new_independent_initial_states']==0 and two['statistical_confidence_interval'] is False


def test_original_inactive_plus_missing_replay01_cannot_promote_only_replay02_into_two_sample_range():
    from test_stage6_tracking_calibration import evidence,calibrate
    module=api();data,context,_=evidence()
    full=module.load_teacher_context(data,context,calibrate(data,context),provenance={})
    missing=module._key(full['samples'][7])
    context['physics_context']['live'][0,7]=False
    original=module.load_teacher_context(data,context,calibrate(data,context),provenance={})
    replay01=run('replay01',[copy.deepcopy(row) for row in full['samples'] if module._key(row)!=missing])
    replay02=run('replay02',copy.deepcopy(full['samples']))
    envelope=module.build_repeat_envelope([original,replay01,replay02])
    assert envelope['ranges'][missing]['q_rad']['reference_samples']==1
    assert envelope['ranges'][missing]['q_rad']['available'] is False
