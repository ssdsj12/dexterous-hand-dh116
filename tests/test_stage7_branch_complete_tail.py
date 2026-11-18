"""Synthetic ownership proofs for unowned branch tails; no native certificates."""
import copy
import gzip
import inspect
import json

import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_vision_physical_divergence as physical


def write_rows(path, rows):
    with gzip.open(path, 'wt') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')


def tail_fixture(tmp_path):
    from test_stage6_physical_divergence import native_files
    files, data, context, calibration, arrays, path = native_files(tmp_path)
    report_path = files['report']['path']; report = json.load(open(report_path))
    root = path.parent.parent; cadence=calibration['clock_mapping']['reported_sdk_cadence_s']
    for name, value in list(arrays.items()): arrays[name]=np.concatenate((value,value.copy()),axis=0)
    arrays['physics_step']=np.arange(1,17)[:,None].repeat(4,axis=1)
    arrays['timestamp_s']=(np.arange(19,35)*cadence)[:,None].repeat(4,axis=1)
    arrays['policy_step'][8:]=1; arrays['terminal'][:]=False
    report.update(mode='branch',status='bounded_branch_completed',event='CONTACT',branch_steps=1,
        completed_count=0,ppo_updates=0,native_success_count=0,pure_student_native_success_count=0)
    for row in report['source_case_definitions']:
        row['event_entry_ticks']={'CONTACT':0 if row['env_id']==0 else 1}
    for row in report['source_rows']: row.update(terminal=None,native_success=False)
    raw, live, requests = [], [], []
    for tick in range(2):
        owners=['student','teacher_prefix','teacher_prefix','teacher_prefix'] if tick==0 else ['branch_complete','student','student','student']
        requests.append(dict(source='actual_issued_SDK24_action',reset_epoch=0,policy_step=tick,
            env_ids=list(range(4)),episode_ids=[0]*4,episode_uids=[f'native:0:{i}:0' for i in range(4)],
            first_trial_eligible=[True]*4,action_owner=owners,actor_request24=[[0.]*24 for _ in range(4)],request_finite=[True]*4,
            physics_step_before_submit=tick*8,timestamp_s_before_submit=(tick*8+18)*cadence,
            submit_physics_step_available=True,submit_timestamp_available=True,submit_timestamp_finite=True))
        for env,owner in enumerate(owners):
            if owner=='branch_complete':continue
            live.append(dict(env_id=env,policy_step=tick,clock_before_action=dict(physics_step=tick*8,
                simulation_step_index=tick*8+18,timestamp_s=(tick*8+18)*cadence),run_uuid='actual_execution',episode_id=0,
                episode_uid=f'actual_execution:{env}:0',recorded_teacher_episode_uid=f'original02:{env}:first',owner=owner,issued_action=[0.]*24))
            for sub in range(1,9):
                step=tick*8+sub
                raw.append(dict(env_id=env,source_episode_id=env,episode_uid=f'actual_execution:{env}:0',policy_step=tick,
                    physics_step=step,timestamp_s=(step+18)*cadence,action_owner=owner,failure_code=0,
                    payload=dict(live=True,done=False,native_success=False,measurement_valid=True,contact_valid=True,
                        bilateral_opposed_contact=True,slip_valid=False,slip_m=None,lift_valid=True,lift_m=.001)))
    raw.sort(key=lambda r:(r['physics_step'],r['env_id']));live.sort(key=lambda r:(r['policy_step'],r['env_id']))
    request_path=path.parent/'policy_requests.jsonl.gz'
    def save():
        np.savez_compressed(path,**arrays)
        write_rows(root/'source_physics_480hz.jsonl.gz',raw);write_rows(root/'live_policy_60hz.jsonl.gz',live);write_rows(request_path,requests)
        summary=json.load(open(files['audit_summary']['path']));summary['files']=[physical.file_identity(path),physical.file_identity(request_path)]
        summary['array_field_semantics']={'actor_request24':'actual_issued_SDK24;join_policy_requests_action_owner'}
        json.dump(summary,open(files['audit_summary']['path'],'w'))
        report['raw_physics']=physical.file_identity(root/'source_physics_480hz.jsonl.gz');json.dump(report,open(report_path,'w'))
        files.update(report=physical.file_identity(report_path),audit_summary=physical.file_identity(files['audit_summary']['path']),
            policy_trace=physical.file_identity(root/'live_policy_60hz.jsonl.gz'))
    save()
    return files,data,context,calibration,arrays,path,report,raw,live,requests,save


def test_default_reader_keeps_hard_refusal_for_npz_without_owned_raw(tmp_path):
    files,data,context,calibration,*_=tail_fixture(tmp_path)
    with pytest.raises(ValueError,match='raw|bridge'):
        physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration)


def test_optin_excludes_only_explicit_zero_branch_complete_tail_and_keeps_all_owned_rows(tmp_path):
    files,data,context,calibration,*_=tail_fixture(tmp_path)
    assert 'verify_branch_complete_tail' in inspect.signature(physical.load_native_run).parameters, 'explicit branch-tail proof API missing'
    result=physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration,verify_branch_complete_tail=True)
    assert len(result['samples'])==56
    excluded=result['provenance']['proven_unowned_branch_complete_samples']
    assert len(excluded)==8 and all(row['env_id']==0 and row['policy_step']==1 for row in excluded)
    assert all(row['action_owner']=='branch_complete' and row['sdk_physics_step_is_derived'] for row in excluded)
    assert all(row['actual_issued_zero24'] is True and row['counted_as_Student_event'] is False for row in excluded)
    assert all(not (row['env_id']==0 and row['policy_step']==1) for row in result['samples'])


@pytest.mark.parametrize('bad',['owner','nonzero_request','nonzero_npz','early','uid','epoch','clock','missing_owned_raw','owned_live_conflict','missing_owner_file'])
def test_no_generic_skip_for_missing_raw_or_unproved_ownership(tmp_path,bad):
    files,data,context,calibration,arrays,path,report,raw,live,requests,save=tail_fixture(tmp_path)
    if bad=='owner':requests[1]['action_owner'][0]='student'
    elif bad=='nonzero_request':requests[1]['actor_request24'][0][0]=.1
    elif bad=='nonzero_npz':arrays['actor_request24'][8,0,0]=.1
    elif bad=='early':report['source_case_definitions'][0]['event_entry_ticks']['CONTACT']=1
    elif bad=='uid':requests[1]['episode_uids'][0]='foreign:0:0:0'
    elif bad=='epoch':arrays['reset_epoch'][8,0]=1
    elif bad=='clock':requests[1]['timestamp_s_before_submit']+=1/480
    elif bad=='missing_owned_raw':raw[:]=[row for row in raw if not(row['env_id']==1 and row['physics_step']==9)]
    elif bad=='owned_live_conflict':
        row=copy.deepcopy(next(row for row in live if row['env_id']==1 and row['policy_step']==1))
        row.update(env_id=0,episode_uid='actual_execution:0:0',recorded_teacher_episode_uid='original02:0:first');live.append(row)
    save()
    if bad=='missing_owner_file':
        summary=json.load(open(files['audit_summary']['path']));summary['files']=summary['files'][:1]
        json.dump(summary,open(files['audit_summary']['path'],'w'));files['audit_summary']=physical.file_identity(files['audit_summary']['path'])
    assert 'verify_branch_complete_tail' in inspect.signature(physical.load_native_run).parameters, 'explicit branch-tail proof API missing'
    with pytest.raises(ValueError):
        physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration,verify_branch_complete_tail=True)


def test_unowned_tail_terminal_is_retained_as_excluded_not_student_first_terminal(tmp_path):
    files,data,context,calibration,arrays,path,report,raw,live,requests,save=tail_fixture(tmp_path)
    arrays['terminal'][8,0]=True;save()
    assert 'verify_branch_complete_tail' in inspect.signature(physical.load_native_run).parameters, 'explicit branch-tail proof API missing'
    result=physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration,verify_branch_complete_tail=True)
    excluded=result['provenance']['proven_unowned_branch_complete_samples']
    assert excluded[0]['unowned_terminal'] is True and not any(row['terminal'] for row in result['samples'])


def test_qualified_extra_field_reader_keeps_only_proven_owned_rows(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_student_divergence import load_native_event_fields
    files,data,context,calibration,*_=tail_fixture(tmp_path)
    result=load_native_event_fields(physical.load_native_run(files,dataset_id=data,context_id=context,
        calibration=calibration,verify_branch_complete_tail=True))
    assert len(result['samples'])==56
    assert len(result['provenance']['proven_unowned_branch_complete_samples'])==8
    assert all(row['action_owner'] in ('student','teacher_prefix') for row in result['samples'])


def test_physical_save_does_not_silently_ignore_branch_optin_on_a_teacher_reference(tmp_path):
    from test_stage6_physical_divergence import full_analysis_config
    config,_=full_analysis_config(tmp_path)
    with pytest.raises(ValueError,match='branch'):
        physical.save_physical_divergence(config,tmp_path/'not_a_branch.json',verify_branch_complete_tail=True)


def test_duplicate_unowned_physical_identity_is_rejected_even_when_all_owner_proofs_match(tmp_path):
    files,data,context,calibration,arrays,path,*_=tail_fixture(tmp_path)
    duplicate={name:value[8:9].copy() for name,value in arrays.items()}
    duplicate['first_trial_eligible'][0,1:]=False
    duplicate['live_before_substep'][0,1:]=False
    extra=path.with_name('physics_00001.npz');np.savez_compressed(extra,**duplicate)
    summary=json.load(open(files['audit_summary']['path']));summary['files'].append(physical.file_identity(extra))
    json.dump(summary,open(files['audit_summary']['path'],'w'));files['audit_summary']=physical.file_identity(files['audit_summary']['path'])
    with pytest.raises(ValueError,match='duplicate'):
        physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration,verify_branch_complete_tail=True)
