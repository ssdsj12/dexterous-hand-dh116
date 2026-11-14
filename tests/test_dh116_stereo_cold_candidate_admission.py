"""Synthetic ledger integration; none of these fixtures are native evidence."""
import copy
import json
from pathlib import Path
import pytest
import torch

from test_dh116_stereo_collection import FakeEnv, TinyTeacher
from test_dh116_stereo_trials import manifest
from go2_pvcnn.control.dh116 import tabletop_stereo_training_cli as cli
from go2_pvcnn.control.dh116 import tabletop_stereo_state_supervision as state
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256, physical_initial_sha256
from go2_pvcnn.control.dh116.tabletop_privileged_evaluation import prepare_cold_trial_batches
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import rows_from_records, validate_rows


def cold_payload(tmp_path):
    env=FakeEnv();env.set_action_owners=lambda owners:None
    teacher=TinyTeacher(); value=manifest(4)
    value['metadata'].update(source_sha256='d'*64,camera_sha256='c'*64)
    plan=prepare_cold_trial_batches(value,2,minimum_per_style=4,schema_only=True)
    folder=tmp_path/'cold_initials';folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps(value))
    records=[];batches=[];ledger=[]
    for batch_index,trials in enumerate(plan['batches']):
        if batch_index:
            env.reset_id+=1;env.store.reset(torch.ones(2,dtype=torch.bool));env._refresh(torch.zeros(2,24))
        batch_records=[cli.collect_candidate_reference(env,teacher,1)[0] for _ in range(4)]
        for record in batch_records:
            record['extra_labels']={}
            record['state_sources']={}
        initials=[]
        for i,trial in enumerate(trials):
            physical=dict(coordinate_frame='environment_local',object_pose=copy.deepcopy(trial['physical_state']['wire_pose']),
                palm_pose=[[0,0,0,1,0,0,0]]*2,joint_position=[0.]*12,joint_velocity=[0.]*12,
                object_velocity=copy.deepcopy(trial['physical_state']['wire_velocity']))
            for j,record in enumerate(batch_records):
                identity=record['identities'][i]
                identity.update(source_episode_group=trial['trial_id'],trial_id=trial['trial_id'],
                    grasp_style=trial['grasp_style'],physical_initial_condition_sha256=physical_initial_sha256(physical),
                    episode_policy_tick=j,trial_episode_index=1)
                record['privileged_identities'][i]=copy.deepcopy(identity)
                record['info']['evidence_scope']='native_runtime_capture_unqualified'
                record['extra_labels']['training_label_evidence_scope']='native_runtime_capture_unqualified'
                record['extra_labels']['training_label_source_scope']='simulation_SDK_encoder_readback_only'
                keys={k:identity[k] for k in ('run_uuid','env_id','episode_uid','reset_id')}
                if j==0:
                    initials.append(copy.deepcopy(identity))
                    ledger.append(dict(**keys,kind='episode_start',initial_measurement_valid=True,
                        physical_initial_condition_source='actual_sdk_readback',physical_initial_condition=physical,
                        source_episode_group=trial['trial_id']))
                for step in range(1,9):
                    ledger.append(dict(**keys,kind='sample',owner='teacher',action_owner='teacher',checkpoint_sha256='d'*64,
                        policy_tick=identity['policy_tick'],physics_step=identity['physics_tick_end']+step,
                        timestamp=identity['simulation_timestamp']+step/480,frame_id=identity['frame_id'],
                        calibration_sha256=identity['calibration_sha256'],pre_action_identity=identity,
                        actor_issued_action=record['issued_actions'][i].tolist(),terminal=False))
        path=folder/f'reset_readback_{batch_index:04d}.json'
        path.write_text(json.dumps(dict(actual_states=[dict(trial_id=t['trial_id'],grasp_style=t['grasp_style'],
            authored_state_sha256=t['state_sha256'],physical_state=t['physical_state']) for t in trials],
            physics_steps_advanced=0,max_abs_writer_readback_error=0.,writer_match_tolerance=2e-5)))
        batches.append(dict(batch_index=batch_index,initial_identities=initials,
            actual_readback_sha256=file_sha256(path),completed_control_steps=4,stopped_on_terminal=False))
        records.extend(batch_records)
    cohort=dict(version='dh116-stereo-cold-candidate-capture-v1',manifest_sha256=plan['manifest_sha256'],
        minimum_per_style=4,steps_per_initial=4,batches=batches,independent_actual_initials=8,samples=32,
        checkpoint_sha256='d'*64,calibration_sha256='c'*64,native_execution_performed=True,
        native_qualified=False,successful_demonstration_pool=False,reset_buffers_restored=True,formal_success_evaluation=False)
    (folder/'cohort_report.json').write_text(json.dumps(cohort))
    path=tmp_path/'candidates.pt';raw=tmp_path/'native'/'native_physics.jsonl';raw.parent.mkdir()
    torch.save(dict(version=cli.CANDIDATE_VERSION,teacher_checkpoint_sha256='d'*64,records=records,
        candidate_only=True,successful_demonstration_pool=False,cold_cohort=cohort),path)
    raw.write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    return path,raw


def test_full_cold8_candidate_ledger_accepts_only_bound_diagnostic_reset(tmp_path,monkeypatch):
    path,raw=cold_payload(tmp_path)
    # Source resolver is isolated here. This test validates complete rows/raw/reset
    # admission, not optical/instance labels or actual Isaac execution.
    monkeypatch.setattr(state,'_verified_wire_observation',lambda *a:False)
    rows=state.read_candidate(path,raw=raw,sha256=file_sha256(path),raw_sha256=file_sha256(raw))
    assert len(rows)==32 and state.split_estimator_rows(rows)['independent_groups']==8
    assert all(not row['verified_canonical_native'].any() for row in rows)
    with pytest.raises(ValueError,match='reset boundary'):
        validate_rows(rows,contract={'calibration_sha256':'c'*64},schema_only=False)


@pytest.mark.parametrize('damage',['no_report','modified_readback','wrong_initial','extra_episode','missing_record'])
def test_cold_reset_candidate_admission_fails_closed(tmp_path,monkeypatch,damage):
    path,raw=cold_payload(tmp_path)
    monkeypatch.setattr(state,'_verified_wire_observation',lambda *a:False)
    payload=torch.load(path,weights_only=True)
    if damage=='no_report':(tmp_path/'cold_initials/cohort_report.json').unlink()
    elif damage=='modified_readback':(tmp_path/'cold_initials/reset_readback_0001.json').write_text('{}')
    elif damage=='wrong_initial':payload['cold_cohort']['batches'][1]['initial_identities'][0]['episode_uid']='different'
    elif damage=='extra_episode':payload['records'][4]['identities'][0]['episode_uid']='different'
    elif damage=='missing_record':payload['records'].pop()
    torch.save(payload,path)
    with pytest.raises((ValueError,FileNotFoundError)):
        state.read_candidate(path,raw=raw,sha256=file_sha256(path),raw_sha256=file_sha256(raw))


def test_teacher_checkpoint_hash_checked_on_both_sides_of_loading(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_privileged_cli as privileged
    path=tmp_path/'teacher.pt';path.write_bytes(b'original checkpoint fixture')
    expected=file_sha256(path);calls=[]
    def replaced(*args):
        calls.append(1);path.write_bytes(b'replaced during loader');return object(),{}
    monkeypatch.setattr(privileged,'_policy_for_evaluation',replaced)
    with pytest.raises(ValueError,match='checkpoint'):
        cli._load_frozen_teacher(None,{},None,path,expected)
    assert calls==[1]
    calls.clear()
    with pytest.raises(ValueError,match='checkpoint'):
        cli._load_frozen_teacher(None,{},None,path,expected)
    assert calls==[]


@pytest.mark.parametrize('damage',['object_pose','object_velocity'])
def test_self_consistent_wrong_native_header_cannot_detach_from_reset_readback(tmp_path,monkeypatch,damage):
    path,raw=cold_payload(tmp_path)
    payload=torch.load(path,weights_only=True)
    ledger=[json.loads(line) for line in raw.read_text().splitlines()]
    header=next(row for row in ledger if row['kind']=='episode_start')
    header['physical_initial_condition'][damage][0]+=.01
    sha=physical_initial_sha256(header['physical_initial_condition'])
    uid=header['episode_uid']
    for record in payload['records']:
        for key in ('identities','privileged_identities'):
            for ident in record[key]:
                if ident['episode_uid']==uid:ident['physical_initial_condition_sha256']=sha
    for batch in payload['cold_cohort']['batches']:
        for ident in batch['initial_identities']:
            if ident['episode_uid']==uid:ident['physical_initial_condition_sha256']=sha
    for sample in ledger:
        if sample['kind']=='sample' and sample['episode_uid']==uid:
            sample['pre_action_identity']['physical_initial_condition_sha256']=sha
    torch.save(payload,path)
    raw.write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    (tmp_path/'cold_initials/cohort_report.json').write_text(json.dumps(payload['cold_cohort']))
    monkeypatch.setattr(state,'_verified_wire_observation',lambda *a:False)
    with pytest.raises(ValueError,match='reset readback'):
        state.read_candidate(path,raw=raw,sha256=file_sha256(path),raw_sha256=file_sha256(raw))


def test_reset_readback_header_compares_world_origin_and_quaternion_sign_correctly(tmp_path,monkeypatch):
    path,raw=cold_payload(tmp_path)
    ledger=[json.loads(line) for line in raw.read_text().splitlines()]
    for row in ledger:
        if row['kind']!='episode_start':continue
        initial=row['physical_initial_condition'];origin=[17.,-31.,9.]
        initial.update(coordinate_frame='world',environment_origin=origin)
        for pose in [initial['object_pose'],*initial['palm_pose']]:
            pose[:3]=[a+b for a,b in zip(pose[:3],origin)]
            pose[3:]=[-x for x in pose[3:]]
    raw.write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    monkeypatch.setattr(state,'_verified_wire_observation',lambda *a:False)
    assert len(state.read_candidate(path,raw=raw,sha256=file_sha256(path),raw_sha256=file_sha256(raw)))==32


def test_consistently_swapped_environment_slot_is_rejected(tmp_path,monkeypatch):
    path,raw=cold_payload(tmp_path);payload=torch.load(path,weights_only=True)
    ledger=[json.loads(line) for line in raw.read_text().splitlines()]
    for record in payload['records']:
        for name in ('identities','privileged_identities'):
            for ident in record[name]:ident['env_id']=1-ident['env_id']
    for batch in payload['cold_cohort']['batches']:
        for ident in batch['initial_identities']:ident['env_id']=1-ident['env_id']
    for row in ledger:
        row['env_id']=1-row['env_id']
        if 'pre_action_identity' in row:row['pre_action_identity']['env_id']=row['env_id']
    torch.save(payload,path);raw.write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    (tmp_path/'cold_initials/cohort_report.json').write_text(json.dumps(payload['cold_cohort']))
    monkeypatch.setattr(state,'_verified_wire_observation',lambda *a:False)
    with pytest.raises(ValueError,match='initial identity'):
        state.read_candidate(path,raw=raw,sha256=file_sha256(path),raw_sha256=file_sha256(raw))
