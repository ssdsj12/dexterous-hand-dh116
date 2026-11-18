"""CPU evidence fixtures are synthetic, never new native trial certificates."""
import copy
import gzip
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def api():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_stage7_event_reference'
    assert importlib.util.find_spec(name), 'Stage7 actual post-contact binding module is missing'
    return importlib.import_module(name)


def fixture(first_contact_tick=1):
    from test_vision_branching_eval import source_fixture
    data, context, collection = source_fixture(8)
    cases, outcomes, policies, raw = [], [], [], []
    for env in range(4):
        rows = [r for r in data['sample_metadata'] if r['env_id'] == env]
        uid = f'replay:{env}:0'
        cases.append(dict(env_id=env, episode_id=str(env), source_episode_uid=rows[0]['episode_uid'],
            source_run_uuid=rows[0]['run_uuid'], recorded_teacher_tape_length=8,
            event_entry_ticks=dict(CONTACT=2, LIFT_ONSET=3, HOLD=6)))
        terminal = dict(policy_step=7, physics_step=64, timestamp_s=84/480,
            failure_code=0, action_owner='teacher_replay')
        outcomes.append(dict(env_id=env, episode_id=str(env), execution_episode_id=0,
            execution_episode_uid=uid, terminal=terminal, native_success=True,
            student_policy_steps_executed=0, teacher_prefix_policy_steps_executed=0,
            teacher_replay_policy_steps_executed=8))
        for tick, meta in enumerate(rows):
            action = data['actions'][env*8+tick].tolist()
            policies.append(dict(env_id=env, policy_step=tick, run_uuid='replay', episode_id=0,
                episode_uid=uid, recorded_teacher_episode_uid=meta['episode_uid'], owner='teacher_replay',
                issued_action=action, recorded_teacher_action=action,
                actual_previous_issued_action=meta['previous_action'],
                clock_before_action=dict(physics_step=8*tick, simulation_step_index=20+8*tick,
                                         timestamp_s=(20+8*tick)/480)))
            for substep in range(1, 9):
                step = tick*8+substep
                raw.append(dict(env_id=env, source_episode_id=str(env), episode_uid=uid,
                    policy_step=tick, physics_step=step, timestamp_s=(20+step)/480,
                    action_owner='teacher_replay', failure_code=0,
                    payload=dict(live=True, done=step==64, native_success=step==64,
                        measurement_valid=True, contact_valid=True,
                        bilateral_opposed_contact=(tick>first_contact_tick or
                            tick==first_contact_tick and substep>=3))))
    raw.sort(key=lambda r:(r['physics_step'],r['env_id']))
    policies.sort(key=lambda r:(r['policy_step'],r['env_id']))
    replay = dict(version='dh116-stage6-native-v1', mode='teacher-replay', status='completed',
        error=None, native_runtime_verified=True, execution_run_uuid='replay', ppo_updates=0,
        evaluated_source_count=4, completed_count=4, native_success_count=4,
        pure_student_native_success_count=0, source_case_definitions=cases, source_rows=outcomes)
    return data, context, collection, replay, raw, policies


def derive(parts):
    data, context, _, replay, raw, policies = parts
    return api().derive_event_reference(data, context, replay, raw, policies, dataset_sha256='d'*64)


def test_postphysics_contact_minus_one_does_not_reinterpret_old_preaction_contact():
    result = derive(fixture())
    for case in result['cases']:
        assert case['events']['CONTACT']['tick'] == 2
        assert case['events']['LIFT_ONSET']['tick'] == 3
        assert case['events']['HOLD']['tick'] == 6
        event = case['events']['PRE_CONTACT']
        assert event['available'] is True and event['tick'] == 0
        assert event['first_actual_post_contact']['policy_step'] == 1
        assert event['first_actual_post_contact']['substep'] == 3
        assert event['first_actual_post_contact']['world_physics_step'] == 11
        assert event['preceding_source_row']['tick'] == 0
        assert event['first_actual_post_contact']['sdk_post_index_is_derived'] is True
    assert result['actor_prediction_used'] is False


def test_first_contact_at_zero_has_no_precontact_row_and_is_not_clamped():
    for case in derive(fixture(0))['cases']:
        event = case['events']['PRE_CONTACT']
        assert event['tick'] is None and event['available'] is False
        assert event['unavailable_reason'] == 'first_actual_post_contact_at_tick_zero'


def test_declared_native_onset_without_onset_field_keeps_original_measured_velocity_fallback():
    parts=fixture();parts[1]['lift_onset_source']='native_reference_event'
    assert all(c['events']['LIFT_ONSET']['tick']==3 for c in derive(parts)['cases'])


def test_contact_already_observed_at_initial_boundary_cannot_be_called_precontact():
    parts=fixture()
    for env in range(4):
        parts[1]['events']['opposed_contact'][env*8]=True
        parts[3]['source_case_definitions'][env]['event_entry_ticks']['CONTACT']=0
    for case in derive(parts)['cases']:
        event=case['events']['PRE_CONTACT']
        assert event['available'] is False and event['tick'] is None
        assert event['unavailable_reason']=='contact_already_established_at_initial_pre_action_boundary'


@pytest.mark.parametrize('bad', ['missing_substep','duplicate','wrong_uid','wrong_owner',
    'invalid_contact_prefix','wrong_clock','wrong_terminal','wrong_issued_tape','wrong_previous_action'])
def test_incomplete_or_mismatched_actual_reference_is_refused(bad):
    parts = fixture(); raw, policies = parts[4], parts[5]
    if bad == 'missing_substep': del raw[0]
    elif bad == 'duplicate': raw.insert(1, copy.deepcopy(raw[0]))
    elif bad == 'wrong_uid': raw[0]['episode_uid'] = 'other-execution'
    elif bad == 'wrong_owner': raw[0]['action_owner'] = 'student'
    elif bad == 'invalid_contact_prefix': raw[0]['payload']['contact_valid'] = False
    elif bad == 'wrong_clock': raw[0]['timestamp_s'] = -1.
    elif bad == 'wrong_terminal': raw[-1]['payload']['done'] = False
    elif bad == 'wrong_issued_tape': policies[0]['issued_action'] = [.9]*24
    elif bad == 'wrong_previous_action': policies[0]['actual_previous_issued_action'] = [.9]*24
    with pytest.raises(ValueError): derive(parts)


def test_context_rows_are_identity_joined_before_legacy_contact_lookup():
    parts = fixture(); context = parts[1]
    order = torch.arange(31, -1, -1)
    def reorder(value):
        if isinstance(value, dict): return {k:reorder(v) for k,v in value.items()}
        if isinstance(value, torch.Tensor) and len(value)==32:return value[order]
        if isinstance(value, list) and len(value)==32:return [value[int(i)] for i in order]
        return value
    parts = (parts[0],reorder(context),*parts[2:])
    assert all(c['events']['CONTACT']['tick']==2 for c in derive(parts)['cases'])


def file_fixture(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import file_sha256
    data, context, collection, replay, raw, policies = fixture()
    def identity(p):return dict(path=str(p.resolve()),sha256=file_sha256(p))
    source=tmp_path/'source.pt';source.write_bytes(b'actual-checkpoint-identity-fixture')
    initial=tmp_path/'initial.pt';initial.write_bytes(b'actual-reset-identity-fixture')
    collection_path=tmp_path/'teacher_collection_report.json'
    for ep in data['episodes'].values():ep['proof_path']=str(collection_path)
    collection['episodes']=copy.deepcopy(data['episodes'])
    dataset=tmp_path/'successful.pt';torch.save(data,dataset)
    context['dataset_sha256']=file_sha256(dataset)
    context_path=tmp_path/'context.pt';torch.save(context,context_path)
    collection.update(status='completed',successful_dataset=identity(dataset),successful_sequence_context=identity(context_path))
    collection_path.write_text(json.dumps(collection))
    native=tmp_path/'native_replay';native.mkdir()
    replay['inputs']=dict(checkpoint=identity(source),teacher_dataset=identity(dataset),
        sequence_context=identity(context_path),source_initial_state=identity(initial),
        source_collection_report=identity(collection_path))
    raw_path=native/'source_physics_480hz.jsonl.gz'
    for p, rows in ((raw_path,raw),(native/'live_policy_60hz.jsonl.gz',policies)):
        with gzip.open(p,'wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    replay['raw_physics']=identity(raw_path)
    report=native/'stage6_native_report.json';report.write_text(json.dumps(replay))
    from go2_pvcnn.control.dh116.tabletop_vision_branching_eval import SOURCE_STATE_FIELDS
    matched=dict(matched=True,tolerance=0.,fields={n:dict(matched=True,dtype_equal=True,
        shape_equal=True,finite=True) for n in SOURCE_STATE_FIELDS})
    (native/'source_reset_comparison.json').write_text(json.dumps(matched))
    (native/'source_camera_warmup_comparison.json').write_text(json.dumps(dict(matched,
        before_clock=dict(physics_step=0),after_clock=dict(physics_step=0))))
    inputs=dict(source_checkpoint=identity(source),teacher_dataset=identity(dataset),
        sequence_context=identity(context_path),source_collection_report=identity(collection_path),
        source_initial_state=identity(initial),native_replay_report=identity(report))
    return inputs,raw_path


def test_file_binding_roundtrip_uses_reference_checkpoint_and_common_inputs(tmp_path):
    inputs,_=file_fixture(tmp_path);out=tmp_path/'event_reference.json'
    identity=api().build_event_reference(inputs,out)
    common={k:inputs[k] for k in ('teacher_dataset','sequence_context','source_collection_report','source_initial_state')}
    result=api().load_event_reference(identity,common,required_event='PRE_CONTACT')
    assert result['inputs']==inputs
    assert all(c['events']['PRE_CONTACT']['tick']==0 for c in result['cases'])
    assert api().load_event_reference(identity,common,required_event='LIFT_ONSET')['cases'][0]['events']['LIFT_ONSET']['tick']==3
    assert api().load_event_reference(identity,common,required_event='HOLD')['cases'][0]['events']['HOLD']['tick']==6
    with pytest.raises(FileExistsError):api().build_event_reference(inputs,out)
    wrong=dict(common,teacher_dataset=inputs['source_initial_state'])
    with pytest.raises(ValueError):api().load_event_reference(identity,wrong)


def test_changed_original_stream_or_forged_manifest_claim_is_refused(tmp_path):
    inputs,raw=file_fixture(tmp_path);out=tmp_path/'reference.json'
    identity=api().build_event_reference(inputs,out)
    content=json.loads(out.read_text());content['cases'][0]['events']['PRE_CONTACT']['tick']=1
    out.write_text(json.dumps(content))
    with pytest.raises(ValueError):api().load_event_reference(identity,inputs)
    identity=api().file_identity(out)
    with pytest.raises(ValueError):api().load_event_reference(identity,inputs)
    out.unlink();identity=api().build_event_reference(inputs,out)
    raw.write_bytes(raw.read_bytes()+b'x')
    with pytest.raises(ValueError):api().load_event_reference(identity,inputs)


def test_identical_code_in_independent_capsule_keeps_original_implementation_identity(tmp_path):
    inputs,_=file_fixture(tmp_path);identity=api().build_event_reference(inputs,tmp_path/'reference.json')
    capsule=tmp_path/'capsule';capsule.mkdir();path=capsule/'event_reference.py'
    path.write_bytes(Path(api().__file__).read_bytes())
    name='go2_pvcnn.control.dh116._event_reference_capsule_fixture'
    spec=importlib.util.spec_from_file_location(name,path)
    copied=importlib.util.module_from_spec(spec);spec.loader.exec_module(copied)
    value=copied.load_event_reference(identity,inputs,required_event='PRE_CONTACT')
    assert value['implementation']['path']==str(Path(api().__file__).resolve())
    assert value['implementation']['sha256']==copied.file_identity(path)['sha256']
    path.write_bytes(path.read_bytes()+b'\n# changed code bytes\n')
    with pytest.raises(ValueError):copied.load_event_reference(identity,inputs)
