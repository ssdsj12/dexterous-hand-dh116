"""Synthetic trust-boundary unit fixtures only; no real native trial counts."""
import copy
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_vision_experiment_checkpoint import checkpoint, sha
from test_vision_evaluation_trials import manifest, sample
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import (
    FirstTerminalAggregator, validate_manifest, state_sha256, _digest,
)


def module():
    name='go2_pvcnn.control.dh116.tabletop_validation_selection'
    assert importlib.util.find_spec(name), 'independent validation selection API is missing'
    return importlib.import_module(name)


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding='utf-8')
    return sha(path)


def evidence(tmp_path, *, successes=0, lift=.031, split='validation', slip_mode='all_valid',
             torque_size=12,action_size=24):
    """Simulates externally trusted evidence shape, never actual native execution."""
    cp,_=checkpoint(tmp_path)
    from go2_pvcnn.control.dh116.tabletop_vision_experiment_checkpoint import retain_checkpoint_bundle
    bundle=retain_checkpoint_bundle(cp,tmp_path/'bundles')
    frozen=validate_manifest(manifest(100,split))
    collector=FirstTerminalAggregator(frozen,checkpoint_sha256=sha(cp),deterministic=True)
    raw=[];conditional=[]
    for i,row in enumerate(frozen['trials']):
        s=sample(lift_m=lift,qualified_hold_s=5.,action=[0.]*action_size,measurement_valid=True,
            estimated_torque_nm=[.1]*torque_size,effort_limit_nm=[.2]*torque_size)
        if slip_mode in ('initial_no_anchor','never_anchor','success_only_anchor'):s['slip_m']=None
        collector.observe(i,row['trial_id'],s)
        raw.append(dict(env_id=i,trial_id=row['trial_id'],sample=copy.deepcopy(s),terminal=False,
            native_success=False,slip_applicable=s['slip_m'] is not None,
            slip_valid=s['slip_m'] is not None,
            prior_anchor_latched=s['slip_m'] is not None,
            slip_inapplicable_reason=None if s['slip_m'] is not None else 'no_anchor'))
        s['action']=[.01]*action_size
        if slip_mode not in ('never_anchor','success_only_anchor') or (slip_mode=='success_only_anchor' and i%100<successes):
            s['slip_m']=.0002
        collector.observe(i,row['trial_id'],s,terminal=True,native_success=i%100<successes)
        raw.append(dict(env_id=i,trial_id=row['trial_id'],sample=copy.deepcopy(s),terminal=True,
            native_success=i%100<successes,slip_applicable=s['slip_m'] is not None,
            slip_valid=s['slip_m'] is not None,
            prior_anchor_latched=s['slip_m'] is not None,
            slip_inapplicable_reason=None if s['slip_m'] is not None else 'no_anchor'))
        app=.5 if slip_mode=='all_valid' else .25 if s['slip_m'] is not None else 0.
        conditional.append(dict(trial_id=row['trial_id'],observed_duration_s=.5,
            slip_applicable_duration_s=app,slip_valid_applicable_duration_s=app,
            slip_inapplicable_duration_s=.5-app,slip_applicable_missing_samples=0,
            slip_applicability_missing_samples=0,
            slip_inapplicable_reason_counts={} if app==.5 else {'no_anchor':1 if app else 2}))
    report=collector.summary()
    report['conditional_telemetry']=dict(per_trial=conditional,raw_sample_count=len(raw))
    report['conditional_telemetry_complete']=True
    report_path=tmp_path/'report.json';write(report_path,report)
    telemetry_path=tmp_path/'samples.jsonl.gz'
    with gzip.open(telemetry_path,'wt',encoding='utf-8') as stream:
        stream.write(json.dumps({'schema_version':'dh116-conditional-telemetry-samples-v1','format':'gzip_jsonl',
            'manifest_sha256':_digest(frozen),'checkpoint_sha256':sha(cp)})+'\n')
        for row in raw:stream.write(json.dumps(row)+'\n')
    actual=[]
    for row in frozen['trials']:
        actual.append(dict(trial_id=row['trial_id'],grasp_style=row['grasp_style'],
            authored_state_sha256=row['state_sha256'],readback_state_sha256=row['state_sha256'],
            physical_state=row['physical_state']))
    readback=dict(schema_version='dh116-evaluation-reset-readback-v1',
        native_validation_complete=False,actual_states=actual,unique_readback_count=200,
        max_abs_writer_readback_error=0.,writer_match_tolerance=2e-6,
        identity='actual_local_float32_state_not_placement_or_label',physics_steps_advanced=0)
    # The unit test simulates the external trust anchor; production fixtures
    # declaring fixture execution/status are explicitly rejected below.
    readback_path=tmp_path/'readback.json';write(readback_path,readback)
    geometry_path=tmp_path/'geometry.json';write(geometry_path,dict(passed=True,
        scope='initial_validation_only_GT_never_returned_to_actor',tolerance_p95_m=.003,
        environments=[dict(env_id=i,point_count=100,median_m=.001,p95_m=.002,max_m=.003)
                      for i in range(200)],failed_env_ids=[]))
    actor_path=tmp_path/'actor.pt'
    history=torch.zeros(200,4,961);history[:,-1,768:896]=1.
    history[:,-1,-3]=1./60;history[:,-1,-2:]=1.
    torch.save(dict(policy=history.flatten(1),previous_action=torch.zeros(200,24),history=history,
        contract={'actor_observation_size':3844,'critic_observation_size':371,'action_contract':{'size':24}},
        sensor_packet=dict(points=torch.zeros(200,2,128,3),point_valid=torch.ones(200,128,dtype=torch.bool),
            image_valid=torch.ones(200,dtype=torch.bool),object_valid=torch.ones(200,dtype=torch.bool))),actor_path)
    frozen_path=tmp_path/'frozen_manifest.json';write(frozen_path,frozen)
    files=dict(raw_reset_readback=readback_path,initial_camera_geometry=geometry_path,
        initial_visual_actor_inputs=actor_path,frozen_manifest=frozen_path,
        conditional_telemetry_samples=telemetry_path)
    cert=dict(schema_version='dh116-native-visual-evaluation-certificate-v1',
        execution='native_isaac_tensor_world',attestation_status='native_verified',
        report_sha256=sha(report_path),checkpoint_sha256=sha(cp),manifest_sha256=_digest(frozen),
        split=split,expected_count=200,runner_source_sha256='9'*64,
        physics_sha256=frozen['metadata']['physics_sha256'],camera_sha256=frozen['metadata']['camera_sha256'],
        command=['python','native_evaluation.py','--vision-evaluate'],
        native_execution_complete=True,conditional_telemetry_complete=True,
        evidence_files={name:dict(path=str(p),sha256=sha(p)) for name,p in files.items()})
    for name in ('raw_reset_readback','initial_camera_geometry','initial_visual_actor_inputs'):
        cert[name+'_sha256']=sha(files[name])
    cert_path=tmp_path/'certificate.json';write(cert_path,cert)
    return dict(bundle_manifest_path=Path(bundle['manifest_path']),report_path=report_path,
        certificate_path=cert_path,trusted_certificate_sha256=sha(cert_path),
        output_dir=tmp_path/'selection'),report,cert


def refresh(args,report,cert):
    write(args['report_path'],report);cert['report_sha256']=sha(args['report_path'])
    write(args['certificate_path'],cert);args['trusted_certificate_sha256']=sha(args['certificate_path'])


def test_zero_success_selects_validation_only_and_retains_immutable_candidate(tmp_path):
    api=module(); args,report,cert=evidence(tmp_path)
    assert report['native_validation_complete'] is False
    rng=torch.get_rng_state().clone()
    result=api.select_validation_candidate(**args)
    root=args['output_dir'];best=json.loads((root/'best_validation.json').read_text())
    assert best['checkpoint_sha256']==cert['checkpoint_sha256']
    assert Path(result['candidate_path']).is_file() and result['native_success_macro_rate']==0.
    assert not (root/'best_success_rate.json').exists()
    assert not list(root.glob('*successful*')) and torch.equal(rng,torch.get_rng_state())
    assert api.select_validation_candidate(**args)==result


def test_lexicographic_native_success_precedes_lift_and_previous_candidates_survive(tmp_path):
    api=module(); args,_,_=evidence(tmp_path,successes=0,lift=.080)
    first=api.select_validation_candidate(**args)
    other=tmp_path/'other';other.mkdir()
    second_args,report,cert=evidence(other,successes=1,lift=.031)
    second_args['output_dir']=args['output_dir']
    second=api.select_validation_candidate(**second_args)
    best=json.loads((args['output_dir']/'best_validation.json').read_text())
    success=json.loads((args['output_dir']/'best_success_rate.json').read_text())
    assert best['candidate_path']==second['candidate_path']==success['candidate_path']
    assert Path(first['candidate_path']).exists()
    assert second['native_success_macro_rate']==.01
    assert set(second['styles'])=={'pinch_thumb_index','tripod_thumb_index_middle'}


@pytest.mark.parametrize('bad',['fixture','authored','trust','report_hash','test_split','incomplete',
    'missing_metric','forged_summary','duplicate_state','raw_hash','command','readback_missing',
    'readback_duplicate','geometry','actor_privilege','actor_previous','actor_history','actor_camera','bundle_hash'])
def test_untrusted_incomplete_forged_or_nonvalidation_evidence_cannot_select(tmp_path,bad):
    api=module();args,report,cert=evidence(tmp_path)
    if bad=='fixture':cert['execution']='fixture_only'
    elif bad=='authored':cert['attestation_status']='authored_unverified'
    elif bad=='trust':args['trusted_certificate_sha256']='0'*64
    elif bad=='report_hash':cert['report_sha256']='0'*64
    elif bad=='test_split':report['split']=cert['split']='test'
    elif bad=='incomplete':report['trial_rows'][0]['status']='incomplete'
    elif bad=='missing_metric':report['styles']['pinch_thumb_index']['slip_quantiles_m']['P95']=None
    elif bad=='forged_summary':report['styles']['pinch_thumb_index']['native_success_rate']=1.
    elif bad=='duplicate_state':report['trial_rows'][1]['state_sha256']=report['trial_rows'][0]['state_sha256']
    elif bad=='raw_hash':cert['evidence_files']['raw_reset_readback']['sha256']='0'*64
    elif bad=='command':cert['command']=[]
    elif bad in ('readback_missing','readback_duplicate'):
        record=cert['evidence_files']['raw_reset_readback'];p=Path(record['path']);v=json.loads(p.read_text())
        if bad=='readback_missing':v['actual_states'].pop()
        else:v['actual_states'][1]['physical_state']=copy.deepcopy(v['actual_states'][0]['physical_state'])
        record['sha256']=write(p,v);cert['raw_reset_readback_sha256']=record['sha256']
    elif bad=='geometry':
        rec=cert['evidence_files']['initial_camera_geometry'];p=Path(rec['path']);v=json.loads(p.read_text())
        v['environments'][0]['p95_m']=.004;rec['sha256']=write(p,v);cert['initial_camera_geometry_sha256']=rec['sha256']
    elif bad in ('actor_privilege','actor_previous','actor_history','actor_camera'):
        rec=cert['evidence_files']['initial_visual_actor_inputs'];p=Path(rec['path']);v=torch.load(p,weights_only=False)
        if bad=='actor_privilege':v['policy']=torch.zeros(200,371)
        elif bad=='actor_previous':v['previous_action'][0,0]=1.
        elif bad=='actor_history':v['history'][0,0,0]=1.
        else:v['sensor_packet']['image_valid'][0]=False
        torch.save(v,p);rec['sha256']=sha(p);cert['initial_visual_actor_inputs_sha256']=rec['sha256']
    elif bad=='bundle_hash':
        p=args['bundle_manifest_path'];v=json.loads(p.read_text());v['checkpoint_sha256']='0'*64;write(p,v)
    if bad!='trust':refresh(args,report,cert)
    if bad=='report_hash':
        cert['report_sha256']='0'*64
        args['trusted_certificate_sha256']=write(args['certificate_path'],cert)
    with pytest.raises((ValueError,FileNotFoundError)):
        api.select_validation_candidate(**args)
    assert not args['output_dir'].exists()


def test_candidates_from_different_frozen_validation_manifest_are_incomparable(tmp_path):
    api=module();args,_,_=evidence(tmp_path);first=api.select_validation_candidate(**args)
    other=tmp_path/'other';other.mkdir();second,report,cert=evidence(other)
    p=Path(cert['evidence_files']['frozen_manifest']['path']);frozen=json.loads(p.read_text())
    frozen['metadata']['seed']+=1;cert['evidence_files']['frozen_manifest']['sha256']=write(p,frozen)
    report['manifest_sha256']=cert['manifest_sha256']=_digest(frozen)
    raw=Path(cert['evidence_files']['conditional_telemetry_samples']['path'])
    with gzip.open(raw,'rt',encoding='utf-8') as stream:lines=[json.loads(v) for v in stream]
    lines[0]['manifest_sha256']=_digest(frozen)
    with gzip.open(raw,'wt',encoding='utf-8') as stream:
        for row in lines:stream.write(json.dumps(row)+'\n')
    cert['evidence_files']['conditional_telemetry_samples']['sha256']=sha(raw)
    refresh(second,report,cert)
    second['output_dir']=args['output_dir']
    with pytest.raises(ValueError,match='frozen validation manifest'):
        api.select_validation_candidate(**second)
    assert Path(first['candidate_path']).is_file()


@pytest.mark.parametrize('mode',['initial_no_anchor','never_anchor'])
def test_legitimate_unobservable_slip_remains_null_and_can_select_zero_success_validation(tmp_path,mode):
    api=module();args,report,cert=evidence(tmp_path,slip_mode=mode)
    assert not report['telemetry_complete'] and not report['validation_complete']
    result=api.select_validation_candidate(**args)
    assert result['slip_ranking_policy']=='unknown_is_worst_after_success_hold_height'
    if mode=='never_anchor':
        assert result['styles']['pinch_thumb_index']['max_slip_m'] is None
        assert result['slip_metric_available'] is False
    assert (args['output_dir']/'best_validation.json').exists()
    assert not (args['output_dir']/'best_success_rate.json').exists()


@pytest.mark.parametrize('bad',['applicable_missing','applicability_unknown','raw_report_mismatch','missing_clock'])
def test_conditional_telemetry_must_be_replayed_and_never_inferred_from_valid_flag(tmp_path,bad):
    api=module();args,report,cert=evidence(tmp_path,slip_mode='initial_no_anchor')
    rec=cert['evidence_files']['conditional_telemetry_samples'];p=Path(rec['path'])
    with gzip.open(p,'rt',encoding='utf-8') as stream:lines=[json.loads(v) for v in stream]
    if bad=='applicable_missing':lines[1]['slip_applicable']=True
    elif bad=='applicability_unknown':lines[1]['slip_applicable']=None
    elif bad=='missing_clock':lines[1]['sample'].pop('dt_s')
    else:lines[1]['sample']['lift_m']=1.
    with gzip.open(p,'wt',encoding='utf-8') as stream:
        for row in lines:stream.write(json.dumps(row)+'\n')
    rec['sha256']=sha(p);refresh(args,report,cert)
    with pytest.raises(ValueError):api.select_validation_candidate(**args)
    assert not args['output_dir'].exists()


def test_selection_lock_is_exclusive_and_never_removes_another_writer_lock(tmp_path):
    api=module();args,_,_=evidence(tmp_path);root=args['output_dir'];root.mkdir()
    lock=root/'.selection.lock';lock.write_bytes(b'competing selection')
    with pytest.raises(FileExistsError):api.select_validation_candidate(**args)
    assert lock.read_bytes()==b'competing selection'
    assert list(root.iterdir())==[lock]


def test_selection_sidecar_race_cannot_overwrite_or_delete_competing_candidate(tmp_path,monkeypatch):
    api=module();args,_,_=evidence(tmp_path);original=Path.open;raced=[]
    def racing(path,mode='r',*a,**kw):
        if path.name.startswith('candidate_') and mode=='xb' and not raced:
            raced.append(path)
            with original(path,'wb') as stream:stream.write(b'competing candidate')
        return original(path,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',racing)
    with pytest.raises(FileExistsError):api.select_validation_candidate(**args)
    assert raced[0].read_bytes()==b'competing candidate'
    assert not (args['output_dir']/'best_validation.json').exists()
    assert not (args['output_dir']/'.selection.lock').exists()


def test_atomic_pointer_failure_retains_previous_best_and_immutable_new_candidate(tmp_path,monkeypatch):
    api=module();args,_,_=evidence(tmp_path);api.select_validation_candidate(**args)
    best=args['output_dir']/'best_validation.json';before=best.read_bytes()
    other=tmp_path/'other';other.mkdir();new,_,_=evidence(other,successes=1)
    new['output_dir']=args['output_dir']
    def failed(*a,**kw):raise OSError('injected pointer publication failure')
    monkeypatch.setattr(api.os,'replace',failed)
    with pytest.raises(OSError,match='pointer publication'):api.select_validation_candidate(**new)
    assert best.read_bytes()==before and len(list(args['output_dir'].glob('candidate_*.json')))==2
    assert not list(args['output_dir'].glob('*.tmp')) and not (args['output_dir']/'.selection.lock').exists()


def test_true_native_successes_remain_eligible_when_failed_trials_never_have_anchor(tmp_path):
    api=module();args,report,cert=evidence(tmp_path,successes=1,slip_mode='success_only_anchor')
    result=api.select_validation_candidate(**args)
    assert result['native_success_macro_rate']==.01 and not result['slip_metric_available']
    assert result['styles']['pinch_thumb_index']['completed_count']==100
    assert (args['output_dir']/'best_success_rate.json').exists()


def test_checkpoint_change_during_raw_replay_cannot_publish_candidate(tmp_path,monkeypatch):
    api=module();args,_,_=evidence(tmp_path);original=api._replay_conditional_samples
    pt=Path(json.loads(args['bundle_manifest_path'].read_text())['checkpoint_path'])
    def mutate(*a,**kw):
        result=original(*a,**kw);pt.write_bytes(pt.read_bytes()+b'external mutation')
        return result
    monkeypatch.setattr(api,'_replay_conditional_samples',mutate)
    with pytest.raises(ValueError,match='changed during selection'):api.select_validation_candidate(**args)
    assert not args['output_dir'].exists()


@pytest.mark.parametrize('torque_size,action_size',[(2,24),(11,24),(13,24),(12,2),(12,23),(12,25)])
def test_native_verifier_rejects_self_consistent_wrong_active_joint_or_action_dimensions(
        tmp_path,torque_size,action_size):
    api=module();args,report,cert=evidence(tmp_path,torque_size=torque_size,action_size=action_size)
    # Generic aggregation accepts a nonempty packet and reports full coverage.
    # A native certificate must independently enforce all 12 active joints and
    # all 24 issued action elements, including older producer output.
    assert report['conditional_telemetry_complete']
    assert report['styles']['pinch_thumb_index']['torque_valid_coverage']==1.
    with pytest.raises(ValueError,match='native raw sample requires complete 12D torque/limits and 24D action'):
        api.select_validation_candidate(**args)
    assert not args['output_dir'].exists()


def test_native_verifier_independently_requires_complete_effort_limit_vector(tmp_path):
    api=module();args,report,cert=evidence(tmp_path)
    record=cert['evidence_files']['conditional_telemetry_samples'];path=Path(record['path'])
    with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    rows[1]['sample']['effort_limit_nm']=rows[1]['sample']['effort_limit_nm'][:11]
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    record['sha256']=sha(path);refresh(args,report,cert)
    with pytest.raises(ValueError,match='native raw sample requires complete 12D torque/limits and 24D action'):
        api.select_validation_candidate(**args)
    assert not args['output_dir'].exists()


def test_known_anchor_with_measured_opposition_cannot_be_disguised_as_no_contact(tmp_path):
    api=module();args,report,cert=evidence(tmp_path,successes=1,slip_mode='success_only_anchor')
    record=cert['evidence_files']['conditional_telemetry_samples'];path=Path(record['path'])
    with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    changed=0
    for raw in rows[1:]:
        if raw['sample']['slip_m'] is None:
            assert raw['sample']['measurement_valid'] and raw['sample']['bilateral_opposed_contact']
            raw['prior_anchor_latched']=True
            raw['slip_applicable']=False
            raw['slip_inapplicable_reason']='no_contact'
            changed+=1
    assert changed==398
    for trial in report['conditional_telemetry']['per_trial']:
        reasons=trial['slip_inapplicable_reason_counts']
        if 'no_anchor' in reasons:trial['slip_inapplicable_reason_counts']={'no_contact':reasons['no_anchor']}
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    record['sha256']=sha(path);refresh(args,report,cert)
    with pytest.raises(ValueError,match='slip applicability disagrees with raw anchor/contact evidence'):
        api.select_validation_candidate(**args)
    assert not args['output_dir'].exists()


@pytest.mark.parametrize('invalid',[None,'contact','measurement'])
def test_known_anchor_na_requires_trustworthy_no_contact_measurement(tmp_path,invalid):
    api=module();args,report,cert=evidence(tmp_path,slip_mode='never_anchor')
    record=cert['evidence_files']['conditional_telemetry_samples'];path=Path(record['path'])
    with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    frozen=json.loads(Path(cert['evidence_files']['frozen_manifest']['path']).read_text())
    replay=FirstTerminalAggregator(frozen,checkpoint_sha256=cert['checkpoint_sha256'],deterministic=True)
    for raw in rows[1:]:
        raw['prior_anchor_latched']=True
        raw['sample']['bilateral_opposed_contact']=False
        raw['slip_inapplicable_reason']='no_contact'
        replay.observe(raw['env_id'],raw['trial_id'],raw['sample'],
            terminal=raw['terminal'],native_success=raw['native_success'])
    conditional=report['conditional_telemetry']
    for trial in conditional['per_trial']:
        trial['slip_inapplicable_reason_counts']={'no_contact':2}
    report=replay.summary();report['conditional_telemetry']=conditional
    report['conditional_telemetry_complete']=True
    if invalid=='contact':rows[1]['sample']['bilateral_opposed_contact']=None
    elif invalid=='measurement':rows[1]['sample']['measurement_valid']=False
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    record['sha256']=sha(path);refresh(args,report,cert)
    if invalid:
        with pytest.raises(ValueError,match='slip applicability disagrees with raw anchor/contact evidence'):
            api.select_validation_candidate(**args)
        assert not args['output_dir'].exists()
    else:
        result=api.select_validation_candidate(**args)
        assert result['native_success_macro_rate']==0. and not result['slip_metric_available']
        assert (args['output_dir']/'best_validation.json').exists()
        assert not (args['output_dir']/'best_success_rate.json').exists()
