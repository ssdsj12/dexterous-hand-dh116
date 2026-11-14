"""CPU schema/statistics only; synthetic records never establish native evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from go2_pvcnn.control.dh116.tabletop_stereo_evaluation import (
    physical_initial_sha256, evaluate_records, aggregate_episodes, wilson_interval,
    select_best_checkpoint,
)
from go2_pvcnn.control.dh116.tabletop_stereo_promotion import (
    verify_native_episode, verify_force_qualification, require_teacher_qualification,
    qualified_episode_identities, promotion_ledger,
    verify_stereo_qualification, verify_student_qualification, stereo_metric_gate,
)


def initial(x=0.,origin=None):
    d=dict(coordinate_frame='environment_local',object_pose=[x,0,.4,1,0,0,0],
           palm_pose=[[0,0,.4,1,0,0,0],[0,.1,.4,1,0,0,0]],
           joint_position=[.1]*12,joint_velocity=[0.]*12,object_velocity=[0.]*6)
    if origin is not None:
        d['coordinate_frame']='world';d['environment_origin']=origin
        for pose in [d['object_pose'],*d['palm_pose']]:
            for j in range(3):pose[j]+=origin[j]
    return d


def records(uid='e0',env=0,x=0.,owner='teacher',style='pinch_thumb_index',count=8,success=True):
    common=dict(run_uuid='CPU_SCHEMA_ONLY',env_id=env,episode_uid=uid,reset_id=0,
                owner=owner,grasp_style=style,source_scope='cpu_fixture')
    header=dict(common,kind='episode_start',physical_initial_condition=initial(x),
                initial_physics_step=0,initial_timestamp=0.,first_episode=True,resumed=False,recovered=False)
    samples=[dict(common,kind='sample',physics_step=i,timestamp=i/480,live=True,
        terminal=i==count,native_success=success and i==count,failure_code=0 if success else 8,
        object_net_lift_m=.03,formal_hold_s=5. if i==count else 0.,formal_drift_m=.0001,
        formal_tracker_source='surface_lift_success_30mm_5s_0.5mm',formal_tracker_valid=True,
        formal_window_completed=i%8==0,formal_window_good=i>=8,formal_anchor_latched=i>=8,
        formal_drift_source='last_completed_window_max_over_raw_pose_points',
        contact_valid=True,bilateral_opposed_contact=True,measurement_valid=True,
        external_support_force_n=None,external_upward_force_n=None,attachment_count=None,
        support_source_qualified=False,force_source_qualified=False) for i in range(1,count+1)]
    return [header,*samples]


def test_actual_initial_state_deduplicates_clones_and_ignores_random_seed_names():
    a=physical_initial_sha256(initial())
    assert a==physical_initial_sha256(initial(origin=[2.,3.,0.]))
    assert a!=physical_initial_sha256(initial(.001))
    invalid=initial();invalid.pop('joint_position');invalid['seed']=17
    with pytest.raises(ValueError,match='physical|initial'):
        physical_initial_sha256(invalid)


def test_wilson_interval_matches_known_small_sample_and_no_data_is_na():
    lo,hi=wilson_interval(0,10)
    assert abs(lo)<1e-12 and .27<hi<.28
    assert wilson_interval(0,0)==(None,None)
    with pytest.raises(ValueError):wilson_interval(11,10)


def test_task_success_and_qualified_success_remain_distinct_for_cpu_clones():
    report=evaluate_records(records()+records('e1',1),provenance={'source_scope':'cpu_fixture'})
    summary=report['summaries'][0]
    assert summary['episodes']==2 and summary['native_task_successes']==2
    assert summary['independent_initial_count']==1 and summary['qualified_successes']==0
    assert summary['lift_m']['p50']==.03
    assert summary['hold_s']['max']==5.
    assert summary['formal_slip_m']['coverage']==1.
    assert report['qualified_for_promotion'] is False


def test_owners_styles_and_failure_attribution_are_not_combined():
    raw=records(owner='mpc')+records('bc',2,owner='bc',success=False)+records('s',3,owner='student',style='tripod_thumb_index_middle')
    report=evaluate_records(raw,provenance={'source_scope':'cpu_fixture'})
    assert {(s['owner'],s['grasp_style']) for s in report['summaries']}=={
        ('mpc','pinch_thumb_index'),('bc','pinch_thumb_index'),('student','tripod_thumb_index_middle')}
    bc=next(s for s in report['summaries'] if s['owner']=='bc')
    assert bc['failure_breakdown']=={'8':1}


@pytest.mark.parametrize('fault',['hole','nonmonotonic_time','mixed_episode','diagnostic_slip','post_terminal','truncated'])
def test_invalid_raw_episode_cannot_qualify(fault):
    raw=records()
    if fault=='hole':del raw[3]
    if fault=='nonmonotonic_time':raw[3]['timestamp']=raw[2]['timestamp']
    if fault=='mixed_episode':raw[3]['episode_uid']='other'
    if fault=='diagnostic_slip':
        for row in raw[1:]:row['formal_tracker_source']='diagnostic_3mm_tracker'
    if fault=='post_terminal':raw.append(copy.deepcopy(raw[-1]))
    if fault=='truncated':raw.pop()
    report=evaluate_records(raw,provenance={'source_scope':'cpu_fixture'})
    assert not report['qualified_for_promotion']
    assert any(e['reasons'] for e in report['episodes'])
    if fault=='diagnostic_slip':assert report['summaries'][0]['formal_slip_m']['coverage']==0.


def test_no_valid_measurements_are_na_instead_of_zero_error_or_motion():
    raw=records()
    for row in raw[1:]:row.update(object_net_lift_m=None,formal_hold_s=None,formal_drift_m=None,contact_valid=False)
    report=evaluate_records(raw,provenance={'source_scope':'cpu_fixture'})
    s=report['summaries'][0]
    assert s['lift_m']['max'] is None and s['formal_slip_m']['max'] is None
    assert s['formal_slip_m']['coverage']==0.
    assert not report['qualified_for_promotion']


def artifact(tmp_path,name,data):
    p=tmp_path/name;p.write_bytes(data)
    return dict(path=str(p),sha256=hashlib.sha256(data).hexdigest())


def test_arbitrary_bytes_plus_native_boolean_do_not_qualify_episode_or_force(tmp_path):
    raw=artifact(tmp_path,'fake.bin',b'not a physical capture')
    proof=dict(version='dh116-native-episode-proof-v1',source_scope='native',schema_only=False,
               process_exit_code=0,raw_artifact=raw,sample_count=2400,qualified=True,
               success_metrics=dict(net_lift_m=.03,continuous_hold_s=5.,formal_anchor_slip_m=0.))
    with pytest.raises(ValueError):verify_native_episode(proof)
    f=artifact(tmp_path,'force.json',json.dumps(dict(proof,version='dh116-force-source-proof-v1')).encode())
    with pytest.raises(ValueError):verify_force_qualification({'proof_path':f['path'],'proof_sha256':f['sha256']})


def test_cpu_fixture_raw_artifact_is_never_native_qualification(tmp_path):
    raw=artifact(tmp_path,'cpu.jsonl','\n'.join(json.dumps(x) for x in records()).encode())
    producer=artifact(tmp_path,'producer.py',b'# CPU schema arithmetic only')
    cfg=artifact(tmp_path,'cfg.json',b'{}')
    kwargs=dict(physics_hz=480,control_hz=60)
    digest=hashlib.sha256(json.dumps(kwargs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    proof=dict(version='dh116-native-episode-proof-v1',source_scope='cpu_fixture',schema_only=True,process_exit_code=0,
               producer=dict(source_artifact=producer,config_artifact=cfg,kwargs=kwargs,kwargs_sha256=digest),
               raw_artifact=raw,sample_count=8)
    with pytest.raises(ValueError,match='native|scope|fixture'):verify_native_episode(proof)


def test_missing_or_tampered_native_artifact_and_wrong_checkpoint_are_rejected(tmp_path):
    report=evaluate_records(records(),provenance={'source_scope':'cpu_fixture','checkpoint_sha256':'1'*64})
    report.update(source_scope='native',owner='teacher',qualified_for_promotion=True)
    with pytest.raises(ValueError):require_teacher_qualification(report,'2'*64)
    with pytest.raises(ValueError):qualified_episode_identities(report)


def test_insufficient_independent_heldout_conditions_never_grant_promotion():
    report=evaluate_records(records(),provenance={'source_scope':'cpu_fixture'})
    assert 'insufficient_independent_initials' in ' '.join(report['blocked_reasons'])
    ledger=promotion_ledger({'phase0':{'implemented':True,'cpu_verified':True},'phase1':{'implemented':True,'cpu_verified':True}})
    assert ledger['formal_dagger_allowed'] is False and ledger['deployment_allowed'] is False
    assert ledger['phases']['phase1']['cpu_verified'] is True
    assert ledger['phases']['phase1']['physical_qualified'] is False


def test_checkpoint_selection_rejects_training_and_shared_initial_or_episode_leakage():
    report=evaluate_records(records(),provenance={'source_scope':'cpu_fixture','validation_split':'heldout','checkpoint_sha256':'1'*64})
    h=report['episodes'][0]['physical_initial_condition_sha256']
    with pytest.raises(ValueError,match='heldout|leak|initial'):
        select_best_checkpoint([report],training_initial_sha256=[h])
    report['validation_split']='train'
    with pytest.raises(ValueError,match='heldout'):
        select_best_checkpoint([report])


def test_missing_or_invalid_measurement_is_excluded_even_when_numeric_value_is_present():
    raw=records()
    for row in raw[1:]:row.update(measurement_valid=False)
    r=evaluate_records(raw,provenance={'source_scope':'cpu_fixture'})
    assert r['summaries'][0]['lift_m']['coverage']==0.
    assert r['summaries'][0]['hold_s']['coverage']==0.


def test_checkpoint_selection_requires_the_actual_checkpoint_artifact():
    r=evaluate_records(records(),provenance={'source_scope':'cpu_fixture','validation_split':'heldout','checkpoint_sha256':'1'*64})
    with pytest.raises(ValueError,match='checkpoint|artifact'):
        select_best_checkpoint([r])


def test_clone_retries_do_not_inflate_independent_success_rate_or_confidence():
    r=evaluate_records(records(success=False)+records('clone_retry',1),provenance={'source_scope':'cpu_fixture'})
    s=r['summaries'][0]
    assert s['raw_success_rate']==.5
    assert s['independent_trials']==1 and s['success_rate']==0.
    assert s['wilson95']==wilson_interval(0,1)


def test_stereo_and_student_native_schema_flags_alone_cannot_qualify():
    with pytest.raises(ValueError):verify_stereo_qualification({'qualified':True,'source_scope':'native'})
    with pytest.raises(ValueError):verify_student_qualification({'qualified':True,'source_scope':'native'},'1'*64)


def test_depth_threshold_arithmetic_has_a_positive_branch_without_native_claim():
    limits=dict(max_mae_m=.005,max_rmse_m=.01,min_coverage=.5,min_valid_pixels=1000)
    metrics=dict(mae_m=.001,rmse_m=.002,coverage=.8,count=2000)
    assert stereo_metric_gate(metrics,limits) is True
    assert stereo_metric_gate(dict(metrics,coverage=.1),limits) is False
    assert stereo_metric_gate(dict(metrics,mae_m=None),limits) is False


def test_promotion_ledger_positive_boolean_logic_uses_verifiers_not_fixture_native_claim(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_stereo_promotion as p
    # Pure control-flow test: these substitutes are NOT native evidence/jobs.
    monkeypatch.setattr(p,'require_teacher_qualification',lambda *a,**k:{'qualified':True})
    monkeypatch.setattr(p,'verify_stereo_qualification',lambda *a,**k:{'qualified':True})
    monkeypatch.setattr(p,'verify_student_qualification',lambda *a,**k:{'qualified':True})
    result=p.promotion_ledger({},teacher_report={},teacher_checkpoint_sha256='1'*64,
                             stereo_native_proof={},student_report={},student_checkpoint_sha256='2'*64)
    assert result['formal_dagger_allowed'] is True and result['deployment_allowed'] is True


def test_cpu_checkpoint_candidate_selection_binds_real_bytes_but_never_grants_physics(tmp_path):
    import io
    import torch
    checkpoint=io.BytesIO();torch.save(torch.nn.Linear(1,1).state_dict(),checkpoint)
    binding=artifact(tmp_path,'cpu_model.pt',checkpoint.getvalue());binding['kind']='checkpoint'
    report=evaluate_records(records(),provenance={'source_scope':'cpu_fixture','validation_split':'heldout','checkpoint_sha256':binding['sha256']},evidence_artifacts=[binding])
    report['qualified_for_promotion']=True  # Self-declaration must be ignored.
    chosen=select_best_checkpoint([report])
    assert chosen['checkpoint_sha256']==binding['sha256']
    assert chosen['physical_qualified'] is False


def test_native_checkpoint_selection_requires_bound_training_source_manifest(tmp_path):
    binding=artifact(tmp_path,'model.pt',b'CPU schema checkpoint bytes');binding['kind']='checkpoint'
    report=evaluate_records(records(),provenance={'source_scope':'native','validation_split':'heldout',
        'checkpoint_sha256':binding['sha256'],'training_initial_sha256':[],'training_episode_groups':[]},
        evidence_artifacts=[binding])
    # Merely writing empty training sets must not certify heldout independence.
    with pytest.raises(ValueError,match='training.*manifest'):
        select_best_checkpoint([report])


def test_training_source_manifest_outer_bytes_are_reopened_before_metadata(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_evaluation import training_partition
    manifest=artifact(tmp_path,'lineage.json',b'{}')
    manifest['sha256']='f'*64
    report={'checkpoint_sha256':'1'*64,'provenance':{'training_manifest_artifact':manifest}}
    with pytest.raises(ValueError,match='manifest.*SHA'):
        training_partition(report)


def test_native_source_verification_alone_does_not_mark_selected_checkpoint_physical(tmp_path,monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_stereo_evaluation as evaluation
    import go2_pvcnn.control.dh116.tabletop_stereo_promotion as promotion
    binding=artifact(tmp_path,'selection_model.pt',b'CPU selection branch test');binding['kind']='checkpoint'
    report=evaluate_records(records(),provenance={'source_scope':'native','validation_split':'heldout',
        'checkpoint_sha256':binding['sha256'],'owner':'teacher'},evidence_artifacts=[binding])
    episodes=copy.deepcopy(report['episodes'])
    for row in episodes:row['native_evidence_verified']=True
    # Control flow substitutes only, never actual native evidence or jobs.
    monkeypatch.setattr(evaluation,'training_partition',lambda r:{'physical_initial_condition_sha256':set(),'source_episode_groups':set()})
    monkeypatch.setattr(promotion,'_report_episodes',lambda r:(r,episodes,tmp_path))
    monkeypatch.setattr(promotion,'require_teacher_qualification',lambda *a,**k:(_ for _ in ()).throw(ValueError('100 independent conditions missing')))
    chosen=select_best_checkpoint([report])
    assert chosen['physical_qualified'] is False
    assert '100 independent' in chosen['physical_qualification_reasons'][0]


def test_reference_created_after_process_completion_cannot_join_the_native_proof(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_promotion import _process_output
    capture=artifact(tmp_path,'completed_raw.bin',b'CPU raw output binding arithmetic')
    completion=artifact(tmp_path,'completion.json',json.dumps({'produced_artifacts':[capture]}).encode())
    proof={'completion_artifact':completion}
    assert _process_output(proof,tmp_path,capture)[0]==b'CPU raw output binding arithmetic'
    inserted=artifact(tmp_path,'late_reference.bin',b'created after observed process exit')
    with pytest.raises(ValueError,match='absent.*completion'):
        _process_output(proof,tmp_path,inserted)


def test_real_surface_tracker_2400_raw_steps_accepts_float_hold_and_pending_first_window():
    import torch
    from test_surface_success import scene
    from go2_pvcnn.control.dh116.tabletop_surface_success import SurfaceLiftSuccess
    tracker=SurfaceLiftSuccess(1,'cpu',.30)
    measurement=scene(n=1)
    raw=records(count=2400)
    raw[0]['physical_initial_condition']=initial()
    for i,row in enumerate(raw[1:],1):
        result=tracker.update(**measurement,dt=1/480)
        def value(name):return measurement[name][0].tolist()
        row.update(native_success=bool(result['event'][0]),terminal=bool(result['event'][0]),
            formal_hold_s=float(tracker.duration[0]),formal_drift_m=float(tracker.drift_m[0]),
            formal_window_completed=result['updated'],formal_window_good=bool(tracker.window_good[0]),
            formal_anchor_latched=bool(tracker._latched[0]),formal_tracker_valid=bool(tracker._physical_window_valid[0]),
            formal_object_pose_w=value('object_pose'),formal_wrist_pose_w=value('wrist_pose'),
            formal_pad_loads_n=value('loads'),formal_pad_vectors_w=value('vectors'),
            formal_pad_depth_m=value('depth'),formal_rejected_load_n=value('ordinary'),
            formal_object_bottom_m=value('bottom'),formal_object_velocity=value('object_velocity'),
            formal_reference_bottom_m=.30,formal_task_active=True)
    assert result['event'][0] and tracker.duration[0]<5.
    assert all(not r['formal_window_good'] for r in raw[1:8])
    report=evaluate_records(raw,provenance={'source_scope':'cpu_fixture'})
    episode=report['episodes'][0]
    assert episode['formal_replay_event'] is True
    assert episode['formal_window_count']==300
    assert 'formal_hold_below_5s_or_NA' not in episode['reasons']
    assert 'raw_hold_lift_formal_slip_contact_not_qualified' not in episode['reasons']
    assert episode['qualified_success'] is False  # CPU regression is not native evidence.
    prefix=copy.deepcopy(raw[:9]);prefix[-1]['formal_tracker_valid']=False
    tampered=evaluate_records(prefix,provenance={'source_scope':'cpu_fixture'})
    assert 'formal_window_physical_validity_source_mismatch' in tampered['episodes'][0]['reasons']


def test_student_task_evidence_rejects_cross_task_samples_and_flexible_claims():
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import digest
    from go2_pvcnn.control.dh116.tabletop_stereo_promotion import _student_task_identity

    def bound(kind, obj, source, rod):
        identity = dict(version='dh116-stereo-native-task-identity-v1', task_kind=kind,
            physical_object_kind=obj, state_source=source, rod_state_available=rod, task_profile_sha256='e'*64)
        identity['task_identity_sha256'] = digest(identity)
        return dict(task_identity=identity, task_identity_sha256=identity['task_identity_sha256'])

    cylinder = bound('rigid_proxy', 'solid_cylinder', 'rigid_body_sdk_pose_velocity', False)
    straw = bound('rigid_proxy', 'hollow_straw', 'rigid_body_sdk_pose_velocity', False)
    flexible = bound('deformable_rod', 'cosserat_rod', 'cosserat_solver_node_readback', True)
    assert _student_task_identity(cylinder, [dict(cylinder)]) == cylinder['task_identity']
    with pytest.raises(ValueError, match='task identity'):
        _student_task_identity(cylinder, [dict(straw)])
    with pytest.raises(ValueError, match='rigid_proxy'):
        _student_task_identity(flexible, [dict(flexible)])
    with pytest.raises(ValueError, match='task identity'):
        _student_task_identity(cylinder, [{}])


@pytest.mark.parametrize('failure, message', [
    ('declared_flexible', 'declare rigid_proxy'),
    ('mixed', 'mixes raw task'),
    ('duplicate', 'initial condition repeated'),
])
def test_student_report_requires_one_task_and_unique_physical_initials(tmp_path, monkeypatch, failure, message):
    from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion

    report = dict(owner='student', checkpoint_sha256='1'*64, validation_split='heldout',
                  task_kind='rigid_proxy', task_identity_sha256='a'*64)
    if failure == 'declared_flexible':
        report['task_kind'] = 'deformable_rod'
    runtime = dict(teacher_execution_enabled=False, mpc_enabled=False, critic_deployed=False,
        privileged_actor_inputs=False, critic_training=False, actor_input_source='stereo_RGB_and_proprio_only',
        depth_source='computed_from_synchronized_left_right_RGB', model_checkpoint_sha256='1'*64)
    episodes = [
        dict(owner='student', checkpoint_sha256='1'*64, task_identity=dict(task_kind='rigid_proxy',
            task_identity_sha256='a'*64), actor_runtime_samples_verified=True, runtime=runtime,
            physical_initial_condition_sha256=key*64, source_episode_group=key, grasp_style=style)
        for key, style in [('b', 'pinch_thumb_index'), ('c', 'tripod_thumb_index_middle')]
    ]
    if failure == 'mixed':
        episodes[1]['task_identity']['task_identity_sha256'] = 'd'*64
    if failure == 'duplicate':
        episodes[1]['physical_initial_condition_sha256'] = episodes[0]['physical_initial_condition_sha256']
    monkeypatch.setattr(promotion, '_report_episodes', lambda _: (report, episodes, tmp_path))
    monkeypatch.setattr(promotion, 'training_partition', lambda *_args, **_kwargs: dict(
        physical_initial_condition_sha256=set(), source_episode_groups=set()))
    with pytest.raises(ValueError, match=message):
        promotion.verify_student_qualification(report, '1'*64)
