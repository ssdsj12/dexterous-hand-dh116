import importlib
import importlib.util
import pytest
import torch
from test_vision_evaluation_trials import manifest
from test_vision_action_derivation import source_checkpoint
from test_vision_training import fixture, Camera
from test_vision_evaluation_reset import world_stub, reset_manifest
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import _digest
from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import file_sha256
from types import SimpleNamespace
import json


def api():
    name='go2_pvcnn.control.dh116.tabletop_vision_evaluation'
    assert importlib.util.find_spec(name), 'visual first-terminal runner is missing'
    return importlib.import_module(name)


def payload():
    return dict(dt_s=.01, live=torch.tensor([True,True]),done=torch.tensor([True,False]),
        native_success=torch.tensor([False,False]),measurement_valid=torch.ones(2,dtype=torch.bool),
        lift_m=torch.tensor([.031,.01]),lift_valid=torch.ones(2,dtype=torch.bool),
        qualified_hold_s=torch.zeros(2),qualified_hold_valid=torch.ones(2,dtype=torch.bool),
        slip_m=torch.full((2,),float('nan')),slip_valid=torch.zeros(2,dtype=torch.bool),
        bilateral_opposed_contact=torch.ones(2,dtype=torch.bool),contact_valid=torch.ones(2,dtype=torch.bool),
        estimated_torque_nm=torch.zeros(2,12),effort_limit_nm=torch.ones(2,12),
        torque_valid=torch.ones(2,12,dtype=torch.bool),
        action=torch.zeros(2,24),action_valid=torch.ones(2,dtype=torch.bool))


def test_collector_waits_for_begin_retains_terminal_and_ignores_repeated_episode():
    c=api().EvaluationTelemetryCollector(manifest(),checkpoint_sha256='a'*64,min_trials_per_style=1)
    p=payload(); c(p)
    assert c.summary()['completed_count']==0
    c.begin_trials(); c(p)
    p['lift_m'][0]=10.; p['done'][:]=True; c(p)
    result=c.summary()
    assert result['completed_count']==2 and result['trial_rows'][0]['peak_lift_m']==pytest.approx(.031)
    assert result['trial_rows'][0]['sample_count']==1
    assert result['trial_rows'][0]['max_slip_m'] is None
    assert result['trial_rows'][0]['slip_valid_coverage']==0
    assert result['trial_rows'][0]['contact_alive_duration_s']==.01
    assert not result['native_validation_complete']
    with pytest.raises(RuntimeError,match='begun'):
        c.begin_trials()


def test_conditional_slip_keeps_unobservable_null_and_rejects_missing_applicable_measurement():
    c=api().EvaluationTelemetryCollector(manifest(),checkpoint_sha256='a'*64,min_trials_per_style=1)
    c.begin_trials(); p=payload()
    p['done'][:]=False
    p['slip_applicable']=torch.tensor([False,True])
    p['slip_inapplicable_reason']=['no_anchor',None]
    c(p); p['done'][:]=True; c(p)
    report=c.summary()
    assert 'conditional_telemetry' in report, 'conditional observability accounting missing'
    rows=report['conditional_telemetry']['per_trial']
    assert rows[0]['slip_inapplicable_duration_s']==.02
    assert rows[0]['slip_applicable_missing_samples']==0
    assert rows[0]['slip_inapplicable_reason_counts']=={'no_anchor':2}
    assert rows[1]['slip_applicable_duration_s']==.02
    assert rows[1]['slip_applicable_missing_samples']==2
    assert not report['conditional_telemetry_complete']
    assert report['trial_rows'][0]['max_slip_m'] is None


def test_conditional_enum_uses_prior_anchor_and_independent_applicability():
    c=api().EvaluationTelemetryCollector(manifest(),checkpoint_sha256='a'*64,min_trials_per_style=1)
    raw=[]; c.attach_raw_sink(raw.append); c.begin_trials(); p=payload()
    p.update(slip_applicable=torch.tensor([False,True]),prior_anchor_latched=torch.tensor([False,True]),
        slip_reason_code=torch.tensor([1,4]),slip_reason_codes={'valid':0,'no_anchor':1,'no_contact':2,
            'measurement_invalid':3,'anchor_invalid':4,'inactive':5})
    p['done'][:]=True; c(p)
    assert raw[0]['slip_inapplicable_reason']=='no_anchor'
    assert raw[0]['prior_anchor_latched'] is False
    assert raw[1]['slip_inapplicable_reason'] is None
    assert raw[1]['slip_reason']=='anchor_invalid'
    rows=c.summary()['conditional_telemetry']['per_trial']
    assert rows[0]['slip_applicability_missing_samples']==0
    assert rows[1]['slip_applicable_missing_samples']==1


def test_valid_never_anchor_trials_are_conditionally_complete_without_zero_slip():
    c=api().EvaluationTelemetryCollector(manifest(),checkpoint_sha256='a'*64,min_trials_per_style=1)
    c.begin_trials(); p=payload()
    p.update(slip_applicable=torch.zeros(2,dtype=torch.bool),prior_anchor_latched=torch.zeros(2,dtype=torch.bool),
        slip_reason_code=torch.ones(2,dtype=torch.long),slip_reason_codes={'valid':0,'no_anchor':1},
        estimated_torque_nm=torch.zeros(2,12),effort_limit_nm=torch.ones(2,12),torque_valid=torch.ones(2,12,dtype=torch.bool))
    p['done'][:]=False; c(p); p['done'][:]=True; c(p)
    report=c.summary()
    assert report['conditional_telemetry_complete']
    assert not report['telemetry_complete'] and not report['validation_complete']
    assert report['aggregate']['max_slip_m'] is None


@pytest.mark.parametrize('missing',['partial','all','limit'])
def test_required_twelve_joint_torque_cannot_gain_full_coverage_from_partial_mask(missing):
    c=api().EvaluationTelemetryCollector(manifest(),checkpoint_sha256='a'*64,min_trials_per_style=1)
    raw=[]; c.attach_raw_sink(raw.append); c.begin_trials(); p=payload()
    if missing=='all':
        p.update(estimated_torque_nm=None,effort_limit_nm=None,torque_valid=None)
    elif missing=='limit':
        p['effort_limit_nm']=None
    else:
        p['torque_valid'][0,11]=False
    p['done'][:]=True; c(p)
    report=c.summary(); row=report['trial_rows'][0]
    assert row['torque_valid_coverage']==0, 'partial 11/12 torques were treated as a full physical sample'
    assert row['torque_valid_samples']==0 and row['invalid_measurement_samples']==1
    assert not report['conditional_telemetry_complete']
    assert raw[0]['sample']['measurement_valid'] is False
    assert raw[0]['sample'].get('estimated_torque_nm') is None
    diagnostic=raw[0]['torque_diagnostics']
    assert diagnostic['reason'] in ('required_torque_unavailable','partial_or_invalid_required_torque')
    if missing!='all':
        assert diagnostic['original_shape']==[12]
        assert len(diagnostic['estimated_torque_nm'])==12
        assert diagnostic['valid_mask']==([True]*11+[False] if missing=='partial' else [True]*12)


def test_certificate_guard_independently_requires_production_minimum_per_grasp():
    m=api()
    assert hasattr(m,'_production_manifest_eligible'), 'native certificate lacks independent production minimum guard'
    assert not m._production_manifest_eligible(manifest(),min_trials_per_style=1)
    assert not m._production_manifest_eligible(manifest(100),min_trials_per_style=1)
    assert m._production_manifest_eligible(manifest(100),min_trials_per_style=100)
    assert m._production_manifest_eligible(manifest(100,'validation'),min_trials_per_style=100)


@pytest.mark.parametrize('terminal,measurement_valid',[(False,True),(True,True),(True,False)])
def test_bounded_runner_uses_actual_reset_and_never_certifies_test_camera(tmp_path,monkeypatch,terminal,measurement_valid):
    source,saved=source_checkpoint(tmp_path)
    _,_,_,args=fixture(tmp_path)
    dw=world_stub(); task=SimpleNamespace(n=2,_seed={'command':torch.zeros(12)})
    import go2_pvcnn.control.dh116.tabletop_vision_evaluation_reset as resets
    for name in resets.TASK_RESET_FIELDS: setattr(task,name,torch.zeros(2,12))
    class Env:
        def enable_evaluation_telemetry(self,callback): self.callback=callback
        def reset(self):
            dw.q=torch.stack(dw.seed_q,1); dw.qd=torch.stack(dw.seed_qd,1)
            dw.roots=dw.root_seed.clone(); dw.roots[...,2]+=dw.initial_offset
            dw.wp=dw.seed_wire[:,:3]; dw.wq=dw.seed_wire[:,3:]; dw.wv=dw.seed_velocity
            return torch.zeros(2,371)
        def step(self,action):
            p=payload(); p['done'][:]=terminal; p['action']=action
            p['measurement_valid'][:]=measurement_valid
            self.callback(p)
            return torch.zeros(2,371),torch.zeros(2),p['done'],{}
    value=reset_manifest(); value['metadata']['camera_sha256']=file_sha256(args.depth_camera_config)
    value['metadata']['physics_sha256']=_digest(saved['contract']['privileged_critic_contract'])
    manifest_path=tmp_path/'frozen.json'; manifest_path.write_text(json.dumps(value))
    args.vision_evaluation_checkpoint=source; args.vision_evaluation_manifest=manifest_path
    args.vision_evaluation_max_control_steps=3; args._vision_new_output=str(args.output.resolve())
    import go2_pvcnn.control.dh116.tabletop_vision_audit as audit
    monkeypatch.setattr(audit,'audit_initial_points',lambda *a,**kw:{'passed':True,'environments':[]})
    report=api().run_vision_evaluation(SimpleNamespace(),dw,task,Env(),args,
        saved['contract']['privileged_critic_contract'],camera=Camera(2),min_trials_per_style=1)
    assert report['completed_count']==(2 if terminal else 0)
    assert report['status']==('completed' if terminal and measurement_valid else 'not_completed')
    assert report['control_steps']==(1 if terminal else 3)
    assert not report['native_validation_complete']
    assert not (args.output/'vision_evaluation_certificate.json').exists()
    raw=json.loads((args.output/'evaluation_reset_readback.json').read_text())
    assert raw['unique_readback_count']==2
    actor=torch.load(args.output/'evaluation_initial_visual_actor_inputs.pt',weights_only=False)
    assert actor['policy'].shape==(2,3844) and not actor['previous_action'].any()
    # First history frame contains only post-reset sensor values.
    assert not actor['history'][:,:3].any()


@pytest.mark.parametrize('replacement_time',['after_prepare','during_load'])
def test_runner_binds_prepared_checkpoint_and_loaded_actor_before_reset_or_evidence(tmp_path,monkeypatch,replacement_time):
    source,saved=source_checkpoint(tmp_path)
    _,_,_,args=fixture(tmp_path)
    args.num_envs=200
    # Retain the source bounded-camera processing size under the explicit
    # evaluation-only batch projection. This test never constructs a camera.
    args.vision_camera_batch_size=2
    args.vision_evaluation_checkpoint=source
    args.vision_evaluation_max_control_steps=3
    args._vision_new_output=str(args.output.resolve())
    frozen=manifest(100)
    frozen['metadata']['camera_sha256']=file_sha256(args.depth_camera_config)
    frozen['metadata']['physics_sha256']=_digest(saved['contract']['privileged_critic_contract'])
    args.vision_evaluation_manifest=tmp_path/'frozen.json'
    args.vision_evaluation_manifest.write_text(json.dumps(frozen))
    class Env:
        reset_calls=0
        def enable_evaluation_telemetry(self,callback): self.callback=callback
        def reset(self):
            self.reset_calls+=1
            raise RuntimeError('reset reached before checkpoint binding validation')
    class NeverSampleCamera:
        sample_calls=0
        def sample(self):
            self.sample_calls+=1
            raise RuntimeError('camera reached before checkpoint binding validation')
    env=Env(); camera=NeverSampleCamera(); dw=SimpleNamespace(n=200)
    module=api()
    collector=module.prepare_vision_evaluation(dw,None,env,args)
    original_sha=file_sha256(source)
    assert collector.summary()['checkpoint_sha256']==original_sha
    # Both versions have the same complete contract. Only actual actor weights
    # differ, so a contract check alone cannot identify this replacement.
    saved['policy']['actor.action_net.4.bias'][0]+=.25
    if replacement_time=='after_prepare':
        torch.save(saved,source)
    else:
        import go2_pvcnn.control.dh116.tabletop_vision_inference as inference
        real_load=inference.load_vision_inference
        def replace_then_load(*a,**kw):
            torch.save(saved,source)
            return real_load(*a,**kw)
        monkeypatch.setattr(inference,'load_vision_inference',replace_then_load)
    with pytest.raises(ValueError,match='evaluation checkpoint changed'):
        module.run_vision_evaluation(SimpleNamespace(),dw,None,env,args,
            saved['contract']['privileged_critic_contract'],collector=collector,camera=camera)
    assert file_sha256(source)!=original_sha
    assert env.reset_calls==0 and camera.sample_calls==0
    assert collector.summary()['completed_count']==0 and not collector._begun
    assert list(args.output.iterdir())==[]
