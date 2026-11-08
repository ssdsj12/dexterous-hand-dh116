"""Synthetic CPU observer parity; fixtures are never native evaluation trials."""
import gzip
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

MODULE='go2_pvcnn.control.dh116.tabletop_vision_evaluation_diagnostics'


def api():
    return importlib.import_module(MODULE)


def fixture(tmp_path):
    n=2
    names=['wrist_x','wrist_y','wrist_z','finger11','finger12']
    class View:
        def get_dof_position_targets(self):return np.full((n,5),.25,dtype=np.float32)
        def get_dof_types(self):return np.broadcast_to(np.array([1,1,1,0,0],dtype=np.uint8),(n,5)).copy()
    hands=[SimpleNamespace(joint_names=names.copy(),root_physx_view=View(),
        data=SimpleNamespace(joint_pos_target=torch.full((n,5),.9))) for _ in range(2)]
    world=SimpleNamespace(n=n,device='cpu',hands=hands,roots=torch.zeros(n,2,7),
        q=torch.full((n,2,5),.2),qd=torch.full((n,2,5),.01),
        root_targets=torch.zeros(n,2,7),commands=torch.full((n,2,6),.21),
        pad_measurements=dict(effective_pad_loads_n=torch.ones(n,2,5),
            effective_pad_valid=torch.ones(n,2,5,dtype=torch.bool)))
    env=SimpleNamespace(physics_action_diagnostics=None,_diagnostic_policy_step=17,
        diagnostic_episode_ids=torch.tensor([3,9]),task=SimpleNamespace(success=torch.zeros(n,dtype=torch.bool)))
    manifest={'trials':[{'trial_id':f'case{i}','grasp_style':'pinch_thumb_index'} for i in range(n)]}
    return env,world,manifest


def audit_payload(n=2):
    return dict(raw_action24=np.zeros((n,24)),requested_palm_target_m=np.ones((n,2,3))*.01,
        pre_projection_palm_target_m=np.ones((n,2,3))*.008,
        projected_palm_target_m=np.ones((n,2,3))*.006,
        final_sdk_palm_target_m=np.ones((n,2,3))*.005,
        actual_palm_pose_m=np.ones((n,2,3))*.004,
        actual_palm_velocity_m_s=np.ones((n,2,3))*.003,
        table_clearance_m=np.ones((n,2))*.002,safe_z_m=np.full((n,2),np.nan),
        projection_reasons=np.zeros((n,2)),contact_loads_n=np.ones((n,2)),
        contact_valid=np.ones((n,2),dtype=bool),failure_code=np.array([3,0]))


def record(writer,step=100,*,live=(True,True),terminal=(False,False),episode=(3,9),payload=None,capture=True):
    method=getattr(writer,'capture_measurement_sources',None)
    if capture and method is not None:method(getattr(writer.env.task,'m',{}))
    writer.record_batch(np.arange(2),np.array(episode),writer.policy_step,
        np.full(2,step),np.full(2,step/480),np.array(live),np.array(terminal),
        audit_payload() if payload is None else payload)


def chunks(output):
    return [dict(np.load(path,allow_pickle=False)) for path in sorted((output/'evaluation_action_diagnostics').glob('physics_*.npz'))]


def test_disabled_does_not_inspect_or_install_or_create(tmp_path):
    class Unreadable:
        def __getattr__(self,name):raise AssertionError(name)
    assert api().install_evaluation_physics_diagnostics(Unreadable(),Unreadable(),None,tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_records_raw_request_backend_cache_missing_sources_and_terminal_before_noop(tmp_path):
    env,world,manifest=fixture(tmp_path)
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True,chunk_batches=2)
    action=torch.linspace(-.4,.4,48).reshape(2,24)
    writer.submit(action,policy_step=4)
    original=action.clone();record(writer,terminal=(True,False))
    world.q.fill_(99);record(writer,101,live=(False,True),episode=(4,9))
    result=writer.finish()
    assert env.physics_action_diagnostics is None and env._diagnostic_policy_step==17
    assert torch.equal(action,original)
    samples=chunks(tmp_path)[0]
    np.testing.assert_array_equal(samples['actor_request24'][0],original.numpy())
    np.testing.assert_allclose(samples['left_backend_target'][0],.25)
    np.testing.assert_allclose(samples['left_command_cache'][0],.9)
    assert samples['left_backend_target__available'].all()
    assert samples['safe_z_m__available'].all() and not samples['safe_z_m__finite'].any()
    assert samples['contact_loads_n__available'].all() and not samples['contact_loads_n__source_valid'].any()
    assert not samples['object_pose_w__available'].any()
    assert np.isnan(samples['object_pose_w']).all()
    assert samples['episode_uid'][0,0]!=samples['episode_uid'][1,0]
    terminal=[json.loads(line) for line in gzip.open(tmp_path/'evaluation_action_diagnostics/terminals.jsonl.gz','rt')]
    assert len(terminal)==1 and terminal[0]['failure_code']==3
    assert terminal[0]['values']['left_q'][0]==pytest.approx(.2)
    assert result['terminal_count']==1 and result['native_trials_certified_by_writer']==0
    assert result['new_independent_initial_states']==0


def test_backend_error_never_falls_back_to_command_cache(tmp_path):
    env,world,manifest=fixture(tmp_path)
    def fail():raise RuntimeError('backend test unavailable')
    world.hands[1].root_physx_view.get_dof_position_targets=fail
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    writer.submit(torch.zeros(2,24),policy_step=0);record(writer);writer.finish()
    samples=chunks(tmp_path)[0]
    assert not samples['right_backend_target__available'].any()
    assert np.isnan(samples['right_backend_target']).all()
    assert samples['right_command_cache__available'].all()


def test_original_measurement_nonfinite_cannot_gain_source_validity_from_finite_audit_fallback(tmp_path):
    env,world,manifest=fixture(tmp_path)
    env.task.m={'hand_object_loads':torch.tensor([[float('nan'),1.],[1.,1.]])}
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    writer.submit(torch.zeros(2,24),policy_step=0);record(writer);writer.finish()
    data=chunks(tmp_path)[0]
    assert data['contact_loads_n__finite'].all()
    assert data['contact_loads_n__source_valid'][0].tolist()==[False,True]


def test_real_zero_load_and_false_contact_are_observed_sources_not_missing_measurements(tmp_path):
    env,world,manifest=fixture(tmp_path)
    env.task.m={'hand_object_loads':torch.zeros(2,2),'contact_valid':torch.zeros(2,2,dtype=torch.bool)}
    payload=audit_payload();payload['contact_loads_n'][:]=0.;payload['contact_valid'][:]=False
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    writer.submit(torch.zeros(2,24),policy_step=0);record(writer,payload=payload);writer.finish()
    data=chunks(tmp_path)[0]
    assert data['contact_valid__source_valid'].all() and not data['contact_valid'].any()
    assert data['contact_loads_n__source_valid'].all() and not data['contact_loads_n'].any()


def test_raw_measurement_source_is_consumed_once_and_cannot_leak_into_next_substep(tmp_path):
    env,world,manifest=fixture(tmp_path)
    env.task.m={'wrist_velocity':torch.zeros(2,2,6)}
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    writer.submit(torch.zeros(2,24),policy_step=0)
    record(writer)
    record(writer,101,capture=False)
    writer.finish()
    data=chunks(tmp_path)[0]
    assert data['actual_palm_velocity_m_s__source_valid'][0].all()
    assert not data['actual_palm_velocity_m_s__source_valid'][1].any()


def test_global_reset_uid_is_unique_without_overwriting_first_terminal(tmp_path):
    env,world,manifest=fixture(tmp_path)
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    writer.submit(torch.zeros(2,24),policy_step=0);record(writer,terminal=(True,False))
    writer.begin_reset()
    env.diagnostic_episode_ids[:]=0
    writer.submit(torch.zeros(2,24),policy_step=0)
    record(writer,1,terminal=(True,False),episode=(0,0));writer.finish()
    samples=chunks(tmp_path)[0]
    assert samples['episode_uid'][0,0]!=samples['episode_uid'][1,0]
    assert samples['reset_epoch'][:,0].tolist()==[0,1]
    assert not samples['first_trial_eligible'][1].any()
    terminals=[json.loads(line) for line in gzip.open(tmp_path/'evaluation_action_diagnostics/terminals.jsonl.gz','rt')]
    assert len(terminals)==2 and terminals[0]['episode_uid']!=terminals[1]['episode_uid']


@pytest.mark.parametrize('clock_available',[False,True])
def test_policy_requests_have_explicit_env_episode_trial_identity_and_true_or_na_clock(tmp_path,clock_available):
    env,world,manifest=fixture(tmp_path)
    if clock_available:
        world._physics_steps=321
        world.w=SimpleNamespace(sim=SimpleNamespace(current_time=.6))
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    writer.submit(torch.zeros(2,24),policy_step=4);record(writer);writer.finish()
    row=json.loads(next(iter(gzip.open(tmp_path/'evaluation_action_diagnostics/policy_requests.jsonl.gz','rt'))))
    data=chunks(tmp_path)[0]
    assert row['env_ids']==[0,1] and row['episode_ids']==[3,9]
    assert row['episode_uids']==data['episode_uid'][0].tolist()
    assert row['trial_ids']==['case0','case1'] and row['policy_step']==4
    assert row['submit_physics_step_available'] is clock_available
    assert row['submit_timestamp_available'] is clock_available
    assert row['physics_step_before_submit']==(321 if clock_available else None)
    assert row['timestamp_s_before_submit']==(.6 if clock_available else None)


def test_stale_clock_and_policy_action_shape_fail_closed(tmp_path):
    env,world,manifest=fixture(tmp_path)
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    with pytest.raises(ValueError):writer.submit(torch.zeros(2,23),policy_step=0)
    writer.submit(torch.zeros(2,24),policy_step=0);record(writer)
    with pytest.raises(ValueError,match='clock'):record(writer,99)
    writer.finish(error='synthetic_expected_clock_failure')


def test_finish_restores_prior_writer_and_closes_streams_even_on_chunk_write_failure(tmp_path,monkeypatch):
    env,world,manifest=fixture(tmp_path)
    writer=api().install_evaluation_physics_diagnostics(env,world,manifest,tmp_path,enabled=True)
    writer.submit(torch.zeros(2,24),policy_step=0);record(writer)
    def failure(*args,**kwargs):raise OSError('synthetic disk failure')
    monkeypatch.setattr(np,'savez_compressed',failure)
    with pytest.raises(OSError,match='synthetic disk'):writer.finish()
    assert env.physics_action_diagnostics is None and env._diagnostic_policy_step==17
    assert writer.policy_stream.closed and writer.terminal_stream.closed
    assert writer.finish() is None


def test_optin_requires_independent_student_evaluation():
    validate=api().validate_evaluation_physics_diagnostics_options
    assert not validate(SimpleNamespace())
    base=dict(vision_evaluation_physics_audit=True,vision_evaluation_checkpoint='new.pt',vision_student=True)
    assert validate(SimpleNamespace(**base))
    for name,value in (('paired_bias_arm','m0'),('vision_teacher_source','teacher.pt'),('vision_m2_replay','replay.pt')):
        with pytest.raises(ValueError):validate(SimpleNamespace(**base,**{name:value}))


def test_actual_tensor_env_transition_reward_and_reset_parity_with_real_terminal(tmp_path):
    from test_dh116_tensor_env import World
    from test_dh116_tensor_task import template
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
    outputs=[]
    for enabled in (False,True):
        task=TensorTask(template(),2,'cpu');world=World(task)
        # Real TensorTask intentionally does not retain this raw measurement
        # key. Source validity must follow this exact physical packet instead.
        world.m['wrist_velocity']=torch.zeros(2,2,6)
        world._physics_steps=0
        world.w=SimpleNamespace(sim=SimpleNamespace(current_time=0.))
        advance=world.advance
        def measured_advance(t,allowed,live):
            m=advance(t,allowed,live);world._physics_steps+=1;world.w.sim.current_time+=world.dt
            world.last_action_diagnostics=dict(audit_payload(),raw_action24=t.last_raw_action24.clone(),
                physics_step=world._physics_steps,timestamp_s=world.w.sim.current_time)
            return m
        world.advance=measured_advance
        env=TensorEnv(world,task)
        output=tmp_path/str(enabled);output.mkdir()
        manifest={'trials':[{'trial_id':str(i),'grasp_style':'pinch_thumb_index'} for i in range(2)]}
        writer=api().install_evaluation_physics_diagnostics(env,world,manifest,output,enabled=enabled)
        action=torch.zeros(2,24);action[:,:12]=torch.linspace(-.1,.1,12)
        if writer:writer.submit(action,policy_step=0)
        returned=env.step(action[:,:20])
        if writer:writer.finish()
        assert 'wrist_velocity' not in task.m
        if writer:
            data=chunks(output)[0]
            live=data['live_before_substep'].astype(bool)
            assert data['actual_palm_velocity_m_s__source_valid'][live].all()
        outputs.append((returned,task.command.clone(),world.q.clone(),world.qd.clone(),world.resets))
    for before,after in zip(outputs[0][:4],outputs[1][:4]):
        if isinstance(before,tuple):
            for a,b in zip(before[:3],after[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        else:torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert outputs[0][4]==outputs[1][4]==[[0]]
    terminal=[json.loads(line) for line in gzip.open(tmp_path/'True/evaluation_action_diagnostics/terminals.jsonl.gz','rt')]
    assert terminal[0]['physics_step']==3 and terminal[0]['failure_code']==6


def test_evaluation_runner_submits_actual_policy_before_step_and_finishes(tmp_path,monkeypatch):
    import test_vision_evaluation as existing
    from go2_pvcnn.control.dh116 import tabletop_vision_evaluation_diagnostics as diagnostics
    original_fixture=existing.fixture
    calls=[]
    def opted_fixture(*args,**kwargs):
        result=original_fixture(*args,**kwargs)
        result[3].vision_evaluation_physics_audit=True
        result[3].vision_student=True
        return result
    class Writer:
        paths=[]
        def submit(self,action,*,policy_step):calls.append(('submit',policy_step,action.clone()))
        def finish(self,*,error=None):
            calls.append(('finish',error))
            target=self.output/'evaluation_action_diagnostics';target.mkdir()
            (target/'summary.json').write_text(json.dumps({'status':'fixture_not_native'}))
            return {'status':'fixture_not_native'}
    def install(*args,**kwargs):
        calls.append(('install',kwargs['enabled']))
        result=Writer();result.output=Path(args[3]);return result
    monkeypatch.setattr(existing,'fixture',opted_fixture)
    monkeypatch.setattr(diagnostics,'install_evaluation_physics_diagnostics',install)
    original_runner=existing.api().run_vision_evaluation
    reports=[]
    def capture_report(*args,**kwargs):
        result=original_runner(*args,**kwargs);reports.append(result);return result
    monkeypatch.setattr(existing.api(),'run_vision_evaluation',capture_report)
    existing.test_bounded_runner_uses_actual_reset_and_never_certifies_test_camera(
        tmp_path,monkeypatch,True,True)
    assert calls[0]==('install',True)
    assert calls[1][0:2]==('submit',0) and calls[1][2].shape==(2,24)
    assert calls[-1]==('finish',None)
    assert 'optional_student_physics_diagnostics' in reports[0]
    assert set(reports[0]['evidence_files'])=={'raw_reset_readback','initial_camera_geometry',
        'initial_visual_actor_inputs','frozen_manifest','conditional_telemetry_samples'}
