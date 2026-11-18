"""Cold-reset cohort execution fixtures exercise the original buffer kernels."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_vision_evaluation_reset import world_stub,templates
from go2_pvcnn.control.dh116.tabletop_vision_evaluation_reset import generate_cold_reset_manifest,TASK_RESET_FIELDS
from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_physics_sha256


ROOT=Path(__file__).resolve().parents[1]


def api():
    name='go2_pvcnn.control.dh116.tabletop_privileged_evaluation'
    assert importlib.util.find_spec(name),'independent cold-trial evaluation wiring is missing'
    return importlib.import_module(name)


def base_contract():
    return dict(num_envs=2,physics_hz=480,control_hz=60,close_seed_sha256='d'*64,
        grasp_initialization_sha256='e'*64,success_bottom_m=[.338,.338],
        surface_force_sharing={'lift_reference_bottom_m':[.308,.308]},
        object_contact_material={'static_friction':1.,'dynamic_friction':1.})


def manifest(per_style=2,*,validation=False):
    value=generate_cold_reset_manifest(templates(),seed=116,split='validation' if validation else 'test',
        provenance=dict(source_sha256='2'*64,physics_sha256=evaluation_physics_sha256(base_contract()),camera_sha256='4'*64))
    if per_style!=100:
        value['trials']=[row for style in ('pinch_thumb_index','tripod_thumb_index_middle')
            for row in [r for r in value['trials'] if r['grasp_style']==style][:per_style]]
    return value


class FrozenPolicy(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.full((24,),.1));self.calls=[]
    def deterministic(self,value):
        assert isinstance(value,torch.Tensor) and value.shape==(2,371)
        self.calls.append(value.detach().clone());return self.weight[None].expand(2,-1).tanh()


class FixtureBridge:
    evidence_scope='cpu_fixture'
    def __init__(self):
        self.n=2;self.device=torch.device('cpu');self.device_world=world_stub();self.contract=base_contract()
        self.task=SimpleNamespace(n=2,_seed={})
        for name in TASK_RESET_FIELDS:setattr(self.task,name,torch.zeros(2,12))
        self.checkpoint_sha256='2'*64;self.policy_tick=0;self.serial=0
        self.uid=['unbound0','unbound1'];self.identities=SimpleNamespace(uid=lambda i:self.uid[i])
        self._terminal_rows={};self.events=[];self.actions=[];self._all_bound_ids=[]
        self.critic=torch.zeros(2,371);self._physics_steps=0
        self.world=SimpleNamespace(sim=SimpleNamespace(current_time=0.))
    def bind_cold_trials(self,trials):
        self.bound=copy.deepcopy(trials);self._all_bound_ids.extend(row['trial_id'] for row in trials)
        self.events.append('bind')
    def reset(self):
        self.serial+=1;self.uid=[f'fixture:{self.serial}:{i}' for i in range(2)]
        self.start_uid=self.uid.copy();self.batch_step=0;self.events.append('reset')
        w=self.device_world
        w.q=torch.stack(w.seed_q,1).clone();w.qd=torch.stack(w.seed_qd,1).clone()
        w.roots=w.root_seed.clone();w.roots[...,2]+=w.initial_offset
        w.wp=w.seed_wire[:,:3].clone();w.wq=w.seed_wire[:,3:].clone();w.wv=w.seed_velocity.clone()
        self.critic=torch.zeros(2,371);self.critic[:,:12]=w.q[:,:,:6].flatten(1)
        return self.critic
    def set_action_owners(self,owners):self.owners=owners.copy()
    def set_execution_context(self,**kwargs):self.execution=kwargs
    def step(self,actions):
        self.actions.append(actions.clone());self.batch_step+=1;self.policy_tick+=1
        self._physics_steps+=8;self.world.sim.current_time+=1/60
        done=torch.tensor([self.batch_step==1,self.batch_step==2])
        for i in done.nonzero().flatten().tolist():
            self._terminal_rows[self.start_uid[i]]=dict(episode_uid=self.start_uid[i],
                trial_id=self.bound[i]['trial_id'],terminal=True,native_success=False,
                physics_step=self._physics_steps,source_scope='cpu_fixture')
            self.uid[i]=f'fixture:auto:{self.serial}:{i}'
        self.critic=self.critic+.001
        return self.critic,torch.zeros(2),done,dict(success=torch.zeros(2,dtype=torch.bool))


def test_required_trial_evaluation_module_has_explicit_red():api()


def test_full_manifest_200_cases_splits_into_small_balanced_batches_without_test_retag():
    m=api();value=manifest(100,validation=True)
    plan=m.prepare_cold_trial_batches(value,8,minimum_per_style=100)
    assert len(plan['batches'])==25 and plan['manifest']['metadata']['split']=='validation'
    assert all([r['grasp_style'] for r in chunk]==['pinch_thumb_index']*4+['tripod_thumb_index_middle']*4
               for chunk in plan['batches'])
    assert len({r['trial_id'] for chunk in plan['batches'] for r in chunk})==200


def test_small_batches_use_original_seed_writer_and_actual_readback_then_first_terminal(tmp_path):
    m=api();bridge=FixtureBridge();policy=FrozenPolicy();before=policy.weight.detach().clone()
    source=copy.deepcopy(bridge.contract)
    result=m.evaluate_cold_trials(bridge,policy,manifest(),tmp_path/'run',minimum_per_style=1,
        schema_only=True,max_control_steps=3)
    assert result['completed_trials']==4 and len(result['trials'])==4 and result['batch_count']==2
    assert len({r['first_episode_uid'] for r in result['trials']})==4
    assert result['native_qualified'] is False and result['native_execution_performed'] is False
    assert all(row['actual_readback_state_sha256'] for row in result['trials'])
    assert len(set(bridge._all_bound_ids))==4
    assert bridge.contract==source and bridge.policy_tick==4
    assert bridge.actions[1][0].eq(0.).all() # completed slot owns diagnostic zero until other first terminal
    torch.testing.assert_close(policy.weight,before);assert policy.weight.grad is None


def test_timeout_preserves_incomplete_first_episode_instead_of_success(tmp_path):
    result=api().evaluate_cold_trials(FixtureBridge(),FrozenPolicy(),manifest(1),tmp_path/'run',
        minimum_per_style=1,schema_only=True,max_control_steps=1)
    assert result['completed_trials']==1 and result['incomplete_trials']==1
    assert result['native_successes'] is None


def test_physics_or_policy_source_mismatch_rejected_before_reset_and_output(tmp_path):
    m=api();bridge=FixtureBridge();value=manifest(1);value['metadata']['physics_sha256']='0'*64
    with pytest.raises(ValueError,match='physics|physical'):
        m.evaluate_cold_trials(bridge,FrozenPolicy(),value,tmp_path/'bad',minimum_per_style=1,schema_only=True)
    assert not bridge.events and not (tmp_path/'bad').exists()
    value=manifest(1);value['metadata']['source_sha256']='0'*64
    with pytest.raises(ValueError,match='checkpoint|source'):
        m.evaluate_cold_trials(bridge,FrozenPolicy(),value,tmp_path/'bad2',minimum_per_style=1,schema_only=True)


def test_bad_late_trial_limits_are_preflighted_before_any_real_buffer_mutation(tmp_path):
    m=api();bridge=FixtureBridge();value=manifest(2)
    bridge.device_world.limits=bridge.device_world.limits.clone();bridge.device_world.limits[0,0,:,1]=.19
    before=bridge.device_world.seed_q[0].clone()
    with pytest.raises(ValueError,match='limits'):
        m.evaluate_cold_trials(bridge,FrozenPolicy(),value,tmp_path/'bad',minimum_per_style=1,schema_only=True)
    assert not bridge.events and torch.equal(before,bridge.device_world.seed_q[0])


def test_formal_minimum_and_existing_output_are_not_silently_relaxed(tmp_path):
    m=api();bridge=FixtureBridge();target=tmp_path/'existing';target.mkdir()
    with pytest.raises(ValueError,match='100|native'):
        m.evaluate_cold_trials(bridge,FrozenPolicy(),manifest(1),tmp_path/'small',minimum_per_style=1)
    with pytest.raises(FileExistsError):
        m.evaluate_cold_trials(bridge,FrozenPolicy(),manifest(1),target,minimum_per_style=1,schema_only=True)
    assert not bridge.events


def test_student_receives_only_sensor_history_and_callback_clears_after_reset(tmp_path):
    from test_dh116_stereo_estimator import sensor
    from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistory
    from go2_pvcnn.control.dh116.tabletop_stereo_teacher_collector import PreActionState,StepResult
    bridge=FixtureBridge();seen=[]
    class Student:
        def deterministic(self,observation):
            from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistoryBatch
            assert type(observation) is StereoHistoryBatch;seen.append(float(observation.left_rgb[:,-1].mean()))
            return torch.zeros(2,24)
    cleared=[];state={}
    def frame(value,action=None,done=None):
        current=sensor();current.left_rgb.fill_(value);current.right_rgb.fill_(value)
        if action is not None:current.proprio[:,38:]=action
        if done is not None:current.proprio[done,38:]=0.
        state['history']=state['store'].update(current,reset_mask=done)
    def clear(b):
        cleared.append(b.serial);state['store']=StereoHistory(2);frame(0.)
    def step(action):
        obs,reward,done,extra=bridge.step(action);frame(bridge.batch_step/10,action,done)
        ids=[dict(run_uuid='sensor_fixture',env_id=i,episode_uid=bridge.uid[i],reset_id=bridge.serial,
            policy_tick=bridge.policy_tick,physics_tick_start=bridge._physics_steps,physics_tick_end=bridge._physics_steps,
            simulation_timestamp=bridge.world.sim.current_time,frame_id=bridge.policy_tick,calibration_sha256='4'*64) for i in range(2)]
        after=PreActionState(state['history'].clone(),obs.clone(),torch.zeros(2,371,dtype=torch.bool),ids)
        return StepResult(after,reward,done,extra,action.clone())
    result=api().evaluate_cold_trials(bridge,Student(),manifest(1),tmp_path/'student',
        owner='student',minimum_per_style=1,schema_only=True,max_control_steps=3,
        observation_callback=lambda:state['history'].clone(),on_batch_reset=clear,step_callback=step)
    assert seen==pytest.approx([0.,.1]) and cleared==[1] and result['owner']=='student'
    assert result['teacher_executed'] is False and result['critic_deployed'] is False


def test_student_privileged_callback_is_rejected(tmp_path):
    bridge=FixtureBridge()
    with pytest.raises((TypeError,ValueError),match='sensor|history|StereoHistory'):
        api().evaluate_cold_trials(bridge,FrozenPolicy(),manifest(1),tmp_path/'poisoned',
            owner='student',minimum_per_style=1,schema_only=True,
            observation_callback=lambda:torch.ones(2,371),step_callback=bridge.step)


def test_cli_defaults_to_no_native_config_check_without_output(tmp_path):
    script=ROOT/'scripts/evaluate_dh116_stereo_trials.py'
    assert script.exists(),'cold-trial evaluation CLI missing'
    result=subprocess.run([sys.executable,str(script),'--check-config'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    payload=json.loads(result.stdout)
    assert payload['native_execution_performed'] is False and payload['minimum_per_style']>=100


def test_policy_derivation_preserves_entire_physical_cohort_and_binds_both_source_files(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    old=tmp_path/'teacher.bin';new=tmp_path/'student.bin';old.write_bytes(b'CPU source fixture');new.write_bytes(b'CPU target fixture')
    value=manifest(1);value['metadata']['source_sha256']=file_sha256(old)
    source=tmp_path/'source_manifest.json';source.write_text(json.dumps(value))
    before={p:file_sha256(p) for p in (source,old,new)};target=tmp_path/'derived.json'
    result=api().derive_policy_trial_manifest(source,old,new,target,minimum_per_style=1,schema_only=True)
    derived=json.loads(target.read_text());proof=json.loads(target.with_suffix('.provenance.json').read_text())
    assert derived['trials']==value['trials'] and derived['metadata']['physics_sha256']==value['metadata']['physics_sha256']
    assert derived['metadata']['source_sha256']==file_sha256(new)
    assert proof['source_policy_sha256']==file_sha256(old) and proof['target_policy_sha256']==file_sha256(new)
    assert proof['checkpoint_contract_rewritten'] is False and result['native_qualified'] is False
    assert {p:file_sha256(p) for p in before}==before
    with pytest.raises(FileExistsError):api().derive_policy_trial_manifest(source,old,new,target,minimum_per_style=1,schema_only=True)


def test_teacher_manifest_freezing_binds_original_base_and_two_compatible_template_artifacts(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    from go2_pvcnn.control.dh116.tabletop_surface_contract import semantic_sha256
    from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import VERSION
    from go2_pvcnn.control.dh116.tabletop_evaluation_trials import validate_manifest
    initials=[]
    for i,state in enumerate(templates().values()):
        path=tmp_path/f'init{i}.json'
        path.write_text(json.dumps(dict(format='dh116-adaptive-grasp-initialization-v1',
            source_contract={'physics_hz':480,'control_hz':60},roots=state['roots'],joint_positions=state['joint_positions'],
            grasp_command=[.3]*12,wire_pose=state['wire_pose'])))
        initials.append(path)
    base=base_contract();base['grasp_initialization_sha256']=file_sha256(initials[0])
    checkpoint=tmp_path/'source.pt'
    # CPU provenance fixture only. Actual policy loading remains the existing
    # strict checkpoint loader and is not blessed by authored manifest creation.
    torch.save(dict(version=VERSION,identity={'source_sha256':{'native_task_contract':semantic_sha256(base)}}),checkpoint)
    original=file_sha256(checkpoint);out=tmp_path/'cohort.json'
    api().freeze_teacher_cold_manifest(checkpoint,base,*initials,out,seed=116,wrist_joint_names=())
    value=validate_manifest(json.loads(out.read_text()))
    assert len(value['trials'])==200 and value['metadata']['source_sha256']==original
    assert file_sha256(checkpoint)==original
    changed=copy.deepcopy(base);changed['object_contact_material']['dynamic_friction']=.9
    with pytest.raises(ValueError,match='original native task contract'):
        api().freeze_teacher_cold_manifest(checkpoint,changed,*initials,tmp_path/'wrong.json',seed=116,wrist_joint_names=())


@pytest.mark.parametrize('failure', ('tuple', 'mutated_action', 'observation_advanced'))
def test_independent_student_requires_unchanged_issued_action_and_clock(tmp_path, failure):
    from test_dh116_stereo_estimator import sensor
    from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistory
    from go2_pvcnn.control.dh116.tabletop_stereo_teacher_collector import StepResult

    bridge = FixtureBridge()
    history = StereoHistory(2).update(sensor())

    class Student:
        def deterministic(self, observation):
            return torch.zeros(2, 24)

    def observe():
        if failure == 'observation_advanced':
            bridge._physics_steps += 1
        return history.clone()

    def step(action):
        if failure == 'tuple':
            return bridge.step(action)
        if failure == 'mutated_action':
            action[0, 7] = .5
        _, reward, done, info = bridge.step(action)
        return StepResult(None, reward, done, info, action)

    match = {'tuple':'StepResult', 'mutated_action':'issued action', 'observation_advanced':'advanced'}[failure]
    with pytest.raises((RuntimeError, TypeError, ValueError), match=match):
        api().evaluate_cold_trials(bridge, Student(), manifest(1), tmp_path/'student',
            owner='student', minimum_per_style=1, schema_only=True, max_control_steps=2,
            observation_callback=observe, step_callback=step)
