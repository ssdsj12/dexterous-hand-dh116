"""Independent full-first-episode runner checks; fixtures are never native trials."""
import copy
import importlib
from types import SimpleNamespace as NS

import pytest
import torch


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_full_episode_tracking_experiment')


def test_only_audited_surface_profile_can_supply_known_no_mandatory_unload():
    from test_surface_exploration import surface_task
    t,w=surface_task();t.pinch_force_hold=False
    audit=module().audit_no_force_unload_profile(t,w)
    producer=module().no_force_unload_feedback(audit)
    t.actions.fill_(-1.)  # Negative policy actions are not guessed to be mandatory.
    masks=producer(w,t,torch.ones(t.n,dtype=torch.bool))
    assert not masks['mandatory_release_mask'].any() and not masks['mandatory_wrist_release_mask'].any()
    assert not masks['overload'].any() and audit['automatic_force_unload_branch'] is False
    t.pinch_force_hold=True
    with pytest.raises(ValueError,match='profile'):producer(w,t,torch.ones(t.n,dtype=torch.bool))


@pytest.mark.parametrize('bad',['owned','pinch','redistribution','callback'])
def test_unknown_priority_profile_is_rejected_before_installing_common_candidate(bad):
    from test_surface_exploration import surface_task
    t,w=surface_task();t.pinch_force_hold=False
    if bad=='owned':t.surface_policy_owned_targets=False
    elif bad=='pinch':t.pinch_force_hold=True
    elif bad=='redistribution':t.surface_redistribution=object()
    else:t.contact_effort=object()
    with pytest.raises(ValueError):module().audit_no_force_unload_profile(t,w)


def payload(n=8):
    return dict(q=torch.zeros(n,12,dtype=torch.float64),qd=torch.zeros(n,12,dtype=torch.float64),
        object_link_loads=torch.zeros(n,2,17,dtype=torch.float64),mimic_errors=torch.zeros(n,10,dtype=torch.float64),
        effective_pad_loads_n=torch.ones(n,2,5,dtype=torch.float64),bottom=torch.ones(n,dtype=torch.float64)*.34)


@pytest.mark.parametrize('arm',['original','unilateral','common'])
def test_recorder_freezes_all_eight_first_terminals_and_separates_hybrid_results(arm):
    rows=[];r=module().FullEpisodeTrackingRecorder(arm,native=True,sink=rows.append)
    ids=torch.arange(8,dtype=torch.long)+11;r.begin(ids)
    live=torch.ones(8,dtype=torch.bool);done=torch.ones(8,dtype=torch.bool)
    success=done.clone();failure=torch.zeros(8,dtype=torch.long)
    metrics=dict(net_lift_m=torch.full((8,),.031),hold_s=torch.full((8,),5.),drift_m=torch.zeros(8))
    guard=dict(pause_hand=torch.zeros(8,2,dtype=torch.bool),state=torch.zeros(8,dtype=torch.long))
    r.observe(physics_step=8,timestamp_s=8/480,live=live,done=done,success=success,failure=failure,
        measurement=payload(),guard_info=guard,metrics=metrics,raw={},episode_ids=ids)
    saved=copy.deepcopy(r.first_terminal)
    r.observe(physics_step=9,timestamp_s=9/480,live=live,done=done,success=torch.zeros(8,dtype=torch.bool),
        failure=torch.ones(8,dtype=torch.long)*6,measurement=payload(),guard_info=guard,metrics=metrics,raw={},episode_ids=ids+1)
    assert r.first_terminal==saved
    report=r.summary()
    assert report['completed_first_episodes']==8 and report['styles']['pinch']['completed']==4 and report['styles']['tripod']['completed']==4
    assert report['observed_original_native_success_count']==8
    assert report['pure_student_original_native_success_count']==(8 if arm=='original' else 0)
    assert report['privileged_hybrid_native_success_count']==(0 if arm=='original' else 8)
    assert report['production_evaluation_certificate_issued'] is False
    assert rows[0]['run_uuid']==rows[1]['run_uuid'] and rows[0]['episode_uids']==rows[1]['episode_uids']
    assert not torch.equal(rows[0]['diagnostic_episode_ids_at_sample'],rows[1]['diagnostic_episode_ids_at_sample'])
    assert r.first_terminal[0]['episode_uid']==rows[0]['episode_uids'][0]


def test_pause_timeout_is_not_fabricated_as_an_original_physical_terminal():
    r=module().FullEpisodeTrackingRecorder('common',native=False,sink=lambda row:None)
    masks=torch.zeros(8,dtype=torch.bool)
    for step in range(1000):
        r.observe(physics_step=step,timestamp_s=step/480,live=~masks,done=masks,success=masks,
            failure=torch.zeros(8,dtype=torch.long),measurement=payload(),
            guard_info=dict(pause_hand=torch.ones(8,2,dtype=torch.bool),state=torch.ones(8,dtype=torch.long)*4),
            metrics={},raw={})
    report=r.summary()
    assert report['completed_first_episodes']==0 and not r.closed.any()
    assert report['candidate_terminated_env_ids']==list(range(8))
    assert report['pause_over_2s_env_ids']==list(range(8))
    assert not report['all_first_episodes_completed']


def test_common_requires_qualified_physx_calibration_with_actual_02_source_hashes(tmp_path):
    import hashlib,json
    def identity(name):
        path=tmp_path/name;path.write_bytes(name.encode());return str(path),hashlib.sha256(path.read_bytes()).hexdigest()
    data,dsha=identity('actual_02_dataset.pt');context,csha=identity('actual_02_context.pt')
    source='a'*64;action='b'*64
    report=dict(version='dh116-teacher-tracking-calibration-v1',qualified=True,
        target_source='physx_backend_get_dof_position_targets',
        feedback_source='original_advance_returned_refresh_before_TensorEnv_terminal_reset',
        dataset_path=data,dataset_sha256=dsha,context_path=context,context_sha256=csha,
        controller_source_sha256=source,action_contract_sha256=action,
        teacher_error_p999_rad=[.002]*12,trigger_rad=[.01]*12)
    path=tmp_path/'calibration02.json';path.write_text(json.dumps(report));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    audit=dict(controller_source_sha256=source,action_contract_sha256=action)
    value=module().load_qualified_calibration(dict(path=str(path),sha256=sha,source_run='02'),audit)
    assert value['teacher_error_p999_rad']==[.002]*12
    report['qualified']=False;path.write_text(json.dumps(report));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='qualified'):
        module().load_qualified_calibration(dict(path=str(path),sha256=sha,source_run='02'),audit)


def runner_scene(tmp_path,arm,monkeypatch):
    import hashlib,json
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    from go2_pvcnn.control.dh116.tabletop_vision_command_shadow import controller_source_sha256,action_contract_sha256
    class World:
        n=8;dt=1/480;decimation=8;device='cpu'
        def advance(self,task,allowed,live):
            self.commands=torch.where(live[:,None,None],task.command.reshape(8,2,6),self.commands)
            for h,hand in enumerate(self.hands):
                hand.data.joint_pos_target=self.commands[:,h].float().clone()
                hand.root_physx_view.targets=hand.data.joint_pos_target.clone()
            self._physics_steps+=1;self.w.sim.current_time_step_index+=1;self.w.sim.current_time+=self.dt
            return payload()
    w=World();w._physics_steps=0;w.q=torch.zeros(8,2,6,dtype=torch.float64);w.qd=w.q.clone()
    w.roots=torch.zeros(8,2,7,dtype=torch.float64);w.roots[...,3]=1.;w.root_seed=w.roots.clone();w.root_targets=w.roots.clone()
    w.palms=torch.zeros(8,2,3,dtype=torch.float64);w.wrist_angles=w.palms.clone();w.commands=w.q.clone()
    w.wp=torch.zeros(8,3,dtype=torch.float64);w.wq=torch.zeros(8,4,dtype=torch.float64);w.wq[:,0]=1.
    w.origins=torch.zeros(8,3,dtype=torch.float64);w.mixed_reset_style_ids=torch.tensor([0]*4+[1]*4)
    native=NS(ids=[list(range(6))]*2,mounts=None,sim=NS(current_time_step_index=0,current_time=0.));w.w=native
    w.hands=[]
    for _ in range(2):
        hand=NS(joint_names=ACTIVE,data=NS(joint_pos_target=torch.zeros(8,6,dtype=torch.float32)),
            root_physx_view=NS(targets=torch.zeros(8,6,dtype=torch.float32)))
        hand.root_physx_view.get_dof_position_targets=lambda view=hand.root_physx_view:view.targets
        hand.root_physx_view.get_dof_types=lambda:torch.zeros(8,6,dtype=torch.int32)
        w.hands.append(hand)
    class Task:
        policy_control=True;free_wrist_grasp=True;surface_force_sharing=True
        surface_policy_owned_targets=True;surface_redistribution=False;pinch_force_hold=False
        episode_timeout_s=30.;n=8
        def control(self,action,live,dt):
            self.command[live]+=action[live,:12]*.3*dt
            self.palm_targets[live]=w.palms[live]+action[live,12:18].reshape(-1,2,3)*.01*dt
            self.rotation_targets[live]=w.wrist_angles[live]
            return live.clone()
    t=Task();t.command=torch.zeros(8,12,dtype=torch.float64);t.low=torch.full((12,),-1.,dtype=torch.float64);t.high=-t.low
    t.offset=torch.zeros(8,dtype=torch.float64);t.done=torch.zeros(8,dtype=torch.bool);t.success=t.done.clone();t.failure=torch.zeros(8,dtype=torch.long)
    t.palm_targets=w.palms.clone();t.rotation_targets=w.wrist_angles.clone();t.surface_reference_bottom_m=torch.zeros(8,dtype=torch.float64)
    t.m=payload();t.m['mimic_errors'].zero_()
    class Env:
        task=t;diagnostic_episode_ids=torch.zeros(8,dtype=torch.long)
        def reset(self):self.tick=0;return torch.zeros(8,371)
        def step(self,action):
            action=action.to(t.command)  # original TensorTask.begin_step conversion
            for substep in range(8):
                live=~t.done;physical=self.bimanual_action_adapter(action) if getattr(self,'bimanual_action_adapter',None) is not None else action
                allowed=t.control(physical,live,w.dt);m=w.advance(t,allowed,live);t.m=m
                if substep==7:
                    if self.tick==1:t.done[:4]=True;t.success[:4]=True
                    if self.tick==3:t.done[4:]=True;t.success[4:]=True
                self.joint_replay_observer(self,m,live,live&t.done)
            self.tick+=1
            return torch.zeros(8,371),torch.zeros(8),t.done.clone(),dict(success=t.success.clone())
    env=Env()
    class Camera:
        def sample(self):
            return dict(points=torch.zeros(8,2,128,3),point_valid=torch.ones(8,128,dtype=torch.bool),
                image_valid=torch.ones(8,dtype=torch.bool),object_valid=torch.ones(8,dtype=torch.bool))
    expected={key:value.clone() for key,value in zip(('q','qd','command','roots','object_position','object_quaternion','reference_bottom'),
        (w.q,w.qd,t.command,w.roots,w.wp,w.wq,t.surface_reference_bottom_m))}
    expected.update(seed=42,source='actual_native_cold_reset_before_camera')
    initial=tmp_path/'reset.pt';torch.save(expected,initial)
    nonce=0
    while True:
        raw=f'fixture-c58-{nonce}'.encode();sha=hashlib.sha256(raw).hexdigest()
        if sha.startswith('c58'):break
        nonce+=1
    checkpoint=tmp_path/'c58.pt';checkpoint.write_bytes(raw)
    monkeypatch.setattr(module(),'BASELINE_C58_SHA256',sha,raising=False)
    config=dict(version='dh116-full-episode-tracking-experiment-v1',arm=arm,
        checkpoint=dict(path=str(checkpoint),sha256=sha),source_initial_state=dict(path=str(initial),sha256=hashlib.sha256(initial.read_bytes()).hexdigest()),
        initial_state_tolerance=0.,scene_seed=42,num_envs=8)
    if arm=='common':
        data=tmp_path/'data02.pt';data.write_bytes(b'fixture-only');context=tmp_path/'context02.pt';context.write_bytes(b'fixture-only-context')
        cal=dict(version='dh116-teacher-tracking-calibration-v1',qualified=True,target_source='physx_backend_get_dof_position_targets',
            feedback_source='original_advance_returned_refresh_before_TensorEnv_terminal_reset',
            dataset_path=str(data),dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
            context_path=str(context),context_sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
            controller_source_sha256=controller_source_sha256(),action_contract_sha256=action_contract_sha256(),
            teacher_error_p999_rad=[.002]*12,trigger_rad=[.01]*12)
        path=tmp_path/'calibration02.json';path.write_text(json.dumps(cal));config['calibration']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_run='02')
    cfg=tmp_path/'tracking.json';cfg.write_text(json.dumps(config))
    class Policy:
        checkpoint_sha256=sha;provenance={'injected_fixture':True}
        def act(self,obs):assert obs['policy'].shape==(8,3844);return torch.ones(8,24)*.1
    args=NS(num_envs=8,seed=42,device='cpu',vision_camera_mode='fixed',vision_action_distribution='tanh_gaussian',
        vision_mirror_preprocessing='m0_raw',vision_tracking_experiment_config=str(cfg),output=str(tmp_path/'output'))
    return native,w,t,env,args,Camera(),Policy()


@pytest.mark.parametrize('arm',['original','unilateral','common'])
def test_finite_runner_evaluates_four_pinch_and_four_tripod_first_episodes_without_certificate(tmp_path,arm,monkeypatch):
    native,w,t,env,args,camera,policy=runner_scene(tmp_path,arm,monkeypatch)
    report=module().run_full_episode_tracking_experiment(native,w,t,env,args,{},camera=camera,policy=policy)
    assert report['completed_first_episodes']==8 and report['control_steps']==4
    assert report['styles']['pinch']['completed']==report['styles']['tripod']['completed']==4
    assert report['native_runtime_verified'] is False and not report['production_evaluation_certificate_issued']
    assert report['pure_student_original_native_success_count']==0
    assert not hasattr(env,'joint_replay_observer') and 'advance' not in vars(w) and 'control' not in vars(t)


def test_left_joint_failure_evidence_is_saved_without_changing_original_boundary():
    r=module().FullEpisodeTrackingRecorder('common',native=False,sink=lambda row:None)
    m=payload();m['q'][:,:6]=-1.006
    r.observe(physics_step=1,timestamp_s=1/480,live=torch.ones(8,dtype=torch.bool),done=torch.ones(8,dtype=torch.bool),
        success=torch.zeros(8,dtype=torch.bool),failure=torch.full((8,),3,dtype=torch.long),measurement=m,guard_info={},metrics={},
        raw=dict(original_joint_lower=torch.full((12,),-1.),original_joint_upper=torch.ones(12)))
    report=r.summary()
    assert report['left_joint_boundary_crossed_sample_count']==[1]*8
    assert report['right_joint_boundary_crossed_sample_count']==[0]*8
    assert report['left_joint_failure_terminal_count']==8


def test_interruption_preserves_incomplete_report_and_restores_hooks(tmp_path,monkeypatch):
    import json
    native,w,t,env,args,camera,policy=runner_scene(tmp_path,'original',monkeypatch)
    def interrupted(action):raise KeyboardInterrupt()
    env.step=interrupted
    with pytest.raises(KeyboardInterrupt):
        module().run_full_episode_tracking_experiment(native,w,t,env,args,{},camera=camera,policy=policy)
    report=json.loads((tmp_path/'output'/'tracking_full_episode_report.json').read_text())
    assert report['status']=='interrupted' and report['completed_first_episodes']==0
    assert report['production_evaluation_certificate_issued'] is False
    assert not hasattr(env,'joint_replay_observer') and 'control' not in vars(t)


def test_short_c58_prefix_cannot_substitute_a_different_full_checkpoint_hash(tmp_path,monkeypatch):
    _,_,_,_,args,_,_=runner_scene(tmp_path,'original',monkeypatch)
    monkeypatch.setattr(module(),'BASELINE_C58_SHA256','c58f9cf0718d36cd9feec181c9180a9657e97b54d6fa8e99ce1b3f16244f6d97')
    with pytest.raises(ValueError,match='complete.*c58|full.*c58'):
        module().load_tracking_config(args.vision_tracking_experiment_config)


def test_episode_identity_change_before_first_terminal_is_rejected():
    r=module().FullEpisodeTrackingRecorder('common',native=True,sink=lambda row:None)
    ids=torch.zeros(8,dtype=torch.long);r.begin(ids);ids[2]=1
    with pytest.raises(RuntimeError,match='episode identity'):
        r.observe(physics_step=1,timestamp_s=1/480,live=torch.ones(8,dtype=torch.bool),done=torch.zeros(8,dtype=torch.bool),
            success=torch.zeros(8,dtype=torch.bool),failure=torch.zeros(8,dtype=torch.long),measurement=payload(),guard_info={},metrics={},raw={},episode_ids=ids)


def test_named_palm_body_motion_never_substitutes_base_origin_motion():
    hands=[]
    for _ in range(2):
        pose=torch.zeros(8,2,7,dtype=torch.float64);pose[...,3]=1.
        hands.append(NS(body_names=['base_link','palm_link'],data=NS(body_link_pose_w=pose)))
    w=NS(n=8,hands=hands)
    initial=module().body_link_snapshot(w)
    for hand in hands:
        hand.data.body_link_pose_w[:,0,0]=.1;hand.data.body_link_pose_w[:,1,0]=.004
    current=module().body_link_motion(w,initial)
    assert current['palm_body_available']==[True,True]
    assert current['palm_body_displacement_m'][:,:,0].eq(.004).all()
    for hand in hands:
        hand.body_names=['base_link'];hand.data.body_link_pose_w=hand.data.body_link_pose_w[:,:1]
    current=module().body_link_motion(w,initial)
    assert current['palm_body_available']==[False,False] and current['palm_body_displacement_m'] is None


def test_success_flag_with_original_failure_is_never_a_style_success():
    r=module().FullEpisodeTrackingRecorder('original',native=True,sink=lambda row:None)
    ids=torch.zeros(8,dtype=torch.long);r.begin(ids)
    r.observe(physics_step=1,timestamp_s=1/480,live=torch.ones(8,dtype=torch.bool),done=torch.ones(8,dtype=torch.bool),
        success=torch.ones(8,dtype=torch.bool),failure=torch.full((8,),6,dtype=torch.long),measurement=payload(),
        guard_info={},metrics={},raw={},episode_ids=ids)
    report=r.summary()
    assert report['observed_original_native_success_count']==0
    assert report['styles']['pinch']['observed_success']==report['styles']['tripod']['observed_success']==0


def test_unilateral_boundary_casts_float32_request_to_actual_float64_chart_without_extra_scaling():
    from test_stage6_common_pause_runtime import scene
    from go2_pvcnn.control.dh116.tabletop_bimanual_tracking_guard import BimanualTrackingGuard
    w,t,_=scene();raw=torch.linspace(-.2,.2,24,dtype=torch.float32)[None]
    # Healthy backend means the old unilateral guard must preserve every value.
    for hand in w.hands:hand.data.joint_pos_target.zero_()
    guard=BimanualTrackingGuard(1,enabled=True,coupled_pause=False)
    projected,info=module().project_unilateral_tracking_action(guard,raw,w,t)
    assert projected.dtype==torch.float64 and torch.equal(projected,raw.to(t.command))
    assert info['adapter_input_action24'].dtype==torch.float32
    assert torch.equal(info['adapter_input_action24'],raw)
    assert torch.equal(info['original_controller_action24'],raw.to(t.command))
    assert torch.equal(info['projected_controller_action24'],projected)


@pytest.mark.parametrize('arm',['original','unilateral','common'])
def test_actual_vision_inference_float32_full_runner_preserves_actor_and_execution_evidence(tmp_path,arm,monkeypatch):
    import gzip,json
    from go2_pvcnn.control.dh116.tabletop_vision_inference import VisionInference
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import SQUASHED_VERSION
    native,w,t,env,args,camera,fixture_policy=runner_scene(tmp_path,arm,monkeypatch)
    actor=torch.nn.Linear(3844,24)
    with torch.no_grad():actor.weight.zero_();actor.bias.fill_(.123456789)
    policy=VisionInference(actor,SQUASHED_VERSION,fixture_policy.checkpoint_sha256,{}, {})
    assert policy.act({'policy':torch.zeros(8,3844)}).dtype==torch.float32
    module().run_full_episode_tracking_experiment(native,w,t,env,args,{},camera=camera,policy=policy)
    with gzip.open(tmp_path/'output'/'tracking_policy_60hz.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
    with gzip.open(tmp_path/'output'/'tracking_physics_480hz.jsonl.gz','rt') as stream:
        physical=[json.loads(line) for line in stream if '"raw"' in line]
    first=rows[0]
    assert first['raw_actor_action24_dtype']=='torch.float32'
    assert first['original_controller_action24_dtype']=='torch.float64'
    assert first['raw_actor_action24']==first['env_step_action24']==first['original_controller_action24']
    assert physical[0]['raw']['raw_actor_action24']==first['raw_actor_action24']
    assert physical[0]['raw']['original_controller_action24']==first['original_controller_action24']
    # Retired first episodes get zero env requests while their actual Actor
    # outputs remain intact in the raw stream; no second episode is counted.
    assert any(row['raw_actor_action24']!=row['env_step_action24'] for row in rows)
    if arm=='unilateral':
        guard=physical[0]['guard_state']
        assert guard['original_controller_action24']==first['original_controller_action24']
        assert guard['projected_controller_action24']==first['original_controller_action24']
