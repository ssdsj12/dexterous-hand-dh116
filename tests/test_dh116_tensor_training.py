import torch
import pytest
from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm, observations


def test_stopped_monitor_retains_counters_and_partial_rollout_peaks(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_tensor_training import publish_stopped_monitor
    path=tmp_path/'monitor.json'
    path.write_text(json.dumps(dict(episodes=8,failures={'timeout':2},physics_peaks_since_start={'link_force_n':12.})))
    publish_stopped_monitor(path,RuntimeError('force fault'),torch.tensor([46.,1.,2.,.01]),7,168,4)
    row=json.loads(path.read_text())
    assert row['episodes']==8 and row['failures']=={'timeout':2}
    assert row['iteration']==7 and row['timesteps']==168
    assert row['error']=='force fault' and row['status']=='stopped'
    assert row['physics_peaks_since_start']['link_force_n']==46.


def test_stopped_monitor_readback_failure_preserves_last_peaks(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_tensor_training import publish_stopped_monitor
    class Broken:
        def cpu(self): raise RuntimeError('device unavailable')
    path=tmp_path/'monitor.json'
    path.write_text(json.dumps(dict(physics_peaks_since_start={'link_force_n':12.})))
    publish_stopped_monitor(path,RuntimeError('original fault'),Broken(),7,168,4)
    row=json.loads(path.read_text())
    assert row['error']=='original fault'
    assert row['physics_peaks_since_start']=={'link_force_n':12.}
    assert 'device unavailable' in row['physics_peak_readback_error']


def test_stopped_monitor_handles_missing_file_and_nonfinite_peaks(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_tensor_training import publish_stopped_monitor
    path=tmp_path/'monitor.json'
    publish_stopped_monitor(path,RuntimeError('nonfinite'),torch.tensor([float('nan'),1.,2.,.01]),0,0,4)
    row=json.loads(path.read_text())
    assert row['physics_peaks_since_start']['link_force_n'] is None
    assert row['physics_peak_nonfinite_fields']==['link_force_n']


def test_stopped_monitor_write_failure_does_not_replace_original_error(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
    def broken(*args): raise OSError('disk full')
    monkeypatch.setattr(training,'write_json',broken)
    error=RuntimeError('original fault')
    training.publish_stopped_monitor(tmp_path/'monitor.json',error,torch.zeros(4),0,0,4)
    assert str(error)=='original fault'
    assert any('disk full' in note for note in error.__notes__)


def test_checkpoint_migration_only_adds_known_controller_guard_fields():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_tensor_env import VELOCITY_GUARD_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_pressure_relief import CONTRACT
    current=dict(num_envs=4096,pressure_relief=CONTRACT,reset_velocity_guard=VELOCITY_GUARD_CONTRACT)
    assert checkpoint_contract_changes(current,current)==[]
    old=dict(num_envs=4096,pressure_relief=CONTRACT)
    assert checkpoint_contract_changes(old,current)==['add_reset_velocity_tolerance']
    assert set(checkpoint_contract_changes(dict(num_envs=4096),current))=={'add_measured_pressure_relief','add_reset_velocity_tolerance'}
    for bad in (dict(old,num_envs=16),dict(current,reset_velocity_guard=dict(VELOCITY_GUARD_CONTRACT,absolute_tolerance_rad_s=.1))):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(bad,current)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(current,old)


def test_checkpoint_migration_records_exact_terminal_reset_change():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TERMINAL_RESET_CONTRACT
    old=dict(num_envs=4096,physics_hz=480)
    current=dict(old,terminal_reset=TERMINAL_RESET_CONTRACT)
    assert checkpoint_contract_changes(old,current)==['add_immediate_validated_terminal_reset']
    with pytest.raises(ValueError,match='contract mismatch'):
        checkpoint_contract_changes(dict(old,terminal_reset={'version':'unvalidated'}),current)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(current,old)


def test_checkpoint_migration_only_adds_exact_live_velocity_guard():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_tensor_env import LIVE_VELOCITY_GUARD_CONTRACT
    old=dict(num_envs=4096)
    new=dict(old,live_velocity_guard=LIVE_VELOCITY_GUARD_CONTRACT)
    assert checkpoint_contract_changes(old,new)==['add_live_substep_velocity_guard']
    for bad in (dict(new,num_envs=16),dict(new,live_velocity_guard=dict(LIVE_VELOCITY_GUARD_CONTRACT,nominal_limit_rad_s=9.))):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)


def test_compliant_mimic_migration_requires_the_replayed_physics_configuration():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_tensor_env import LIVE_VELOCITY_GUARD_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract, material_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_world import ROOT_POSE_WRITE_CONTRACT
    old=dict(physics_hz=480,solver_type=0,position_iterations=64,velocity_iterations=16,
        num_envs=4096,mimic_natural_frequency_s_inv=0.,mimic_damping_ratio=1.,
        active_velocity_limit_rad_s=.3,passive_velocity_limit_rad_s=2.97,
        control_hz=60,active_effort_limit_nm=.2,active_stiffness_range=[10.,20.],
        active_stiffness_reset=10.,active_stiffness_rate_per_s=10.,active_damping=.2,
        root_pose_write=ROOT_POSE_WRITE_CONTRACT,full_hand_contact=True,
        **material_contract('extreme_grip'),
        live_velocity_guard=LIVE_VELOCITY_GUARD_CONTRACT,
        **hand_material_contract('grip_compliant'))
    new=dict(old,mimic_natural_frequency_s_inv=400.)
    assert checkpoint_contract_changes(old,new)==['compliant_mimic_0_to_400']
    assert checkpoint_contract_changes(new,new)==[]
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)
    for key,value in [('mimic_natural_frequency_s_inv',200.),('mimic_damping_ratio',2.),
            ('physics_hz',240),('solver_type',1),('position_iterations',16),('velocity_iterations',4),
            ('active_velocity_limit_rad_s',2.97),('passive_velocity_limit_rad_s',3.),
            ('num_envs',512),('live_velocity_guard',{}),('object_kind','hollow_straw'),
            ('control_hz',30),('active_effort_limit_nm',20.),('active_stiffness_range',[10.,2000.]),
            ('active_stiffness_reset',20.),('active_stiffness_rate_per_s',100.),('active_damping',0.),
            ('root_pose_write',{}),('full_hand_contact',False),('object_contact_material',{}),
            ('hand_contact_material',hand_material_contract('grip')['hand_contact_material'])]:
        # Also reject changing an otherwise matching baseline to an untested
        # physics configuration; full contract equality alone is insufficient.
        source=dict(old,**{key:value}) if key!='mimic_natural_frequency_s_inv' else old
        target=dict(new,**{key:value})
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(source,target)
    with pytest.raises(ValueError,match='contract mismatch'):
        checkpoint_contract_changes(old,dict(new,unrelated_field='changed'))


def test_checkpoint_migration_allows_only_declared_passive_headroom_candidate():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import VELOCITY_HEADROOM_CONTRACT
    old=dict(passive_velocity_limit_rad_s=3.,num_envs=4096)
    new=dict(old,passive_velocity_limit_rad_s=2.97,velocity_headroom=VELOCITY_HEADROOM_CONTRACT)
    assert checkpoint_contract_changes(old,new)==['add_passive_velocity_headroom']
    for bad in (dict(new,passive_velocity_limit_rad_s=2.9),dict(new,num_envs=16),
                dict(new,velocity_headroom={}),dict(old,passive_velocity_limit_rad_s=2.97)):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)


def test_close_force_behavior_requires_new_controller_contract():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_coordinated_grasp import CONTRACT
    from go2_pvcnn.control.dh116.tabletop_tensor_task import CLOSE_DOWNWARD_PENALTY_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    assert CONTRACT['version']=='measured_motion_grasp_v2'
    assert CLOSE_DOWNWARD_PENALTY_CONTRACT['version']=='close_downward_impulse_reset_v3'
    old=dict(num_envs=4096,
        grasp_controller=dict(CONTRACT,version='measured_motion_grasp_v1',command_rate_rad_s=.05),
        close_downward_penalty=dict(CLOSE_DOWNWARD_PENALTY_CONTRACT,
            version='close_downward_event_penalty_v1',impulse_limit_n_s=None))
    with pytest.raises(ValueError,match='contract mismatch'):
        checkpoint_contract_changes(old,dict(num_envs=4096,grasp_controller=CONTRACT,
            close_downward_penalty=CLOSE_DOWNWARD_PENALTY_CONTRACT))


def test_checkpoint_migration_only_accepts_exact_contact_window_reward():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_contact_window import reward_contract,REWARD_VERSION,OBSERVATION_VERSION
    old=dict(num_envs=4096,phase_reward_version='phase_potential_v1_10n_5',
        reward='full_hand_object_support_progress_v1_10n_5',
        observation_version='dh116-free-grasp-v1',observation_size=148)
    new=dict(old,phase_reward_version=REWARD_VERSION,reward=REWARD_VERSION,
        observation_version=OBSERVATION_VERSION,observation_size=150,
        contact_reward_window=reward_contract('first_contact'))
    assert checkpoint_contract_changes(old,new)==['contact_window_reward_v2']
    for bad in (dict(new,observation_size=151),dict(new,num_envs=16),
        dict(new,contact_reward_window=dict(new['contact_reward_window'],window_s=2.))):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)


def test_checkpoint_window_start_can_change_only_to_close_start():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_contact_window import reward_contract,REWARD_VERSION,OBSERVATION_VERSION
    old=dict(num_envs=4096,phase_reward_version=REWARD_VERSION,reward=REWARD_VERSION,
        observation_version=OBSERVATION_VERSION,observation_size=150,
        contact_reward_window=reward_contract('first_contact'))
    new=dict(old,contact_reward_window=reward_contract('close_start'))
    assert checkpoint_contract_changes(old,new)==['contact_window_start_at_close']
    assert checkpoint_contract_changes(new,new)==[]
    for bad in (dict(new,num_envs=16),dict(new,observation_size=148),
                dict(new,contact_reward_window=dict(new['contact_reward_window'],window_s=2.))):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)


def test_checkpoint_load_target_can_expand_from_four_to_twelve_only():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    from go2_pvcnn.control.dh116.tabletop_force_feedback import force_feedback_contract
    old=dict(force_feedback_contract(LoadParameters(max_force_n=4.)),num_envs=4096)
    new=dict(force_feedback_contract(LoadParameters(max_force_n=12.,force_rate_n_s=2.)),num_envs=4096)
    assert checkpoint_contract_changes(old,new)==['hand_load_target_4n_to_12n']
    assert checkpoint_contract_changes(new,new)==[]
    for bad in (dict(new,num_envs=16),
                dict(force_feedback_contract(LoadParameters(max_force_n=12.,force_rate_n_s=3.)),num_envs=4096),
                dict(force_feedback_contract(LoadParameters(max_force_n=12.,effort_limit_nm=.5)),num_envs=4096),
                dict(force_feedback_contract(LoadParameters(max_force_n=15.)),num_envs=4096)):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)


@pytest.mark.parametrize('old_size,new_size',[(148,150),(150,151)])
def test_extended_reward_observation_preserves_network_and_optimizer_prefix(old_size,new_size):
    import copy
    from go2_pvcnn.control.dh116.tabletop_tensor_training import extend_checkpoint_observations
    old_obs=observations(torch.randn(4,old_size));old=make_algorithm(old_obs,2)
    with torch.inference_mode():
        for _ in range(2):
            a=old.act(old_obs);old.process_env_step(old_obs,-a.square().sum(1),torch.zeros(4,dtype=torch.bool),{})
        old.compute_returns(old_obs)
    old.update()
    source=dict(policy=copy.deepcopy(old.policy.state_dict()),optimizer=copy.deepcopy(old.optimizer.state_dict()))
    extended_obs=observations(torch.cat([old_obs['policy'],torch.randn(4,new_size-old_size)],1));new=make_algorithm(extended_obs,2)
    policy,optimizer=extend_checkpoint_observations(source,new)
    new.policy.load_state_dict(policy);new.optimizer.load_state_dict(optimizer)
    with torch.no_grad():
        torch.testing.assert_close(old.policy.act_inference(old_obs),new.policy.act_inference(extended_obs))
        torch.testing.assert_close(old.policy.evaluate(old_obs),new.policy.evaluate(extended_obs))
    for key in ('actor.0.weight','critic.0.weight'):
        assert source['policy'][key].shape==(128,old_size)
        assert torch.equal(policy[key][:,:old_size],source['policy'][key]) and not policy[key][:,old_size:].any()
    names={id(p):name for name,p in new.policy.named_parameters()}
    for group in new.optimizer.param_groups:
        for p in group['params']:
            if names[id(p)] in ('actor.0.weight','critic.0.weight'):
                for field in ('exp_avg','exp_avg_sq'):
                    assert new.optimizer.state[p][field].shape==p.shape
                    assert not new.optimizer.state[p][field][:,old_size:].any()
    with torch.inference_mode():
        for _ in range(2):
            a=new.act(extended_obs);new.process_env_step(extended_obs,-a.square().sum(1),torch.zeros(4,dtype=torch.bool),{})
        new.compute_returns(extended_obs)
    assert all(torch.isfinite(torch.tensor(v)) for v in new.update().values())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_rsl_rollout_update_stays_on_device(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    obs=observations(torch.zeros((16,148),device=device))
    alg=make_algorithm(obs,4)
    before=[p.detach().clone() for p in alg.policy.parameters()]
    with torch.inference_mode():
        for i in range(4):
            action=alg.act(obs)
            assert action.device.type==device
            assert torch.allclose(alg.policy.action_mean[:,-2:],torch.full((16,2),-.95,device=device),atol=1e-6)
            reward=-action.square().sum(1)
            done=torch.zeros(16,device=device,dtype=torch.bool);done[0]=i==2
            alg.process_env_step(obs,reward,done,{})
        alg.compute_returns(obs)
    assert alg.storage.observations.device.type==device
    assert alg.storage.returns.device.type==device
    assert alg.storage.dones[2,0].item()==1
    result=alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert any(not torch.equal(a,b) for a,b in zip(before,alg.policy.parameters()))
