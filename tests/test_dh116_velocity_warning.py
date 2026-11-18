import pytest
import torch
from test_dh116_tensor_env import World, TerminalWorld
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv, physical_guard


@pytest.mark.parametrize('speed', [3.10246, -3.10246, 8.83416])
def test_finite_speed_warning_continues_without_reset_with_scaled_penalty(speed):
    t=TensorTask(template(),2,'cpu'); w=World(t); w.tick=100
    e=TensorEnv(w,t); w.qd[0,1,3]=speed
    assert physical_guard(w,w.m)[0].all()
    _,reward,done,extras=e.step(torch.zeros((2,20)))
    assert not done.any() and not w.resets and w.tick==108
    stats=e.velocity_diagnostics()
    assert stats['exceedance_env_substeps']==8
    assert stats['affected_envs']==1
    assert stats['peak_abs_rad_s']==pytest.approx(abs(speed))
    assert stats['longest_exceedance_s']==pytest.approx(8*w.dt)
    expected=-.1*((abs(speed)-3.)/3.)**2*8*w.dt
    assert extras['terms']['velocity_excess'][0]==pytest.approx(expected,rel=1e-5)
    assert extras['terms']['velocity_excess'][1]==0.
    assert reward[0]==pytest.approx(sum(float(v[0]) for v in extras['terms'].values()),abs=1e-7)
    w.qd.zero_();e.step(torch.zeros((2,20)))
    assert e.velocity_exceedance_streak_s.tolist()==[0.,0.]
    assert e.velocity_diagnostics()['exceedance_env_substeps']==8


def test_speed_warning_does_not_block_force_recovery_or_join_streak_across_reset():
    t=TensorTask(template(),2,'cpu'); w=World(t); e=TensorEnv(w,t)
    w.qd[0]=3.2
    _,_,done,extras=e.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False] and w.resets==[[0]]
    assert e.velocity_diagnostics()['exceedance_env_substeps']==8
    assert e.velocity_diagnostics()['longest_exceedance_s']==pytest.approx(5*w.dt)
    assert extras['terms']['velocity_excess'][0]==pytest.approx(-.1*((3.2-3.)/3.)**2*3*w.dt,rel=1e-5)


@pytest.mark.parametrize('speed',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_speed_still_fatal_before_task_or_reset(speed):
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    w.qd[0,1,3]=speed
    with pytest.raises(RuntimeError,match='nonfinite_velocity'):
        e.step(torch.zeros((2,20)))
    assert w.tick==1 and not w.resets and not t.elapsed.any()


def test_finite_speed_after_reset_is_warning_only():
    t=TensorTask(template(),2,'cpu');w=TerminalWorld(t,'velocity');e=TensorEnv(w,t)
    t.elapsed[0]=4.01
    _,_,done,_=e.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False] and w.resets==[[0]]
    assert e.velocity_diagnostics()['exceedance_env_substeps']==7


def test_changed_velocity_reward_and_stop_contract_rejects_old_checkpoint():
    from go2_pvcnn.control.dh116.tabletop_tensor_env import LIVE_VELOCITY_GUARD_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    current=dict(live_velocity_guard=LIVE_VELOCITY_GUARD_CONTRACT)
    old=dict(live_velocity_guard=dict(nominal_limit_rad_s=3.,absolute_tolerance_rad_s=1e-5,
        comparison='finite_abs_le_limit_plus_atol',scope='all_joints_every_physics_substep',
        action='fatal_before_reward_or_reset'))
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,current)
    assert checkpoint_contract_changes(current,current)==[]
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes({},current)


def test_stopped_monitor_reads_current_velocity_diagnostics(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_tensor_training import publish_stopped_monitor
    p=tmp_path/'monitor.json';p.write_text(json.dumps(dict(velocity_diagnostics={'affected_envs':0})))
    fresh=dict(affected_envs=2,exceedance_env_substeps=19)
    publish_stopped_monitor(p,RuntimeError('force fault'),torch.zeros(4),0,0,2,
        velocity_diagnostics=lambda:fresh)
    assert json.loads(p.read_text())['velocity_diagnostics']==fresh


def test_velocity_diagnostic_failure_keeps_original_exception(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_tensor_training import publish_stopped_monitor
    def broken():raise RuntimeError('diagnostics readback failed')
    error=RuntimeError('original fault');p=tmp_path/'monitor.json'
    publish_stopped_monitor(p,error,torch.zeros(4),0,0,2,velocity_diagnostics=broken)
    row=json.loads(p.read_text())
    assert row['error']=='original fault'
    assert row['velocity_diagnostics_readback_error']=='diagnostics readback failed'


@pytest.mark.parametrize('speed,rate', [(2.,0.),(3.,0.),(6.,-.1),(30.,-2.5)])
def test_velocity_penalty_changes_only_reward_and_is_time_scaled(speed,rate):
    results=[]
    for dt in (1/480,1/960):
        t=TensorTask(template(),2,'cpu');w=World(t);w.tick=100;w.dt=dt
        e=TensorEnv(w,t);w.qd[0]=speed
        _,reward,done,extra=e.step(torch.zeros((2,20)))
        assert not done.any() and not w.resets
        penalty=extra['terms']['velocity_excess'][0]
        assert float(penalty)==pytest.approx(rate*8*dt)
        # Both environments otherwise have identical state/action/reward.
        assert float(reward[0]-reward[1])==pytest.approx(float(penalty),abs=1e-7)
        results.append(float(penalty))
    assert results[0]==pytest.approx(2*results[1])
