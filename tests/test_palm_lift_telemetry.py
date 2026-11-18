import importlib

import pytest
import torch

from test_dh116_tensor_env import World
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv


def telemetry(n=2):
    cls=importlib.import_module('go2_pvcnn.control.dh116.tabletop_palm_lift_telemetry').PalmLiftTelemetry
    return cls(n,'cpu')


def measured(z):
    z=torch.tensor(z,dtype=torch.float64)
    pose=torch.zeros((*z.shape,7),dtype=torch.float64);pose[...,2]=z;pose[...,3]=1.
    return {'wrist_pose':pose,'palm_targets':torch.full_like(z,99.),'root_seed':torch.full_like(z,-99.)}


def test_actual_measurement_and_left_right_order_with_signed_tick_peaks():
    t=telemetry();t.reset(torch.arange(2),measured([[.5,.7],[.6,.8]]));t.begin_tick()
    t.observe(measured([[.53,.69],[.62,.84]]),torch.tensor([True,False]))
    t.observe(measured([[.51,.68],[.65,.86]]),torch.tensor([True,False]))
    out=t.snapshot()
    assert torch.allclose(out['last_m'][0],torch.tensor([.01,-.02],dtype=torch.float64))
    assert torch.allclose(out['peak_m'][0],torch.tensor([.03,-.01],dtype=torch.float64))
    assert out['valid'].tolist()==[[True,True],[False,False]]


def test_selected_reset_updates_actual_refs_without_destroying_terminal_tick_snapshot():
    t=telemetry();t.reset(torch.arange(2),measured([[.5,.7],[.6,.8]]));t.begin_tick()
    t.observe(measured([[.53,.76],[.62,.84]]),torch.ones(2,dtype=torch.bool))
    before=t.snapshot()
    t.reset(torch.tensor([0]),measured([[.9,1.1],[88.,99.]]))
    assert torch.equal(t.snapshot()['peak_m'],before['peak_m'])
    t.begin_tick();t.observe(measured([[.91,1.12],[.63,.86]]),torch.ones(2,dtype=torch.bool))
    assert torch.allclose(t.snapshot()['last_m'],torch.tensor([[.01,.02],[.03,.06]],dtype=torch.float64))
    before['last_m'].fill_(999.)
    assert not (t.snapshot()['last_m']==999.).any()


def test_missing_reference_and_nonfinite_palm_rows_are_explicitly_invalid():
    t=telemetry();t.begin_tick();t.observe(measured([[.5,.7],[.6,.8]]),torch.ones(2,dtype=torch.bool))
    assert not t.snapshot()['valid'].any()
    baseline=measured([[.5,.7],[float('nan'),.8]])
    t.reset(torch.arange(2),baseline);t.begin_tick()
    m=measured([[.51,.71],[.61,.81]]);m['wrist_pose'][0,1,0]=float('nan')
    t.observe(m,torch.ones(2,dtype=torch.bool))
    assert t.snapshot()['valid'].tolist()==[[True,False],[False,True]]
    assert torch.isfinite(t.snapshot()['last_m']).all()


def test_snapshot_is_detached_and_observation_does_not_consume_rng():
    t=telemetry();m=measured([[.5,.7],[.6,.8]]);m['wrist_pose'].requires_grad_()
    rng=torch.get_rng_state().clone()
    t.reset(torch.arange(2),m);t.begin_tick();t.observe(m,torch.ones(2,dtype=torch.bool))
    out=t.snapshot();assert not out['last_m'].requires_grad
    assert torch.equal(torch.get_rng_state(),rng)
    out['valid'].zero_();assert t.snapshot()['valid'].all()


class PalmWorld(World):
    def __init__(self,t):
        super().__init__(t)
        self.reset_counts=torch.zeros(2,dtype=torch.long)
        self.base_z=torch.tensor([[.5,.7],[.6,.8]],dtype=torch.float64)
        self.m.update(measured(self.base_z.tolist()))
        # These command-like values intentionally disagree with actual poses.
        self.palms=torch.full((2,2,3),33.)
        self.root_seed=torch.full((2,2,7),44.)
    def reset(self,ids):
        super().reset(ids);self.reset_counts[ids]+=1
        self.m['wrist_pose'][ids,:,2]=self.base_z[ids]+.4*(self.reset_counts[ids,None]-1)
        return self.m
    def advance(self,t,allowed,live):
        m=super().advance(t,allowed,live)
        # Continue physical evolution for reset rows while task snapshots freeze.
        m['wrist_pose'][:,:,2]+=torch.tensor([.01,.02],dtype=torch.float64)
        return m


def env(enabled=False,continuous=False):
    task=TensorTask(template(),2,'cpu');task.continuous_motion=continuous
    world=PalmWorld(task);result=TensorEnv(world,task)
    if enabled:result.enable_palm_lift_telemetry()
    result.reset()
    return result,world,task


def test_env_captures_terminal_before_physical_reset_and_next_tick_uses_that_actual_reset():
    e,w,t=env(True)
    _,_,done,extra=e.step(torch.zeros(2,20))
    assert done.tolist()==[True,False]
    saved=extra['palm_lift_telemetry']
    assert torch.allclose(saved['last_m'],torch.tensor([[.03,.06],[.08,.16]],dtype=torch.float64))
    assert torch.allclose(saved['peak_m'],saved['last_m'])
    assert saved['valid'].all()
    _,_,_,next_extra=e.step(torch.zeros(2,20))
    # Five post-reset substeps in the prior tick plus eight new substeps; do not
    # silently replace the actual physical-reset baseline at task.reset(tick end).
    assert torch.allclose(next_extra['palm_lift_telemetry']['last_m'],
        torch.tensor([[.13,.26],[.16,.32]],dtype=torch.float64))
    assert torch.allclose(saved['last_m'][0],torch.tensor([.03,.06],dtype=torch.float64))


def test_invalid_action_reset_refreshes_reference_without_fabricating_terminal_measurement():
    e,w,t=env(True,continuous=True)
    w.tick=10  # This test exercises invalid-action recovery, not the force fault.
    action=torch.zeros(2,20);action[0,0]=float('nan')
    _,_,done,extra=e.step(action)
    assert done[0] and not extra['palm_lift_telemetry']['valid'][0].any()
    _,_,_,extra=e.step(torch.zeros(2,20))
    assert torch.allclose(extra['palm_lift_telemetry']['last_m'][0],
        torch.tensor([.16,.32],dtype=torch.float64))


def test_enabled_and_disabled_keep_existing_behavior_and_rng_identical():
    baseline,wb,tb=env(False);observed,wo,to=env(True)
    assert observed.enable_palm_lift_telemetry() is observed.palm_lift_telemetry
    rng=torch.get_rng_state().clone()
    for _ in range(2):
        a=baseline.step(torch.zeros(2,20));b=observed.step(torch.zeros(2,20))
        for x,y in zip(a[:3],b[:3]):assert torch.equal(x,y)
        assert 'palm_lift_telemetry' not in a[3]
        assert wb.resets==wo.resets
        assert torch.equal(a[3]['terminal_observation'],b[3]['terminal_observation'])
        for key in a[3]['terms']:assert torch.equal(a[3]['terms'][key],b[3]['terms'][key])
    assert torch.equal(torch.get_rng_state(),rng)
