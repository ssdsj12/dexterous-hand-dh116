import pytest
import torch
from test_dh116_close_downward_penalty import task
from test_dh116_tensor_contact_window import loads


def enabled(n=2):
    t=task(n)
    t.enable_policy_control()
    t.m=loads(t,4.)
    t.command.copy_(t.m['q'])
    return t


@pytest.mark.parametrize('phase',[1,3])
@pytest.mark.parametrize('sign',[-1.,1.])
def test_policy_moves_both_directions_at_actuator_rate_even_with_contact(phase,sign):
    t=enabled();t.phase[:]=phase
    t.m['mimic_errors'][:]=.007
    t.slip[:]=.01
    before=t.command.clone();offset=t.offset.clone()
    a=torch.full((2,20),sign,dtype=torch.float64)
    allowed=t.control(a,~t.done,1/480)
    assert allowed.all() and not t.paused.any()
    torch.testing.assert_close(t.command-before,torch.full_like(before,sign*.3/480))
    torch.testing.assert_close(t.offset,offset)


def test_lift_keeps_policy_palm_and_stiffness_authority_without_contact():
    t=enabled();t.phase[:]=3;t.m=loads(t,0.)
    a=torch.zeros((2,20),dtype=torch.float64)
    a[:,14]=1.;a[:,17]=-1.;a[:,18]=-1.;a[:,19]=1.
    assert t.control(a,~t.done,1/480).all()
    torch.testing.assert_close(t.palm_targets[:,:,2],torch.tensor([[.05,-.004]]*2,dtype=torch.float64))
    torch.testing.assert_close(t.policy_stiffness,torch.tensor([[10.,t.parameters.stiffness_max]]*2,dtype=torch.float64))
    assert not hasattr(t,'lift_grasp_hold')


def test_severe_mimic_and_overload_still_override_unsafe_requests():
    t=enabled();t.m['mimic_errors'][0,1]=-.013
    t.m['hand_object_loads'][1]=32.
    before=t.command.clone()
    allowed=t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    assert not allowed.any() and t.paused.all()
    assert t.command[0,2]<before[0,2]
    assert (t.command[1]<=before[1]+1e-12).all()
    assert ((t.command-before).abs()<=.3/480+1e-12).all()
    assert (t.command>=t.low).all() and (t.command<=t.high).all()


def test_inactive_and_selected_reset_do_not_change_other_environment():
    t=enabled();before=t.command.clone();palms=t.palm_targets.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),torch.tensor([True,False]),1/480)
    torch.testing.assert_close(t.command[1],before[1]);torch.testing.assert_close(t.palm_targets[1],palms[1])
    first=t.command[0].clone();t.reset([1],t.m)
    torch.testing.assert_close(t.command[0],first)
    assert torch.isfinite(t.observe()).all()


def test_world_submits_policy_targets_at_carriage_rate_and_policy_gains():
    from test_dh116_tensor_world import palm_world
    w,t=palm_world();w.device="cpu"
    del t.coordinated_control
    t.policy_control=True;t.phase=torch.full((3,),3)
    t.palm_targets=torch.zeros((3,2,3),dtype=torch.float64);t.palm_targets[:,:,2]=.05
    t.policy_stiffness=torch.tensor([[11.,19.]]*3,dtype=torch.float64)
    w.advance(t,torch.ones(3,dtype=torch.bool),torch.tensor([True,True,False]))
    torch.testing.assert_close(w.palms[:2,:,2],torch.full((2,2),.05/480,dtype=torch.float64))
    torch.testing.assert_close(w.gains[:2],t.policy_stiffness[:2])
    assert (w.palms[2]==0).all() and (w.gains[2]==10).all()


def test_policy_contract_rejects_legacy_action_semantics():
    from go2_pvcnn.control.dh116.tabletop_policy_control import apply_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    old=dict(num_envs=2,action='20D_piecewise_joint_envelope_palm_load_v1',force_feedback={'enabled':True})
    new=apply_contract(dict(old),enabled())
    assert new['action'] != old['action']
    assert new['force_feedback']['enabled'] is False
    assert checkpoint_contract_changes(new,new)==[]
    with pytest.raises(ValueError):checkpoint_contract_changes(old,new)
    with pytest.raises(ValueError):checkpoint_contract_changes(new,old)


@pytest.mark.parametrize('option', ['release_reposition','force_direction_reposition','contact_effort_feedback','teacher_search_cases','control_probe_checkpoint','play_checkpoint'])
def test_policy_configuration_rejects_scripted_or_old_action_paths(option):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_policy_control import validate_configuration
    args=SimpleNamespace(tensor_pipeline=True,palm_transport='driven_carriage',
        pressure_relief=True,skip_hold=True,grasp_readiness='measured_load',
        palm_exploration_axis='xyz',reset_height_jitter_m=0.)
    setattr(args,option,True)
    with pytest.raises(ValueError):validate_configuration(args)


def test_policy_world_preserves_independent_targets_and_restores_table_clearance():
    from test_dh116_tensor_world import palm_world
    w,t=palm_world();w.device='cpu';del t.coordinated_control
    t.policy_control=True;t.phase=torch.ones(3,dtype=torch.long)
    t.palm_targets=torch.zeros((3,2,3),dtype=torch.float64)
    t.palm_targets[:,0,2]=.05;t.palm_targets[:,1,2]=-.004
    t.policy_stiffness=w.gains.clone()
    w.clearances[:,1]=.0001
    w.advance(t,torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(w.palms[:,0,2],torch.full((3,),.05/480,dtype=torch.float64))
    torch.testing.assert_close(w.palms[:,1,2],torch.full((3,),.0002,dtype=torch.float64))


def test_policy_mode_rejects_later_scripted_controller_enable():
    t=enabled()
    with pytest.raises(ValueError):t.enable_coordinated_control(torch.ones(2,dtype=torch.bool))


def test_shaped_policy_reports_real_requested_transport_and_guard_masks():
    t=task(2,grasp_shaping=True);t.enable_policy_control();t.phase[:]=3
    t.m=loads(t,4.);t.command.copy_(t.m['q'])
    a=torch.zeros((2,20),dtype=torch.float64);a[:,14]=a[:,17]=1.
    t.control(a,~t.done,1/480)
    assert t.lift_diagnostic_masks[:,0].all() and t.lift_diagnostic_masks[:,1].all()
    t.m['mimic_errors'][0,1]=.013
    t.control(a,~t.done,1/480)
    assert t.lift_diagnostic_masks[0,2] and not t.lift_diagnostic_masks[0,1]


def test_policy_contract_describes_actual_independent_clearance_and_gain_submission():
    from go2_pvcnn.control.dh116.tabletop_policy_control import apply_contract
    t=enabled();c=apply_contract({},t)
    assert c['palm_target_safety']['synchronization']=='none_per_hand'
    assert c['palm_descent_clearance_m']==t.palm_descent_clearance_m==.0003
    assert c['active_stiffness_rate_per_s'] is None


def test_default_shell_selects_executable_policy_arguments(tmp_path):
    import subprocess,os,json
    from pathlib import Path
    stub=tmp_path/'bash'
    stub.write_text('#!/usr/bin/env python3\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o755)
    # Process availability is external to this argument-wiring test.
    probe=tmp_path/'pgrep';probe.write_text('#!/bin/sh\nexit 1\n');probe.chmod(0o755)
    env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'])
    for key in ('DH116_GRASP_CONTROLLER','DH116_FORCE_DIRECTION_REPOSITION','DH116_RESUME'):
        env.pop(key,None)
    launcher=Path(__file__).resolve().parents[1]/'scripts/train_dh116_tensor_ppo.sh'
    # Stub only the final interpreter launch; do not start Isaac.
    run=subprocess.run(['/bin/bash',str(launcher)],env=env,text=True,capture_output=True,check=True)
    args=json.loads(run.stdout)
    assert args[args.index('--grasp-controller')+1]=='policy'
    assert args[args.index('--palm-exploration-axis')+1]=='xyz'
    assert '--release-reposition' not in args


def test_overload_and_severe_mimic_conflict_uses_neutral_measured_target():
    t=enabled();t.m['mimic_errors'][0,1]=.013;t.m['hand_object_loads'][0]=32.
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    torch.testing.assert_close(t.joint_governed_target[0,2],t.m['q'][0,2])
    torch.testing.assert_close(t.command[0,2],t.m['q'][0,2])


def test_policy_shaped_environment_steps_and_keeps_terminal_guards():
    from test_dh116_tensor_env import World
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
    t=task(2,grasp_shaping=True);t.enable_policy_control()
    w=World(t);w.tick=100
    w.vertical_components=torch.zeros((2,2,2),dtype=torch.float64)
    w.object_weight_n=torch.ones(2,dtype=torch.float64)
    e=TensorEnv(w,t)
    e.lift_gate_counts=torch.zeros(8,dtype=torch.long)
    observation,reward,done,_=e.step(torch.zeros((2,20)))
    assert torch.isfinite(observation).all() and torch.isfinite(reward).all()
    assert not done.any()
    # A policy mode cannot bypass the existing nonfinite-velocity hard stop.
    w.qd[:]=float("nan")
    with pytest.raises(RuntimeError,match='physical safety failure'):
        e.step(torch.zeros((2,20)))
