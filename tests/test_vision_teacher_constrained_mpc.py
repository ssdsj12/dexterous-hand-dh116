"""Opt-in constrained geometry integration; these are CPU contract tests."""
import importlib

import pytest
import torch

from test_vision_teacher_acquisition_mpc import fixture, controller as original_controller
from test_vision_teacher_pressure_mpc import PINCH, TRIPOD


def controller(env, atlas, sequence, styles=None):
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_teacher_constrained_mpc')
    return module.ConstrainedAcquisitionTeacherReferenceMPC(env,'a'*64,sequence,
        grasp_styles=styles or [TRIPOD]*env.world.n,cylinder_radius_m=.008,
        cylinder_length_m=.5,pad_atlas=atlas)


@pytest.mark.parametrize('case',['active','below_air','invalid_q','invalid_command','bad_sensor'])
def test_pinch_preserves_frozen_acquisition_actions_bitwise(case):
    before,_=fixture();after,atlas=fixture();sequence=torch.full((1,5,24),.1)
    for env in (before,after):
        env.task.m['effective_pad_loads_n'][:]=env.task.command.new_tensor([1.,.5,.5,0,0])
        env.task.m['effective_pad_vectors_w'][...,0]=env.task.m['effective_pad_loads_n']*env.task.command.new_tensor([1.,-1.,-1.,0,0])
        env.task.m['effective_pad_valid']=env.task.m['effective_pad_loads_n']>0
        if case=='active':env.task.m['bottom'][:]=.39
    old=original_controller(before,atlas,sequence,[PINCH]);new=controller(after,atlas,sequence,[PINCH])
    for env in (before,after):
        if case=='invalid_q':env.task.m['q'][0,0]=float('nan')
        if case=='invalid_command':env.task.command[0,0]=float('nan')
        if case=='bad_sensor':
            env.world.pad_measurements={'effective_pad_bad_env':torch.ones(1,dtype=torch.bool)}
            env.world.adapter_fault_bits=torch.ones(1,dtype=torch.long)
    expected,a=old.refine(sequence[:,0],context={'tick':0})
    actual,b=new.refine(sequence[:,0],context={'tick':0})
    assert torch.equal(actual,expected)
    assert b['per_env'][0]['reason']==a['per_env'][0]['reason']
    assert b['per_env'][0]['profile']==a['per_env'][0]['profile']


@pytest.mark.parametrize('case',['bearing','wrong_facing','inside','bad_sensor'])
def test_no_geometry_uses_original_planner_and_fallback_bitwise(case):
    before,atlas=fixture(facing=case!='wrong_facing',gap=-.0002 if case=='inside' else .0002)
    after,_=fixture(facing=case!='wrong_facing',gap=-.0002 if case=='inside' else .0002)
    for env in (before,after):
        if case=='bearing':
            env.task.m['effective_pad_loads_n'][:]=env.task.command.new_tensor([5.,.75,5.,0,0])
            env.task.m['effective_pad_vectors_w'][...,0]=env.task.m['effective_pad_loads_n']*env.task.command.new_tensor([1.,-1.,-1.,0,0])
            env.task.m['effective_pad_valid']=env.task.m['effective_pad_loads_n']>0
        if case=='bad_sensor':env.task.m['effective_pad_bad_env']=torch.ones(1,dtype=torch.bool)
    sequence=torch.full((1,5,24),.1)
    old=original_controller(before,atlas,sequence);new=controller(after,atlas,sequence)
    expected,a=old.refine(sequence[:,0],context={'tick':0})
    actual,b=new.refine(sequence[:,0],context={'tick':0})
    assert torch.equal(actual,expected)
    assert b['per_env'][0]['reason']==a['per_env'][0]['reason']
    assert b['per_env'][0]['constrained_solver_calls']==[]


def test_geometry_logs_complete_replay_inputs_and_actual_encoded_first_step_without_sensor_mutation():
    env,atlas=fixture();sequence=torch.zeros(1,5,24);sequence[:,:,13]=.1
    new=controller(env,atlas,sequence)
    measured={key:value.clone() for key,value in env.task.m.items()}
    issued,diag=new.refine(sequence[:,0],context={'tick':0})
    row=diag['per_env'][0]
    assert diag['profile']=='pressure_geometry_constrained'
    assert diag['force_prediction'] is False and diag['optimized_action_dim']==12
    assert torch.equal(issued[:,12:],sequence[:,0,12:])
    assert row['active'] and row['constrained_encoded_first_check']['passed']
    assert row['constrained_solver_calls']
    for call in row['constrained_solver_calls']:
        assert set(('q','command','nominal','points','jacobian','target_points','wrist_motion','weights','lower','upper','response'))<=call['planner_inputs'].keys()
        trajectory=torch.tensor(call['planner_diagnostics']['trajectory'],dtype=torch.float64)
        data=call['planner_inputs'];command=torch.tensor(data['command'],dtype=torch.float64)
        increments=torch.cat((trajectory[:,:1]-command[:,None],trajectory[:,1:]-trajectory[:,:-1]),1)
        assert increments.abs().max()<=.005+1e-12
        assert (trajectory>=torch.tensor(data['lower'])-1e-7).all()
        assert (trajectory<=torch.tensor(data['upper'])+1e-7).all()
        assert call['geometry_context']['cylinder_radius_m']==.008
    actual_command=env.task.command+issued[:,:12].double()*.005
    torch.testing.assert_close(torch.tensor(row['constrained_encoded_first_check']['command_rad']),actual_command.reshape(2,6).float())
    for key,value in measured.items():assert torch.equal(value,env.task.m[key])
    assert new.native.pressure_planner.geometry_context is None


def test_infeasible_geometry_keeps_original_permanent_code4_and_budget_accounting():
    before,atlas=fixture();after,_=fixture();sequence=torch.zeros(1,6,24)
    for env in (before,after):env.world.hands[0].root_physx_view.get_jacobians().zero_()
    old=original_controller(before,atlas,sequence);new=controller(after,atlas,sequence)
    for tick in range(3):
        expected,a=old.refine(sequence[:,tick],context={'tick':tick})
        actual,b=new.refine(sequence[:,tick],context={'tick':tick})
        assert torch.equal(actual,expected)
        for key in ('geometry_stop_code','geometry_requested_approach_m','geometry_request_time_s','geometry_no_progress_s'):
            assert a['per_env'][0][key]==b['per_env'][0][key]
    assert (torch.tensor(b['per_env'][0]['geometry_stop_code'])[...,:3]==4).all()


def test_bad_sibling_and_terminal_do_not_leak_constraint_context_or_restart():
    env,atlas=fixture(3);sequence=torch.zeros(3,5,24)
    new=controller(env,atlas,sequence);env.task.m['q'][1,0]=float('nan')
    issued,diag=new.refine(sequence[:,0],context={'tick':0})
    assert torch.isfinite(issued).all() and not issued[1,:12].any()
    assert diag['per_env'][0]['active'] and diag['per_env'][2]['active']
    assert diag['per_env'][1]['constrained_solver_calls']==[]
    new.observe_result(torch.tensor([True,False,False]))
    issued,diag=new.refine(sequence[:,1],context={'tick':1},alive=torch.tensor([False,True,True]))
    assert not issued[0].any() and diag['per_env'][0]['constrained_solver_calls']==[]
    with pytest.raises(ValueError,match='ended'):
        new.refine(sequence[:,2],context={'tick':2},alive=torch.ones(3,dtype=torch.bool))


def test_encoded_command_is_checked_against_geometry_instead_of_trusting_predicted_points(monkeypatch):
    env,atlas=fixture();sequence=torch.zeros(1,5,24);new=controller(env,atlas,sequence)
    original=new.native.pressure_planner.plan
    def inconsistent(**kwargs):
        target,diag=original(**kwargs)
        return target+.1,diag  # Good stored model points cannot authorize a different command.
    monkeypatch.setattr(new.native.pressure_planner,'plan',inconsistent)
    issued,diag=new.refine(sequence[:,0],context={'tick':0})
    assert not issued[:,:12].any()
    assert not diag['per_env'][0]['active']
    assert diag['per_env'][0]['constrained_encoded_first_check']['passed'] is False
    assert diag['per_env'][0]['reason']=='constrained_encoded_first_step_rejected'
