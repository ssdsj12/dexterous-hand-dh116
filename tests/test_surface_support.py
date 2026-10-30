"""A lift-support wrench comes from bearing fingers, within motor capacity."""
import importlib
import importlib.util
import torch
import pytest


def module():
    name='go2_pvcnn.control.dh116.tabletop_surface_support'
    assert importlib.util.find_spec(name) is not None,'real-contact support missing'
    return importlib.import_module(name)


def inputs():
    j=torch.zeros(1,2,2,3,6,dtype=torch.float64)
    j[:,:,0,2,0]=.03;j[:,:,1,2,1]=.03
    n=torch.zeros(1,2,2,3,dtype=torch.float64);n[:,:,0,0]=1.;n[:,:,1,0]=-1.
    loads=torch.full((1,2,2),2.5,dtype=torch.float64)
    normal_z=torch.zeros(1,2,dtype=torch.float64);friction_z=normal_z.clone()
    weight=torch.tensor([1.1772],dtype=torch.float64);velocity=weight*0
    enabled=torch.tensor([True])
    return j,n,loads,normal_z,friction_z,weight,velocity,enabled


def test_stationary_lift_target_exceeds_weight_but_tracks_weight_at_goal_speed():
    values=inputs();tau,d=module().support_torques(*values)
    assert d['target_total_fz_n'][0]>values[5][0]
    values=list(values);values[6]=torch.tensor([.003],dtype=torch.float64)
    _,d=module().support_torques(*values)
    torch.testing.assert_close(d['target_total_fz_n'],values[5])
    assert (tau[:,:,0:2]>0).all()


def test_downward_normal_force_increases_required_upward_tangent_support():
    values=list(inputs());_,baseline=module().support_torques(*values)
    values[3]=torch.full((1,2),-.5,dtype=torch.float64)
    _,compensated=module().support_torques(*values)
    assert (compensated['target_tangent_z_n']>baseline['target_tangent_z_n']).all()
    torch.testing.assert_close(compensated['measured_total_fz_n'],torch.tensor([-1.],dtype=torch.float64))


def test_no_measured_contact_or_no_lift_request_produces_no_support_torque():
    values=list(inputs());values[2].zero_();tau,_=module().support_torques(*values)
    assert not tau.any()
    values=list(inputs());values[-1]=torch.tensor([False]);tau,_=module().support_torques(*values)
    assert not tau.any()


def test_tangential_wrench_has_no_contact_normal_component():
    values=list(inputs());n=values[1].clone();n[:,:,:,0]*=(1-.25**2)**.5;n[:,:,:,2]=.25;values[1]=n
    _,d=module().support_torques(*values)
    assert torch.allclose((d['digit_tangent_force_w_n']*n).sum(-1),torch.zeros(1,2,2,dtype=torch.float64),atol=1e-12)


def test_torque_direction_matches_virtual_work_and_does_not_exceed_total_cap():
    values=list(inputs());values[0]*=100
    tau,d=module().support_torques(*values,effort_limit_nm=.2)
    assert tau.abs().max()<=.2 and d['torque_saturated'].any()
    assert (tau[:,:,0:2]>0).all()


def test_traction_request_respects_measured_normal_load():
    values=list(inputs());values[2].fill_(.1)
    _,d=module().support_torques(*values)
    assert (d['digit_tangent_force_w_n'].norm(dim=-1)<=.5*values[2]+1e-12).all()


def test_invalid_force_measurement_fails_before_actuation():
    values=list(inputs());values[3][0,0]=float('nan')
    with pytest.raises(ValueError):module().support_torques(*values)


def test_continuous_servo_holds_force_across_steps_and_corrects_disturbance():
    m=module();servo=m.SupportServo(1,'cpu');values=inputs()
    requested=torch.tensor([[3.,8.]],dtype=torch.float64)
    torques=[]
    for _ in range(40):
        tau,d=servo.step(*values,requested_normal_n=requested,dt=1/480)
        torques.append(tau)
    assert (torch.stack(torques)[-10:].abs().sum((1,2,3))>0).all()
    torch.testing.assert_close(d['requested_normal_n'],requested)
    assert d['normal_force_error_n'][0,1]>d['normal_force_error_n'][0,0]
    disturbed=list(values);disturbed[4]=torch.full((1,2),-.3,dtype=torch.float64)
    before=d['requested_tangent_with_integral_n'].clone()
    for _ in range(20):_,d=servo.step(*disturbed,requested_normal_n=requested,dt=1/480)
    assert (d['requested_tangent_with_integral_n']>before).all()


def test_servo_reset_isolates_selected_candidates():
    m=module();servo=m.SupportServo(2,'cpu');v=inputs()
    batch=[x.repeat(2,*([1]*(x.ndim-1))) for x in v]
    goal=torch.full((2,2),6.,dtype=torch.float64)
    for _ in range(4):servo.step(*batch,requested_normal_n=goal,dt=1/480)
    before=servo.normal_integral_n[1].clone();servo.reset(torch.tensor([0]))
    assert not servo.normal_integral_n[0].any()
    torch.testing.assert_close(servo.normal_integral_n[1],before)


def test_lost_digit_cannot_keep_its_old_torque_while_other_digit_bears():
    servo=module().SupportServo(1,'cpu');values=list(inputs())
    requested=torch.full((1,2),5.,dtype=torch.float64)
    for _ in range(8):servo.step(*values,requested_normal_n=requested,dt=1/480)
    assert (servo.previous_torque_nm[:,:,0]>0).all()
    values[2][:,:,0]=0.
    torque,_=servo.step(*values,requested_normal_n=requested,dt=1/480)
    assert not torque[:,:,0].any() and (torque[:,:,1]>0).all()


def test_bearing_digit_without_valid_actual_contact_model_cannot_actuate():
    servo=module().SupportServo(1,'cpu');values=inputs();requested=torch.full((1,2),5.,dtype=torch.float64)
    valid=torch.ones(1,2,2,dtype=torch.bool)
    for _ in range(8):servo.step(*values,requested_normal_n=requested,dt=1/480,contact_model_valid=valid)
    valid[:,:,0]=False
    torque,diagnostic=servo.step(*values,requested_normal_n=requested,dt=1/480,contact_model_valid=valid)
    assert not torque[:,:,0].any() and (torque[:,:,1]>0).all()
    assert not diagnostic['measured_contact'][:,:,0].any()


def test_reachable_high_force_command_is_not_blocked_by_small_integral_cap():
    # Static contact plant: 30mm moment arm, 0.5N preload per finger.
    # 6N/finger is reachable with 0.165Nm, below the real 0.2Nm motor cap.
    values=list(inputs());values[0].zero_()
    values[0][:,:,0,0,0]=.03;values[0][:,:,1,0,1]=-.03
    values[-1].fill_(False);values[2].fill_(.5)
    servo=module().SupportServo(1,'cpu');requested=torch.full((1,2),12.,dtype=torch.float64)
    for _ in range(2400):
        torque,_=servo.step(*values,requested_normal_n=requested,dt=1/480)
        values[2]=.5+torque[:,:,:2]/.03
    assert ((values[2].sum(-1)-requested).abs()<=.05*requested).all()
    assert torque.abs().max()<=.2


@pytest.mark.parametrize('position_only',[False,True])
def test_installed_servo_updates_each_physics_step_and_shares_motor_budget(position_only):
    from types import SimpleNamespace
    m=module();values=inputs();j,n,loads,nz,fz,weight,vz,enabled=values
    class Hand:
        def write_joint_effort_limit_to_sim(self,value,**kwargs):self.budget=value.clone()
        def set_joint_effort_target(self,value,**kwargs):self.effort=value.clone()
    hands=[Hand(),Hand()];calls=[]
    task=SimpleNamespace(n=1,device='cpu',done=torch.tensor([False]),f_requested=torch.tensor([[3.,8.]],dtype=torch.float64))
    raw=torch.zeros(1,2,6,dtype=torch.float64);raw[:,:,:2]=loads
    world=SimpleNamespace(n=1,device='cpu',dt=1/480,hands=hands,w=SimpleNamespace(ids=[[0,1,2,3,4,5]]*2),
        parameters=SimpleNamespace(effort_limit_nm=.2),object_weight_n=weight,wv=torch.zeros(1,6,dtype=torch.float64),
        vertical_components=torch.stack((nz,fz),-1))
    world.measure=lambda:dict(grouped_object_loads=raw,hand_object_loads=raw.sum(-1))
    def advance(*args):
        assert hasattr(hands[0],'effort');calls.append(hands[0].effort.clone());return world.measure()
    world.advance=advance;world.reset=lambda ids:{'reset':ids.tolist()}
    jac=torch.zeros(1,2,2,3,12,dtype=torch.float64);jac[...,6:]=j
    controller=SimpleNamespace(pinch_hold=position_only,last=dict(jacobian=jac,inward=n,lift_requested=enabled,
        contact_model_valid=torch.ones(1,2,2,dtype=torch.bool)),refreshes=0)
    def refresh(measurement):
        assert measurement['hand_object_loads'].shape==(1,2)
        controller.refreshes+=1
    controller.refresh_contact_model=refresh
    servo=m.install_support_servo(world,task,controller)
    for tick in range(12):
        if position_only and tick==8:raw[:,:,0]=0.
        world.advance(task,torch.tensor([True]),torch.tensor([True]))
    assert len(calls)==12
    if position_only:
        assert not calls[-1].any()
        for hand in hands:torch.testing.assert_close(hand.budget,torch.full_like(hand.budget,.2))
        torch.testing.assert_close(controller.force_feedback_loads[:,:,0],torch.full((1,2),1.25,dtype=torch.float64))
        assert not world.measure()['grouped_object_loads'][:,:,0].any(),'raw sensor data must remain unchanged'
    else:assert calls[-1].abs().sum()>0
    assert controller.refreshes==12,'actual contact model must refresh at each physics step'
    for hand in hands:assert (hand.budget+hand.effort.abs()<=.2+1e-7).all()
    assert world.surface_support_diagnostic['requested_normal_n'].tolist()==[[3.,8.]]
    world.reset(torch.tensor([0]))
    assert not servo.normal_integral_n.any()
