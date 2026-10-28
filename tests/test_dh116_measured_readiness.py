"""Stable physical support must not be reset by stochastic force-goal exploration."""
import pytest
import torch
from test_dh116_tensor_task import template,measurement
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters


def task():
    e=template();e.force_controller.parameters=LoadParameters(max_force_n=12.,force_rate_n_s=None)
    return TensorTask(e,1,'cpu',pressure_relief=True,contact_reward_window='close_start',
                      grasp_readiness='measured_load')


def test_changing_force_requests_does_not_reset_measured_stable_grasp():
    t=task();t.begin_step(torch.zeros((1,20)));m=loads(t,4.)
    for k in range(240):
        t.f_requested[:]=2. if (k//8)%2 else 12.
        t.consume(m,~t.done,1/480)
        if k==94:assert t.phase.item()==1
        if k==95:assert t.phase.item()==2
        if k==238:assert t.phase.item()==2
    assert t.phase.item()==3 and t.terms['grasp_ready'].item()==3.
    assert not t.success.item()  # Object still rests on the table.
    assert t.contact_reward_paid.item()==pytest.approx(1.)
    for _ in range(240):t.consume(m,~t.done,1/480)
    assert t.terms['grasp_ready'].item()==3. and t.contact_reward_paid.item()==pytest.approx(2.)
    for _ in range(240):t.consume(m,~t.done,1/480)
    assert t.contact_reward_paid.item()==pytest.approx(2.)


@pytest.mark.parametrize('condition',['one_hand','weak','overforce','moving','slipping'])
def test_unstable_or_insufficient_grip_resets_dwell(condition):
    t=task();t.begin_step(torch.zeros((1,20)));m=loads(t,4.)
    for _ in range(120):t.consume(m,~t.done,1/480)
    assert t.phase.item()==2
    if condition=='one_hand':
        m['object_link_loads'][:,1]=0.;m['hand_object_loads'][:,1]=0.;m['grouped_object_loads'][:,1]=0.
    elif condition=='weak':m=loads(t,1.8)
    elif condition=='overforce':m=loads(t,32.)
    elif condition=='moving':m['object_velocity'][:,0]=.021
    elif condition=='slipping':m['object_relative'][:,0]+=.004
    t.consume(m,~t.done,1/480)
    assert t.phase.item()==1 and t.contact_time.item()==0.
    assert not t.grasp_ready_awarded.item()


def test_lift_support_does_not_require_tracking_new_random_force_goal():
    t=task();m=loads(t,4.);t.m=m;t.phase[:]=3;t.command.copy_(m['q'])
    t.f_requested[:]=12.;t.f_target[:]=12.
    a=torch.zeros((1,20),dtype=torch.float64);a[:,18:]=1.
    before=t.offset.clone();t.control(a,~t.done,1/480)
    assert bool((t.offset>before).all())
    # Real contact loss and pressure relief still stop lift.
    t.m=loads(t,0.);before=t.offset.clone();t.control(a,~t.done,1/480)
    assert torch.equal(t.offset,before)
    # The minimum action requests 2 N; 3 N per hand exceeds its 2.52 N band.
    t.m=loads(t,6.);a[:,18:]=-1.;before=t.offset.clone();t.control(a,~t.done,1/480)
    assert t.f_target.tolist()==[[2.,2.]] and t.pressure_limited.all()
    assert torch.equal(t.offset,before)


def test_new_readiness_contract_is_exact_and_one_way():
    from go2_pvcnn.control.dh116.tabletop_tensor_task import MEASURED_READINESS_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    old=dict(physics_hz=480,num_envs=4096)
    new=dict(old,grasp_readiness=MEASURED_READINESS_CONTRACT)
    assert checkpoint_contract_changes(old,new)==['measured_hand_load_readiness_v1']
    for wrong in (dict(new,physics_hz=240),dict(new,grasp_readiness=dict(MEASURED_READINESS_CONTRACT,min_hand_load_n=.01))):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,wrong)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)


def test_invalid_readiness_mode_rejected():
    with pytest.raises(ValueError,match='readiness'):TensorTask(template(),1,'cpu',grasp_readiness='unsupported')


@pytest.mark.parametrize('old_target,new_target,should_lift',[(4.,2.,False),(2.,4.,True)])
def test_lift_and_pressure_release_use_same_current_force_target(old_target,new_target,should_lift):
    t=task();t.m=loads(t,6.);t.phase[:]=3;t.command.copy_(t.m['q'])
    t.f_target[:]=old_target;t.f_requested[:]=old_target
    a=torch.zeros((1,20),dtype=torch.float64)
    a[:,18:]=2*(new_target-2.)/10.-1
    before=t.offset.clone();allowed=t.control(a,~t.done,1/480)
    assert t.f_target.tolist()==[[new_target,new_target]]
    assert bool(t.offset[0]>before[0])==should_lift
    assert bool(allowed[0])==should_lift
    assert bool(t.pressure_limited.any())!=should_lift
