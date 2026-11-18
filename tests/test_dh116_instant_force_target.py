import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
from go2_pvcnn.control.dh116.tabletop_force_feedback import ReactionForceController, force_feedback_contract
from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask
from test_dh116_tensor_task import template, measurement


def instant_parameters():
    return LoadParameters(max_force_n=12.,force_rate_n_s=None)


def test_immediate_scalar_target_and_preserved_offset_rate():
    c=ReactionForceController(instant_parameters())
    for action,expected in (([1.,-1.],[12.,.1]),([-1.,1.],[.1,12.])):
        previous=c.offset.copy()
        c.update(action=action,forces=[0.,0.,0.,0.],blocked=[False,False],slip_m=0.,phase='close',dt=1/480)
        np.testing.assert_allclose(c.target,expected)
        assert np.max(abs(c.offset-previous))<=.03/480+1e-12
    assert force_feedback_contract(instant_parameters())['force_feedback']['target_rate_n_s'] is None
    assert 'force_rate_n_s' in instant_parameters().as_dict()
    with pytest.raises(ValueError):LoadParameters(effort_limit_nm=None)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_immediate_tensor_batch_parity_inactive_rows_and_real_readiness(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    e=template();e.force_controller=ReactionForceController(instant_parameters(),full_hand_contact=True)
    t=TensorTask(e,2,device,contact_reward_window='close_start');b=BatchTask([e,e])
    live=np.array([True,False]);tlive=torch.tensor(live,device=device)
    for force_action in (1.,-1.,0.,1.):
        a=np.zeros((2,20));a[:,18:]=force_action
        previous=t.command.clone()
        t.control(torch.tensor(a,device=device),tlive,1/480);b.control(a,live,1/480)
        expected=2.+(force_action+1)*.5*10.
        np.testing.assert_allclose(t.f_target[0].cpu(),[expected]*2)
        np.testing.assert_allclose(t.f_target[1].cpu(),[2.]*2)
        for name in ('f_target','f_offset','f_requested','command'):
            np.testing.assert_allclose(getattr(t,name).cpu(),getattr(b,name),atol=1e-12)
        assert float((t.command-previous).abs().max())<=.20/480+1e-12
    # A large requested target is not evidence of an established grasp.
    m=measurement(t);m['hand_object_loads'].zero_();m['object_link_loads'].zero_()
    t.begin_step(torch.zeros((2,20),device=device))
    for _ in range(60):t.consume(m,tlive,.01)
    assert t.phase[0].item()==1 and not t.success[0]
    t.reset([0],m)
    assert t.f_target[0].tolist()==[2.,2.]


def test_exact_immediate_target_checkpoint_migration():
    old=dict(force_feedback_contract(LoadParameters(max_force_n=12.,force_rate_n_s=2.)),num_envs=4096)
    new=dict(force_feedback_contract(instant_parameters()),num_envs=4096)
    assert checkpoint_contract_changes(old,new)==['immediate_force_target_v1']
    assert checkpoint_contract_changes(new,new)==[]
    for bad in (dict(new,num_envs=16),dict(new,active_effort_limit_nm=.5),
                dict(new,load_parameters=dict(new['load_parameters'],max_force_n=15.))):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)
