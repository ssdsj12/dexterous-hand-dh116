import pytest
import torch
from test_dh116_tensor_task import template, measurement
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask


def task(device='cpu'):
    e=template();e.force_controller.parameters=LoadParameters(max_force_n=12.,force_rate_n_s=None)
    return TensorTask(e,2,device,pressure_relief=True,contact_reward_window='close_start',
        grasp_readiness='measured_load',vertical_force_reward=True,vertical_force_reward_mode='continuous',
        force_reward_start='downward_after_close',skip_hold=True)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_close_direct_lift_timing_reward_and_reset(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=task(device);t.begin_step(torch.zeros((2,20),device=device));m=loads(t,4.)
    force=torch.tensor([2.,-2.],device=device)
    for _ in range(95):t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.phase.tolist()==[1,1] and not t.force_scoring_started.any()
    t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.phase.tolist()==[3,1] and not t.force_scoring_started.any()
    assert t.grip_valid[0] and not t.grip_valid[1] and not t.success.any()
    assert t.terms['contact'][0].item()==pytest.approx(.2)
    assert t.terms['downward_force'][1].item()==0.
    assert t.terms['grasp_ready'].tolist()==[3.,0.]
    t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.force_scoring_started[0] and not t.force_scoring_started[1]
    assert t.terms['downward_force'][1].item()==0.
    assert t.terms['grasp_ready'].tolist()==[3.,0.]
    t.reset([1],measurement(t))
    assert t.phase.tolist()==[3,1] and t.force_scoring_started.tolist()==[True,False]
    assert t.observe().shape==(2,151)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_downward_force_cannot_admit_direct_lift(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=task(device);t.begin_step(torch.zeros((2,20),device=device));m=loads(t,4.)
    force=torch.tensor([-1.,1.],device=device)
    for _ in range(96):
        t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.phase.tolist()==[1,3]


@pytest.mark.parametrize('bad',['one_hand','moving'])
def test_unstable_close_must_rebuild_full_dwell(bad):
    t=task();t.begin_step(torch.zeros((2,20)));m=loads(t,4.);force=torch.ones(2)
    for _ in range(95):t.consume(m,~t.done,1/480,vertical_force_n=force)
    broken={k:v.clone() for k,v in m.items()}
    if bad=='one_hand':
        for key in ('hand_object_loads','object_link_loads','grouped_object_loads'):
            broken[key][:,1]=0
    else:broken['object_velocity'][:,0]=.03
    t.consume(broken,~t.done,1/480,vertical_force_n=force)
    assert not t.failure.any()
    assert t.phase.tolist()==[1,1] and not t.contact_time.any()
    for _ in range(95):t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.phase.tolist()==[1,1]
    t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.phase.tolist()==[3,3]


def test_skip_hold_contract_migration_is_exact_and_one_way():
    from go2_pvcnn.control.dh116.tabletop_tensor_task import measured_readiness_contract
    from go2_pvcnn.control.dh116.tabletop_vertical_force import vertical_reward_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    def contract(skip):
        v=vertical_reward_contract('continuous','downward_after_close',skip_hold=skip)
        return dict(reward=v['version'],phase_reward_version=v['version'],vertical_force_reward=v,
            grasp_readiness=measured_readiness_contract(skip),observation_size=151,num_envs=4096,
            observation_version='dh116-after-close-force-v3')
    old,new=contract(False),contract(True)
    assert checkpoint_contract_changes(old,new)==['close_direct_lift_v1']
    assert new['grasp_readiness']['hold_dwell_s']==0.
    assert 'close_to_lift' in new['vertical_force_reward']['force_reward_activation']
    for a,b in [(new,old),(old,dict(new,num_envs=16)),(old,dict(new,grasp_readiness=dict(new['grasp_readiness'],close_dwell_s=.01)))]:
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(a,b)
