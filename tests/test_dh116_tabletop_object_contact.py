import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_object_contact import aggregate_object_contacts, validate_object_contacts

LINKS=('finger14_link','finger23_link','finger32_link','palm_link')
def test_nonpad_opposite_vectors_aggregate_without_cancellation():
    vectors=np.zeros((3,2,4,3));vectors[:,0,2,0]=2;vectors[:,1,2,0]=-2
    m=aggregate_object_contacts(vectors,LINKS)
    assert np.all(m['hand_object_loads']==2)
    assert np.all(m['grouped_object_loads'][:,:,2]==2)
    validate_object_contacts(m,LINKS,batch_size=3)

@pytest.mark.parametrize('kind',['missing','negative','nonfinite','inconsistent'])
def test_bad_measurements_rejected(kind):
    m=aggregate_object_contacts(np.zeros((2,4,3)),LINKS)
    if kind=='missing':m.pop('object_link_loads')
    elif kind=='negative':m['object_link_loads'][0,0]=-1
    elif kind=='nonfinite':m['hand_object_loads'][0]=np.nan
    else:m['hand_object_loads'][0]=1
    with pytest.raises(ValueError):validate_object_contacts(m,LINKS)

def test_duplicate_links_rejected():
    with pytest.raises(ValueError):aggregate_object_contacts(np.zeros((2,2,3)),('finger14_link',)*2)

def test_full_hand_controller_rejects_legacy_shape_and_uses_nonpad_load():
    from go2_pvcnn.control.dh116.tabletop_force_feedback import ReactionForceController
    c=ReactionForceController(full_hand_contact=True)
    c.update(action=-np.ones(2),forces=np.full(2,.1),blocked=np.zeros(2),slip_m=0,phase='close',dt=1/480)
    assert c.ready(np.full(2,.1),speed=0,slip=0)
    with pytest.raises(ValueError):c.ready(np.zeros(4),speed=0,slip=0)


def free_env():
    from test_dh116_batched_task import Backend
    from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv
    b=Backend(force=0.)
    b.full_hand_contact=True;b.object_contact_links=LINKS
    original=b.measure
    def measure():
        m=original();vectors=np.zeros((2,4,3));vectors[:,2,0]=.1+b.t
        m.update(aggregate_object_contacts(vectors,LINKS));return m
    b.measure=measure
    e=TabletopEnv(b,stage='full',reset_phase='close',control_mode='adaptive_grasp',grasp_mode='object_stability',initial_offset=.004,reset_jitter=0,force_direction=np.ones((2,6))/np.sqrt(6))
    e.reset(seed=11);return e


def test_nonpad_scalar_batch_observation_reward_and_gate_parity():
    from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask
    scalar=free_env();fast=free_env()
    for _ in range(20):
        action=np.r_[np.zeros(18),-1.,-1.]
        expected=scalar.step(action)
        batch=BatchTask([fast])
        def advance(offset,command,allowed,live):
            fast.backend.advance(offset[0],command[0],palm_motion_allowed=bool(allowed[0]),palm_action=batch.palm_actions[0],grip_action=np.ones(2))
            return batch.stack_measurements([fast.backend.measure()])
        actual=batch.step(action[None],advance)[0]
        assert actual[0].shape==(148,)
        np.testing.assert_allclose(actual[0],expected[0],atol=1e-6)
        assert actual[1]==pytest.approx(expected[1],abs=1e-10)
        assert fast.task.phase==scalar.task.phase
        assert np.all(fast.measurement['forces']==0)
    assert batch.terms['contact'][0]>0

def test_reset_clears_selected_slots_only():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_object_contact import clear_object_contact_slots
    vectors=np.ones((3,2,4,3))
    owner=SimpleNamespace(object_link_vectors=vectors,**aggregate_object_contacts(vectors,LINKS))
    clear_object_contact_slots(owner,[1])
    for a in vars(owner).values():
        assert np.all(a[1]==0)
        assert np.all(a[[0,2]]>0) if a.shape[-1]!=6 else np.any(a[[0,2]]>0)

def test_same_hand_opposed_links_do_not_cancel():
    vectors=np.zeros((2,4,3));vectors[0,0,0]=2.;vectors[0,2,0]=-2.
    m=aggregate_object_contacts(vectors,LINKS)
    assert m['hand_object_loads'][0]==4.
    assert m['grouped_object_loads'][0,0]==2.
    assert m['grouped_object_loads'][0,2]==2.


def test_nonpad_local_overforce_fails_scalar_and_batch_with_zero_pad_loads():
    from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask
    scalar=free_env();fast=free_env()
    for e in (scalar,fast):
        original=e.backend.measure
        def measure(original=original):
            m=original();vectors=np.zeros((2,4,3));vectors[0,2,0]=31.
            m.update(aggregate_object_contacts(vectors,LINKS));return m
        e.backend.measure=measure
    action=np.r_[np.zeros(18),-1.,-1.]
    expected=scalar.step(action)
    batch=BatchTask([fast])
    def advance(offset,command,allowed,live):
        fast.backend.advance(offset[0],command[0],palm_motion_allowed=bool(allowed[0]),palm_action=batch.palm_actions[0],grip_action=np.ones(2))
        return batch.stack_measurements([fast.backend.measure()])
    actual=batch.step(action[None],advance)[0]
    assert scalar.task.failure==fast.task.failure=='excess_force'
    assert expected[2] and actual[2]
    assert scalar.metrics['max_force_n']==fast.metrics['max_force_n']==31.
    assert np.all(fast.metrics['pad_forces_n']==0)
    assert actual[1]==pytest.approx(expected[1],abs=1e-10)

@pytest.mark.parametrize('kind',['nan','negative','inconsistent'])
def test_bad_contact_row_terminates_only_that_slot_like_scalar(kind):
    from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask
    scalar_bad=free_env();scalar_good=free_env();fast=[free_env(),free_env()]
    def corrupt(e):
        original=e.backend.measure
        def measure():
            m=original()
            if kind=='nan':m['object_link_loads'][0,2]=np.nan
            elif kind=='negative':m['object_link_loads'][0,2]=-1.
            else:m['hand_object_loads'][0]+=1.
            return m
        e.backend.measure=measure
    corrupt(scalar_bad);corrupt(fast[0])
    action=np.r_[np.zeros(18),-1.,-1.]
    expected=[scalar_bad.step(action),scalar_good.step(action)]
    batch=BatchTask(fast)
    def advance(offset,command,allowed,live):
        for i,e in enumerate(fast):
            if live[i]:e.backend.advance(offset[i],command[i],palm_motion_allowed=bool(allowed[i]),palm_action=batch.palm_actions[i],grip_action=np.ones(2))
        return batch.stack_measurements([e.backend.measure() for e in fast])
    actual=batch.step(np.tile(action,(2,1)),advance)
    assert fast[0].task.failure==scalar_bad.task.failure=='nonfinite_state'
    assert fast[1].task.failure is None
    for x,y in zip(actual,expected):
        np.testing.assert_allclose(x[0],y[0],atol=1e-6)
        assert x[1]==pytest.approx(y[1],abs=1e-10)
        assert x[2]==y[2]
