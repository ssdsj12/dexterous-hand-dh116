import numpy as np
import pytest
import torch
from types import SimpleNamespace
from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_heights as reference_heights
from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_points as reference_points
from go2_pvcnn.control.dh116.tabletop_tensor_world import (
    contact_geometry_from_sensors,
    rotated_heights,
    rotated_points,
    signed_hand_normal_vertical_force,
    signed_hand_vertical_force,
)


def test_contact_geometry_from_sensors_uses_physx_contact_positions():
    def sensor(position, force):
        return SimpleNamespace(data=SimpleNamespace(
            contact_pos_w=torch.tensor([[[position]]], dtype=torch.float64),
            force_matrix_w=torch.tensor([[[force]]], dtype=torch.float64)))

    sensors = [
        sensor([.008, 0., -.004], [3., 0., 0.]),
        sensor([-.006, 0., -.003], [0., 0., 4.]),
        sensor([-.008, 0., -.004], [-3., 0., 0.]),
        sensor([.006, 0., -.003], [0., 0., 4.]),
    ]
    result = contact_geometry_from_sensors(sensors, torch.zeros((1, 3), dtype=torch.float64))
    assert result['contact_count'].tolist() == [[2, 2]]
    assert result['contact_center_w'][0, 0, 2].item() == pytest.approx(-.0034285714)
    assert result['all_below_centerline'].tolist() == [[True, True]]


def test_geometry_matches_exact_hull_reference():
    rng = np.random.default_rng(42)
    pose = rng.normal(size=(32,7)); pose[:,3:] /= np.linalg.norm(pose[:,3:],axis=1)[:,None]
    points = rng.normal(size=(125,3))*.03
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        actual = rotated_heights(torch.tensor(pose,device=device),torch.tensor(points,device=device))
        assert actual.device.type == device
        np.testing.assert_allclose(actual.cpu(),reference_heights(pose,points),atol=1e-12,rtol=0)
        actual = rotated_points(torch.tensor(pose,device=device),torch.tensor(points[0],device=device))
        np.testing.assert_allclose(actual.cpu(),reference_points(pose,points[:1])[:,0],atol=1e-12,rtol=0)


def test_signed_hand_vertical_force_sums_normal_and_friction_components():
    components = torch.tensor([[[1., -2.], [3., -4.]]], dtype=torch.float64)
    torch.testing.assert_close(
        signed_hand_vertical_force(components),
        torch.tensor([[-1., -1.]], dtype=torch.float64),
    )


def test_signed_hand_normal_vertical_force_ignores_friction_components():
    components = torch.tensor([[[1., -20.], [3., -40.]]], dtype=torch.float64)
    torch.testing.assert_close(
        signed_hand_normal_vertical_force(components),
        torch.tensor([[1., 3.]], dtype=torch.float64),
    )


@pytest.mark.parametrize('components', [
    torch.zeros((2, 2, 3), dtype=torch.float64),
    torch.zeros((2, 2), dtype=torch.float64),
    torch.tensor([[[1., float('nan')], [3., 4.]]], dtype=torch.float64),
])
def test_signed_hand_normal_vertical_force_rejects_malformed_or_nonfinite_components(components):
    with pytest.raises(ValueError):
        signed_hand_normal_vertical_force(components)


def palm_world():
    """Exercise the real target/advance path without creating a PhysX instance."""
    from test_dh116_root_pose_writes import world
    w=world();w.n=3;w.dt=1/480
    w.palm_limit=torch.tensor([.003,.003,.004],dtype=torch.float64)
    w.palms=torch.zeros((3,2,3),dtype=torch.float64)
    w.clearances=torch.full((3,2),.003,dtype=torch.float64)
    w.gains=torch.full((3,2),10.,dtype=torch.float64)
    w.parameters=SimpleNamespace(stiffness_rate=10.,stiffness_max=20.)
    w.root_seed=w._root_pose_written.double().clone()
    # The authored bases differ in height; synchronization is of exploration
    # offsets, not an attempt to overwrite the authored hand geometry.
    w.root_seed[:,:,2]=torch.tensor([.429,.419],dtype=torch.float64)
    w.root_targets=w.root_seed.clone();w.roots=w.root_seed.clone()
    w.commands=torch.zeros((3,2,6),dtype=torch.float64)
    w.refresh=lambda: None
    w.w=SimpleNamespace(ids=[list(range(6))]*2,args=SimpleNamespace(headless=True),
                        sim=SimpleNamespace(step=lambda **kw: None))
    for hand in w.hands:
        hand.set_joint_position_target=lambda *a,**kw: None
        hand.write_joint_stiffness_to_sim=lambda *a,**kw: None
        hand.write_data_to_sim=lambda: None
        hand.actuators={'active':SimpleNamespace(stiffness=torch.zeros((3,6)))}
    task=SimpleNamespace(palm_actions=torch.zeros((3,2,3),dtype=torch.float64),
        offset=torch.zeros(3,dtype=torch.float64),command=torch.zeros((3,12),dtype=torch.float64),
        coordinated_control=torch.ones(3,dtype=torch.bool),palm_descent_clearance_m=.0003)
    return w,task


def test_world_shared_z_uses_the_stricter_hands_measured_clearance():
    w,task=palm_world();task.palm_actions[:,:,2]=-1.
    w.clearances[:,0]=.0003
    w.advance(task,torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(w.palms[:,:,2],torch.zeros((3,2),dtype=torch.float64))
    torch.testing.assert_close(w.root_targets[:,:,2]-w.root_seed[:,:,2],w.palms[:,:,2])


@pytest.mark.parametrize('allowed', [False,True])
def test_world_cancels_existing_descent_target_backlog_at_clearance_floor(allowed):
    w,task=palm_world();task.palm_actions[:,:,2]=-1.
    w.palms[:,:,2]=-.002;w.root_targets[:,:,2]-=.002
    # Wrist drivers have reached only -1 mm; their queued target is -2 mm.
    w.roots[:,:,2]-=.001;w.clearances[:,0]=.0003
    w.advance(task,torch.full((3,),allowed),torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(w.palms[:,:,2],torch.full((3,2),-.001,dtype=torch.float64))
    torch.testing.assert_close(w.root_targets[:,:,2],w.roots[:,:,2])


@pytest.mark.parametrize('clearance', [.0001,-.00036])
@pytest.mark.parametrize('limiting_hand', [0,1])
@pytest.mark.parametrize('allowed', [False,True])
def test_world_restores_clearance_below_floor_during_held_lift(clearance,limiting_hand,allowed):
    w,task=palm_world();task.phase=torch.full((3,),3,dtype=torch.long)
    task.offset[:]=.0040625
    held_palm=-.0003508021189448929
    task.palm_actions[:,:,2]=held_palm/w.palm_limit[2]
    w.palms[:,:,2]=held_palm
    w.root_targets[:,:,2]+=task.offset[:,None]+held_palm
    w.roots[:,:,2]+=task.offset[:,None]-.0001
    w.clearances[:]=.0011;w.clearances[:,limiting_hand]=clearance
    task.command[:]=.4;w.commands[:]=task.command.reshape(3,2,6)
    before_target=w.root_targets.clone();before_roots=w.roots.clone()
    before_offset=task.offset.clone();before_actions=task.palm_actions.clone()
    before_command=task.command.clone()
    w.advance(task,torch.full((3,),allowed),torch.tensor([True,True,False]))
    # Restore only the deficient hand, even while the lift/pressure governor
    # freezes ordinary palm motion. Its partner retains its accepted target.
    expected_z=before_target[:2,:,2].clone()
    expected_z[:,limiting_hand]=before_roots[:2,limiting_hand,2]+.0003-clearance
    torch.testing.assert_close(w.root_targets[:2,:,2],expected_z)
    torch.testing.assert_close(w.root_targets[:2,1-limiting_hand,2],
                               before_target[:2,1-limiting_hand,2])
    torch.testing.assert_close(w.palms[:2,:,2],
        expected_z-w.root_seed[:2,:,2]-task.offset[:2,None])
    torch.testing.assert_close(w.root_targets[2],before_target[2])
    torch.testing.assert_close(w.root_targets[:,:,:2],before_target[:,:,:2])
    torch.testing.assert_close(task.offset,before_offset)
    torch.testing.assert_close(task.palm_actions,before_actions)
    torch.testing.assert_close(task.command,before_command)
    torch.testing.assert_close(w.commands,before_command.reshape(3,2,6))
    assert task.phase.tolist()==[3,3,3]


@pytest.mark.parametrize('allowed', [False,True])
def test_world_clearance_restoration_target_does_not_ratchet_during_wrist_lag(allowed):
    w,task=palm_world();task.palm_actions[:,:,2]=-1.
    w.palms[:,:,2]=-.002;w.root_targets[:,:,2]-=.002
    w.roots[:,:,2]-=.001
    w.clearances[:,0]=-.00036;w.clearances[:,1]=.0011
    w.advance(task,torch.full((3,),allowed),torch.ones(3,dtype=torch.bool))
    target=w.root_targets.clone()
    torch.testing.assert_close(target[:,:,2]-w.roots[:,:,2],
                               torch.full((3,2),.00066,dtype=torch.float64))
    # With unchanged fingers, measured clearance rises by the same amount as
    # the driven bases. The resulting absolute safe target must stay fixed.
    for rise in (.0001,.0002,.00036):
        w.roots[:,:,2]+=rise;w.clearances+=rise
        w.advance(task,torch.full((3,),allowed),torch.ones(3,dtype=torch.bool))
        torch.testing.assert_close(w.root_targets,target)


def test_world_clearance_projection_preserves_common_lift_and_authored_root_heights():
    w,task=palm_world();w.palms[:,:,2]=-.001
    # Existing 10 mm lift and -1 mm exploration; request a further 0.02 mm lift.
    w.root_targets[:,:,2]+=.009;w.roots[:,:,2]+=.009
    task.offset[:]=.01002;task.palm_actions[:,:,2]=-.25;w.clearances[:]=.0003
    w.advance(task,torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(w.palms[:,:,2],torch.full((3,2),-.001,dtype=torch.float64))
    torch.testing.assert_close(w.root_targets[:,:,2]-w.roots[:,:,2],
                               torch.full((3,2),.00002,dtype=torch.float64))


def test_world_shared_clearance_projection_does_not_change_xy_or_inactive_rows():
    w,task=palm_world();task.palm_actions[:]=-.5;w.clearances[:,0]=.0003
    before=w.root_targets[2].clone()
    w.advance(task,torch.ones(3,dtype=torch.bool),torch.tensor([True,True,False]))
    torch.testing.assert_close(w.palms[:2,:,:2],torch.full((2,2,2),-.002/480,dtype=torch.float64))
    torch.testing.assert_close(w.palms[2],torch.zeros((2,3),dtype=torch.float64))
    torch.testing.assert_close(w.root_targets[2],before)


@pytest.mark.parametrize('allowed', [False,True])
@pytest.mark.parametrize('waiting_hand', [0,1])
def test_world_release_hold_cancels_descent_backlog_with_ample_clearance(allowed,waiting_hand):
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    w,task=palm_world();task.phase=torch.ones(3,dtype=torch.long)
    task.palm_reposition=LoadedPalmReposition(torch.ones(3,dtype=torch.bool))
    task.palm_reposition.state[:]=2;task.palm_reposition.state[:,waiting_hand]=1
    task.palm_reposition.hold[:,:,2]=-.001
    # All offsets are relative to the authored roots plus an existing 10 mm
    # common lift. The controller now holds -1 mm, but -2 mm is still queued.
    task.offset[:]=.01;task.palm_actions[:,:,2]=-.25;w.palms[:,:,2]=-.002
    w.roots[:,:,2]+=.009;w.root_targets[:,:,2]+=.008;w.clearances[:]=.006
    before=w.root_targets[2].clone()
    w.advance(task,torch.full((3,),allowed),torch.tensor([True,True,False]))
    torch.testing.assert_close(w.palms[:2,:,2],torch.full((2,2),-.001,dtype=torch.float64))
    torch.testing.assert_close(w.root_targets[:2,:,2],w.roots[:2,:,2])
    torch.testing.assert_close(w.root_targets[2],before)
    torch.testing.assert_close(w.palms[2,:,2],torch.full((2,),-.002,dtype=torch.float64))


def test_world_stale_release_state_does_not_override_lift_or_disabled_reposition():
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    w,task=palm_world();task.phase=torch.tensor([3,1,1])
    task.palm_reposition=LoadedPalmReposition(torch.tensor([True,False,True]))
    task.palm_reposition.state[:,0]=1;task.palm_reposition.hold[:,:,2]=-.001
    task.offset[:]=.01;task.offset[0]+=.00002;task.palm_actions[:,:,2]=-.25
    w.palms[:,:,2]=-.002;w.roots[:,:,2]+=.009;w.root_targets[:,:,2]+=.008
    w.clearances[:]=.006;before=w.root_targets[2].clone()
    w.advance(task,torch.ones(3,dtype=torch.bool),torch.tensor([True,True,False]))
    expected=torch.full((2,2),-.002+.002/480,dtype=torch.float64)
    torch.testing.assert_close(w.palms[:2,:,2],expected)
    torch.testing.assert_close(w.root_targets[:2,:,2]-w.root_seed[:2,:,2],
                               expected+task.offset[:2,None])
    torch.testing.assert_close(w.root_targets[2],before)


def test_world_reposition_arrives_at_goal_raised_by_clearance_projection():
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    w,task=palm_world();task.phase=torch.ones(3,dtype=torch.long)
    c=LoadedPalmReposition(torch.ones(3,dtype=torch.bool));task.palm_reposition=c
    c.state[:]=2;c.goal[:,:,2]=-.001
    task.palm_actions[:,:,2]=-.25
    w.clearances[:,0]=-.0002
    live=torch.tensor([True,True,False])
    w.advance(task,torch.ones(3,dtype=torch.bool),live)
    torch.testing.assert_close(c.goal[:2,:,2],w.palms[:2,:,2])
    torch.testing.assert_close(c.goal[2,:,2],torch.full((2,),-.001,dtype=torch.float64))
    # Arrival still requires measured travel and the existing unloaded dwell.
    current=w.palms.clone();load=torch.zeros((3,2))
    for _ in range(5):c.apply(c.goal.clone(),current,load,task.phase,live,.01)
    assert c.state[:2].tolist()==[[3,3],[3,3]]


def test_world_reposition_goal_is_not_replaced_by_slew_or_another_hands_hold():
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    for waiting in (False,True):
        w,task=palm_world();task.phase=torch.ones(3,dtype=torch.long)
        c=LoadedPalmReposition(torch.ones(3,dtype=torch.bool));task.palm_reposition=c
        c.state[:]=2;c.goal[:,:,2]=-.001
        if waiting:
            c.state[:,0]=1;c.hold[:,:,2]=.0005
        task.palm_actions[:,:,2]=-.25
        before=c.goal.clone()
        w.advance(task,torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
        torch.testing.assert_close(c.goal,before)


@pytest.mark.parametrize('clearance', [.0003,.0008])
def test_world_reposition_goal_tracks_safety_bound_even_when_slew_point_is_higher(clearance):
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    w,task=palm_world();task.phase=torch.ones(3,dtype=torch.long)
    c=LoadedPalmReposition(torch.ones(3,dtype=torch.bool));task.palm_reposition=c
    c.state[:]=2;c.goal[:,:,2]=-.001;task.palm_actions[:,:,2]=-.25
    w.clearances[:]=clearance
    w.advance(task,torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    expected=torch.full((3,2),-(clearance-.0003),dtype=torch.float64)
    torch.testing.assert_close(c.goal[:,:,2],expected)
    assert c.goal_clipped.all()


def test_contact_effort_submission_shares_budget_and_clears_only_reset_rows():
    from go2_pvcnn.control.dh116.tabletop_tensor_world import write_contact_effort
    class Hand:
        def __init__(self): self.calls=[]
        def set_joint_effort_target(self,value,**kw): self.calls.append(('effort',value.clone(),kw))
        def write_joint_effort_limit_to_sim(self,value,**kw): self.calls.append(('limit',value.clone(),kw))
    hand=Hand();ids=[4,1];rows=torch.tensor([2])
    ff=torch.tensor([[.1,.2]],dtype=torch.float64)
    write_contact_effort(hand,ids,ff,.2,env_ids=rows)
    assert [c[0] for c in hand.calls]==['limit','effort']
    torch.testing.assert_close(hand.calls[0][1]+hand.calls[1][1].abs(),torch.full((1,2),.2))
    assert all(c[2]['joint_ids']==ids and torch.equal(c[2]['env_ids'],rows) for c in hand.calls)
    hand.calls=[];write_contact_effort(hand,ids,torch.zeros_like(ff),.2,env_ids=rows)
    torch.testing.assert_close(hand.calls[0][1],torch.full((1,2),.2))
    assert hand.calls[1][1].count_nonzero()==0


def test_position_pause_allows_effort_until_severe_coupling_guard():
    from test_dh116_coordinated_grasp import enabled
    from go2_pvcnn.control.dh116.tabletop_tensor_world import contact_effort_permission
    t=enabled(3);t.m['q'][:,[1,7]]=t.low[[1,7]]+.1
    t.m['mimic_errors'][:,0]=torch.tensor([.007,.009,.013],dtype=torch.float64)
    t.force_guard_conflict=torch.zeros((3,2),dtype=torch.bool)
    allowed=contact_effort_permission(t,torch.ones(3,dtype=torch.bool))
    assert allowed[:,0].tolist()==[True,True,False]
    assert allowed[:,1].all()
