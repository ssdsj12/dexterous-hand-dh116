import torch
import pytest
from test_dh116_close_downward_penalty import task
from test_dh116_tensor_contact_window import loads


def enabled(n=2):
    t=task(n,grasp_shaping=True)
    assert hasattr(t,'enable_coordinated_control'), 'coordinated controller not implemented'
    t.enable_coordinated_control(torch.ones(n,dtype=torch.bool))
    return t


def test_recovery_can_retreat_below_policy_floor_but_not_mechanical_bounds():
    t=enabled();t.envelope={};t.lower=t.low+.1;t.upper=t.high-.1;t.center=(t.lower+t.upper)/2
    t.m['q'][:]=t.lower;t.command[:]=t.lower
    t.m['mimic_errors'][:,1]=-.016
    a=torch.zeros((2,20),dtype=torch.float64)
    before=t.command.clone();t.control(a,~t.done,1/480)
    assert (t.command[:,2]<t.lower[2]).all()
    assert (t.command>=t.low).all() and ((t.command-before).abs()<=.20/480+1e-12).all()


def test_measured_midpoint_compensation_ignores_unrealized_force_request():
    t=enabled();t.f_offset[:]=.07;t.compensation[:]=-.03;t.has_compensation[:]=True
    a=torch.zeros((2,20),dtype=torch.float64);a[:,18:]=1.
    t.control(a,~t.done,1/480)
    torch.testing.assert_close(t.palm_targets,torch.zeros_like(t.palm_targets))
    t.m['pad_relative'][:,:,2]+=.001
    t.control(a,~t.done,1/480)
    torch.testing.assert_close(t.palm_targets[:,:,2],torch.full((2,2),-.001,dtype=torch.float64))


def test_lift_accepts_bounded_loaded_error_and_blocks_severe_error():
    t=enabled();t.m=loads(t,4.);t.phase[:]=3;t.command.copy_(t.m['q'])
    t.m['mimic_errors'][:,1]=torch.tensor([-.007,-.013]);before=t.offset.clone()
    a=torch.zeros((2,20),dtype=torch.float64);a[:,18:]=-.35
    t.control(a,~t.done,1/480)
    assert t.offset[0]>before[0] and t.offset[1]==before[1]
    assert t.paused.tolist()==[False,True]


def test_close_command_slew_uses_the_faster_rate():
    t=enabled();t.m=loads(t,4.);t.phase[:]=1;t.command.copy_(t.m['q'])
    before=t.command.clone();a=torch.ones((2,20),dtype=torch.float64)
    t.control(a,~t.done,1/480)
    delta=(t.command-before).abs()
    assert (delta.max(1).values>.10/480).all()
    assert (delta<=.20/480+1e-12).all()


def test_overpressure_blocks_lift_and_releases_with_slew_limit():
    t=enabled();t.phase[:]=3;t.m=loads(t,32.);t.command.copy_(t.m['q'])
    before=t.command.clone();lift=t.offset.clone();a=torch.ones((2,20),dtype=torch.float64);a[:,18:]=-1.
    t.control(a,~t.done,1/480)
    assert t.pressure_limited.all();torch.testing.assert_close(t.offset,lift)
    sign=torch.where(t.direction==0,1.,t.direction.sign()).flatten(1)
    assert (((t.command-before)*sign)<=1e-12).all()
    assert (abs(t.command-before)<=.20/480+1e-12).all()


def test_partial_mask_preserves_legacy_behavior_and_inactive_commands():
    a=torch.zeros((2,20),dtype=torch.float64)
    original=task(2,grasp_shaping=True);mixed=task(2,grasp_shaping=True)
    assert hasattr(mixed,'enable_coordinated_control')
    mixed.enable_coordinated_control(torch.tensor([False,True]))
    before=mixed.command[1].clone();live=torch.tensor([True,False])
    original.control(a,live,1/480);mixed.control(a,live,1/480)
    torch.testing.assert_close(mixed.command[0],original.command[0]);torch.testing.assert_close(mixed.command[1],before)


def test_anchor_updates_only_for_reset_environment():
    t=enabled();anchor=t.control_pad_anchor.clone();m={k:v.clone() for k,v in t.m.items()}
    m['pad_relative'][0,:,2]+=.001;t.reset([0],m)
    torch.testing.assert_close(t.control_pad_anchor[0,:,2],anchor[0,:,2]+.001)
    torch.testing.assert_close(t.control_pad_anchor[1],anchor[1])


def test_midpoint_reference_rejects_object_motion_and_common_lift_as_finger_motion():
    from go2_pvcnn.control.dh116.tabletop_coordinated_grasp import midpoint_coordinates
    t=enabled();before=midpoint_coordinates(t.m).clone()
    t.m['pad_relative'][:,:,2]+=.002;t.m['object_relative'][:,2]-=.002
    torch.testing.assert_close(midpoint_coordinates(t.m),before)
    t.m['pad_relative'][:,:,0]-=.003;t.m['object_relative'][:,0]+=.003
    torch.testing.assert_close(midpoint_coordinates(t.m),before)


def test_controller_contract_requires_exact_recovery_semantics_on_resume():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_coordinated_grasp import CONTRACT
    old=dict(num_envs=4096,physics_hz=480,observation_size=156)
    new=dict(old,grasp_controller=CONTRACT)
    with pytest.raises(ValueError):checkpoint_contract_changes(old,new)
    assert checkpoint_contract_changes(new,new)==[]
    for a,b in [(new,old),(old,dict(new,physics_hz=240)),(old,dict(new,grasp_controller=dict(CONTRACT,position_lead_rad=.1)))]:
        with pytest.raises(ValueError):checkpoint_contract_changes(a,b)


def test_thumb_near_limit_releases_opposing_index_and_stops_lift():
    t=enabled();t.m=loads(t,4.);t.phase[:]=3
    t.m['q'][:,7]=t.low[7]+.002;t.command.copy_(t.m['q']);before=t.command.clone();lift=t.offset.clone()
    a=torch.ones((2,20),dtype=torch.float64)
    t.control(a,~t.done,1/480)
    assert (t.command[:,8]<before[:,8]).all()
    assert (t.command[:,7]>=before[:,7]).all()
    torch.testing.assert_close(t.offset,lift)
    assert t.f_blocked[:,1].all() and t.paused.all()


def test_policy_floor_does_not_override_position_lead_during_loaded_displacement():
    from go2_pvcnn.control.dh116.tabletop_coordinated_grasp import CONTRACT
    t=enabled();t.envelope={};t.lower=t.low+.1;t.upper=t.high-.1;t.center=(t.lower+t.upper)/2
    t.m['q'][:]=t.center;t.m['q'][:,7]=t.lower[7]-.02;t.command.copy_(t.m['q'])
    a=torch.zeros((2,20),dtype=torch.float64)
    for _ in range(120):t.control(a,~t.done,1/480)
    assert ((t.command-t.m['q']).abs()<=CONTRACT['position_lead_rad']+1e-12).all()
    assert (t.command>=t.low).all()


def test_force_direction_controller_uses_per_hand_vertical_force_gate():
    from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition
    t = enabled()
    t.force_direction_reposition = True
    t.palm_reposition = ForceDirectionReposition(torch.ones(2, dtype=torch.bool))
    t.m = loads(t, 4.)
    t.m['vertical_hand_force_n'] = torch.tensor([[-.30, .02], [-.30, .02]], dtype=torch.float64)
    t.command.copy_(t.m['q'])
    action = torch.zeros((2, 20), dtype=torch.float64)
    action[:, 14] = action[:, 17] = -.5
    t.control(action, ~t.done, 1/480)
    assert t.palm_reposition.state[:, 0].tolist() == [1, 1]
    assert t.palm_reposition.state[:, 1].tolist() == [0, 0]


def test_near_limit_thumb_reduces_stale_overshoot_without_opening_position_bias():
    t=enabled();t.m=loads(t,4.);t.phase[:]=3
    t.m['q'][:,7]=t.low[7]+.002;t.command.copy_(t.m['q'])
    t.command[:,7]+=.02
    a=torch.ones((2,20),dtype=torch.float64);lift=t.offset.clone()
    for _ in range(150):
        before=t.command.clone();t.control(a,~t.done,1/480)
        assert (t.command[:,7]>=t.m['q'][:,7]).all()
        assert ((t.command-before).abs()<=.20/480+1e-12).all()
    torch.testing.assert_close(t.command[:,7],t.m['q'][:,7]+.008)
    assert (t.command[:,8]<t.m['q'][:,8]).all()
    torch.testing.assert_close(t.offset,lift)
    assert t.paused.all()


@pytest.mark.parametrize('loaded_hand', [0, 1])
def test_shared_z_cannot_override_one_loaded_hands_release_hold(loaded_hand):
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    t=enabled();t.palm_reposition=LoadedPalmReposition(torch.ones(2,dtype=torch.bool))
    t.m=loads(t,4.);t.m['hand_object_loads'][:,1-loaded_hand]=0.
    t.command.copy_(t.m['q'])
    action=torch.zeros((2,20),dtype=torch.float64);action[:,14]=action[:,17]=-.5
    t.control(action,~t.done,1/480)
    assert t.palm_reposition.state[0,loaded_hand]==1
    torch.testing.assert_close(t.palm_targets[:,:,2],torch.zeros((2,2),dtype=torch.float64))


@pytest.mark.parametrize('waiting_hand', [0, 1])
def test_shared_z_waits_for_slower_release_without_discarding_moving_goal(waiting_hand):
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    t=enabled();c=LoadedPalmReposition(torch.ones(2,dtype=torch.bool));t.palm_reposition=c
    t.m=loads(t,0.);t.m['hand_object_loads'][:,waiting_hand]=2.
    c.state[:,waiting_hand]=1;c.state[:,1-waiting_hand]=2;c.goal[:,:,2]=-.001
    t.command.copy_(t.m['q'])
    action=torch.zeros((2,20),dtype=torch.float64);action[:,14]=action[:,17]=-.5
    t.control(action,~t.done,1/480)
    torch.testing.assert_close(t.palm_targets[:,:,2],torch.zeros((2,2),dtype=torch.float64))
    torch.testing.assert_close(c.goal[:,:,2],torch.full((2,2),-.001,dtype=torch.float64))


def test_shared_reposition_preserves_inactive_rows_and_lift_behavior():
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    t=enabled(3);baseline=enabled(3)
    t.palm_reposition=LoadedPalmReposition(torch.ones(3,dtype=torch.bool))
    for task in (t,baseline):
        task.m=loads(task,4.);task.phase[:]=torch.tensor([1,3,1]);task.command.copy_(task.m['q'])
        task.palm_targets[2]=.0007
    t.palm_reposition.state[:]=1;t.palm_reposition.hold[:,:,2]=.001
    live=torch.tensor([True,True,False]);action=torch.zeros((3,20),dtype=torch.float64)
    action[:,14]=action[:,17]=-.5
    before=t.palm_targets[2].clone()
    for task in (t,baseline):task.control(action,live,1/480)
    torch.testing.assert_close(t.palm_targets[1],baseline.palm_targets[1])
    torch.testing.assert_close(t.palm_targets[2],before)


def test_shared_clipped_reposition_goals_can_finish_at_the_common_safe_height():
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    t=enabled();c=LoadedPalmReposition(torch.ones(2,dtype=torch.bool));t.palm_reposition=c
    t.m=loads(t,0.);t.command.copy_(t.m['q'])
    c.state[:]=2;c.goal[:,:,2]=torch.tensor([-.0001,-.001],dtype=torch.float64)
    action=torch.zeros((2,20),dtype=torch.float64)
    t.control(action,~t.done,1/480)
    t.m['adaptive_palm_offset'].copy_(t.palm_targets)
    for _ in range(24):t.control(action,~t.done,1/480)
    assert c.state.tolist()==[[3,3],[3,3]]
    torch.testing.assert_close(c.goal[:,:,2],torch.full((2,2),-.0001,dtype=torch.float64))


@pytest.mark.parametrize('error_m,expected_m', [(1e-5,1e-5),(-1e-5,-1e-5),(.002,.0005)])
def test_coordinated_contact_correction_adds_meters_to_meter_targets(error_m,expected_m):
    t=enabled();t.contact_target_z_m=torch.full((2,2),-.004+error_m,dtype=torch.float64)
    t.m['contact_center_relative_m'][:,:,2]=-.004
    t.m['contact_valid'][:]=True;t.m['contact_all_below_centerline'][:]=True
    action=torch.zeros((2,20),dtype=torch.float64)
    t.control(action,~t.done,1/480)
    torch.testing.assert_close(t.palm_targets[:,:,2],torch.full((2,2),expected_m,dtype=torch.float64))


def test_force_direction_release_supports_joint_envelope():
    from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition
    t=enabled();t.envelope={};t.lower=t.low+.1;t.upper=t.high-.1;t.center=(t.lower+t.upper)/2
    t.force_direction_reposition=True;t.palm_reposition=ForceDirectionReposition(torch.ones(2,dtype=torch.bool))
    t.m=loads(t,4.);t.m['vertical_hand_force_n']=torch.full((2,2),-.3,dtype=torch.float64)
    t.command.copy_(t.m['q'])
    t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,1/480)
    assert torch.isfinite(t.command).all()
