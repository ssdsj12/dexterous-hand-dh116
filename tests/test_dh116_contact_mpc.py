"""Behavioral checks for geometry-driven, hand-only predictive control."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from go2_pvcnn.control.dh116.contact_mpc import (
    ContactMPCConfig, ParallelContactMPC, wire_surface_targets,
)


def observation(device="cpu"):
    q = torch.full((2, 6), .2, device=device)
    points = torch.zeros((2, 3, 3), device=device)
    points[:, :, 1] = torch.tensor([-.02, .02, 0.], device=device)
    jac = torch.zeros((2, 3, 3, 6), device=device)
    for digit, joint in enumerate((0, 2, 3)):
        jac[:, digit, 0, joint] = .02
    target = points.clone()
    target[:, :, 0] += torch.tensor([.0012, -.0008, .001], device=device)
    return dict(active_joints=q, previous_action=q.reshape(12),
        pad_positions=points, pad_jacobians=jac, target_points=target)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_contact_mpc_reduces_geometry_error_and_respects_previous_command(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = ContactMPCConfig(horizon=6, candidates=64, elites=8, iterations=3,
                              control_dt=.1, joint_speed_limit=.3)
    planner = ParallelContactMPC(config, device=device)
    obs = observation(device)

    action, diag = planner.plan_contact(**obs)

    assert action.shape == (2, 6)
    assert action.device == torch.device(device)
    assert diag["trajectory"].shape == (6, 12)
    assert float(diag["optimized_cost"]) < float(diag["hold_cost"])
    assert action[0, 0] > .2 and action[0, 2] < .2 and action[0, 3] > .2
    assert (action.reshape(12) - obs["previous_action"]).abs().max() <= .030001
    # Joints with no observed effect on any selected pad should hold.
    torch.testing.assert_close(action[:, [1, 4, 5]], obs["active_joints"][:, [1, 4, 5]])
    assert diag["model"] == "linearized_pad_kinematics"
    assert diag["action_dim"] == 12


def test_contact_mpc_holds_an_already_satisfied_grasp_target():
    obs = observation()
    obs["target_points"] = obs["pad_positions"].clone()
    action, diag = ParallelContactMPC().plan_contact(**obs)
    torch.testing.assert_close(action, obs["active_joints"], atol=1e-7, rtol=0)
    assert float(diag["optimized_cost"]) == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_preloaded_hand_is_exactly_held_in_prediction_while_other_hand_closes(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    obs=observation(device)
    planner=ParallelContactMPC(ContactMPCConfig(horizon=4,control_dt=.01,
        candidates=16,elites=4,iterations=1),device=device)
    held=torch.tensor([True]*6+[False]*6,device=device)
    load=torch.full((12,),.01,device=device)
    # The held command must not be silently changed by equilibrium projection,
    # even when the loaded lower-bound inset would ordinarily move it.
    obs['previous_action'][0]=planner.lower[0]+.0001
    action,diag=planner.plan_contact(**obs,actuator_load_offset=load,held_coordinates=held)
    torch.testing.assert_close(action[0],obs['previous_action'][:6],atol=0,rtol=0)
    torch.testing.assert_close(diag['trajectory'][:,:6],obs['previous_action'][:6].expand(4,6),atol=0,rtol=0)
    assert action[1,0]>obs['previous_action'][6]
    # Physical Jacobians and tracking predictions must not disappear on hold.
    expected=obs['active_joints'].reshape(12)+.5*(action.reshape(12)-load-obs['active_joints'].reshape(12))
    torch.testing.assert_close(diag['predicted_active_joints'],expected)
    assert diag['held_coordinates'].tolist()==held.tolist()
    # Releasing the latch permits subsequent optimization again.
    resumed,_=planner.plan_contact(**observation(device),held_coordinates=torch.zeros_like(held))
    assert resumed[0,0]>.2


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_loaded_mpc_learns_measured_tracking_bias_without_unbounded_commands(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    obs=observation(device)
    config=ContactMPCConfig(horizon=4,candidates=16,elites=4,iterations=1,control_dt=.01)
    planner=ParallelContactMPC(config,device=device)
    nominal=torch.zeros(12,device=device);nominal[[0,2,3,6,8,9]]=.004
    extra=torch.zeros_like(nominal);extra[[0,2,3,6,8,9]]=.006
    anchor=obs['active_joints'].clone();initial_points=obs['pad_positions'].clone()
    errors=[]
    for _ in range(18):
        obs['pad_positions']=initial_points+torch.einsum('hpcj,hj->hpc',
            obs['pad_jacobians'],obs['active_joints']-anchor)
        action,diag=planner.plan_contact(**obs,actuator_load_offset=nominal)
        assert 'tracking_bias_rad' in diag
        assert (action.reshape(12)-obs['previous_action']).abs().max()<=.003001
        actual=obs['active_joints']+.5*(action-(nominal+extra).reshape(2,6)-obs['active_joints'])
        errors.append(float((diag['predicted_active_joints']-actual.reshape(12)).square().mean().sqrt()))
        obs['active_joints']=actual;obs['previous_action']=action.reshape(12)
    torch.testing.assert_close(diag['tracking_bias_rad'],extra,atol=.0001,rtol=0)
    assert errors[-1]<errors[0]*.02
    torch.testing.assert_close(action[:,[1,4,5]],anchor[:,[1,4,5]])


def test_tracking_observer_is_bounded_and_does_not_leak_into_legacy_free_space_calls():
    planner=ParallelContactMPC(ContactMPCConfig(candidates=16,elites=4,iterations=1))
    obs=observation()
    planner.plan_contact(**obs,actuator_load_offset=torch.zeros(12))
    obs['active_joints']-=.1
    _,diag=planner.plan_contact(**obs,actuator_load_offset=torch.zeros(12))
    assert 'tracking_bias_rad' in diag
    assert diag['tracking_bias_rad'].abs().max()<=np.pi/180+1e-7
    _,legacy=planner.plan_contact(**observation())
    torch.testing.assert_close(legacy['tracking_bias_rad'],torch.zeros(12))


def test_loaded_prediction_keeps_equilibrium_inside_limits_without_command_teleport():
    planner=ParallelContactMPC(ContactMPCConfig(control_dt=.01,candidates=16,elites=4,iterations=1))
    obs=observation();q=obs['active_joints'].clone()
    q[0,1]=planner.lower[1]+.0001
    obs['active_joints']=q;obs['previous_action']=q.reshape(12).clone()
    obs['previous_action'][1]+=.01
    obs['pad_jacobians'][0,0]=0.;obs['pad_jacobians'][0,0,0,1]=.02
    obs['target_points'][0,0,0]=-.001
    nominal=torch.zeros(12);nominal[1]=.004
    extra=torch.zeros(12);extra[1]=.006
    planner.restore_tracking_bias(extra)
    action,diag=planner.plan_contact(**obs,actuator_load_offset=nominal)
    assert diag['predicted_active_joints'][1]>=planner.lower[1]
    assert (action.reshape(12)-obs['previous_action']).abs().max()<=.003001
    assert action[0,1]>obs['previous_action'][1]


def test_contact_mpc_changes_joint_direction_with_measured_jacobian_sign():
    obs = observation()
    a, _ = ParallelContactMPC().plan_contact(**obs)
    obs["pad_jacobians"] = -obs["pad_jacobians"]
    b, _ = ParallelContactMPC().plan_contact(**obs)
    assert a[0, 0] > .2 > b[0, 0]
    assert a[0, 2] < .2 < b[0, 2]


def test_contact_mpc_limits_action_slew_from_previous_command_when_tracking_lags():
    obs = observation()
    obs["previous_action"] = torch.full((12,), .25)
    config = ContactMPCConfig(control_dt=.02, joint_speed_limit=.3)
    action, _ = ParallelContactMPC(config).plan_contact(**obs)
    assert (action.reshape(12) - obs["previous_action"]).abs().max() <= .006001


def test_wire_surface_targets_follow_actual_centerline_and_load_preload():
    points = torch.tensor([[[.43, -.15, .325], [.43, -.15, .290], [.445, -.15, .308]],
                           [[.43, .15, .325], [.43, .15, .290], [.445, .15, .308]]])
    wire = torch.stack((torch.full((41,), .43), torch.linspace(-.25, .25, 41),
                        torch.full((41,), .308)))
    targets = wire_surface_targets(points, wire, radius=.008, target_loads=torch.ones(2, 3),
                                   contact_stiffness=1e4)
    distance = torch.linalg.vector_norm(targets[:, :, [0, 2]] -
                                       torch.tensor([.43, .308]), dim=-1)
    torch.testing.assert_close(distance, torch.full((2, 3), .0079), atol=1e-7, rtol=0)
    translation = torch.tensor([.01, 0., .02])
    moved = wire_surface_targets(points + translation, wire + translation[:, None],
        radius=.008, target_loads=torch.ones(2, 3), contact_stiffness=1e4)
    torch.testing.assert_close(moved, targets + translation, atol=1e-7, rtol=0)


def test_contact_mpc_does_not_accept_panda_jacobian_columns():
    obs = observation()
    obs["pad_jacobians"] = torch.zeros((2, 3, 3, 13))
    with pytest.raises(ValueError, match="Jacobian|jacobian"):
        ParallelContactMPC().plan_contact(**obs)


def test_pinch_targets_are_on_opposite_sides_of_the_actual_wire():
    points=torch.tensor([[[.442,.2,.318],[.418,.2,.316],[.36,.2,.318]],
                         [[.440,-.2,.317],[.419,-.2,.315],[.36,-.2,.318]]])
    nodes=torch.stack((torch.full((41,),.43),torch.linspace(-.25,.25,41),
                       torch.full((41,),.308)))
    target=wire_surface_targets(points,nodes,opposed_pinch=True)
    for side in range(2):
        center=torch.tensor([.43, points[side,0,1], .308])
        a=target[side,0]-center; b=target[side,1]-center
        assert float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))) < -.999
        torch.testing.assert_close(torch.linalg.vector_norm(a),torch.tensor(.008),atol=1e-7,rtol=0)


def test_float32_mpc_limit_commands_remain_legal_in_float64_coupling():
    from go2_pvcnn.control.dh116.linear_coupling import expand
    obs=observation()
    obs['active_joints'][:,1]=.009
    obs['previous_action']=obs['active_joints'].reshape(12).clone()
    obs['pad_jacobians'][:,0,0,0]=0.
    obs['pad_jacobians'][:,0,0,1]=.02
    obs['target_points'][:,0,0]=-.001
    planner=ParallelContactMPC()
    for _ in range(3):
        action,_=planner.plan_contact(**obs)
        for hand in action.cpu().numpy():expand(hand)
        obs['previous_action']=action.reshape(12)


def test_normal_contact_objective_allows_required_tangential_pad_motion():
    obs=observation()
    obs['pad_jacobians'].zero_()
    obs['pad_jacobians'][:,0,:,0]=torch.tensor([.01,.03,0.])
    obs['target_points']=obs['pad_positions'].clone()
    obs['target_points'][:,0,0]+=.001
    planner=ParallelContactMPC(ContactMPCConfig(horizon=6,control_dt=.1))
    _,diag=planner.plan_contact(**obs)
    # Strict XYZ point tracking stalls at dq=.01; the normal-contact objective
    # permits the tangential component needed to approach the wire surface.
    assert diag['trajectory'][-1,0] > .25


def test_wire_targets_accept_closed_loop_penetration_instead_of_load_proxy():
    obs=observation()
    nodes=torch.tensor([[0.,1.],[0.,0.],[0.,0.]])
    points=obs['pad_positions'].clone();points[:,:,2]=.02
    target=wire_surface_targets(points,nodes,penetration_m=torch.full((2,3),.0005))
    assert torch.linalg.vector_norm(target[:,:,[1,2]],dim=-1).mean().item()==pytest.approx(.0075)


def test_loaded_hold_predicts_measured_equilibrium_not_free_space_tracking():
    obs=observation()
    obs['target_points']=obs['pad_positions'].clone()
    obs['previous_action']=obs['active_joints'].reshape(12).clone()+.01
    planner=ParallelContactMPC()
    action,diag=planner.plan_contact(**obs,actuator_load_offset=torch.full((12,),.01))
    torch.testing.assert_close(action.reshape(12),obs['previous_action'],atol=1e-7,rtol=0)
    torch.testing.assert_close(diag['predicted_pad_points'],obs['pad_positions'].expand(8,2,3,3),atol=1e-7,rtol=0)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_mpc_rotates_thumb_in_contact_nullspace_to_clear_non_pad_link(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    obs=observation(device)
    obs['pad_jacobians'].zero_()
    obs['pad_jacobians'][:,0,0,0]=.02
    obs['pad_jacobians'][:,0,0,1]=.06
    obs['target_points']=obs['pad_positions'].clone()
    gradient=torch.zeros((2,12),device=device)
    gradient[0,0]=.02;gradient[0,1]=-.01
    gradient[1,6]=.02;gradient[1,7]=-.01
    planner=ParallelContactMPC(ContactMPCConfig(horizon=16,control_dt=.01),device=device)
    action,diag=planner.plan_contact(**obs,clearances=torch.full((2,),.0011,device=device),
                                    clearance_jacobians=gradient)
    assert action[0,0]>.2 and action[0,1]<.2
    assert action[1,0]>.2 and action[1,1]<.2
    assert diag['predicted_clearances'][-1].min()>.0015
    assert (diag['predicted_pad_points'][-1]-obs['pad_positions']).abs().max()<.0001
    assert (action.reshape(12)-obs['previous_action']).abs().max()<=.003001
    torch.testing.assert_close(action[:,2:],obs['active_joints'][:,2:])


def test_clearance_prediction_rejects_panda_columns_and_partial_inputs():
    obs=observation()
    with pytest.raises(ValueError,match='clearance'):
        ParallelContactMPC().plan_contact(**obs,clearances=[.001],clearance_jacobians=np.zeros((1,26)))
    with pytest.raises(ValueError,match='clearance'):
        ParallelContactMPC().plan_contact(**obs,clearances=[.001])
