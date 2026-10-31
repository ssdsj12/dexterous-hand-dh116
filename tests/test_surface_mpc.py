"""CPU-only contract tests for the independent local pad MPC."""
import importlib

import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand


def planner(**kwargs):
    module = importlib.import_module(
        "go2_pvcnn.control.dh116.tabletop_surface_mpc")
    return module.SurfaceGripMPC(**kwargs)


def observation(n=2, d=5, h=4, dtype=torch.float64, device="cpu"):
    q = torch.full((n, 2, 6), .4, dtype=dtype, device=device)
    points = torch.zeros(n, 2, d, 3, dtype=dtype, device=device)
    jacobian = torch.zeros(n, 2, d, 3, 6, dtype=dtype, device=device)
    for digit in range(d):
        jacobian[:, :, digit, 0, digit % 6] = .03
    return dict(q=q, command=q.clone(), nominal=q[:, None].repeat(1, h, 1, 1),
                points=points, jacobian=jacobian,
                target_points=points[:, None].repeat(1, h, 1, 1, 1),
                wrist_motion=torch.zeros(n, h, 2, d, 3, dtype=dtype, device=device),
                weights=torch.ones(n, 2, d, dtype=dtype, device=device),
                lower=torch.zeros(2, 6, dtype=dtype, device=device),
                upper=torch.full((2, 6), 2., dtype=dtype, device=device))


def project_nominal(obs):
    lo, hi = (obs["q"].new_tensor(v) for v in active_limits())
    lo = torch.maximum(lo, obs["lower"])
    hi = torch.minimum(hi, obs["upper"])
    previous = obs["command"]
    result = []
    for desired in obs["nominal"].unbind(1):
        previous = desired.clamp(torch.maximum(lo, previous - .005),
                                 torch.minimum(hi, previous + .005))
        result.append(previous)
    return torch.stack(result, 1)


def rollout(obs, trajectory, response=.5):
    state = obs["q"].clone()
    load = obs["command"] - obs["q"]
    states = []
    for command in trajectory.unbind(1):
        state = (1 - response) * state + response * (command - load)
        states.append(state)
    states = torch.stack(states, 1)
    points = obs["points"][:, None] + torch.einsum(
        "nsdcj,nhsj->nhsdc", obs["jacobian"], states - obs["q"][:, None])
    return states, points + obs["wrist_motion"]


def geometry_cost(obs, predicted_points):
    weights = obs["weights"]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-15)
    return (((predicted_points - obs["target_points"]) / .001).square().sum(-1)
            * weights[:, None]).sum(-1).mean(1)


def test_multistep_optimization_reduces_geometry_and_reports_actual_rollout():
    obs = observation(d=6)
    obs["target_points"][..., 0] = .00045
    original = {name: value.clone() for name, value in obs.items()}
    first, diag = planner().plan(**obs)
    trajectory = diag["trajectory"]
    assert trajectory.shape == (2, 4, 2, 6)
    assert diag["predicted_joints"].shape == (2, 4, 2, 6)
    assert diag["predicted_points"].shape == (2, 4, 2, 6, 3)
    torch.testing.assert_close(first, trajectory[:, 0], atol=0, rtol=0)
    states, points = rollout(obs, trajectory)
    torch.testing.assert_close(diag["predicted_joints"], states)
    torch.testing.assert_close(diag["predicted_points"], points)
    baseline = geometry_cost(obs, rollout(obs, project_nominal(obs))[1])
    assert (geometry_cost(obs, points) < baseline * .6).all()
    assert (trajectory[:, -1] > trajectory[:, 0]).all()
    assert (diag["optimized_cost"] < diag["nominal_cost"]).all()
    increments = torch.diff(trajectory, dim=1, prepend=obs["command"][:, None])
    expected = (geometry_cost(obs, points)
                + .02 * ((trajectory - obs["nominal"]) / .005).square().mean((1, 3))
                + .002 * (increments / .005).square().mean((1, 3))).sum(-1)
    torch.testing.assert_close(diag["optimized_cost"], expected)
    assert diag["model"] == "local_pad_kinematics_not_object_dynamics"
    assert diag["response"] == .5
    for name in obs:
        torch.testing.assert_close(obs[name], original[name], atol=0, rtol=0)


def test_future_target_changes_first_command_even_when_first_target_is_satisfied():
    obs = observation(n=1, d=1)
    obs["target_points"][:, 1:, :, :, 0] = .001
    first, _ = planner().plan(**obs)
    short = {name: value[:, :1] if name in
             ("nominal", "target_points", "wrist_motion") else value
             for name, value in obs.items()}
    one_step, _ = planner(horizon=1).plan(**short)
    torch.testing.assert_close(one_step, obs["command"], atol=1e-14, rtol=0)
    assert (first[:, :, 0] > one_step[:, :, 0] + .001).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_contacts_returns_exact_rate_limited_student_nominal(dtype):
    obs = observation(dtype=dtype)
    obs["command"] += .06  # Limits are relative to applied targets, never q.
    obs["weights"].zero_()
    obs["nominal"][:, :, 0] += .3
    obs["nominal"][:, :, 1] -= .3
    obs["target_points"] += .8
    first, diag = planner().plan(**obs)
    nominal = project_nominal(obs)
    torch.testing.assert_close(first, nominal[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(diag["trajectory"], nominal, atol=0, rtol=0)
    torch.testing.assert_close(diag["optimized_cost"], diag["nominal_cost"], atol=0, rtol=0)


def test_contact_free_hand_retains_its_nominal_while_other_hand_optimizes():
    obs = observation()
    obs["weights"][:, 0] = 0
    obs["nominal"][:, :, 0, :] += torch.arange(1, 5)[:, None] * .002
    obs["target_points"][..., 0] = .0005
    _, diag = planner().plan(**obs)
    torch.testing.assert_close(diag["trajectory"][:, :, 0],
                               project_nominal(obs)[:, :, 0], atol=0, rtol=0)
    assert (diag["trajectory"][:, 0, 1, 0] > obs["command"][:, 1, 0]).all()


def test_nominal_intent_and_smoothness_both_enter_objective():
    obs = observation(n=1, d=1)
    obs["target_points"][..., 0] = .0004
    weak, _ = planner(nominal_weight=.001).plan(**obs)
    strong, _ = planner(nominal_weight=100.).plan(**obs)
    smooth, _ = planner(nominal_weight=.001, smoothness_weight=100.).plan(**obs)
    assert (weak[:, :, 0] > strong[:, :, 0] + .001).all()
    assert (weak[:, :, 0] > smooth[:, :, 0] + .001).all()
    obs["nominal"] += .003
    follows_student, _ = planner(nominal_weight=100.).plan(**obs)
    assert (follows_student[:, :, 0] > strong[:, :, 0] + .002).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_whole_trajectory_obeys_rate_and_actual_coupled_joint_limits(dtype):
    obs = observation(dtype=dtype)
    lo, hi = active_limits()
    obs["command"][:] = obs["q"].new_tensor(hi) - .002
    obs["command"][1] = obs["q"].new_tensor(lo) + .002
    obs["q"] = obs["command"] - .02
    obs["nominal"][0] = 100
    obs["nominal"][1] = -100
    obs["lower"][0, 0] = .001
    obs["upper"][1, 5] = float(hi[5]) - .001
    obs["target_points"][0] += 100
    obs["target_points"][1] -= 100
    _, diag = planner().plan(**obs)
    trajectory = diag["trajectory"]
    deltas = torch.diff(trajectory, dim=1, prepend=obs["command"][:, None])
    assert deltas.abs().max() <= .0050001
    assert (trajectory.double() >= torch.as_tensor(lo)).all()
    assert (trajectory.double() <= torch.as_tensor(hi)).all()
    assert (trajectory >= obs["lower"]).all()
    assert (trajectory <= obs["upper"]).all()
    for command in trajectory.reshape(-1, 6):
        expand(command.cpu().numpy())


@pytest.mark.parametrize("d", [1, 2, 5, 7])
def test_arbitrary_pad_counts_and_digit_permutations(d):
    obs = observation(d=d)
    obs["target_points"][..., 0] = torch.linspace(.0001, .0006, d)
    obs["weights"][:] = torch.arange(1, d + 1)
    first, diag = planner().plan(**obs)
    order = torch.arange(d - 1, -1, -1)
    permuted = dict(obs, points=obs["points"][:, :, order],
                    jacobian=obs["jacobian"][:, :, order],
                    weights=obs["weights"][:, :, order],
                    target_points=obs["target_points"][:, :, :, order],
                    wrist_motion=obs["wrist_motion"][:, :, :, order])
    reordered, reordered_diag = planner().plan(**permuted)
    torch.testing.assert_close(reordered, first)
    torch.testing.assert_close(reordered_diag["predicted_points"],
                               diag["predicted_points"][:, :, :, order])
    assert (diag["optimized_cost"] <= diag["nominal_cost"]).all()


def test_zero_weight_pad_cannot_change_solution_or_cost():
    obs = observation(d=5)
    obs["weights"][:, :, -1] = 0
    obs["target_points"][..., 0] = .0004
    first, diag = planner().plan(**obs)
    obs["target_points"][:, :, :, -1] = 10
    obs["jacobian"][:, :, -1] = 20
    other, other_diag = planner().plan(**obs)
    torch.testing.assert_close(other, first, atol=0, rtol=0)
    torch.testing.assert_close(other_diag["optimized_cost"], diag["optimized_cost"], atol=0, rtol=0)


def test_batch_and_left_right_are_independent():
    obs = observation(n=3)
    obs["target_points"][..., 0] = torch.tensor(
        [[.0004, -.0004], [-.0003, .0005], [.0001, -.0002]])[:, None, :, None]
    obs["jacobian"][1] *= -1
    first, diag = planner().plan(**obs)
    for env in range(3):
        single = {k: v if k in ("lower", "upper") else v[env:env + 1]
                  for k, v in obs.items()}
        single_first, single_diag = planner().plan(**single)
        torch.testing.assert_close(first[env:env + 1], single_first)
        torch.testing.assert_close(diag["trajectory"][env:env + 1], single_diag["trajectory"])
    changed = {k: v.clone() for k, v in obs.items()}
    changed["target_points"][0, :, 0] += 1
    changed_first, _ = planner().plan(**changed)
    torch.testing.assert_close(changed_first[1:], first[1:], atol=0, rtol=0)
    torch.testing.assert_close(changed_first[0, 1], first[0, 1], atol=0, rtol=0)


@pytest.mark.parametrize("response", [.2, .5, 1.])
def test_loaded_equilibrium_and_future_wrist_motion_are_forecast(response):
    obs = observation(n=1)
    obs["command"] += .04
    obs["nominal"][:] = obs["command"][:, None]
    obs["wrist_motion"][..., 2] = torch.arange(1, 5)[None, :, None, None] * .0003
    obs["target_points"] = obs["points"][:, None] + obs["wrist_motion"]
    first, diag = planner().plan(**obs, response=response)
    torch.testing.assert_close(first, obs["command"], atol=0, rtol=0)
    torch.testing.assert_close(diag["predicted_joints"], obs["q"][:, None].expand(-1, 4, -1, -1), atol=0, rtol=0)
    torch.testing.assert_close(diag["predicted_points"], obs["target_points"], atol=0, rtol=0)
    assert diag["response"] == response
    obs["wrist_motion"][..., 0] = .0004
    corrected, corrected_diag = planner().plan(**obs, response=response)
    assert (corrected[:, :, 0] < first[:, :, 0]).all()
    torch.testing.assert_close(corrected_diag["predicted_points"],
                               rollout(obs, corrected_diag["trajectory"], response)[1])


def test_replanning_uses_fresh_state_and_reset_ids_has_no_cross_env_history():
    obs = observation()
    obs["target_points"][..., 0] = .0005
    controller = planner()
    with torch.inference_mode():
        _, diag = controller.plan(**obs)
    obs["q"] = diag["predicted_joints"][:, 0].clone()
    obs["command"] = diag["trajectory"][:, 0].clone()
    obs["points"] = diag["predicted_points"][:, 0].clone()
    controller.reset(torch.tensor([1]))
    first, _ = controller.plan(**obs)
    fresh, _ = planner().plan(**obs)
    torch.testing.assert_close(first, fresh, atol=0, rtol=0)
    assert (first - obs["command"]).abs().max() <= .005000001


@pytest.mark.parametrize("name", list(observation()))
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_input_is_rejected_for_nominal_fallback(name, bad):
    obs = observation()
    obs[name].flatten()[0] = bad
    with pytest.raises(ValueError, match=name):
        planner().plan(**obs)


@pytest.mark.parametrize("response", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_response_is_rejected(response):
    with pytest.raises(ValueError, match="response"):
        planner().plan(**observation(), response=response)


@pytest.mark.parametrize("name", ["q", "command", "nominal", "points", "jacobian",
                                  "target_points", "wrist_motion", "weights", "lower", "upper"])
def test_bad_shapes_are_rejected(name):
    obs = observation()
    obs[name] = obs[name].unsqueeze(0)
    with pytest.raises(ValueError, match=name):
        planner().plan(**obs)


@pytest.mark.parametrize("case", ["negative_weight", "empty_interval", "infeasible_command"])
def test_invalid_weights_or_infeasible_constraints_raise(case):
    obs = observation()
    if case == "negative_weight":
        obs["weights"][0, 0, 0] = -1
    elif case == "empty_interval":
        obs["lower"][:] = 3
    else:
        obs["command"][:] = 3
    with pytest.raises(ValueError):
        planner().plan(**obs)


@pytest.mark.parametrize("kwargs", [dict(horizon=0), dict(horizon=1.5), dict(iterations=0),
                                    dict(point_scale=0), dict(nominal_weight=-1),
                                    dict(smoothness_weight=float("nan"))])
def test_invalid_optimizer_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        planner(**kwargs)
