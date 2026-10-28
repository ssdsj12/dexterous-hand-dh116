import torch
import pytest


def _setup(mask=(True,)):
    from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition

    c = ForceDirectionReposition(torch.tensor(mask))
    n = len(mask)
    current = torch.zeros((n, 2, 3), dtype=torch.float64)
    target = current.clone()
    target[:, :, 2] = -.001
    loads = torch.ones((n, 2), dtype=torch.float64)
    force = torch.tensor([[-.30, .02]] * n, dtype=torch.float64)
    phase = torch.ones(n, dtype=torch.long)
    live = torch.ones(n, dtype=torch.bool)
    return c, target, current, loads, force, phase, live


def test_negative_vertical_force_releases_before_downward_motion():
    c, target, current, loads, force, phase, live = _setup()
    out, active = c.apply(target, current, loads, phase, live, .01, force)
    torch.testing.assert_close(out[0, 0], current[0, 0])
    torch.testing.assert_close(out[0, 1], target[0, 1])
    assert active.tolist() == [[True, False]]
    assert c.state[0, 0].item() == 1


def test_move_requires_both_load_and_vertical_force_release():
    c, target, current, loads, force, phase, live = _setup()
    c.apply(target, current, loads, phase, live, .01, force)
    loads[:, 0] = 0.
    force[:, 0] = -.10
    for _ in range(5):
        out, _ = c.apply(target, current, loads, phase, live, .01, force)
    torch.testing.assert_close(out[0, 0], current[0, 0])
    assert c.state[0, 0].item() == 1
    force[:, 0] = 0.
    for _ in range(5):
        out, _ = c.apply(target, current, loads, phase, live, .01, force)
    assert c.state[0, 0].item() == 2
    torch.testing.assert_close(out[0, 0, 2], torch.tensor(-.001, dtype=torch.float64))


def test_recontact_or_negative_force_stops_motion_and_reclose_needs_force_gate():
    c, target, current, loads, force, phase, live = _setup()
    c.apply(target, current, loads, phase, live, .01, force)
    loads.zero_()
    force.zero_()
    for _ in range(5):
        c.apply(target, current, loads, phase, live, .01, force)
    assert c.state[0, 0].item() == 2
    current[:, :, 2] = -.001
    loads[:, 0] = 1.
    force[:, 0] = -.20
    out, _ = c.apply(target, current, loads, phase, live, .01, force)
    torch.testing.assert_close(out[0, 0], current[0, 0])
    assert c.state[0, 0].item() == 1
    for _ in range(205):
        c.apply(target, current, loads, phase, live, .01, force)
    assert c.aborted[0, 0].item()


def test_lift_phase_and_unloaded_positioning_are_not_force_repositioned():
    c, target, current, loads, force, phase, live = _setup()
    phase.fill_(3)
    loads.fill_(1.)
    force.fill_(-1.)
    out, active = c.apply(target, current, loads, phase, live, .01, force)
    torch.testing.assert_close(out, target)
    assert not active.any()
    phase.fill_(1)
    loads.zero_()
    out, active = c.apply(target, current, loads, phase, live, .01, force)
    torch.testing.assert_close(out, target)
    assert not active.any()


def test_diagnostic_floor_allows_a_lower_below_axis_goal():
    c, target, current, loads, force, phase, live = _setup()
    current[:, :, 2] = .005
    target[:, :, 2] = 0.
    clearance = torch.full((1, 2), .004, dtype=torch.float64)
    c.apply(target, current, loads, phase, live, .01, force, clearance=clearance)
    default_goal = c.goal[0, 0, 2].item()
    c2, target, current, loads, force, phase, live = _setup()
    c2.goal_table_clearance_m=.0003
    current[:, :, 2] = .005
    target[:, :, 2] = 0.
    c2.apply(target, current, loads, phase, live, .01, force, clearance=clearance)
    assert default_goal == pytest.approx(.0031)
    assert c2.goal[0, 0, 2].item() == pytest.approx(.003)
