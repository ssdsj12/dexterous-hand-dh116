import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import active_limits


def test_coordinated_24d_route_keeps_left_fixed_and_exact_joins():
    from tools import audit_dh116_power_synchronized_relocation as audit

    low, _ = active_limits()
    start = np.stack((np.r_[np.zeros(6), low], np.r_[np.zeros(6), low]))
    middle = start.copy()
    middle[1, 0] += .002
    middle[1, 3] += .032
    middle[1, 8:10] += [.02, .04]
    end = middle.copy()
    end[1, 1] += .004
    end[1, 9] += .08
    states, actions, joins = audit.compile_schedule(start, [middle, end])
    assert np.array_equal(states[0], start)
    assert np.array_equal(states[joins[0]], middle)
    assert np.array_equal(states[-1], end)
    assert np.array_equal(states[:, 0], np.broadcast_to(start[0], states[:, 0].shape))
    assert np.all(actions[:, audit.LEFT_ACTIONS] == 0.)
    assert np.any((actions[:, 14:18] != 0).any(axis=1) & (actions[:, 6:12] != 0).any(axis=1))
    end[0, 0] += .01
    with pytest.raises(ValueError, match="left"):
        audit.compile_schedule(start, [end])


def test_moving_palm_cannot_return_to_a_previous_material_station():
    from tools import audit_dh116_power_synchronized_relocation as audit

    owner = audit.palm_progress(9, [9], [9, 10, 11])
    assert audit.palm_progress(owner, [], [9, 10, 11]) == 9
    owner = audit.palm_progress(owner, [10, 11], [9, 10, 11])
    assert owner == 11
    with pytest.raises(ValueError, match="retreated"):
        audit.palm_progress(owner, [10], [9, 10, 11])
    with pytest.raises(ValueError, match="owned"):
        audit.palm_progress(9, [9, 12], [9, 10, 11])


def test_geometric_command_must_respect_original_workspace_even_when_rates_fit():
    from tools import audit_dh116_power_synchronized_relocation as audit

    targets = np.zeros((3, 2, 12))
    targets[:, 1, 2] = .003
    targets[2, 1, 1] = .030001
    failure = audit.check_workspace(targets[:, 1], np.zeros(3))
    assert failure["window"] == 2
    assert failure["axis"] == "world_y"
    assert failure["actual"] == pytest.approx(.030001)


def test_common_release_opens_all_three_prior_sites_to_first_order():
    from tools import audit_dh116_power_synchronized_relocation as audit

    normals = np.array([[-.305, .250, -.919], [-.580, .202, .789], [.937, .102, .333]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    direction = audit.common_release_direction(normals)
    assert np.linalg.norm(direction) == pytest.approx(1.)
    assert np.all(normals @ direction < -.05)
    assert direction[2] >= 0.
    with pytest.raises(ValueError, match="common"):
        audit.common_release_direction(np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]))
