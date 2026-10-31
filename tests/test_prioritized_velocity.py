"""CPU-only teacher velocity regressions; no Isaac or runs/ dependency."""
import importlib
import importlib.util

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import bounded_dls


def solver():
    name = 'go2_pvcnn.control.dh116.tabletop_prioritized_velocity'
    assert importlib.util.find_spec(name) is not None, 'prioritized velocity solver is missing'
    return importlib.import_module(name)


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def w_fixture():
    """W left env0, two pads, columns finger11/12/21, wrist XYZ/roll/pitch/yaw.

    Exported from the original COM Jacobian, shifted to the teacher's .8/.2
    blended points. The +1 m environment X origin IS included. Inactive finger
    columns are omitted. These constants require no external data at test time.
    At near-setpoint force correction, the old mixed solve's small normal leak
    can reverse unloading; this is not the previously misframed depth example.
    """
    point = tensor([
        [-.01805574870375, -.09136482734379, 0, 1, 0, 0, .10269104802092, -.07508312159869, -.01187775977397],
        [-.04784263027912, .01232327005654, 0, 0, 1, 0, -.08185681050063, -.07792211739870, -.06672063426432],
        [.02904581817935, .04234870979330, 0, 0, 0, 1, -.01523350243929, .05282795067130, .02481666846206],
        [0, 0, .08062028433280, 1, 0, 0, .10189515529806, -.08071374234151, -.00819325419119],
        [0, 0, .00348953707049, 0, 1, 0, -.09336379241343, -.08704273713564, -.06043787686174],
        [0, 0, .06589680416804, 0, 0, 1, -.02752914220989, .06205315671519, .01919858931545],
    ]).reshape(2, 3, 9)
    orientation = tensor([
        [-.09074458650498, -.31330264434823, 0, 0, 0, 0, .17333615423127, -.00472318632984, -.09060341965428],
        [.49639086336911, .76613562978296, 0, 0, 0, 0, -.66707724549656, -.51083910609538, .51711657696417],
        [.07387596096246, 1.68910748436288, 0, 0, 0, 0, -.56648316958580, .81594180535305, .04123042931523],
        [0, 0, -.51214383910126, 0, 0, 0, -.26237241209652, .39806796591394, -.08890822830930],
        [0, 0, 1.43504517589537, 0, 0, 0, .75661604077613, .62924280448196, -.16076402245179],
        [0, 0, 1.16197610871399, 0, 0, 0, .59867392766195, -.62720739218501, .13688772193071],
    ])
    normals = tensor([[-.98067491144223, -.16322264774155, -.10786605273212],
                      [.87452717136650, -.07576623192041, .47902161187393]])
    tangent = torch.eye(3, dtype=torch.float64)[None] - normals[:, :, None]*normals[:, None, :]
    scale = tensor([.02]*3 + [.001]*3 + [.05]*3)
    normal_j = torch.einsum('dc,dck->dk', normals, point)*scale
    normal_v = tensor([-1e-7, -1e-7])
    tangent_v = (tangent @ tensor([0, 0, .009])).flatten()
    secondary_a = torch.cat(((tangent @ point).reshape(6, 9), .0008*orientation))*scale
    secondary_b = torch.cat((tangent_v, torch.zeros(6, dtype=torch.float64)))
    lower = -tensor([.15]*3 + [.008]*3 + [.2]*3)/scale
    upper = -lower
    lower[5] = 6.  # Physical wrist Z lower bound +.006 m/s, in scaled coordinates.
    bias = torch.zeros(9, dtype=torch.float64); bias[5] = 6.
    mixed_a = torch.cat((point.reshape(6, 9), .0008*orientation))*scale
    mixed_b = secondary_b.clone(); mixed_b[:6] += (normal_v[:, None]*normals).flatten()
    return normal_j, normal_v, secondary_a, secondary_b, bias, lower, upper, mixed_a, mixed_b


def assert_preserved(n, result, info, lower, upper):
    assert (result >= lower-1e-12).all() and (result <= upper+1e-12).all()
    torch.testing.assert_close(n @ result.unsqueeze(-1), n @ info['primary'].unsqueeze(-1),
                               atol=1e-8, rtol=0)
    assert (info['normal_preservation_error'] <= 1e-8).all()


def test_correctly_framed_w_old_mixed_dls_reverses_a_normal_force_correction():
    n, _, _, _, bias, lower, upper, mixed_a, mixed_b = w_fixture()
    old = bounded_dls(mixed_a, mixed_b, bias, lower, upper, 1e-5)
    assert (n @ old > 0).any(), 'fixture must reproduce old normal sign reversal'


def test_w_priority_preserves_unloading_and_original_positive_wrist_box():
    n, v, a, b, bias, lower, upper, _, _ = w_fixture()
    result, info = solver().prioritized_velocity(n, v, a, b, bias, lower, upper, 1e-5)
    assert (n @ result < 0).all()
    assert not info['primary_direction_infeasible'].any()
    assert_preserved(n, result, info, lower, upper)
    before = (a @ info['primary']-b).square().sum()
    assert (a @ result-b).square().sum() < before


@pytest.mark.parametrize('sign', [-1., 1.])
def test_all_normal_corrections_are_preserved_under_secondary_saturation(sign):
    n = tensor([[1, 1, 0]]); v = tensor([sign*.5])
    a = torch.eye(3, dtype=torch.float64); b = tensor([2, -2, 2])
    lower = tensor([-1, -1, -1]); upper = -lower
    result, info = solver().prioritized_velocity(n, v, a, b, tensor([0, 0, 0]), lower, upper, 1e-5)
    assert_preserved(n, result, info, lower, upper)
    assert (result.abs() >= 1-1e-9).any()
    assert (a @ result-b).square().sum() < (a @ info['primary']-b).square().sum()
    assert info['secondary_iterations'] <= 4


def test_rank_deficiency_and_inactive_zero_rows_keep_secondary_in_nullspace():
    n = tensor([[1, 1, 0], [2, 2, 0], [0, 0, 0]])
    v = tensor([.6, 1.2, 0]); a = torch.eye(3, dtype=torch.float64)
    lower = tensor([-1, -1, -1]); upper = -lower
    result, info = solver().prioritized_velocity(n, v, a, tensor([1, -1, .75]),
                                                tensor([0, 0, 0]), lower, upper, 1e-5)
    assert_preserved(n, result, info, lower, upper)
    assert abs(result[2]-.75) < 1e-8


def test_zero_normal_rows_leave_the_box_available_to_secondary():
    n = torch.zeros(3, 3, dtype=torch.float64); v = tensor([0, 0, 0])
    a = torch.eye(3, dtype=torch.float64); b = tensor([2, -.25, .5])
    lower = tensor([-1, -1, -1]); upper = -lower
    result, info = solver().prioritized_velocity(n, v, a, b, tensor([0, 0, 0]), lower, upper, 1e-5)
    torch.testing.assert_close(result, tensor([1, -.25, .5]), atol=1e-8, rtol=0)
    assert_preserved(n, result, info, lower, upper)


def test_nearly_rank_deficient_projector_has_no_soft_damping_leak():
    n = tensor([[1, 0, 1, 0], [1, 1e-7, 1, 0]])
    v = tensor([.2, .2]); a = torch.eye(4, dtype=torch.float64)
    lower = tensor([-1]*4); upper = -lower
    result, info = solver().prioritized_velocity(n, v, a, tensor([0, 1, 0, 1]),
                                                tensor([0]*4), lower, upper, 1e-5)
    assert_preserved(n, result, info, lower, upper)
    assert abs(result[1]-info['primary'][1]) < 1e-7
    assert abs(result[3]-1) < 1e-8


def test_primary_reversal_is_explicit_and_preserves_the_existing_box():
    m = solver(); n = tensor([[1, 0]]); v = tensor([-1])
    lower = tensor([.2, -1]); upper = tensor([1, 1])
    result, info = m.prioritized_velocity(n, v, torch.eye(2, dtype=torch.float64),
        tensor([1, 1]), tensor([0, 0]), lower, upper, 1e-5)
    assert info['primary_direction_infeasible'].item()
    assert info['status'].item() == m.PRIMARY_DIRECTION_INFEASIBLE
    assert not info['primary_target_met'].item()
    torch.testing.assert_close(result, info['primary'], atol=0, rtol=0)
    assert_preserved(n, result, info, lower, upper)
    # The caller may relax the teacher's positive wrist floor and retry.
    relaxed, retry = m.prioritized_velocity(n, v, torch.eye(2, dtype=torch.float64),
        tensor([1, 1]), tensor([0, 0]), tensor([-1, -1]), upper, 1e-5)
    assert not retry['primary_direction_infeasible'].item()
    assert (n @ relaxed < 0).all()


def test_nonreversed_but_unattainable_primary_is_not_reported_as_target_met():
    m = solver(); n = torch.eye(2, dtype=torch.float64); v = tensor([.5, .5])
    result, info = m.prioritized_velocity(n, v, n, tensor([-1, -1]), tensor([0, 0]),
        tensor([0, 0]), tensor([.1, 1]), 1e-5)
    assert not info['primary_target_met'].item()
    assert not info['primary_direction_infeasible'].item()
    assert info['status'].item() == m.PRIMARY_TARGET_UNMET
    assert_preserved(n, result, info, tensor([0, 0]), tensor([.1, 1]))


def test_equal_box_bounds_are_pinned_without_losing_normal_equality():
    n = tensor([[1, 1, 1]]); v = tensor([.6])
    lower = tensor([.1, -1, -1]); upper = tensor([.1, 1, 1])
    result, info = solver().prioritized_velocity(n, v, torch.eye(3, dtype=torch.float64),
        tensor([-.8, -1, 1]), tensor([.1, .1, .1]), lower, upper, 1e-5)
    assert result[0].item() == .1
    assert_preserved(n, result, info, lower, upper)


def test_batch_two_env_two_hand_matches_independent_solves():
    n = tensor([[1, 1, 0], [0, 0, 0]]).expand(2, 2, 2, 3).clone()
    v = tensor([[[.2, 0], [-.4, 0]], [[.6, 0], [-.8, 0]]])
    a = torch.eye(3, dtype=torch.float64).expand(2, 2, 3, 3).clone()
    b = tensor([[[2, -2, 1], [-2, 2, -1]], [[.1, .2, .3], [.4, .5, .6]]])
    bias = torch.zeros(2, 2, 3, dtype=torch.float64)
    lower = torch.full_like(bias, -1); upper = -lower
    lower[0, 0, 2] = .25
    solve = solver().prioritized_velocity
    result, info = solve(n, v, a, b, bias, lower, upper, 1e-5)
    assert result.shape == (2, 2, 3) and info['status'].shape == (2, 2)
    assert_preserved(n, result, info, lower, upper)
    for env in range(2):
        for hand in range(2):
            single, report = solve(n[env, hand], v[env, hand], a[env, hand], b[env, hand],
                bias[env, hand], lower[env, hand], upper[env, hand], 1e-5)
            torch.testing.assert_close(result[env, hand], single, atol=1e-10, rtol=0)
            assert info['status'][env, hand] == report['status']


def test_numerical_projector_leak_falls_back_to_primary(monkeypatch):
    m = solver()
    monkeypatch.setattr(torch.linalg, 'pinv', lambda x: torch.zeros(x.shape[-1], x.shape[-2], dtype=x.dtype))
    n = tensor([[1, 0]]); lower = tensor([-1, -1]); upper = -lower
    result, info = m.prioritized_velocity(n, tensor([.3]), torch.eye(2, dtype=torch.float64),
        tensor([-1, 1]), tensor([0, 0]), lower, upper, 1e-5)
    assert info['fallback_to_primary'].item()
    assert info['status'].item() == m.SECONDARY_FALLBACK
    torch.testing.assert_close(result, info['primary'], atol=0, rtol=0)
    assert_preserved(n, result, info, lower, upper)


def test_invalid_box_is_rejected():
    with pytest.raises(ValueError, match='box'):
        solver().prioritized_velocity(tensor([[1]]), tensor([0]), tensor([[1]]), tensor([0]),
            tensor([0]), tensor([1]), tensor([-1]), 1e-5)
