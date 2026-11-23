import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.search_dh116_power_surface_pairs import anchored_translation, plane_ray_candidates


def test_anchor_keeps_the_owned_node_under_a_rotated_pad_surface():
    rotation = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    pad = np.array([.02, -.01, .003])
    node = np.array([.42, .17, .34])
    origin = anchored_translation(rotation, pad, node)
    np.testing.assert_allclose(rotation @ pad + origin, node, atol=1e-14)


def test_surface_ray_is_placed_at_the_locked_negative_sdf_gap():
    step = .001
    origin = np.array([-.015, -.015, -.015])
    coords = origin[0] + step * np.arange(31)
    # A locally planar interpolation in the +X direction.
    values = np.broadcast_to((coords - .001)[:, None, None], (31, 31, 31))
    grid = dict(origin=origin, spacing=step, values=values)
    cooked = np.array([[.001, 0., 0.]])
    rays = np.array([[1., 0., 0.]])
    points, normals = plane_ray_candidates(cooked + [.00005, 0., 0.], cooked, rays, grid, radius=.005)
    np.testing.assert_allclose(points, [[.00595, 0., 0.]], atol=1e-10)
    np.testing.assert_allclose(normals, rays, atol=1e-12)
    none, _ = plane_ray_candidates(cooked + [.00020, 0., 0.], cooked, rays, grid, radius=.005)
    assert not len(none)


def test_axial_error_uses_world_contact_normal_and_material_tangent():
    from scripts.search_dh116_power_surface_pairs import axial_contact_error

    n = np.array([[1., 0., 0.], [0., 1., 0.]])
    t = np.array([[0., 1., 0.], [0., -1., 0.]])
    np.testing.assert_allclose(axial_contact_error(n, t, radius=.005), [0., -.005])


def test_clearance_penalty_covers_both_rod_segment_endpoints_in_world():
    from scripts.search_dh116_power_surface_pairs import separation_deficit

    hull = np.array([[.012, 0., .012], [.012, .01, .013]])
    a = np.array([0., 0., 0.])
    b = np.array([.01, 0., .001])
    # The +X axis loses clearance at endpoint b despite a large distance from a.
    np.testing.assert_allclose(separation_deficit(hull, a, b, np.array([[1., 0., 0.]]),
                                                  radius=.005, required=.00025), .00325)
    # A second valid supporting direction removes the stale-axis penalty.
    assert separation_deficit(hull, a, b, np.array([[1., 0., 0.], [0., 0., 1.]]),
                              radius=.005, required=.00025) == 0.


def test_sliding_sdf_residual_keeps_target_and_penalizes_another_penetrating_node():
    from scripts.search_dh116_power_surface_pairs import selected_station_residuals

    gaps = np.array([.00032, -.00005, -.00007])
    residuals = selected_station_residuals(gaps, owned_node=1, target=-.00005)
    np.testing.assert_allclose(residuals, [0., 0., .00007])


def test_final_candidate_gate_rejects_wrist_and_coupled_joint_overtravel():
    from scripts.search_dh116_power_surface_pairs import evaluate, route

    hand = {"origin": np.zeros(3), "preload_targets": {}}
    scene = {"wire_radius_m": .005}
    nodes = np.zeros((2, 3))
    state = np.zeros(12)
    state[3] = route.WRIST["rotation_limit_rad"] + .001
    assert evaluate(hand, state, nodes, scene, side="left")[0] == "wrist_workspace"

    state[3] = 0.
    state[6:] = -1.
    assert evaluate(hand, state, nodes, scene, side="left")[0] == "coupled_joint_limits"
    state[0] = np.nan
    assert evaluate(hand, state, nodes, scene, side="left")[0] == "invalid_state"


def test_free_refit_can_decouple_wrist_origin_from_the_seed_thumb_ray(monkeypatch):
    from scripts import search_dh116_power_surface_pairs as search

    def transform(y=0.):
        value = np.eye(4)
        value[1, 3] = y
        return value

    class FixedFK:
        def poses(self, q):
            return {"/base_link": transform(), "/finger14_link": transform(.02),
                    "/finger23_link": transform(-.02)}

    monkeypatch.setattr(search, "grid_field", lambda _, **kwargs: lambda points: np.linalg.norm(
        np.atleast_2d(points), axis=1))
    fitted_stations = set()
    original_residuals = search.selected_station_residuals

    def record_station(gaps, *, owned_node, target):
        fitted_stations.add(owned_node)
        return original_residuals(gaps, owned_node=owned_node, target=target)

    monkeypatch.setattr(search, "selected_station_residuals", record_station)
    model = dict(pose=[0., 0., 0., 1., 0., 0., 0.], fk=FixedFK(),
                 source_colliders={}, cooked_pads={})
    hand = dict(model=model, origin=np.zeros(3), seed=Rotation.identity(),
                preload_targets={"base_link": 2, "finger14_link": 3, "finger23_link": 1},
                palm_grid={}, grids={"finger14_link": {}, "finger23_link": {}},
                pad_offsets={"finger14_link": np.eye(4), "finger23_link": np.eye(4)})
    nodes = np.array([[.00495, y, 0.] for y in [-.04, -.02, 0., .02, .04]])
    low, high = search.active_limits()
    start = np.r_[np.zeros(6), (low + high) / 2.]
    # This seed ray would anchor the wrist beyond the original +Z workspace.
    thumb = np.array([[.00495, 0., -.10], [1., 0., 0.]])
    partner = np.array([[.00495, 0., 0.], [-1., 0., 0.]])
    state, error = search.fit_pair(hand, start, nodes, thumb, partner, radius=.005,
                                   align_wire_normals=False, sliding_contact=True,
                                   free_wrist=True)
    assert np.isfinite(error)
    assert fitted_stations == {1, 2, 3}
    assert np.all(state[:3] >= search.route.WRIST["palm_lower_m"])
    assert np.all(state[:3] <= search.route.WRIST["palm_upper_m"])
    rotation = Rotation.from_euler("XYZ", state[3:6])
    fitted_thumb_ray = state[:3] + rotation.apply(thumb[0] + [0., .02, 0.])
    assert np.linalg.norm(fitted_thumb_ray - nodes[3]) > .06
    translated_nodes = nodes + [.008, 0., 0.]
    aligned_thumb = np.array([[.00495, 0., 0.], [1., 0., 0.]])
    state, _ = search.fit_pair(hand, start, translated_nodes, aligned_thumb, partner, radius=.005,
                               align_wire_normals=True, sliding_contact=True, free_wrist=True)
    assert state[0] == pytest.approx(.008, abs=.0002)
    with pytest.raises(ValueError, match="sliding"):
        search.fit_pair(hand, start, nodes, thumb, partner, radius=.005,
                        align_wire_normals=False, free_wrist=True)


def test_refit_grid_outside_residual_increases_with_distance():
    from scripts.search_dh116_power_surface_pairs import grid_field

    grid = dict(origin=np.zeros(3), spacing=.01, values=np.full((2, 2, 2), .008))
    points = np.array([[.005, .005, .005], [.012, .005, .005], [.014, .005, .005]])
    np.testing.assert_allclose(grid_field(grid, track_outside=True)(points), [.008, .010, .012])
    np.testing.assert_allclose(grid_field(grid)(points), [.008, .02, .02])


def test_free_wrist_search_requires_sliding_and_bounded_refits_for_direct_callers():
    from scripts.search_dh116_power_surface_pairs import search_side

    with pytest.raises(ValueError, match="sliding and bounded refits"):
        search_side(None, "left", None, None, None, None, None, align_wire_normals=True,
                    clearance_rounds=0, sliding_contact=True, free_wrist=True)


def test_sliding_normal_follows_the_sdf_gradient_at_the_owned_node():
    from scripts.search_dh116_power_surface_pairs import sliding_world_normal

    def parabolic_field(points):
        points = np.asarray(points)
        return points[:, 0] + points[:, 1] ** 2

    rotation = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    normal = sliding_world_normal(parabolic_field, np.array([0., .5, 0.]), rotation)
    np.testing.assert_allclose(normal, [-2**-.5, 2**-.5, 0.], atol=1e-8)
    assert sliding_world_normal(lambda points: np.zeros(len(points)), np.zeros(3), rotation) is None
