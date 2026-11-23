import numpy as np
import pytest

from scripts.refit_dh116_power_left_contact import (
    bounded_seeds,
    owned_station_constraints,
    projected_segment_margin,
)


def test_neighbor_segment_jointly_limits_a_contact_hull():
    vertices = np.array([[.012, 0., .010], [.013, .004, .010]])
    polyline = np.array([[0., 0., 0.], [.005, 0., 0.], [.012, 0., 0.]])
    along_x = np.array([1., 0., 0.])
    gaps = [projected_segment_margin(vertices, polyline[i:i + 2], along_x, radius=.005, required=.00025)
            for i in (0, 1)]
    assert gaps[0] > 0. > gaps[1]


def test_owned_station_allows_varied_shallow_contact_and_rejects_an_extra_node():
    deep = owned_station_constraints(np.array([.001, -.00006, .0001]), owned_node=1)
    light = owned_station_constraints(np.array([.001, -.00001, .0001]), owned_node=1)
    assert min(deep) > 0. and min(light) > 0.
    extra = owned_station_constraints(np.array([.001, -.00001, -.00006]), owned_node=1)
    open_pad = owned_station_constraints(np.array([.001, .00001, .0001]), owned_node=1)
    assert min(extra) < 0. and min(open_pad) < 0.


def test_seed_selection_uses_signed_lower_bound_even_if_sdf_was_rejected_first():
    def attempt(lower):
        return {"result": "locked_sdf_or_station",
                "geometry": {"limiting_rod": {"required_m": -.00015, "lower_m": lower}}}

    farther = attempt(-.0004)
    near = attempt(-.000153)
    prior = {"clearance_refits": [{"ray_indices": [0, 1], "attempts": [farther]},
                                 {"ray_indices": [1, 2], "attempts": [{"geometry": {}}, near]}]}
    assert bounded_seeds(prior, count=1) == [([1, 2], near)]
    with pytest.raises(ValueError, match="one to three"):
        bounded_seeds(prior, count=4)
