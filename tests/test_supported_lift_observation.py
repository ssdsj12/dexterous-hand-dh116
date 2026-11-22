"""Read the flexible rod, hands and support from one physical endpoint."""
import importlib

import numpy as np
import pytest

from test_dh116_power_palm_contact_evidence import build


def api():
    name = "go2_pvcnn.control.dh116.tabletop_supported_lift_observation"
    assert importlib.util.find_spec(name), "supported lift observation is missing"
    return importlib.import_module(name)


def sample(**kwargs):
    row = build(proximal=True, load=.20, **kwargs)
    row["solver_time_s"] = 1 / 480
    return row


def at_reset(rod):
    return dict(rod, physics_step=2, rod_tick=0, timestamp_s=2 / 480, solver_time_s=0.)


def reference(rod):
    return api().FixedLiftReference.from_rod_reset(at_reset(rod), radius_m=.005, table_top_m=.3)


@pytest.mark.parametrize("grasp", ["pinch", "tripod", "power"])
def test_lift_uses_lowest_rod_node_and_owns_measurement(grasp):
    rod = sample()
    fixed = reference(rod)
    rod["node_positions"][0, :, 2] += .04
    rod["node_positions"][0, 0, 2] -= .02
    observed = api().observe_supported_lift(rod, fixed, grasp=grasp)
    assert observed.bilateral
    assert observed.registered_support_clear
    assert observed.off_support_lift_m == pytest.approx(.02)
    assert observed.median_delta_m == pytest.approx(.04)
    rod["node_positions"][0, :, 2] = 0
    assert observed.node_positions_world_m[0][2] == pytest.approx(.385)
    assert fixed.minimum_z_m == pytest.approx(.365)


def test_wire_on_support_cannot_get_off_support_lift_credit():
    rod = sample(table=True)
    fixed = reference(rod)
    rod["node_positions"][0, :, 2] += .04
    observed = api().observe_supported_lift(rod, fixed, grasp="tripod")
    assert observed.bilateral and observed.minimum_delta_m == pytest.approx(.04)
    assert not observed.registered_support_clear
    assert observed.off_support_lift_m is None
    assert all(load > 0 for _, load in observed.peak_support_load_n)


@pytest.mark.parametrize("kwargs", [
    {"missing_palm": "right.base_link"}, {"crossed": True}, {"crossed_link": "left.finger52_link"},
])
def test_power_requires_both_palms_and_correct_material_side(kwargs):
    rod = sample(**kwargs)
    fixed = reference(rod)
    rod["node_positions"][0, :, 2] += .04
    observed = api().observe_supported_lift(rod, fixed, grasp="power")
    assert not observed.bilateral
    assert observed.off_support_lift_m is None


def test_other_reset_cannot_supply_the_height_origin():
    rod = sample()
    fixed = reference(rod)
    rod["reset_id"] = 2
    with pytest.raises(ValueError, match="Reset identity"):
        api().observe_supported_lift(rod, fixed, grasp="tripod")


def test_stale_solver_boundary_cannot_provide_a_lift_observation():
    rod = sample()
    fixed = reference(rod)
    rod["solver_time_s"] = 0.
    with pytest.raises(ValueError, match="solver clock"):
        api().observe_supported_lift(rod, fixed, grasp="tripod")


def test_power_keeps_observed_proximal_links_separate_from_required_links():
    rod = sample()
    observed = api().observe_supported_lift(rod, reference(rod), grasp="power")
    assert observed.additional_hand_links == ("left.finger52_link", "right.finger52_link")
    assert all("finger52_link" not in links for _, links in observed.required_links)


def test_reset_radius_rejects_boolean():
    with pytest.raises(ValueError, match="radius"):
        api().FixedLiftReference.from_rod_reset(at_reset(sample()), radius_m=True, table_top_m=.3)
