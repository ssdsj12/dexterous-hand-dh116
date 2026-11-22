"""Reaction channels retain link identity across serialized binding order; CPU only."""
import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from test_fresh_supported_physical_exchange import prepared, attach
from test_dh116_flexible_physical_exchange import prepare_supported_handoff


def test_fresh_reset_zero_route_and_next_measured_feedback_share_canonical_body_order(monkeypatch):
    c, _, plan, receipt = prepared(monkeypatch)
    c.rod.bindings = dict(sorted(c.rod.bindings.items()))
    exchange = attach(c, plan, receipt)
    rows = []
    exchange.control(torch.zeros(1, 24), observe=rows.append)
    assert c.sim.current_time_step_index == 10 and c.rope.steps == 8
    for side in ("left", "right"):
        first = rows[0]["issued_composer_wrench"][side]
        assert first["body_ids"] == [0, 1, 2, 3, 4, 5, 6]
        assert not first["forces"].any() and not first["torques"].any()
        for previous, current in zip(rows, rows[1:]):
            actual = current["issued_composer_wrench"][side]
            measured = previous["rod"]["hand_wrenches_com_world"][side]
            assert actual["body_ids"] == measured["body_ids"] == [0, 1, 2, 3, 4, 5, 6]
            for name in ("forces", "torques"):
                np.testing.assert_array_equal(actual[name], measured[name])
    assert rows[1]["issued_composer_wrench"]["right"]["forces"][0, 6, 0] == pytest.approx(-.2)


@pytest.mark.parametrize("proximal,count", [(False, 6), (True, 7)])
def test_supported_handoff_serialization_order_cannot_reassign_force_or_torque(monkeypatch, proximal, count):
    old, rod, sim, _, read, calls = prepare_supported_handoff(monkeypatch, proximal=proximal)
    rod.bindings = dict(sorted(rod.bindings.items()))
    # Distinct measured channel values make any force/torque permutation observable.
    expected = {}
    for side_index, side in enumerate(("left", "right")):
        route = rod._base.last["hand_wrenches_com_world"][side]
        route["forces"][:] = np.arange(count * 3).reshape(1, count, 3) * .01 + side_index
        route["torques"][:] = np.arange(count * 3).reshape(1, count, 3) * -.001 - side_index
        expected[side] = {name: route[name].copy() for name in ("forces", "torques")}
    before = len(calls)
    exchange = FlexiblePhysicalExchange(sim, rod, old._bindings, root_seed_w=old._seed,
        read_geometry=read, physics_budget_end=20, continue_supported_rod=True)
    rows = []
    exchange.control(torch.zeros(1, 24), observe=rows.append)
    assert sim.current_time_step_index == 11
    assert rows[0]["reaction_source_physics_step"] == 3
    for side in ("left", "right"):
        first = rows[0]["issued_composer_wrench"][side]
        assert first["body_ids"] == list(range(count))
        for name in ("forces", "torques"):
            np.testing.assert_array_equal(first[name], expected[side][name])
        # The first writer call, rather than an internal constructor field, is the contract.
        first_write = next(call[2] for call in calls[before:] if call[:2] == ("reaction", side))
        assert first_write["body_ids"] == list(range(count))
        for name in ("forces", "torques"):
            np.testing.assert_allclose(first_write[name].numpy(), expected[side][name], atol=1e-7, rtol=1e-7)
