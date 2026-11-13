"""Connect the real clock guards and SDK-shaped pose/rod readers on CPU."""
import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import LiveGeometryReader
from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from test_dh116_flexible_live_geometry import setup as geometry_setup
from test_dh116_flexible_physical_exchange import setup as exchange_setup


def test_reader_allows_only_explicit_one_tick_sdk_to_rod_handoff():
    reader, hands, sim, rod_clock = geometry_setup()
    previous = dict(rod_clock)
    sim.current_time_step_index += 1
    sim.current_time += 1/480
    hands[1].transforms[..., 2] += .001
    with pytest.raises(ValueError, match="clock"):
        reader()
    row = reader.read_after_sdk_step(previous)
    assert row["physics_step"] == 3
    assert row["geometry_phase"] == "sdk_complete_rod_pending"
    assert row["rod_source_physics_step"] == 2
    assert row["pad_poses"]["right.finger14_link"][0][2] == pytest.approx(.361)
    rod_clock["episode_uid"] = "foreign"
    with pytest.raises(ValueError, match="clock"):
        reader.read_after_sdk_step(previous)


def test_handoff_rejects_clock_changes_during_read():
    reader, hands, sim, rod_clock = geometry_setup()
    previous = dict(rod_clock)
    sim.current_time_step_index += 1
    sim.current_time += 1/480
    def change_clock():
        rod_clock["solver_time_s"] += .001
        return hands[1].coms
    hands[1].root_physx_view.get_coms = change_clock
    with pytest.raises(ValueError, match="clock"):
        reader.read_after_sdk_step(previous)


def test_exchange_uses_live_sdk_geometry_at_each_boundary(monkeypatch):
    existing, sim, base, _ = exchange_setup(monkeypatch)
    example, hands, _, _ = geometry_setup()
    # Match the exchange fake's five original pad slots, and add a real base link.
    names = [name for name in hands[0].body_names if name != "base_link"] + ["base_link"]
    for binding in existing._bindings:
        hand = binding.hand
        side = binding.side
        root = binding.env_paths[7]
        hand.body_names = names.copy()
        hand.root_physx_view.link_paths = [[f"{root}/{name}" for name in names]]
        transforms = np.zeros((1, 6, 7), dtype=np.float32)
        transforms[..., 2] = .40
        transforms[..., 6] = 1
        hand.root_physx_view.get_link_transforms = lambda t=transforms: t
        hand.root_physx_view.get_coms = lambda t=transforms: np.zeros_like(t)
        binding.mount.base_id = 5
    clock = existing._rod.clock
    inspected = dict(identity=[clock()], pad_bindings=existing._rod.bindings,
        shapes=example._shapes, table_top_m=.30)
    reader = LiveGeometryReader(sim, existing._bindings, inspected, read_clock=clock)
    original = sim.step
    right = existing._bindings[1].hand
    def advance(*, render):
        original(render=render)
        right.root_physx_view.get_link_transforms()[0, 0, 0] = .001 * sim.current_time_step_index
    sim.step = advance
    exchange = FlexiblePhysicalExchange(sim, existing._rod, existing._bindings,
        root_seed_w=existing._seed, read_geometry=reader,
        read_geometry_after_sdk=reader.read_after_sdk_step, physics_budget_end=18)
    rows = []
    result = exchange.control(torch.zeros((1,24)), observe=rows.append)
    assert result["physical_samples"] == 8 and base.rope.steps == 8
    assert [row["physics_step"] for row in rows] == list(range(3,11))
    assert [pose[1][0] for pose in base.rope.poses if pose[0] == 6 and not pose[3]] == pytest.approx(
        [i * .001 for i in range(3,11)])
    assert clock()["physics_step"] == sim.current_time_step_index == 10


@pytest.mark.parametrize("skipped", [0, 2])
def test_handoff_refuses_same_or_skipped_sdk_step(skipped):
    reader, _, sim, clock = geometry_setup()
    sim.current_time_step_index += skipped
    sim.current_time += skipped/480
    with pytest.raises(ValueError, match="SDK clock"):
        reader.read_after_sdk_step(dict(clock))


def test_failed_handoff_poisoning_prevents_replay(monkeypatch):
    existing, sim, base, _ = exchange_setup(monkeypatch)
    def stale(previous):
        return dict(geometry_phase="sdk_complete_rod_pending",
                    rod_source_physics_step=previous["physics_step"] - 1)
    exchange = FlexiblePhysicalExchange(sim, existing._rod, existing._bindings,
        root_seed_w=existing._seed, read_geometry=existing._read_geometry,
        read_geometry_after_sdk=stale, physics_budget_end=18)
    with pytest.raises(ValueError, match="handoff geometry"):
        exchange.control(torch.zeros((1,24)), observe=lambda row: pytest.fail("unexpected row"))
    assert sim.current_time_step_index == 3 and base.rope.steps == 0
    with pytest.raises(RuntimeError, match="closed"):
        exchange.control(torch.zeros((1,24)), observe=lambda row: None)
