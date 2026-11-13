"""Compose the production SDK-shaped adapters with a CPU-only physical double."""
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding, DT
from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
from test_dh116_deformable_bimanual_step import Hand
from test_dual_hand_rod_runtime import runtime, reset


def setup(monkeypatch, *, budget_end=18):
    rod, base, _, poses, centers = runtime(monkeypatch, pairs=True)
    reset(rod, poses)
    hands = {side: Hand(side, (7,)) for side in ("left", "right")}
    calls = []
    for side, hand in hands.items():
        hand.update = lambda dt, side=side: calls.append(("refresh", side))
        hand.permanent_wrench_composer = SimpleNamespace(
            reset=lambda side=side: calls.append(("reset_composer", side)),
            set_forces_and_torques=lambda side=side, **values: calls.append(("reaction", side, values)))
    bindings = [HandBinding(s, h, {7: f"/World/envs/env_7/{s}Hand"}, MountView(h))
                for s, h in hands.items()]
    # Native scene would bind both adapters to the very same live link identities.
    for key, value in rod.bindings.items():
        side, body = key.split(".")
        hand = hands[side]
        path = f"/World/envs/env_7/{side}Hand/{body}"
        value["prim_path"] = path
        hand.root_physx_view.link_paths = [list(
            f"/World/envs/env_7/{side}Hand/{name}" for name in
            ("finger14_link", "finger23_link", "finger33_link", "finger43_link", "finger53_link"))]
    sim = SimpleNamespace(current_time_step_index=2, current_time=2*DT, is_playing=lambda: True)
    def step(*, render):
        assert render is False
        calls.append(("physics",))
        sim.current_time_step_index += 1
        sim.current_time += DT
    sim.step = step
    seed = torch.stack([h.data.root_pose_w[0] for h in hands.values()])[None]
    def measure():
        return dict(physics_step=sim.current_time_step_index, root_poses_w=seed.clone(),
                    clearance_m=torch.full((1, 2), .012, dtype=torch.float64),
                    pad_poses=poses, centers_of_mass=centers)
    exchange = FlexiblePhysicalExchange(sim, rod, bindings, root_seed_w=seed,
        read_geometry=measure, physics_budget_end=budget_end)
    return exchange, sim, base, calls


def test_one_control_window_sequences_physx_rod_sdk_readback_and_next_reaction(monkeypatch):
    e, sim, base, calls = setup(monkeypatch)
    action = torch.zeros((1, 24), dtype=torch.float64)
    action[0, 17] = .4
    observed = []
    report = e.control(action, observe=lambda row: observed.append(row) or None)
    assert sim.current_time_step_index == 10 and base.rope.steps == 8
    assert [row["physics_step"] for row in observed] == list(range(3, 11))
    assert [row["reaction_source_physics_step"] for row in observed] == [None, *range(3, 10)]
    assert all(row["action_readback"]["request"]["action24"][0, 17] == .4 for row in observed)
    assert report["control_step"] == 0 and report["physical_samples"] == 8
    assert report["training_reward"] is None and report["teacher_admission"] is False
    reactions = [x for x in calls if x[0] == "reaction"]
    assert len(reactions) == 16 and not reactions[0][2]["forces"].any()
    assert reactions[2][2]["forces"][0, 0, 2] == -1
    assert reactions[3][2]["forces"][0, 0, 2] == -2


def test_terminal_observed_once_and_stops_before_noop(monkeypatch):
    e, sim, base, _ = setup(monkeypatch)
    report = e.control(torch.zeros(1, 24), observe=lambda row: "test_terminal")
    assert report["termination_reason"] == "test_terminal"
    assert sim.current_time_step_index == 3 and base.rope.steps == 1
    with pytest.raises(RuntimeError, match="closed"):
        e.control(torch.zeros(1, 24), observe=lambda row: None)


def test_insufficient_whole_control_budget_rejects_before_writes(monkeypatch):
    e, sim, base, calls = setup(monkeypatch, budget_end=9)
    with pytest.raises(RuntimeError, match="budget"):
        e.control(torch.zeros(1, 24), observe=lambda row: None)
    assert sim.current_time_step_index == 2 and base.rope.steps == 0 and not calls


def test_failed_observation_after_step_cannot_replay(monkeypatch):
    e, sim, base, _ = setup(monkeypatch)
    def fail(row):
        raise ValueError("ledger failure")
    with pytest.raises(ValueError, match="ledger failure"):
        e.control(torch.zeros(1, 24), observe=fail)
    assert sim.current_time_step_index == 3 and base.rope.steps == 1
    with pytest.raises(RuntimeError, match="closed"):
        e.control(torch.zeros(1, 24), observe=lambda row: None)


def test_original_joint_guard_keeps_failure_row_before_stopping(monkeypatch):
    e, sim, base, _ = setup(monkeypatch)
    right = e._bindings[1].hand
    update = right.update
    def fail_after_first_step(dt):
        update(dt)
        right.measured[0, right.joint_names.index("finger12")] = -.05
    right.update = fail_after_first_step
    rows = []
    result = e.control(torch.zeros(1, 24), observe=lambda row: rows.append(row))
    assert len(rows) == 1
    assert rows[0]["guard_failure"] == "right:joint_limit"
    assert result["termination_reason"] == "right:joint_limit"
    assert base.rope.steps == 1 and sim.current_time_step_index == 3


def test_rod_reaction_over_original_cap_is_never_submitted(monkeypatch):
    e, sim, base, calls = setup(monkeypatch)
    solver = base.rope.step_free
    def high_reaction(dt, steps):
        solver(dt, steps)
        base.rope.force *= 40
        base.rope.wrenches *= 40
    base.rope.step_free = high_reaction
    capture = base.rope.contact_pair_samples
    def high_capture():
        packet = capture()
        packet["vectors"][:, 8] *= 40
        return packet
    base.rope.contact_pair_samples = high_capture
    rows = []
    result = e.control(torch.zeros(1, 24), observe=lambda row: rows.append(row))
    assert result["termination_reason"] == "excess_force"
    assert len(rows) == 1 and base.rope.steps == 1
    assert all(not c[2]["forces"].any() for c in calls if c[0] == "reaction")
