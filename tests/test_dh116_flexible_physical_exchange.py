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


def prepare_supported_handoff(monkeypatch, *, proximal=False):
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PowerSharedRodRuntime
    from test_dual_hand_rod_runtime import Grid

    old, sim, base, calls = setup(monkeypatch, budget_end=18)
    palms, initial, grid = {}, {}, {}
    for binding in old._bindings:
        hand = binding.hand
        name = binding.side + ".base_link"
        view = hand.root_physx_view
        view.link_paths[0].append(f"/World/envs/env_7/{binding.side}Hand/base_link")
        palms[name] = dict(old._rod.bindings[binding.side+".finger14_link"], body_index=5,
                           body_name="base_link", prim_path=view.link_paths[0][5])
        initial[name]=(np.zeros(3), np.array([0.,0.,0.,1.]))
        grid[name]=Grid()
    support = {name:Grid() for name in ("support_0","support_1")}
    support_poses = {name:(np.zeros(3),np.array([0.,0.,0.,1.])) for name in support}
    options = {}
    if proximal:
        bound, grids, poses = {}, {}, {}
        for binding, token in zip(old._bindings, ("a", "b"), strict=True):
            view = binding.hand.root_physx_view
            name = binding.side + ".finger52_link"
            view.link_paths[0].append(f"/World/envs/env_7/{binding.side}Hand/finger52_link")
            bound[name] = dict(palms[binding.side+".base_link"], body_index=6, body_name="finger52_link",
                prim_path=view.link_paths[0][6], capture_sha256=token*64, source_geometry_sha256=token*64,
                cooked_geometry_sha256=token*64, sdf_sha256=token*64)
            grids[name] = Grid()
            poses[name] = initial[binding.side+".base_link"]
        options = dict(proximal_bindings=bound, proximal_grids=grids, proximal_poses=poses)
    base.rope.enable_contact_pair_capture = lambda count: None
    original_capture=base.rope.contact_pair_samples
    count = 17 if proximal else 15
    base.rope.contact_pair_samples=lambda: dict(original_capture(), capacity=len(base.nodes)*count*12, mesh_count=count)
    old._rod.require_pairs = True
    supported = PowerSharedRodRuntime.from_original_reset(old._rod, palm_bindings=palms,
        palm_poses=initial, palm_grids=grid, expected_clock=old._rod.clock(),
        support_grids=support, support_poses=support_poses, **options)
    initial = {**initial, **options.get("proximal_poses", {})}
    scene = old._read_geometry
    def read():
        geometry=scene()
        geometry["pad_poses"]={**geometry["pad_poses"], **initial}
        geometry["centers_of_mass"]={**geometry["centers_of_mass"], **{key:np.zeros(3) for key in initial}}
        return geometry

    sim.step(render=False)
    geometry=read()
    before=supported.step(geometry["pad_poses"],geometry["centers_of_mass"],physics_step=3)
    return old, supported, sim, before, read, calls


@pytest.mark.parametrize("proximal, count", [(False, 6), (True, 7)])
def test_native_power_action_exchange_retains_real_clock_and_next_palm_reactions(monkeypatch, proximal, count):
    old, rod, sim, previous, read, calls = prepare_supported_handoff(monkeypatch, proximal=proximal)
    attached=FlexiblePhysicalExchange(sim, rod, old._bindings, root_seed_w=old._seed,
        read_geometry=read, physics_budget_end=20, continue_supported_rod=True)
    rows=[]
    attached.control(torch.zeros(1,24),observe=lambda row:rows.append(row))
    assert sim.current_time_step_index==11 and [row["physics_step"] for row in rows]==list(range(4,12))
    assert rows[0]["reaction_source_physics_step"]==3
    assert rows[0]["rod"]["rod_tick"]==24
    for side in ("left", "right"):
        actual=rows[0]["issued_composer_wrench"][side]
        expected=previous["hand_wrenches_com_world"][side]
        assert actual["body_ids"]==expected["body_ids"] and len(actual["body_ids"])==count
        assert (actual["forces"]==expected["forces"]).all()
    with pytest.raises(ValueError,match="already attached"):
        FlexiblePhysicalExchange(sim,rod,old._bindings,root_seed_w=old._seed,
            read_geometry=read,physics_budget_end=28,continue_supported_rod=True)


def test_supported_handoff_rejects_a_stale_measured_rod_row_before_writes(monkeypatch):
    old, rod, sim, previous, read, calls=prepare_supported_handoff(monkeypatch)
    rod._base.last["physics_step"]=2
    before=list(calls)
    with pytest.raises(ValueError,match="same-clock"):
        FlexiblePhysicalExchange(sim,rod,old._bindings,root_seed_w=old._seed,
            read_geometry=read,physics_budget_end=20,continue_supported_rod=True)
    assert calls==before
