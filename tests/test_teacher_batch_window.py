"""Batch-window lifecycle with the real exchange and CPU-native adapters."""

import pytest
import torch

from test_dh116_flexible_physical_exchange import setup
from test_dh116_poststep_clearance_guard import guarded


@pytest.mark.parametrize("truncated", [False, True])
def test_window_end_requires_ppo_record_before_local_reset(monkeypatch, truncated):
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import TeacherPhysicalGroup
    from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
    from go2_pvcnn.control.dh116.tabletop_deformable_rollout_admission import gae_boundary
    from test_dh116_privileged_teacher import trainer

    a, sim, _, _ = setup(monkeypatch, env_id=0, budget_end=26)
    b, _, _, _ = setup(monkeypatch, env_id=1, sim=sim, budget_end=26)
    group = TeacherPhysicalGroup((a, b))
    engine = trainer(steps=2, environments=2, epochs=1, minibatches=1)
    observations = torch.zeros(2, 371)
    next_observations = torch.full((2, 371), .1)
    sample = engine.collect_step(observations, observations)
    event = "time_limit" if truncated else "task_success"
    events = []
    rows = []

    def observe(row):
        rows.append(row)
        if row["rod"]["env_id"] == 0 and row["substep"] == 7:
            return event
        return None

    def record(sample, recent_rows, *, first):
        with torch.no_grad():
            values = engine.model.value(next_observations)
        boundaries = []
        issued = []
        for row in recent_rows[-2:]:
            end = {key: row["rod"][key] for key in (
                "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick", "timestamp_s"
            )}
            end["phase"] = "post_action_pre_reset"
            done = first and end["env_id"] == 0
            terminated = done and not truncated
            boundaries.append(gae_boundary(terminated, done and truncated,
                endpoint_identity=end, next_value_identity=None if terminated else end))
            issued.append(row["action_readback"]["request"]["action24"][0].clone())
            if terminated:
                values[end["env_id"]] = 0.
        engine.record_step(torch.tensor([1., 2.]),
            torch.tensor([b["done"] for b in boundaries]), values,
            truncated=torch.tensor([b["truncated"] for b in boundaries]),
            bootstrap_mask=torch.tensor([b["bootstrap_mask"] for b in boundaries]),
            executed_actions=torch.stack(issued), collection_token=sample["collection_token"])
        events.append(("commit", sim.current_time_step_index))

    def commit(window):
        assert window["terminal_reasons"] == {0: event}
        with pytest.raises(RuntimeError, match="complete"):
            group.reset_done(lambda *_: pytest.fail("Reset ran during commit"))
        record(sample, rows, first=True)

    assert not torch.equal(sample["actions"][0], sample["actions"][1])
    group.control(sample["actions"], observe=observe, commit=commit)
    assert sim.current_time_step_index == 10
    assert b._rod.clock()["rod_tick"] == 96
    with pytest.raises(RuntimeError, match="Reset"):
        group.control(sample["actions"], observe=rows.append, commit=commit)

    def reset_lane(env_id, previous):
        assert env_id == 0 and previous is a
        assert events == [("commit", 10)] and len(engine.rollout) == 1
        events.append(("reset", sim.current_time_step_index))
        rod = previous._rod
        rod.reset(physics_step=sim.current_time_step_index, episode_uid="episode-0-next",
                  reset_id=2, pad_poses=previous._read_geometry()["pad_poses"])
        return FlexiblePhysicalExchange(sim, rod, previous._bindings,
            root_seed_w=previous._seed, read_geometry=previous._read_geometry,
            physics_budget_end=26)

    other_before = b._rod.clock()
    group.reset_done(reset_lane)
    assert b._rod.clock() == other_before
    next_sample = engine.collect_step(next_observations, next_observations)
    following = group.control(next_sample["actions"], observe=rows.append,
                              commit=lambda _: record(next_sample, rows, first=False))
    assert following["control_step"] == 1
    assert events == [("commit", 10), ("reset", 10), ("commit", 18)]
    assert rows[16]["rod"]["episode_uid"] == "episode-0-next"
    assert rows[16]["policy_step"] == 0 and rows[17]["policy_step"] == 1
    assert rows[16]["reaction_source_physics_step"] is None
    assert rows[17]["reaction_source_physics_step"] == 10
    assert b._rod.clock()["rod_tick"] == 192
    ledger = engine.rollout.batch()
    assert ledger["done"].tolist() == [True, False, False, False]
    assert ledger["bootstrap_mask"].tolist() == [truncated, True, True, True]
    _, advantage = engine.rollout.returns_and_advantages(engine.config)
    delta = ledger["rewards"] + engine.config.gamma * ledger["next_value"] * ledger["bootstrap_mask"] - ledger["values"]
    torch.testing.assert_close(advantage[0], delta[0])
    torch.testing.assert_close(advantage[1], delta[1] + engine.config.gamma * engine.config.lam * delta[3])


@pytest.mark.parametrize("breach", [3, 10])
def test_clearance_failure_rejects_group_before_commit(monkeypatch, breach):
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import TeacherPhysicalGroup

    a, sim, _, _ = guarded(monkeypatch, env_id=0, breach_at=breach)
    b, _, _, _ = guarded(monkeypatch, env_id=1, sim=sim)
    group = TeacherPhysicalGroup((a, b))
    rows = []
    report = group.control(torch.zeros(2, 24), observe=rows.append,
                           commit=lambda _: pytest.fail("guarded window was committed"))
    assert report["terminal_reasons"] == {0: "post_step_source_clearance:right"}
    assert [row["rod"]["env_id"] for row in rows[-2:]] == [0, 1]
    assert sim.current_time_step_index == breach
    with pytest.raises(RuntimeError, match="closed"):
        group.control(torch.zeros(2, 24), observe=rows.append, commit=rows.append)


def test_reset_cannot_renew_the_owned_physics_budget(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import TeacherPhysicalGroup
    from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange

    a, sim, _, _ = setup(monkeypatch, env_id=0, budget_end=18)
    b, _, _, _ = setup(monkeypatch, env_id=1, sim=sim, budget_end=18)
    group = TeacherPhysicalGroup((a, b))
    group.control(torch.zeros(2, 24), observe=lambda row:
        "time_limit" if row["substep"] == 7 and row["rod"]["env_id"] == 0 else None,
        commit=lambda row: None)

    def extend(_, old):
        old._rod.reset(physics_step=10, episode_uid="episode-0-next", reset_id=2,
                       pad_poses=old._read_geometry()["pad_poses"])
        return FlexiblePhysicalExchange(sim, old._rod, old._bindings,
            root_seed_w=old._seed, read_geometry=old._read_geometry,
            physics_budget_end=26)

    with pytest.raises(ValueError, match="physics budget"):
        group.reset_done(extend)
    assert sim.current_time_step_index == 10
    with pytest.raises(RuntimeError, match="closed"):
        group.control(torch.zeros(2, 24), observe=lambda row: None, commit=lambda row: None)
