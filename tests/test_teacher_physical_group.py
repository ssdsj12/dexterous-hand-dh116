"""CPU multi-environment physical barrier using the real DH116 exchange code."""
from collections import Counter

import numpy as np
import pytest
import torch

from test_dh116_flexible_physical_exchange import setup
from test_dh116_poststep_clearance_guard import guarded


@pytest.mark.parametrize("num_envs", [2, 32])
def test_shared_stage_steps_once_and_keeps_every_action_bound_to_its_lane(monkeypatch, num_envs):
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import control_group

    lane, sim, base, calls = setup(monkeypatch, env_id=0)
    lanes = [lane]
    bases = [base]
    phase = []
    step = sim.step
    def stage(*, render):
        phase.append(("physx", sim.current_time_step_index))
        step(render=render)
    sim.step = stage
    for env_id in range(1, num_envs):
        lane, _, base, _ = setup(monkeypatch, env_id=env_id, sim=sim)
        lanes.append(lane)
        bases.append(base)
    for env_id, lane in enumerate(lanes):
        write = lane._write_reactions
        def write_reaction(*args, write=write, env_id=env_id, **kwargs):
            phase.append(("submit", env_id))
            write(*args, **kwargs)
        lane._write_reactions = write_reaction
        step_rod = lane._rod.step
        def step_solver(*args, step_rod=step_rod, env_id=env_id, **kwargs):
            phase.append(("solver", env_id))
            return step_rod(*args, **kwargs)
        lane._rod.step = step_solver

    actions = torch.zeros((num_envs, 24), dtype=torch.float64)
    actions[:, 12] = torch.arange(num_envs, dtype=torch.float64) / (num_envs * 2)
    rows = []
    result = control_group(lanes, actions, observe=rows.append)
    assert result["physical_samples_per_env"] == 8 and result["terminal_reasons"] == {}
    assert sim.current_time_step_index == 10 and len(rows) == 8 * num_envs
    assert [base.rope.steps for base in bases] == [8] * num_envs
    for substep in range(8):
        start = substep * (2 * num_envs + 1)
        stop = start + num_envs
        assert phase[start:stop] == [("submit", e) for e in range(num_envs)]
        assert phase[stop] == ("physx", 2 + substep)
        assert phase[stop + 1:stop + num_envs + 1] == [("solver", e) for e in range(num_envs)]
    assert Counter(row["rod"]["env_id"] for row in rows) == Counter({e: 8 for e in range(num_envs)})
    for row in rows:
        env_id = row["rod"]["env_id"]
        np.testing.assert_array_equal(row["action_readback"]["request"]["action24"][0], actions[env_id])
        assert row["rod"]["rod_tick"] == (row["physics_step"] - 2) * 12


def test_group_retains_terminal_clearance_and_other_lane_row_before_stopping(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import control_group

    left, sim, base_a, _ = guarded(monkeypatch, env_id=0, breach_at=3)
    right, _, base_b, _ = guarded(monkeypatch, env_id=1, sim=sim)
    rows = []
    result = control_group((left, right), torch.zeros(2, 24), observe=rows.append)
    assert [row["rod"]["env_id"] for row in rows] == [0, 1]
    assert result["terminal_reasons"] == {0: "post_step_source_clearance:right"}
    assert rows[0]["post_step_clearance"]["passed"] is False
    assert rows[1]["post_step_clearance"]["passed"] is True
    assert base_a.rope.steps == base_b.rope.steps == 1
    with pytest.raises(RuntimeError, match="closed"):
        control_group((left, right), torch.zeros(2, 24), observe=rows.append)
    assert sim.current_time_step_index == 3


def test_group_rejects_shared_solver_before_first_write(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import control_group

    a, sim, base, calls = setup(monkeypatch, env_id=0)
    b, _, other, remaining_calls = setup(monkeypatch, env_id=1, sim=sim)
    other.rope = base.rope
    with pytest.raises(ValueError, match="distinct.*solver"):
        control_group((a, b), torch.zeros(2, 24), observe=lambda row: None)
    assert not calls and not remaining_calls
    assert sim.current_time_step_index == 2 and base.rope.steps == 0


@pytest.mark.parametrize("fault_sdk_step", [3, 10])
def test_source_failure_stops_the_batch_without_committing_even_on_last_tick(monkeypatch, fault_sdk_step):
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import TeacherPhysicalGroup

    left, sim, _, _ = setup(monkeypatch, env_id=0)
    right, _, _, _ = setup(monkeypatch, env_id=1, sim=sim)
    complete = left._complete_substep

    def broken_reader(pending, sdk):
        row = complete(pending, sdk)
        if sdk == fault_sdk_step:
            row["native_teacher_source_error"] = {"type": "SourceFailure", "message": "fixture reader failed"}
        return row

    left._complete_substep = broken_reader
    rows = []
    result = TeacherPhysicalGroup((left, right)).control(
        torch.zeros((2, 24)), observe=rows.append,
        commit=lambda _: pytest.fail("source-failed window was committed"),
    )
    assert result["aborted"] and result["terminal_reasons"] == {0: "native_teacher_source_error"}
    assert sim.current_time_step_index == fault_sdk_step
    assert [row["rod"]["env_id"] for row in rows[-2:]] == [0, 1]
    assert rows[-2]["native_teacher_source_error"]["message"] == "fixture reader failed"
