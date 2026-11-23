"""Shared native step ownership, distinct resets and reaction consumption."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_dh116_batched_pair_capture import packet
from go2_pvcnn.control.dh116.tabletop_shared_rod_owner import SharedRodOwner


class Solver:
    num_envs = 3
    time = 0.
    last_error = 0

    def __init__(self):
        self.steps = 0
        self.capture = None
        self.nodes = torch.arange(9, dtype=torch.float32).reshape(3, 1, 3)
        self.initial = self.nodes.clone()

    def step_free(self, dt, substeps):
        assert dt == 1/480 and substeps == 2
        self.steps += 1
        self.time += dt
        self.nodes += 1
        self.capture = packet()
        for lane in self.capture["lanes"]:
            lane["owner_time_s_before"] = self.time
            lane["owner_time_s_after"] = self.time

    def contact_pair_samples_batched_v1(self):
        if self.capture is None:
            raise ValueError("no completed pair capture")
        return deepcopy(self.capture)

    def positions(self):
        return self.nodes.clone()

    def reset_envs(self, ids):
        self.nodes[ids] = self.initial[ids]
        self.capture = None


def owner(solver):
    return SharedRodOwner(solver, run_uuid="test-run", physics_step=2,
        episodes=("episode0", "episode1", "episode2"), mesh_names=("pad", "table"),
        solver_substeps=2, device="cpu")


def test_each_completed_shared_step_consumed_once_on_next_sdk_tick():
    solver = Solver()
    runtime = owner(solver)
    data = runtime.advance(physics_step=3)
    assert solver.steps == 1 and data["rod_ticks"] == (2, 2, 2)
    reaction = runtime.consume(physics_step=4)
    assert reaction["run_uuid"] == "test-run"
    assert reaction["source_solver_id"] == id(solver)
    assert reaction["source_physics_step"] == 3
    assert reaction["episode_uids"] == ("episode0", "episode1", "episode2")
    assert reaction["reset_ids"] == (1, 1, 1)
    with pytest.raises(RuntimeError):
        runtime.consume(physics_step=4)
    runtime.advance(physics_step=4)
    assert solver.steps == 2


def test_zero_step_rod_reset_clears_its_reaction_and_preserves_neighbor():
    solver = Solver()
    runtime = owner(solver)
    runtime.advance(physics_step=3)
    before = solver.positions()
    runtime.reset_rod_lane(1, episode_uid="new-episode1")
    after = solver.positions()
    assert torch.equal(before[[0, 2]], after[[0, 2]])
    reaction = runtime.consume(physics_step=4)
    assert reaction["rod_ticks"] == (2, 0, 2)
    assert reaction["reset_ids"] == (1, 2, 1)
    assert reaction["episode_uids"][1] == "new-episode1"
    assert not reaction["mesh_wrenches"][1].any()
    runtime.advance(physics_step=4)
    assert runtime.consume(physics_step=5)["rod_ticks"] == (4, 2, 4)


@pytest.mark.parametrize("damage", ["skip_sdk", "external_time", "native_error", "overflow"])
def test_failed_step_closes_owner_and_cannot_replay_old_reaction(damage):
    solver = Solver()
    runtime = owner(solver)
    runtime.advance(physics_step=3)
    runtime.consume(physics_step=4)
    if damage == "skip_sdk":
        sdk = 5
    else:
        sdk = 4
        if damage == "external_time":
            solver.time += 1
        elif damage == "native_error":
            solver.last_error = 1
        else:
            original = solver.contact_pair_samples_batched_v1

            def overflow():
                raw = original()
                raw["lanes"][2]["overflow"] = True
                return raw

            solver.contact_pair_samples_batched_v1 = overflow
    with pytest.raises((ValueError, RuntimeError)):
        runtime.advance(physics_step=sdk)
    with pytest.raises(RuntimeError):
        runtime.consume(physics_step=5)


def test_rod_reset_cannot_clear_an_existing_native_error():
    solver = Solver()
    runtime = owner(solver)
    runtime.advance(physics_step=3)
    solver.last_error = 1
    original_reset = solver.reset_envs

    def reset_clears_error(ids):
        original_reset(ids)
        solver.last_error = 0

    solver.reset_envs = reset_clears_error
    with pytest.raises(ValueError):
        runtime.reset_rod_lane(1, episode_uid="after-error")
    with pytest.raises(RuntimeError):
        runtime.consume(physics_step=4)
