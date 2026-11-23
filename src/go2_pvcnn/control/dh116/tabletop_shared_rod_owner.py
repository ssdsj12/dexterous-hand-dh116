"""Exclusive step and reaction ownership for an already configured shared solver."""
import math

import numpy as np
import torch

from .tabletop_batched_pair_capture import read_compact_pair_capture
from .tabletop_deformable_solver_clock import solver_clock_matches


class SharedRodOwner:
    """Own one native step per SDK boundary and one reaction read for the next.

    Callers own mesh pose submission and SDK stepping. Rod-only Reset here does
    not reset an Isaac articulation or qualify a complete task episode Reset.
    """

    def __init__(self, solver, *, run_uuid, physics_step, episodes, mesh_names,
                 solver_substeps, device):
        self.solver = solver
        self.run_uuid = run_uuid
        self.mesh_names = tuple(mesh_names)
        self.episodes = tuple(episodes)
        n = solver.num_envs
        if (not isinstance(run_uuid, str) or not run_uuid or type(physics_step) is not int
                or physics_step < 0 or type(n) is not int or not 2 <= n <= 512
                or len(self.episodes) != n or len(set(self.episodes)) != n
                or any(not isinstance(uid, str) or not uid for uid in self.episodes)
                or not self.mesh_names or len(set(self.mesh_names)) != len(self.mesh_names)
                or any(not isinstance(name, str) or not name for name in self.mesh_names)
                or type(solver_substeps) is not int or not 1 <= solver_substeps <= 64
                or solver.last_error != 0):
            raise ValueError("fresh configured shared solver, distinct episodes and mesh names required")
        self.physics_step = physics_step
        self.substeps = solver_substeps
        self.device = device
        self.ticks = [0] * n
        self.resets = [1] * n
        self._time = float(solver.time)
        if not math.isfinite(self._time) or self._time < 0:
            raise ValueError("finite solver owner time required")
        self._initial = torch.as_tensor(solver.positions()).detach().clone()
        self._reaction = None
        self._consumed = True
        self._failed = False

    def _close(self):
        self._failed = True
        self._reaction = None

    def advance(self, *, physics_step):
        """Advance the shared CUDA solver once after exactly one SDK step."""
        if self._failed:
            raise RuntimeError("failed shared solver owner")
        try:
            if (type(physics_step) is not int or physics_step != self.physics_step + 1
                    or not self._consumed or self.solver.last_error != 0
                    or float(self.solver.time) != self._time):
                raise ValueError("one SDK step, consumed reaction and unchanged solver boundary required")
            before = self._time
            self.solver.step_free(1 / 480, self.substeps)
            after = float(self.solver.time)
            if self.solver.last_error != 0 or not solver_clock_matches(before, after, steps=1):
                raise ValueError("shared native solver failed or changed its clock unexpectedly")
            ticks = [tick + self.substeps for tick in self.ticks]
            packet = read_compact_pair_capture(self.solver, physics_step=physics_step,
                rod_ticks=ticks, mesh_names=self.mesh_names, expected_owner_time_s=after, device=self.device)
            residual = packet["node_forces"].sum(1) + packet["mesh_wrenches"][:, :, :3].sum(1)
            if bool((torch.linalg.vector_norm(residual, dim=-1) > 1e-3).any()):
                raise ValueError("rod and all registered mesh forces do not balance")
            self.physics_step, self._time, self.ticks = physics_step, after, ticks
            self._reaction = packet
            self._consumed = False
            return {key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in packet.items()}
        except BaseException:
            self._close()
            raise

    def consume(self, *, physics_step):
        """Transfer one owned reaction for the immediate next SDK write."""
        if self._failed or self._consumed or self._reaction is None:
            raise RuntimeError("no unconsumed measured reaction at this shared boundary")
        if type(physics_step) is not int or physics_step != self.physics_step + 1:
            self._close()
            raise ValueError("reaction must be applied at exactly the next SDK step")
        if self.solver.last_error != 0 or float(self.solver.time) != self._time:
            self._close()
            raise ValueError("solver changed before its reaction was consumed")
        self._consumed = True
        packet, self._reaction = self._reaction, None
        return dict(packet, source_solver_id=id(self.solver), run_uuid=self.run_uuid,
                    source_physics_step=self.physics_step, applied_physics_step=physics_step,
                    rod_ticks=tuple(self.ticks), episode_uids=self.episodes,
                    reset_ids=tuple(self.resets))

    def reset_rod_lane(self, env_id, *, episode_uid):
        """Reset original native rod nodes without advancing time or neighbors.

        This does not relocate meshes. The complete Isaac task Reset must also
        restore its hand and collider states before the next shared advance.
        """
        if self._failed:
            raise RuntimeError("failed shared solver owner")
        try:
            if (type(env_id) is not int or not 0 <= env_id < len(self.ticks)
                    or not isinstance(episode_uid, str) or not episode_uid
                    or episode_uid in self.episodes or self._consumed
                    or self._reaction is None or self.solver.last_error != 0
                    or float(self.solver.time) != self._time):
                raise ValueError("fresh same-boundary rod episode and pending reaction required")
            before = torch.as_tensor(self.solver.positions()).detach().clone()
            self.solver.reset_envs(np.array([env_id], dtype=np.int32))
            after = torch.as_tensor(self.solver.positions()).detach().clone()
            keep = torch.arange(len(self.ticks), device=before.device) != env_id
            if (float(self.solver.time) != self._time or self.solver.last_error != 0
                    or not torch.equal(before[keep], after[keep])
                    or not torch.equal(after[env_id], self._initial[env_id])):
                raise ValueError("rod-only Reset changed time or another lane")
            packet = self._reaction
            retained = packet["env_ids"] != env_id
            for key in ("env_ids", "indices", "vectors", "normal_force"):
                packet[key] = packet[key][retained].clone()
            for key in ("counts", "node_forces", "mesh_wrenches"):
                packet[key][env_id] = 0
            ids = list(self.episodes)
            ids[env_id] = episode_uid
            self.episodes = tuple(ids)
            self.ticks[env_id] = 0
            self.resets[env_id] += 1
            packet["rod_ticks"] = tuple(self.ticks)
        except BaseException:
            self._close()
            raise
