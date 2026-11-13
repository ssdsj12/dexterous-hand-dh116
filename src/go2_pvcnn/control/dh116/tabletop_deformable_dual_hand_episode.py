"""Compose the real two-hand reset, SDK/rod exchange and owned evidence.

This lifecycle consumes an existing scene. The caller owns SDK initialization,
the locked source inspection and teardown. Rewards and PPO are outside this
boundary. A failed episode stays closed, including partial SDK advancement.
"""
from copy import deepcopy

import torch

from .tabletop_deformable_live_geometry import LiveGeometryReader
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_deformable_rod_boundary import RodBoundaryState
from .tabletop_deformable_wrist_projection import CLEARANCE_M
from .tabletop_dual_hand_reset import DualHandResetTransaction


class DualHandPhysicalEpisode:
    """One explicit reset and a bounded sequence of original 24D control steps."""

    def __init__(self, sim, rod, bindings, *, backend_artifact, backend_sha256,
                 inspect_geometry, physics_budget_end, record):
        if not callable(inspect_geometry) or not callable(record):
            raise ValueError("scene geometry inspection and evidence recorder required")
        if type(physics_budget_end) is not int or physics_budget_end < sim.current_time_step_index + 8:
            raise ValueError("budget must cover at least one complete control interval")
        self._sim, self._rod, self._bindings = sim, rod, tuple(bindings)
        self._boundary = RodBoundaryState(rod, backend_artifact, backend_sha256)
        self._inspect_geometry = inspect_geometry
        self._end, self._record = physics_budget_end, record
        self._exchange = None
        self._state = "new"

    def reset(self, seed, *, episode_uid, owner_resets):
        if self._state != "new":
            raise RuntimeError("one reset per physical episode owner")
        self._state = "failed"
        reset = DualHandResetTransaction(seed, episode_uid=episode_uid)
        receipt = reset.run(self._sim, self._rod, self._bindings, owner_resets=owner_resets)
        packet = self._boundary.capture(receipt["rod_reset"])
        reader = LiveGeometryReader(self._sim, self._bindings, self._inspect_geometry(),
                                    read_clock=self._rod.clock)
        initial = reader()
        clear = bool((initial["clearance_m"] >= CLEARANCE_M).all())
        evidence = dict(kind="reset", reset=receipt, rod_boundary=packet,
                        clearance_m=initial["clearance_m"],
                        clearance_floor_m=CLEARANCE_M, clearance_passed=clear,
                        native_qualification=False, teacher_admission=False)
        self._record(deepcopy(evidence))
        if not clear:
            raise ValueError("initial source-envelope clearance below original wrist floor")
        # Keep the actual reset pose as the initial world target. The earlier
        # authored pose excludes the configured reset Z offset.
        root = torch.tensor(receipt["seed"]["reset_root_pose_w"], dtype=torch.float64)[None]
        self._exchange = FlexiblePhysicalExchange(self._sim, self._rod, self._bindings,
            root_seed_w=root, read_geometry=reader,
            read_geometry_after_sdk=reader.read_after_sdk_step, physics_budget_end=self._end)
        self._state = "active"
        return evidence

    def control(self, action):
        if self._state != "active":
            raise RuntimeError("active physical episode required")
        def observe(row):
            self._record(dict(kind="physical", exchange=row,
                              rod_boundary=self._boundary.capture(row["rod"]),
                              native_qualification=False, teacher_admission=False))
        self._state = "failed"
        result = self._exchange.control(action, observe=observe)
        self._state = "closed" if result["termination_reason"] else "active"
        return result
