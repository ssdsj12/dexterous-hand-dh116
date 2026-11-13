"""Compose the real two-hand reset, SDK/rod exchange and owned evidence.

This lifecycle consumes an existing scene. The caller owns SDK initialization,
the locked source inspection and teardown. Rewards and PPO are outside this
boundary. A failed episode stays closed, including partial SDK advancement.
"""
from copy import deepcopy
import math
from pathlib import PurePosixPath

import torch

from .tabletop_deformable_bimanual_step import DT
from .tabletop_deformable_live_geometry import LiveGeometryReader
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_deformable_rod_boundary import RodBoundaryState
from .tabletop_deformable_wrist_projection import CLEARANCE_M
from .tabletop_dual_hand_reset import DualHandResetTransaction
from .tabletop_teacher_mounted_wrist_source import (
    SEED_SOURCE, SEED_VERSION, adapt_mounted_hands, capture_mounted_hand,
)
from .tabletop_teacher_preaction_readback import PHASE


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
        self._control_index = 0
        self._rod_packet = self._state_seed = None

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
        self._rod_packet = packet
        self._state_seed = dict(version=SEED_VERSION, source=SEED_SOURCE,
            identity=self._read_state_clock(),
            authored_hand_roots_w=receipt["seed"]["hand_roots"],
            environment_origin_w=receipt["seed"]["origin"], root_xy=receipt["seed"]["root_xy"],
            root_seed_w=receipt["seed"]["authored_root_seed_w"])
        self._state = "active"
        return deepcopy(evidence)

    def _read_state_clock(self):
        clock = self._rod.clock()
        if (clock["clock_valid"] is not True
                or clock["physics_step"] != self._sim.current_time_step_index
                or not math.isclose(float(self._sim.current_time), clock["physics_step"] * DT,
                                    abs_tol=2e-7, rel_tol=0)):
            raise ValueError("pre-action SDK and rod clock differ")
        return dict(clock, phase=PHASE, policy_step=self._control_index, substep=0)

    def pre_action_state(self):
        """Return the same-boundary measured subset of the original 371D contract."""
        if self._state != "active":
            raise RuntimeError("active physical episode required")
        clock = self._read_state_clock()
        snapshot = self._rod_packet["rod_snapshot"]
        if (snapshot["env_ids"] != [clock["env_id"]]
                or snapshot["episode_uids"] != [clock["episode_uid"]]
                or snapshot["reset_ids"] != [clock["reset_id"]]
                or snapshot["rod_uids"] != [clock["rod_uid"]]
                or snapshot["physics_step"] != clock["physics_step"]
                or snapshot["rod_tick"] != clock["rod_tick"]
                or snapshot["timestamp_s"] != clock["timestamp_s"]):
            raise ValueError("pre-action rod snapshot does not belong to the current boundary")
        packets = []
        for binding in self._bindings:
            path = binding.env_paths[clock["env_id"]]
            container = str(PurePosixPath(
                self._rod.bindings[binding.side + ".finger14_link"]["prim_path"]).parent)
            packets.append(capture_mounted_hand(binding.hand, binding.mount,
                side=binding.side, read_identity=self._read_state_clock, data_env_index=0,
                expected_articulation_path=path, expected_hand_container_path=container,
                reset_seed=self._state_seed))
        common = adapt_mounted_hands(packets, expected_identity=clock)
        if self._read_state_clock() != clock:
            raise ValueError("pre-action clock changed during the observation")
        return dict(common=common, rod_boundary=deepcopy(self._rod_packet),
                    native_qualification=False, teacher_admission=False)

    def control(self, action):
        if self._state != "active":
            raise RuntimeError("active physical episode required")
        def observe(row):
            packet = self._boundary.capture(row["rod"])
            self._record(dict(kind="physical", exchange=row,
                              rod_boundary=deepcopy(packet),
                              native_qualification=False, teacher_admission=False))
            self._rod_packet = packet
        self._state = "failed"
        result = self._exchange.control(action, observe=observe)
        self._control_index += 1
        self._state = "closed" if result["termination_reason"] else "active"
        return result
