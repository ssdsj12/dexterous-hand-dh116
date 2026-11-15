"""Independent zero-step v2 Reset over the unchanged original control path.

CPU continuation is not native admission. Factories are explicit and source
locked; the original C04 owner cannot silently treat this as a v1 episode.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Protocol

import torch

from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
from .tabletop_deformable_clearance_guard import ClearanceGuardedExchange, _pins
from .tabletop_deformable_live_geometry import LiveGeometryReader
from .tabletop_deformable_wrist_projection import CLEARANCE_M
from .tabletop_mesh_reset_readback_v2 import MeasuredMeshResetV2
from .tabletop_native_training_clock import ClockPolicyV2, read_clock_v2
from .tabletop_teacher_mounted_wrist_source import SEED_SOURCE, SEED_VERSION
from .tabletop_teacher_preaction_readback import PHASE

_PINNED_GUARD_SOURCES = {'tabletop_deformable_clearance_guard.py': '98d95efd26b1cb7fb833649e897c30d361bde78807b28671f7132f118ceec9dc', 'tabletop_deformable_physical_exchange.py': 'c910ebd45e246d4aa20ec94429001d2c0e2365ff5a0a0ded922212a68a2d3a36', 'tabletop_deformable_live_geometry.py': '19a318aa643d47a73e7dc49e226d2aa6fb4b5275e072dd2257c426e76ec7bbc9', 'tabletop_batch_geometry.py': 'f375078c59270acd8b7f30993ddd30576ad563031486af178db001ed9274da03', 'tabletop_deformable_wrist_projection.py': 'fbe593155d992b0c4a1c02df893648e67a3294ca11e00d5e12bc208796359ab3'}
_MODULE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_EXCHANGE_TYPE = ClearanceGuardedExchange
_RESET_TYPE = MeasuredMeshResetV2


class ExchangeFactoryV2(Protocol):
    def __call__(self, sim, rod, bindings, *, root_seed_w, read_geometry,
                 physics_budget_end, read_geometry_after_sdk) -> ClearanceGuardedExchange: ...


def guarded_exchange_factory_v2(sim, rod, bindings, *, root_seed_w, read_geometry,
                                physics_budget_end, read_geometry_after_sdk):
    return ClearanceGuardedExchange(sim, rod, bindings, root_seed_w=root_seed_w,
        read_geometry=read_geometry, physics_budget_end=physics_budget_end,
        read_geometry_after_sdk=read_geometry_after_sdk)


_GUARDED_FACTORY = guarded_exchange_factory_v2


def exchange_source_inventory_v2():
    """Current bytes must match the reviewed guard and this imported factory."""
    if (CLEARANCE_M != .0021 or _pins() != _PINNED_GUARD_SOURCES
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _MODULE_SOURCE_SHA256):
        raise ValueError("v2 exchange source bytes or original 2.1mm floor differ")
    return dict(factory_module_sha256=_MODULE_SOURCE_SHA256,
        guard_sources=deepcopy(_PINNED_GUARD_SOURCES), clearance_floor_m=.0021)


def _verify_factories(reset_factory, exchange_factory):
    inventory=exchange_source_inventory_v2()
    if (reset_factory is not _RESET_TYPE or exchange_factory is not _GUARDED_FACTORY
            or guarded_exchange_factory_v2 is not _GUARDED_FACTORY):
        raise ValueError("exact source-locked v2 reset and guarded exchange factory required")
    return inventory


class FixedS0PhysicalEpisodeV2(DualHandPhysicalEpisode):
    """Only Reset and state-clock differ; original action/control is inherited."""

    def __init__(self, sim, rod, bindings, *, backend_artifact, backend_sha256,
                 inspect_geometry, physics_budget_end, record, reset_factory,
                 exchange_factory, clock_policy, mesh_plan):
        inventory=_verify_factories(reset_factory,exchange_factory)
        if type(clock_policy) is not ClockPolicyV2:
            raise ValueError("exact explicit ClockPolicyV2 required")
        super().__init__(sim,rod,bindings,backend_artifact=backend_artifact,
            backend_sha256=backend_sha256,inspect_geometry=inspect_geometry,
            physics_budget_end=physics_budget_end,record=record)
        self._reset_factory,self._exchange_factory=reset_factory,exchange_factory
        self.clock_policy,self.mesh_plan=clock_policy,mesh_plan
        self.source_inventory=inventory

    def clock_readiness(self, *, source_map_path=None):
        from .tabletop_native_clock_consumers_v2 import (
            DEFAULT_SOURCE_MAP_PATH,clock_consumers_gate_v2)
        gate=clock_consumers_gate_v2(clock_policy=self.clock_policy,
            source_map_path=DEFAULT_SOURCE_MAP_PATH if source_map_path is None else source_map_path)
        return dict(gate,source_inventory=exchange_source_inventory_v2())

    @property
    def clock_consumers_ready(self):
        return self.clock_readiness()["clock_consumers_ready"]

    @property
    def native_ready(self):
        # W5 has no actual worker/source adapter and grants no native admission.
        return False

    def reset(self, seed, *, episode_uid, owner_resets):
        if self._state != "new":
            raise RuntimeError("one reset per physical episode owner")
        self._state="failed"
        if _verify_factories(self._reset_factory,self._exchange_factory)!=self.source_inventory:
            raise ValueError("v2 exchange source inventory changed")
        reset=self._reset_factory(seed,episode_uid=episode_uid,
            mesh_plan=self.mesh_plan,clock_policy=self.clock_policy)
        if type(reset) is not _RESET_TYPE:
            raise ValueError("exact MeasuredMeshResetV2 factory result required")
        receipt=reset.run(self._sim,self._rod,self._bindings,owner_resets=owner_resets)
        packet=self._boundary.capture(receipt["rod_reset"])
        reader=LiveGeometryReader(self._sim,self._bindings,self._inspect_geometry(),
                                  read_clock=self._rod.clock)
        initial=reader()
        clear=bool((initial["clearance_m"]>=CLEARANCE_M).all())
        evidence=dict(kind="reset",reset=receipt,rod_boundary=packet,
            clearance_m=initial["clearance_m"],clearance_floor_m=CLEARANCE_M,
            clearance_passed=clear,source_inventory=deepcopy(self.source_inventory),
            clock_policy_sha256=self.clock_policy.sha256,reset_epoch=receipt["reset_epoch"],
            native_qualification=False,teacher_admission=False)
        self._record(deepcopy(evidence))
        if not clear:
            raise ValueError("initial source-envelope clearance below original wrist floor")
        root=torch.tensor(receipt["seed"]["reset_root_pose_w"],dtype=torch.float64)[None]
        self._exchange=self._exchange_factory(self._sim,self._rod,self._bindings,
            root_seed_w=root,read_geometry=reader,physics_budget_end=self._end,
            read_geometry_after_sdk=reader.read_after_sdk_step)
        if type(self._exchange) is not _EXCHANGE_TYPE:
            self._exchange=None
            raise ValueError("exact ClearanceGuardedExchange factory result required")
        self._rod_packet=packet
        self._state_seed=dict(version=SEED_VERSION,source=SEED_SOURCE,
            identity=self._read_state_clock(),authored_hand_roots_w=receipt["seed"]["hand_roots"],
            environment_origin_w=receipt["seed"]["origin"],root_xy=receipt["seed"]["root_xy"],
            root_seed_w=receipt["seed"]["authored_root_seed_w"])
        self._state="active"
        return deepcopy(evidence)

    def _read_state_clock(self):
        packet=read_clock_v2(self._sim,self._rod,policy=self.clock_policy)
        clock=packet["rod"]
        if clock["clock_valid"] is not True:
            raise ValueError("pre-action SDK and rod clock differ")
        return dict(clock,phase=PHASE,policy_step=self._control_index,substep=0)
