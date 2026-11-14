"""Positive, restricted owner provenance for the CPU collector harness only.

This is not a sandbox for malicious Python. It accepts the reviewed repository
CPU doubles and exact pure-Python adapters, rejecting arbitrary runtime owners.
No native module is imported and no getter/step is invoked during inspection.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace, FunctionType, MethodType, BuiltinFunctionType

import numpy as np
import torch

from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_dual_hand_rod_runtime import DualHandRodRuntime
from .tabletop_deformable_grasp_runtime import FreeRodRuntime
from .tabletop_deformable_bimanual_step import HandBinding
from .tabletop_palm_mount import MountView
from .tabletop_deformable_live_geometry import LiveGeometryReader

VERSION = "dh116-restricted-cpu-fixture-owners-v1"
FIXTURE_BYTES = b"CPU double"
FIXTURE_SHA = hashlib.sha256(FIXTURE_BYTES).hexdigest()
SOURCE_SHA256 = {'src/go2_pvcnn/control/dh116/tabletop_deformable_bimanual_step.py': 'e144640a0310ab4824e482a2d643165239a74bbfdee66cae42289ff8cb3ee7f2',
 'src/go2_pvcnn/control/dh116/tabletop_deformable_dual_hand_episode.py': '4c50d414fb983f9189b73cefd3ec2ffe0a7633c105aecd6cf1d01749845e74f7',
 'src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py': '4e0f7762589b761c2129d17013e021703f67dde485275c484c5c9ceed95565c7',
 'src/go2_pvcnn/control/dh116/tabletop_deformable_live_geometry.py': '19a318aa643d47a73e7dc49e226d2aa6fb4b5275e072dd2257c426e76ec7bbc9',
 'src/go2_pvcnn/control/dh116/tabletop_deformable_physical_exchange.py': 'c910ebd45e246d4aa20ec94429001d2c0e2365ff5a0a0ded922212a68a2d3a36',
 'src/go2_pvcnn/control/dh116/tabletop_dual_hand_rod_runtime.py': 'e568a5b1951d43c278bff9949f3773514b1cc8f13e5d948ee0bd982a8c894688',
 'src/go2_pvcnn/control/dh116/tabletop_no_camera_collector.py': '5a15a993b73623d3b7cbc1ef379aa8a5a775dfef741a3cb8c1bd783c89a6fbac',
 'src/go2_pvcnn/control/dh116/tabletop_palm_mount.py': '4eb43511df607c5fee1503d9b389bf49ed52102dd7304c19e4bca7028f60e307',
 'tests/test_deformable_r0_replay.py': 'bd798ff40742459a90e15d8f018aed98c53e295a649334effa8f19a53dd6c608',
 'tests/test_dh116_deformable_bimanual_step.py': 'd8b8fc9dcf892628a2902c444370a96b2b46e64ba0482fa0b3ae4b998902bdbc',
 'tests/test_dh116_dual_hand_episode.py': 'a5dddecf74d23e4c95b926eabefe0e2d7150eb2cf5cf2f0bc61b35e803864744',
 'tests/test_dh116_no_camera_collector.py': '657fb57f4839805689c26e200c7b74188703f70ac6689b7c60f2607fa4cbb5c9',
 'tests/test_dh116_no_camera_fixture_owners.py': '20b60497b585233b51770bcd6dcf22aca6088d64b5b3b4b0ca5becb15d012d9b',
 'tests/test_dual_hand_reset.py': 'da8edd7676af069f8b0813d5de4a4c3287d2a391c0b90c04c78fe9ce55eaeae9',
 'tests/test_dual_hand_rod_runtime.py': '1f177c94993d10d9a7eeb2dd548e853fffb8d91d8c76eada0c41a1dd399ccaf6'}


class CPUFixtureOwners:
    def __init__(self, episode):
        self.project = Path(__file__).resolve().parents[4]
        if type(episode) is not DualHandPhysicalEpisode:
            raise ValueError("exact original CPU fixture episode owner required")
        self.episode = episode
        self.sim, self.rod, self.exchange = episode._sim, episode._rod, episode._exchange
        self.bindings = episode._bindings
        self.artifact = episode._boundary._artifact
        self.validator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.validate()

    def _classes(self):
        for relative, expected in SOURCE_SHA256.items():
            path = self.project/relative
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("CPU fixture owner source SHA differs: "+relative)
        modules = {}
        for name in ("test_dual_hand_rod_runtime","test_dual_hand_reset",
                     "test_dh116_deformable_bimanual_step"):
            module = sys.modules.get(name)
            if (module is None or Path(module.__file__).resolve()
                    != self.project/"tests"/(name+".py")):
                raise ValueError("loaded reviewed CPU fixture owner module required")
            modules[name] = module
        return (modules["test_dual_hand_rod_runtime"].FakeRope,
                modules["test_dual_hand_rod_runtime"].Grid,
                modules["test_dual_hand_reset"].Hand,
                modules["test_dual_hand_reset"].Composer,
                modules["test_dh116_deformable_bimanual_step"].Hand)

    def validate(self):
        from .tabletop_no_camera_collector import CPUFixtureCollector, _ObserveTerminal
        ep = self.episode
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self.validator_sha256:
            raise ValueError("CPU owner validator source changed")
        # A known positive fixture artifact, not a denylist of native versions.
        if (ep._boundary._artifact != self.artifact
                or ep._boundary._sha256 != FIXTURE_SHA
                or self.artifact.stat().st_size != len(FIXTURE_BYTES)
                or self.artifact.read_bytes() != FIXTURE_BYTES):
            raise ValueError("exact CPU fixture backend artifact required")
        fake, grid, hand, composer, action_hand = self._classes()
        current_exchange = ep._exchange
        if type(current_exchange) is _ObserveTerminal:
            current_exchange = current_exchange._inner
        if (ep._sim is not self.sim or ep._rod is not self.rod
                or current_exchange is not self.exchange or ep._bindings is not self.bindings
                or type(self.sim) is not SimpleNamespace
                or type(self.sim.physics_sim_view) is not SimpleNamespace
                or type(self.rod) is not DualHandRodRuntime
                or type(self.rod._base) is not FreeRodRuntime
                or type(self.rod.rope) is not fake
                or type(self.exchange) is not FlexiblePhysicalExchange
                or self.exchange._sim is not self.sim or self.exchange._rod is not self.rod):
            raise ValueError("restricted CPU fixture owner topology differs")
        for binding in self.bindings:
            if (type(binding) is not HandBinding or type(binding.hand) is not hand
                    or type(binding.mount) is not MountView
                    or binding.mount.hand is not binding.hand
                    or type(binding.hand.root_physx_view) is not SimpleNamespace
                    or type(binding.hand.permanent_wrench_composer) is not composer
                    or type(binding.hand.instantaneous_wrench_composer) is not composer):
                raise ValueError("reviewed CPU hand/PhysX/composer owner required")
        allowed = (SimpleNamespace, DualHandRodRuntime, FreeRodRuntime,
                   FlexiblePhysicalExchange, HandBinding, MountView, LiveGeometryReader,
                   fake, grid, hand, composer, action_hand)
        source_paths = {str(self.project/path) for path in SOURCE_SHA256}
        source_paths.add(str(Path(__file__).resolve()))
        seen = set()

        def inspect(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if len(seen) > 50000:
                raise ValueError("CPU owner graph exceeds bounded inspection")
            if type(value) is type and value in allowed:
                return  # __class__ closure used by reviewed super() calls.
            if value is None or type(value) in (str,bytes,bool,int,float):
                return
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu" or type(value) is not torch.Tensor:
                    raise ValueError("CPU owner tensor required; CUDA/subclass rejected")
                return
            if type(value) is np.ndarray:
                if value.dtype.kind not in "biuf":
                    raise ValueError("plain numeric CPU owner array required")
                return
            if isinstance(value, np.generic):
                if value.dtype.kind not in "biuf":
                    raise ValueError("plain numeric CPU scalar required")
                return
            if type(value) in (torch.device, torch.dtype):
                if type(value) is torch.device and value.type != "cpu":
                    raise ValueError("CPU owner device required")
                return
            if isinstance(value, Path):
                return
            if type(value) in (tuple,list,set,frozenset):
                for item in value:inspect(item)
                return
            if type(value) is dict:
                for key,item in value.items():inspect(key);inspect(item)
                return
            if type(value) is FunctionType:
                if str(Path(value.__code__.co_filename).resolve()) not in source_paths:
                    raise ValueError("unreviewed CPU owner callback source")
                inspect(value.__defaults__);inspect(value.__kwdefaults__)
                for cell in value.__closure__ or ():
                    inspect(cell.cell_contents)
                return
            if type(value) is MethodType:
                inspect(value.__func__);inspect(value.__self__)
                return
            if type(value) is BuiltinFunctionType and type(value.__self__) in (list,dict,set):
                inspect(value.__self__)
                return
            # Do not dereference opaque native wrappers, properties or getters.
            if (type(value) not in allowed and type(value) not in (CPUFixtureCollector, _ObserveTerminal)
                    and value is not self):
                raise ValueError("unknown CPU owner type: "+type(value).__module__+"."+type(value).__name__)
            values = object.__getattribute__(value, "__dict__")
            if any(name.startswith("_native") and item is not None for name,item in values.items()):
                raise ValueError("native owner wrapper is forbidden in CPU fixture")
            if type(value) is CPUFixtureCollector:
                if value._episode is not ep:
                    raise ValueError("foreign collector callback owner")
                return  # Known test callbacks can inspect this collector's journal.
            if value is self:
                return
            if type(value) not in allowed and type(value) is not _ObserveTerminal:
                raise ValueError("unknown CPU owner type: "+type(value).__module__+"."+type(value).__name__)
            for item in values.values():inspect(item)

        for root in (self.sim,self.rod,self.bindings,self.exchange):
            inspect(root)
        return dict(version=VERSION,scope="reviewed_CPU_doubles_only",backend_sha256=FIXTURE_SHA,
                    source_sha256=dict(SOURCE_SHA256),validator_source_sha256=self.validator_sha256,
                    native_qualification=False)
