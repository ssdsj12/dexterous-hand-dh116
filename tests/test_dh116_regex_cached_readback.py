"""CPU execution of the pinned installed SDK cache source, without Isaac/CUDA.

The source class executes against a spy view. These tests do not qualify native
physics equivalence or use a replacement cache implementation as the oracle.
"""
import ast
from collections import Counter
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import JOINTS
from go2_pvcnn.control.dh116.tabletop_palm_mount import ANGULAR_JOINT_NAMES, JOINT_NAMES
from go2_pvcnn.control.dh116.tabletop_regex_articulation import RegexArticulationSide

MODULE = "go2_pvcnn.control.dh116.tabletop_regex_cached_readback"
SDK_ROOT = Path("/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab")


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, "SDK cache readback helper is missing"
    return importlib.import_module(MODULE)


def execute_source_node(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class SpyView:
    def __init__(self):
        self.count = 3
        self.prim_paths = [f"/World/envs/env_{i}/leftHand/wrist_fixed" for i in (10, 0, 2)]
        self.calls = Counter()
        self.q = torch.arange(51, dtype=torch.float32).reshape(3, 17)
        self.qd = self.q + 0.25
        self.targets = self.q + 10
        self.root = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        self.links = torch.arange(3 * 23 * 7, dtype=torch.float32).reshape(3, 23, 7)
        # Non-unit quaternions, signed zero and distinct NaN payloads must survive
        # pure storage/reordering operations exactly. No normalization is allowed.
        self.root.view(torch.int32)[0, 3:] = torch.tensor([-2147483648, 2143289377, 1056964608, -1082130432])
        self.links.view(torch.int32)[1, 0, 3:] = torch.tensor([2143289378, -2147483648, -1082130432, 1048576000])
        self.root_velocity = torch.arange(18, dtype=torch.float32).reshape(3, 6) + 1000
        self.link_velocity = torch.arange(3 * 23 * 6, dtype=torch.float32).reshape(3, 23, 6) + 2000

    def _get(self, name, value):
        self.calls[name] += 1
        return value

    def get_dof_positions(self): return self._get("q", self.q)
    def get_dof_velocities(self): return self._get("qd", self.qd)
    def get_dof_position_targets(self): return self._get("target", self.targets)
    def get_root_transforms(self): return self._get("root", self.root)
    def get_root_velocities(self): return self._get("root_velocity", self.root_velocity)
    def get_link_transforms(self): return self._get("link", self.links)
    def get_link_velocities(self): return self._get("link_velocity", self.link_velocity)


@pytest.fixture
def owner():
    if not SDK_ROOT.exists():
        pytest.skip("pinned installed SDK source is unavailable; no native claim is made")
    buffer = runpy.run_path(str(SDK_ROOT / "utils/buffers/timestamped_buffer.py"))["TimestampedBuffer"]
    convert = execute_source_node(SDK_ROOT / "utils/math.py", "convert_quat",
                                  {"torch": torch, "np": np, "Literal": Literal})
    namespace = {"torch": torch, "math_utils": SimpleNamespace(convert_quat=convert),
                 "physx": SimpleNamespace(ArticulationView=object), "__name__": "cpu_sdk_source_fixture"}
    data_type = execute_source_node(SDK_ROOT / "assets/articulation/articulation_data.py",
                                    "ArticulationData", namespace)
    data = data_type.__new__(data_type)
    view = SpyView()
    data._root_physx_view = view
    data._sim_timestamp = 0.0
    data._previous_joint_vel = view.qd.clone()
    physics = SimpleNamespace(calls=0)
    def kinematic_update(): physics.calls += 1
    physics.update_articulations_kinematic = kinematic_update
    data._physics_sim_view = physics
    # Buffer inventory comes from the installed __init__, whose unrelated
    # SimulationManager/gravity initialization is outside this CPU fixture.
    tree = ast.parse((SDK_ROOT / "assets/articulation/articulation_data.py").read_text())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "TimestampedBuffer":
                setattr(data, node.targets[0].attr, buffer())
    hand = SimpleNamespace(data=data, root_physx_view=view, device="cpu", is_initialized=True,
        num_instances=3, joint_names=list(JOINTS + JOINT_NAMES + ANGULAR_JOINT_NAMES),
        body_names=["base_link"] + [f"link_{i}" for i in range(22)],
        permanent_wrench_composer=SimpleNamespace(_link_poses_updated=True, active=True),
        instantaneous_wrench_composer=SimpleNamespace(_link_poses_updated=True, active=True))
    return RegexArticulationSide(hand, (0, 2, 10),
        tuple(f"/World/envs/env_{i}" for i in (0, 2, 10)), "/wrist_fixed")


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def assert_packet_bits(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(bits(actual[name]), bits(expected[name])), name


def test_source_getters_share_one_fetch_and_preserve_raw_bits(owner):
    module = implementation()
    reader = module.CachedSideReadback(owner)
    oracle = owner.read()
    view, data = owner.hand.root_physx_view, owner.hand.data
    view.calls.clear()
    data.update(1 / 480)
    assert view.calls == {"qd": 1}  # SDK update itself eagerly acquires qd.
    actual = module.read_cached_side(owner, reader=reader)
    # The real SDK consumers request these properties after capture.
    data.joint_pos; data.joint_vel; data.body_link_pos_w; data.body_link_quat_w; data.root_pose_w
    assert view.calls == dict.fromkeys(("q", "qd", "target", "root", "root_velocity", "link", "link_velocity"), 1)
    assert data._physics_sim_view.calls == 1
    assert_packet_bits(actual, oracle)
    metadata = reader.metadata()
    assert metadata["native_equivalence_verified"] is False
    assert metadata["actual_producer"]["device"] == "cpu"
    assert metadata["actual_producer"]["view_type"].endswith("SpyView")
    assert metadata["sdk_version"] == "0.54.4"


def test_owned_capturedpost_survives_next_update_and_selected_row_order(owner):
    reader = implementation().CachedSideReadback(owner)
    data, view = owner.hand.data, owner.hand.root_physx_view
    data.update(1 / 480)
    captured = reader.read()
    frozen = {key: value.clone() for key, value in captured.items()}
    for name in ("q", "qd", "targets", "root", "root_velocity", "links", "link_velocity"):
        getattr(view, name).fill_(42.25)
    data.update(1 / 480)
    fresh = reader.read(env_ids=[10, 0])
    oracle = owner.read(env_ids=[10, 0])
    assert_packet_bits(fresh, oracle)
    assert_packet_bits(captured, frozen)
    assert torch.equal(fresh["q"], torch.full((2, 17), 42.25))
    view.q[:, 0] = torch.tensor([100., 200., 300.])
    assert reader.read(env_ids=[10, 0])["q"][:, 0].tolist() == [100., 200.]


def test_explicit_invalidation_refreshes_same_clock_external_write_and_composers(owner):
    reader = implementation().CachedSideReadback(owner)
    data, view = owner.hand.data, owner.hand.root_physx_view
    data.update(1 / 480)
    captured = reader.read()
    frozen = {key: value.clone() for key, value in captured.items()}
    clock, acc_clock = data._sim_timestamp, data._joint_acc.timestamp
    local_com_clock = data._body_com_pose_b.timestamp
    view.root.fill_(7.5)
    view.links.fill_(8.5)
    view.root_velocity.fill_(9.5)
    view.link_velocity.fill_(10.5)
    # A raw setter/reset can mutate the SDK without touching its cache clock.
    reader.invalidate(reason="selected Reset or external PhysX state write completed")
    actual = reader.read()
    assert_packet_bits(actual, owner.read())
    assert_packet_bits(captured, frozen)
    assert data._sim_timestamp == clock
    assert data._joint_acc.timestamp == acc_clock
    assert data._body_com_pose_b.timestamp == local_com_clock
    assert owner.hand.permanent_wrench_composer._link_poses_updated is False
    assert owner.hand.instantaneous_wrench_composer._link_poses_updated is False
    assert owner.hand.permanent_wrench_composer.active is True


def test_pinned_source_and_version_fail_closed_before_reading(owner, monkeypatch):
    module = implementation()
    before = owner.hand.root_physx_view.calls.copy()
    with monkeypatch.context() as change:
        change.setattr(importlib.metadata, "version", lambda name: "0.0.0")
        with pytest.raises(ValueError, match="version"):
            module.CachedSideReadback(owner)
    wrong = dict(module.SDK_SOURCE_SHA256)
    wrong["assets/articulation/articulation_data.py"] = "0" * 64
    monkeypatch.setattr(module, "SDK_SOURCE_SHA256", wrong)
    with pytest.raises(ValueError, match="SHA"):
        module.CachedSideReadback(owner)
    assert owner.hand.root_physx_view.calls == before


def test_reader_rejects_rebound_view_and_wrong_owner(owner):
    module = implementation()
    reader = module.CachedSideReadback(owner)
    with pytest.raises(ValueError, match="owner"):
        module.read_cached_side(SimpleNamespace(), reader=reader)
    owner.hand.root_physx_view = SpyView()
    with pytest.raises(ValueError, match="identity"):
        reader.read()

def test_batch_optin_reader_owns_side_without_reference_cycle(monkeypatch):
    import gc,weakref
    import go2_pvcnn.control.dh116.tabletop_regex_cached_readback as module
    from go2_pvcnn.control.dh116.tabletop_regex_articulation import RegexArticulationBatch
    class Side:
        def read(self,ids=None):raise AssertionError("opt-in must use official audited reader")
    class Reader:
        def __init__(self,owner):self.owner=owner
        def read(self,ids=None):return {"owner_id":id(self.owner)}
        def metadata(self):return {"audited":True}
        def invalidate(self,*,reason):self.reason=reason
    monkeypatch.setenv("DH116_NATIVE_READBACK_BACKEND","official_cache")
    monkeypatch.setattr(module,"CachedSideReadback",Reader)
    batch=RegexArticulationBatch((object(),object()),(0,1),("a","b"),("x","x"),())
    batch.sides={"left":Side(),"right":Side()}
    left=weakref.ref(batch.sides["left"]);right=weakref.ref(batch.sides["right"])
    assert set(batch.read())=={"left","right"}
    assert batch.readback_metadata()["mode"]=="official_cache"
    batch.invalidate_readback(reason="zero-step external write")
    active=gc.isenabled()
    gc.disable()
    try:
        del batch
        assert left() is None and right() is None
    finally:
        if active:gc.enable()

def test_batch_rejects_unknown_readback_backend_before_acquiring_views(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_regex_articulation import RegexArticulationBatch
    monkeypatch.setenv("DH116_NATIVE_READBACK_BACKEND","silent_stale")
    with pytest.raises(ValueError,match="readback"):
        RegexArticulationBatch((object(),object()),(0,1),("a","b"),("x","x"),())

def test_native_cache_oracle_audit_refuses_stale_values(monkeypatch):
    import torch
    import go2_pvcnn.control.dh116.tabletop_regex_cached_readback as module
    from go2_pvcnn.control.dh116.tabletop_regex_articulation import RegexArticulationBatch
    class Side:
        def read(self,ids=None):return {"q":torch.ones(2,17)}
    class Reader:
        def __init__(self,owner):self.owner=owner
        def read(self,ids=None):return {"q":torch.zeros(2,17)}
    monkeypatch.setenv("DH116_NATIVE_READBACK_BACKEND","official_cache")
    monkeypatch.setenv("DH116_NATIVE_READBACK_AUDIT","1")
    monkeypatch.setattr(module,"CachedSideReadback",Reader)
    batch=RegexArticulationBatch((object(),object()),(0,1),("a","b"),("x","x"),())
    batch.sides={"left":Side(),"right":Side()}
    with pytest.raises(ValueError,match="cache/raw"):
        batch.read()
    assert batch.readback_audit()["mismatches"]
