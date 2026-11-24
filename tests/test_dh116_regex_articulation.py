"""CPU contract fixtures for the installed Articulation/PhysX-shaped API.

No AppLauncher, USD assets, native physics or CUDA context is created.
"""
from copy import deepcopy
from pathlib import Path
import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import JOINTS
from go2_pvcnn.control.dh116.tabletop_palm_mount import ANGULAR_JOINT_NAMES, JOINT_NAMES

MODULE = "go2_pvcnn.control.dh116.tabletop_regex_articulation"
ENV_IDS = (0, 2, 10)
PREFIXES = tuple(f"/World/envs/env_{i}" for i in ENV_IDS)
NAMES = list(JOINTS + JOINT_NAMES + ANGULAR_JOINT_NAMES)
BODIES = ["base_link"] + [f"link_{i}" for i in range(22)]


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, "two-regex Articulation implementation is missing"
    return importlib.import_module(MODULE)


class Composer:
    def __init__(self, count):
        self.composed_force_as_torch = torch.full((count, 23, 3), 7.)
        self.composed_torque_as_torch = torch.full((count, 23, 3), 9.)
        self.active = True
        self.calls = []

    def reset(self, env_ids=None):
        self.calls.append(("reset", env_ids))
        ids = slice(None) if env_ids is None else env_ids
        self.composed_force_as_torch[ids] = 0
        self.composed_torque_as_torch[ids] = 0
        if env_ids is None:
            self.active = False

    def set_forces_and_torques(self, forces=None, torques=None, positions=None,
                              body_ids=None, env_ids=None, is_global=False):
        self.calls.append(("set", forces.clone(), env_ids.clone(), body_ids, positions, is_global))
        ids = torch.arange(23) if body_ids is None else torch.as_tensor(body_ids)
        self.composed_force_as_torch[env_ids[:, None], ids] = forces
        if torques is not None:
            self.composed_torque_as_torch[env_ids[:, None], ids] = torques
        self.active = True

    def add_forces_and_torques(self, **kwargs):
        raise AssertionError("fixture only expects replacement wrenches")


class View:
    """Full-view payload plus selected-row indices, as the installed SDK requires."""
    def __init__(self, env_ids, side):
        self.count = len(env_ids)
        self.prim_paths = [f"/World/envs/env_{i}/{side}Hand/wrist_fixed" for i in env_ids]
        self.link_paths = [[path.rsplit("/", 1)[0] + "/" + n for n in BODIES] for path in self.prim_paths]
        self.q = torch.tensor(env_ids, dtype=torch.float32)[:, None].repeat(1, 17)
        self.qd = self.q + .25
        self.targets = self.q + 100
        self.velocity_targets = self.q + 200
        self.effort = self.q + 300
        self.root = torch.zeros(len(env_ids), 7)
        self.root[:, 0] = torch.tensor(env_ids)
        self.root[:, 6] = 1
        self.root_velocity = self.q[:, :6].clone()
        self.links = self.root[:, None].repeat(1, 23, 1)
        self.calls = []

    def get_dof_positions(self): return self.q
    def get_dof_velocities(self): return self.qd
    def get_dof_position_targets(self): return self.targets
    def get_dof_velocity_targets(self): return self.velocity_targets
    def get_dof_actuation_forces(self): return self.effort
    def get_root_transforms(self): return self.root
    def get_root_velocities(self): return self.root_velocity
    def get_link_transforms(self): return self.links
    def get_link_velocities(self): return self.q[:, None, :6].expand(-1, 23, -1)

    def _set(self, field, value, indices):
        target = getattr(self, field)
        assert value.shape == target.shape, "PhysX expects a full-view payload, not selected rows"
        target[indices] = value[indices]
        self.calls.append((field, indices.clone()))

    def set_dof_position_targets(self, data, indices): self._set("targets", data, indices)
    def set_dof_velocity_targets(self, data, indices): self._set("velocity_targets", data, indices)
    def set_dof_actuation_forces(self, data, indices): self._set("effort", data, indices)

    def apply_forces_and_torques_at_position(self, force_data, torque_data,
                                            position_data, indices, is_global):
        assert force_data.numel() == self.count * 23 * 3
        assert torque_data.numel() == self.count * 23 * 3
        self.calls.append(("force", indices.clone(), force_data.clone(), torque_data.clone()))


class Hand:
    def __init__(self, env_ids, side):
        self.device = "cpu"
        self.joint_names = NAMES.copy()
        self.body_names = BODIES.copy()
        self.num_instances = len(env_ids)
        self.num_joints = 17
        self.num_bodies = 23
        self.is_initialized = True
        self.root_physx_view = View(env_ids, side)
        self.permanent_wrench_composer = Composer(len(env_ids))
        self.instantaneous_wrench_composer = Composer(len(env_ids))
        self.data = SimpleNamespace(
            joint_pos_target=self.root_physx_view.targets.clone(),
            joint_vel_target=self.root_physx_view.velocity_targets.clone(),
            joint_effort_target=self.root_physx_view.effort.clone())
        self.actuators = {"active": SimpleNamespace(stiffness=torch.ones(len(env_ids), 6))}
        self.calls = []

    def reset(self, env_ids=None):
        self.calls.append(("reset", env_ids.clone()))
        self.permanent_wrench_composer.reset(env_ids)
        self.instantaneous_wrench_composer.reset(env_ids)

    def _cache(self, field, value, joint_ids, env_ids):
        ids = torch.arange(17) if joint_ids is None else torch.as_tensor(joint_ids)
        getattr(self.data, field)[env_ids[:, None], ids] = value

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self._cache("joint_pos_target", target, joint_ids, env_ids)
        self.calls.append(("target", env_ids.clone()))

    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        self._cache("joint_vel_target", target, joint_ids, env_ids)

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        self._cache("joint_effort_target", target, joint_ids, env_ids)

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        assert joint_ids is None
        self.root_physx_view.q[env_ids] = position
        self.root_physx_view.qd[env_ids] = velocity
        self.calls.append(("state", env_ids.clone()))

    def write_root_pose_to_sim(self, root_pose, env_ids=None):
        self.root_physx_view.root[env_ids] = root_pose[:, [0, 1, 2, 4, 5, 6, 3]]

    def write_root_velocity_to_sim(self, root_velocity, env_ids=None):
        self.root_physx_view.root_velocity[env_ids] = root_velocity

    def write_joint_stiffness_to_sim(self, stiffness, joint_ids=None, env_ids=None):
        self.calls.append(("stiffness", env_ids.clone(), stiffness.clone()))

    def write_joint_effort_limit_to_sim(self, limits, joint_ids=None, env_ids=None):
        self.calls.append(("limits", env_ids.clone(), limits.clone()))

    def write_data_to_sim(self):
        self.calls.append(("write",))
        self.root_physx_view.targets.copy_(self.data.joint_pos_target)
        self.instantaneous_wrench_composer.reset()

    def update(self, dt): self.calls.append(("update", dt))


def side():
    hand = Hand((10, 0, 2), "left")
    owner = implementation().RegexArticulationSide(hand, ENV_IDS, PREFIXES, "/wrist_fixed")
    return owner, hand


def test_actual_prim_paths_remap_all_batch_reads_and_selected_targets():
    owner, hand = side()
    assert owner.row_env_ids == (10, 0, 2)
    assert owner.row_indices.tolist() == [1, 2, 0]
    state = owner.read()
    assert state["q"][:, 0].tolist() == [0, 2, 10]
    assert state["qd"][:, 0].tolist() == [.25, 2.25, 10.25]
    assert state["backend_targets"][:, 0].tolist() == [100, 102, 110]
    assert state["root_pose_wxyz"][:, [0, 3]].tolist() == [[0, 1], [2, 1], [10, 1]]
    assert state["link_pose_wxyz"][:, 0, 0].tolist() == [0, 2, 10]
    assert state["link_velocity"][:, 0, 0].tolist() == [0, 2, 10]
    hand.root_physx_view.q.add_(40)
    assert state["q"][:, 0].tolist() == [0, 2, 10], "readback must own its snapshot"
    owner.set_joint_position_target(torch.tensor([[31., 32.], [41., 42.]]),
                                    joint_ids=[0, 3], env_ids=[10, 0])
    assert hand.data.joint_pos_target[[0, 1], :][:, [0, 3]].tolist() == [[31, 32], [41, 42]]
    assert sum(call[0] == "target" for call in hand.calls) == 1
    with pytest.raises(ValueError, match="environment"):
        owner.rows([3])
    with pytest.raises(ValueError, match="unique"):
        owner.rows([2, 2])
    with pytest.raises(ValueError, match="integer"):
        owner.rows([2.7])


def test_selected_reset_preserves_other_q_targets_and_pending_reactions():
    owner, hand = side()
    untouched = [0, 1]  # Actual rows env10/env0; selected env2 is row2.
    before = owner.read()
    force_before = hand.instantaneous_wrench_composer.composed_force_as_torch.clone()
    q = torch.arange(17, dtype=torch.float32)[None] / 100
    q[:, 11:] = 0
    qd = torch.full_like(q, .03)
    qd[:, 11:] = 0
    pose = torch.tensor([[.3, .4, .5, 1., 0., 0., 0.]])
    owner.reset(env_ids=[2], root_pose_wxyz=pose, joint_pos=q, joint_vel=qd,
                active_effort_limit_nm=.2)
    after = owner.read()
    for field in ("q", "qd", "backend_targets", "root_pose_wxyz", "root_velocity"):
        assert torch.equal(after[field][[0, 2]], before[field][[0, 2]]), field
    assert torch.equal(after["q"][1], q[0])
    assert torch.equal(after["qd"][1], qd[0])
    assert torch.equal(after["backend_targets"][1], q[0])
    assert torch.equal(after["root_pose_wxyz"][1], pose[0])
    assert torch.count_nonzero(after["root_velocity"][1]) == 0
    assert not any(call[0] == "write" for call in hand.calls), "global write consumes other instantaneous wrenches"
    for composer in (hand.permanent_wrench_composer, hand.instantaneous_wrench_composer):
        assert torch.equal(composer.composed_force_as_torch[untouched], force_before[untouched])
        assert torch.count_nonzero(composer.composed_force_as_torch[2]) == 0
    assert hand.actuators["active"].stiffness[:, 0].tolist() == [1, 1, 10]
    limits = [call for call in hand.calls if call[0] == "limits"]
    assert len(limits) == 1 and torch.equal(limits[0][2], torch.full((1, 6), .2))
    assert [call[0] for call in hand.root_physx_view.calls].count("targets") == 1
    assert [call[0] for call in hand.root_physx_view.calls].count("force") == 1


def test_reaction_batch_reaches_native_composer_once_in_selected_env_order():
    owner, hand = side()
    force = torch.tensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
    torque = -force
    owner.set_wrenches(forces=force, torques=torque, body_ids=[0, 22],
                       env_ids=[2, 10], is_global=True, composer="permanent")
    calls = hand.permanent_wrench_composer.calls
    assert len(calls) == 1 and calls[0][2].tolist() == [2, 0]
    assert calls[0][-1] is True
    assert torch.equal(hand.permanent_wrench_composer.composed_force_as_torch[[2, 0]][:, [0, 22]], force)
    assert torch.equal(hand.permanent_wrench_composer.composed_torque_as_torch[[2, 0]][:, [0, 22]], torque)
    assert torch.equal(hand.permanent_wrench_composer.composed_force_as_torch[1], torch.full((23, 3), 7.))


def test_rejects_non_bijective_native_identity():
    impl = implementation()
    hand = Hand((10, 0, 0), "left")
    with pytest.raises(ValueError, match="identity|bijective"):
        impl.RegexArticulationSide(hand, ENV_IDS, PREFIXES, "/wrist_fixed")


def test_author_only_reuses_scene_configuration_and_constructs_exactly_two_sdk_assets(monkeypatch):
    impl = implementation()
    authors, constructors, originals, physical_spawns = [], [], [], []
    nodes = {}
    originals_layout, original_sources, args = object(), object(), object()

    class Cfg(SimpleNamespace):
        def copy(self): return deepcopy(self)

    class Root:
        def __init__(self, path): self.path = path
        def HasAPI(self, api): return True
        def GetPath(self): return SimpleNamespace(pathString=self.path)

    def author(sim, passed_args, layout, sources, prefix, origin, *, external_wire=False):
        assert passed_args is args and layout is originals_layout and sources is original_sources
        assert external_wire
        authors.append(prefix)
        hands = []
        for name in ("left", "right"):
            path = prefix + "/" + name + "Hand"
            cfg = Cfg(prim_path=path, spawn=SimpleNamespace(func=physical_spawn), articulation_root_prim_path=None,
                      init_state=SimpleNamespace(pos=tuple(origin), rot=(1., 0., 0., 0.), joint_pos={"finger11": .4}),
                      actuators={"active": {"stiffness": 10, "effort_limit": .2},
                                 "passive": {"stiffness": 0}, "wrist": {"stiffness": 100000}})
            originals.append(cfg)
            hands.append(Articulation(cfg))
        return dict(hands=hands, sensors=[], wire=None, prefix=prefix, origin=origin)

    def physical_spawn(path, cfg, translation, orientation):
        physical_spawns.append((path, translation, orientation))
        nodes[path] = [Root(path + "/wrist_fixed")]

    original_constructor = object()
    monkeypatch.setitem(author.__globals__, "Articulation", original_constructor)

    def constructor(cfg):
        assert len(authors) == 3, "author all physical USDs before callback construction"
        assert cfg.spawn is None
        assert cfg.articulation_root_prim_path == "/wrist_fixed"
        hand = Hand((10, 0, 2) if "leftHand" in cfg.prim_path else (2, 10, 0),
                    "left" if "leftHand" in cfg.prim_path else "right")
        hand.cfg = cfg
        constructors.append(hand)
        return hand

    fake_assets, fake_sim, fake_vec, fake_pxr = (ModuleType(n) for n in (
        "isaaclab.assets", "isaaclab.sim", "go2_pvcnn.control.dh116.tabletop_vec_isaac", "pxr"))
    fake_assets.Articulation = constructor
    fake_sim.get_all_matching_child_prims = lambda path, predicate, stage, traverse_instance_prims: [p for p in nodes[path] if predicate(p)]
    fake_vec.spawn_scene = author
    fake_vec.__file__ = str(Path(impl.__file__).with_name("tabletop_vec_isaac.py"))
    fake_pxr.UsdPhysics = SimpleNamespace(ArticulationRootAPI=object())
    fake_lab = ModuleType("isaaclab")
    fake_lab.__path__ = []
    fake_lab.sim = fake_sim
    for name, module in (("isaaclab", fake_lab), ("isaaclab.assets", fake_assets),
                         ("isaaclab.sim", fake_sim), (fake_vec.__name__, fake_vec), ("pxr", fake_pxr)):
        monkeypatch.setitem(sys.modules, name, module)
    batch = impl.spawn_regex_articulations(SimpleNamespace(stage=object()), args, originals_layout,
                                         original_sources, [(0, 0, 0), (1, 0, 0), (2, 0, 0)], env_ids=ENV_IDS)
    assert authors == list(PREFIXES)
    assert len(constructors) == 2
    assert len(physical_spawns) == 6
    assert physical_spawns[0][1:] == ((0., 0., 0.), (1., 0., 0., 0.))
    assert author.__globals__["Articulation"] is original_constructor, "process-global SDK constructor was mutated"
    assert [h.cfg.prim_path for h in constructors] == ["/World/envs/env_.*/leftHand", "/World/envs/env_.*/rightHand"]
    assert all(cfg.spawn is not None and cfg.articulation_root_prim_path is None for cfg in originals)
    for index, hand in enumerate(constructors):
        assert hand.cfg.actuators == originals[index].actuators
        assert hand.cfg.init_state == originals[index].init_state
    batch.bind()
    assert batch.sides["left"].row_indices.tolist() == [1, 2, 0]
    assert batch.sides["right"].row_indices.tolist() == [2, 0, 1]
    batch.write_data_to_sim()
    batch.update(1 / 480)
    assert all([c[0] for c in h.calls] == ["write", "update"] for h in constructors)


def test_author_original_scene_rejects_unpinned_source_before_any_usd_write(monkeypatch, tmp_path):
    impl = implementation()
    assert hasattr(impl, "author_original_scene"), "source-bound author-only clone is missing"
    source = tmp_path / "tabletop_vec_isaac.py"
    source.write_text("unexpected source bytes")
    fake_vec = ModuleType("go2_pvcnn.control.dh116.tabletop_vec_isaac")
    fake_vec.__file__ = str(source)
    fake_vec.spawn_scene = lambda *args, **kwargs: pytest.fail("unverified source executed")
    monkeypatch.setitem(sys.modules, fake_vec.__name__, fake_vec)
    with pytest.raises(ValueError, match="SHA|source"):
        impl.author_original_scene(None, None, None, None, "/World/envs/env_0", (0, 0, 0))
