"""Two real regex Articulations over unchanged, independently authored scenes.

Importing this module is CPU-safe. ``spawn_regex_articulations`` is called only
AFTER AppLauncher, before the caller's single SimulationContext.reset(). The
existing scene author owns USD, materials, mounts, mimic and actuator settings;
this module replaces only the number of Articulation Python/PhysX views.

All public tensors use the supplied environment order. PhysX rows are resolved
from actual prim_paths after initialization, independently for each side. There
are no per-environment Articulation wrappers or per-environment hot-path calls.
This layer does not step physics, reset a rod, certify clearance or alter the
original controller/safety/reward contracts.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
import re
import stat
from types import FunctionType

import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS
from .tabletop_palm_mount import ANGULAR_JOINT_NAMES, JOINT_NAMES

SIDES = ("left", "right")
# protected-ubuntu.json (inventory SHA fb6b58cfd2fe86408eb498a4390ecda35df2221d7198ca3a01ff59f86dea05ec)
# seals this unchanged source at the same digest. No protected file is patched.
ORIGINAL_SCENE_SHA256 = "f1f0f268c552572344ae8d13b8ceaa88471599fd67630b5d5c7a9ab1290b7512"


def author_original_scene(sim, args, original_layout, sources, prefix, origin):
    """Execute the SHA-bound original author in an isolated globals namespace.

    Only its Articulation constructor is replaced. The replacement invokes the
    exact cfg.spawn.func with cfg.init_state.pos/rot, then returns the config.
    Original mount, material, compliance and Mimic code runs unchanged. The
    original module/function globals and protected source bytes stay untouched.
    external_wire=True is mandatory, so no per-environment sensors/rod owners
    or Articulation callbacks are created by this author-only call.
    """
    original = importlib.import_module(__package__ + ".tabletop_vec_isaac")
    source = Path(original.__file__)
    if (source.is_symlink() or not stat.S_ISREG(source.stat().st_mode)
            or hashlib.sha256(source.read_bytes()).hexdigest() != ORIGINAL_SCENE_SHA256):
        raise ValueError("protected original scene source SHA differs")

    def author_cfg(cfg):
        if cfg.spawn is None:
            raise ValueError("original physical USD spawn configuration required")
        cfg.spawn.func(cfg.prim_path, cfg.spawn,
                       translation=cfg.init_state.pos, orientation=cfg.init_state.rot)
        return cfg

    function = original.spawn_scene
    namespace = function.__globals__.copy()
    namespace["Articulation"] = author_cfg
    author = FunctionType(function.__code__, namespace, function.__name__ + "_author_only",
                          function.__defaults__, function.__closure__)
    author.__kwdefaults__ = None if function.__kwdefaults__ is None else function.__kwdefaults__.copy()
    return author(sim, args, original_layout, sources, prefix, origin, external_wire=True)


def spawn_regex_articulations(sim, args, original_layout, sources, env_origins,
                             *, env_ids=None, env_root="/World/envs"):
    """Author original physical scenes, then construct exactly two SDK assets.

    ``author_original_scene`` executes the unchanged protected spawn_scene code
    with a private constructor override. It authors the original physical USDs,
    mount/material/mimic and returns two configs in ``hands``; no protected
    signature or process-global SDK constructor is changed.

    Call ``sim.reset()`` once outside this function, then ``batch.bind()``.
    Reset seeds must be supplied explicitly: the single regex config's default
    root state is not a substitute for each authored environment's world pose.
    """
    # Isaac/USD imports must stay after AppLauncher.
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from pxr import UsdPhysics

    origins = np.asarray(env_origins, dtype=float)
    if origins.ndim != 2 or origins.shape[1] != 3 or not len(origins) or not np.isfinite(origins).all():
        raise ValueError("finite nonempty environment origins [N,3] required")
    ids = tuple(range(len(origins))) if env_ids is None else tuple(env_ids)
    if (len(ids) != len(origins) or len(set(ids)) != len(ids)
            or any(type(value) is not int or value < 0 for value in ids)):
        raise ValueError("unique nonnegative integer environment identities required")
    if not isinstance(env_root, str) or re.fullmatch(r"/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*", env_root) is None:
        raise ValueError("literal absolute USD environment root required")
    prefixes = tuple(f"{env_root}/env_{value}" for value in ids)
    scenes = []
    suffixes = {side: [] for side in SIDES}
    for prefix, origin in zip(prefixes, origins, strict=True):
        scene = author_original_scene(sim, args, original_layout, sources, prefix, origin)
        if (len(scene["hands"]) != 2 or scene["wire"] is not None or scene["sensors"]
                or scene["prefix"] != prefix):
            raise ValueError("original bilateral author-only scene without duplicate wire/sensors required")
        for side, cfg in zip(SIDES, scene["hands"], strict=True):
            path = f"{prefix}/{side}Hand"
            if cfg.prim_path != path or cfg.spawn is None:
                raise ValueError("unchanged original authored hand configuration required")
            roots = sim_utils.get_all_matching_child_prims(
                path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                stage=sim.stage, traverse_instance_prims=False)
            if len(roots) != 1:
                raise ValueError("exactly one actual USD ArticulationRootAPI per hand required")
            root_path = roots[0].GetPath().pathString
            if root_path != path and not root_path.startswith(path + "/"):
                raise ValueError("actual articulation root must belong to its authored hand")
            suffixes[side].append(root_path[len(path):])
        scenes.append(scene)
    if any(len(set(suffixes[side])) != 1 for side in SIDES):
        raise ValueError("one identical physical root layout per side required")
    hands = []
    for index, side in enumerate(SIDES):
        cfg = scenes[0]["hands"][index].copy()
        cfg.prim_path = f"{env_root}/env_.*/{side}Hand"
        cfg.spawn = None
        # Installed Articulation._initialize_impl appends this actual USD suffix
        # then converts .* to the PhysX wildcard. No guessed '/root' path.
        cfg.articulation_root_prim_path = suffixes[side][0] or None
        hands.append(Articulation(cfg))
    return RegexArticulationBatch(hands, ids, prefixes,
                                  tuple(suffixes[side][0] for side in SIDES), scenes)


class RegexArticulationSide:
    """One SDK Articulation, with public environment IDs mapped to view rows."""

    def __init__(self, hand, env_ids, prefixes, root_suffix):
        self.hand = hand
        self.env_ids = tuple(env_ids)
        self.prefixes = tuple(prefixes)
        self.device = torch.device(hand.device)
        self.joint_names = tuple(hand.joint_names)
        self.body_names = tuple(hand.body_names)
        if (not hand.is_initialized or len(self.env_ids) != len(self.prefixes)
                or not self.env_ids or len(set(self.env_ids)) != len(self.env_ids)
                or len(set(self.prefixes)) != len(self.prefixes)):
            raise ValueError("initialized view and unique environment identity layout required")
        expected_names = set(JOINTS + JOINT_NAMES + ANGULAR_JOINT_NAMES)
        if (len(self.joint_names) != 17 or set(self.joint_names) != expected_names
                or len(self.body_names) != 23 or len(set(self.body_names)) != 23):
            raise ValueError("original 17DOF/23-body mounted DH116 hand required")
        paths = tuple(hand.root_physx_view.prim_paths)
        # PhysX versions may report either the articulation's container or its
        # API-bearing fixed joint. Both must resolve to the same authored hand.
        pattern = re.compile(r"^(.+)/(left|right)Hand(?:" + re.escape(root_suffix) + r")?$")
        parsed = [pattern.fullmatch(path) for path in paths]
        if (len(paths) != len(self.env_ids) or hand.num_instances != len(paths)
                or any(match is None for match in parsed)
                or len({match.group(2) for match in parsed}) != 1):
            raise ValueError("bijective actual articulation identity per environment required")
        prefix_to_id = dict(zip(self.prefixes, self.env_ids, strict=True))
        native_prefixes = tuple(match.group(1) for match in parsed)
        if len(set(native_prefixes)) != len(paths) or set(native_prefixes) != set(self.prefixes):
            raise ValueError("bijective actual articulation identity per environment required")
        self.side = parsed[0].group(2)
        self.row_env_ids = tuple(prefix_to_id[prefix] for prefix in native_prefixes)
        row_for_id = {value: row for row, value in enumerate(self.row_env_ids)}
        self.row_indices = torch.tensor([row_for_id[value] for value in self.env_ids],
                                        device=self.device, dtype=torch.long)
        ordered = sorted(row_for_id)
        self._sorted_env_ids = torch.tensor(ordered, device=self.device, dtype=torch.long)
        self._sorted_rows = torch.tensor([row_for_id[value] for value in ordered],
                                         device=self.device, dtype=torch.long)
        self.active_ids = [self.joint_names.index(name) for name in ACTIVE]
        self.mount_ids = [self.joint_names.index(name) for name in JOINT_NAMES + ANGULAR_JOINT_NAMES]

    def rows(self, env_ids=None):
        """Return actual PhysX rows in the caller's supplied environment order."""
        if env_ids is None:
            return self.row_indices
        values = torch.as_tensor(env_ids, device=self.device)
        if values.dtype not in (torch.int32, torch.int64):
            raise ValueError("integer environment identities required")
        values = values.to(dtype=torch.long)
        if values.ndim != 1 or not len(values):
            raise ValueError("nonempty environment identity vector required")
        if len(torch.unique(values)) != len(values):
            raise ValueError("unique environment identities required")
        index = torch.searchsorted(self._sorted_env_ids, values)
        safe = index.clamp_max(len(self._sorted_env_ids) - 1)
        if bool(((index >= len(self._sorted_env_ids)) | (self._sorted_env_ids[safe] != values)).any()):
            raise ValueError("unknown environment identity")
        return self._sorted_rows[index]

    def read(self, env_ids=None):
        """Own actual backend q/qd/root/link/targets; quaternions are explicit."""
        rows = self.rows(env_ids)
        view = self.hand.root_physx_view
        getters = dict(q="get_dof_positions", qd="get_dof_velocities",
                       backend_targets="get_dof_position_targets",
                       root_pose_xyzw="get_root_transforms", root_velocity="get_root_velocities",
                       link_pose_xyzw="get_link_transforms", link_velocity="get_link_velocities")
        state = {key: getattr(view, method)().index_select(0, rows) for key, method in getters.items()}
        order = [0, 1, 2, 6, 3, 4, 5]
        state["root_pose_wxyz"] = state["root_pose_xyzw"][..., order]
        state["link_pose_wxyz"] = state["link_pose_xyzw"][..., order]
        return state

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        """Stage one native batched target; call batch.write_data_to_sim next."""
        rows = self.rows(env_ids)
        count = len(self.joint_names) if joint_ids is None else len(joint_ids)
        value = self._tensor(target, (len(rows), count), "joint target")
        self.hand.set_joint_position_target(value, joint_ids=joint_ids, env_ids=rows)

    def set_wrenches(self, *, forces=None, torques=None, positions=None, body_ids=None,
                     env_ids=None, is_global=False, composer="permanent", add=False):
        """One native composer call for a complete side, including selected envs.

        Positions and global/local frames have the original WrenchComposer
        semantics. This method does not negate, redistribute or clip rod loads.
        Incoming reaction forces already carry their physical sign.
        """
        if composer not in ("permanent", "instantaneous"):
            raise ValueError("permanent or instantaneous composer required")
        rows = self.rows(env_ids)
        count = len(self.body_names) if body_ids is None else len(body_ids)
        shape = (len(rows), count, 3)
        values = {name: None if value is None else self._tensor(value, shape, name).contiguous()
                  for name, value in (("forces", forces), ("torques", torques), ("positions", positions))}
        native = getattr(self.hand, composer + "_wrench_composer")
        method = native.add_forces_and_torques if add else native.set_forces_and_torques
        method(**values, body_ids=body_ids, env_ids=rows, is_global=is_global)

    def reset(self, *, env_ids, root_pose_wxyz, joint_pos, joint_vel, active_effort_limit_nm):
        """Restore selected original hand rows without stepping or global write.

        Full 17-DOF inputs must retain all eleven authored finger q/qd values and
        zero the six wrist coordinates. Effort and stiffness match the original
        Reset transaction. Controller/filter/contact/rod reset remains the
        caller's transaction; this method only owns the articulation rows.
        """
        rows = self.rows(env_ids)
        pose = self._tensor(root_pose_wxyz, (len(rows), 7), "reset WXYZ root")
        q = self._tensor(joint_pos, (len(rows), 17), "reset q")
        qd = self._tensor(joint_vel, (len(rows), 17), "reset qd")
        if (not all(bool(torch.isfinite(value).all()) for value in (pose, q, qd))
                or not torch.allclose(pose[:, 3:].norm(dim=-1), torch.ones_like(pose[:, 0]), atol=1e-6, rtol=0)
                or bool((q[:, self.mount_ids] != 0).any()) or bool((qd[:, self.mount_ids] != 0).any())
                or not math.isfinite(active_effort_limit_nm) or active_effort_limit_nm <= 0):
            raise ValueError("finite original reset values, normalized roots and zero wrist state required")
        hand, view = self.hand, self.hand.root_physx_view
        hand.reset(rows)
        hand.write_joint_state_to_sim(q, qd, env_ids=rows)
        hand.write_root_pose_to_sim(pose, env_ids=rows)
        hand.write_root_velocity_to_sim(q.new_zeros((len(rows), 6)), env_ids=rows)
        hand.set_joint_position_target(q, env_ids=rows)
        stiffness = q.new_full((len(rows), len(self.active_ids)), 10.)
        hand.write_joint_stiffness_to_sim(stiffness, joint_ids=self.active_ids, env_ids=rows)
        hand.actuators["active"].stiffness[rows] = 10.
        hand.set_joint_velocity_target(torch.zeros_like(q), env_ids=rows)
        hand.set_joint_effort_target(torch.zeros_like(q), env_ids=rows)
        limits = q.new_full((len(rows), len(self.active_ids)), active_effort_limit_nm)
        hand.write_joint_effort_limit_to_sim(limits, joint_ids=self.active_ids, env_ids=rows)
        # Installed PhysX setters require [view.count, max_dofs], even for
        # selected rows. Articulation.write_data_to_sim has no selected API and
        # resets EVERY instantaneous wrench, so it cannot be used here.
        view.set_dof_position_targets(hand.data.joint_pos_target, indices=rows)
        view.set_dof_velocity_targets(hand.data.joint_vel_target, indices=rows)
        view.set_dof_actuation_forces(hand.data.joint_effort_target, indices=rows)
        zeros = q.new_zeros((hand.num_instances, len(self.body_names), 3))
        view.apply_forces_and_torques_at_position(force_data=zeros.view(-1, 3),
            torque_data=zeros.view(-1, 3), position_data=None, indices=rows, is_global=False)

    def _tensor(self, value, shape, label):
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{label} shape {shape} required")
        return tensor


class RegexArticulationBatch:
    """Own the two original SDK callbacks and their startup-only scene records."""

    def __init__(self, hands, env_ids, prefixes, root_suffixes, scenes):
        self.hands = tuple(hands)
        if len(self.hands) != 2:
            raise ValueError("exactly two SDK Articulations required")
        self.env_ids = tuple(env_ids)
        self.prefixes = tuple(prefixes)
        self.root_suffixes = tuple(root_suffixes)
        self.scenes = tuple(scenes)
        self.sides = {}

    def bind(self):
        """Resolve row identity once after the caller's single SDK reset."""
        if not self.sides:
            sides = {side: RegexArticulationSide(hand, self.env_ids, self.prefixes, suffix)
                     for side, hand, suffix in zip(SIDES, self.hands, self.root_suffixes, strict=True)}
            if any(owner.side != side for side, owner in sides.items()):
                raise ValueError("actual bilateral hand identity differs")
            self.sides = sides
        return self

    def read(self, env_ids=None):
        return {side: self.sides[side].read(env_ids) for side in SIDES}

    def write_data_to_sim(self):
        """One SDK write per side; called once before the shared physics step."""
        for hand in self.hands:
            hand.write_data_to_sim()

    def update(self, dt):
        """One SDK cache update per side; this never advances physics."""
        for hand in self.hands:
            hand.update(dt)

    def reset_composers(self, env_ids=None):
        """Clear both native composers per side with the same row mapping."""
        for side in SIDES:
            owner = self.sides[side]
            rows = None if env_ids is None else owner.rows(env_ids)
            owner.hand.permanent_wrench_composer.reset(rows)
            owner.hand.instantaneous_wrench_composer.reset(rows)
