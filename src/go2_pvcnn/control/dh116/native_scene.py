"""Asset and pose contracts for the native Panda + separate DH116 scene."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SIDES = ("left", "right")
_PANDA_GRIPPER_FINGER_LINKS = frozenset({"panda_leftfinger", "panda_rightfinger"})

_HAND_ASSET_NAMES = {
    "left": "d134680ab619_DH116_left.usd",
    "right": "3046ad2ed855_DH116_right.usd",
}


def wire_orientation_wxyz() -> tuple[float, float, float, float]:
    """Rotate the cylinder's local Z axis into the tabletop +Y direction."""
    return (0.70710678, -0.70710678, 0.0, 0.0)


@dataclass(frozen=True)
class NativeSceneSpec:
    root_path: str
    panda_paths: dict[str, str]
    hand_paths: dict[str, str]
    hand_usd: dict[str, Path]
    flange_link: str
    # Base translations and DH116 palm poses are the reviewed candidate3
    # mounting contract.  They keep native Panda geometry while reproducing
    # the reference image's palm-down, fingers-forward placement.
    panda_base_pos: dict[str, tuple[float, float, float]] | None = None
    hand_root_pose: dict[str, tuple[float, float, float, float, float, float, float]] | None = None


@dataclass
class NativeSceneRuntime:
    spec: NativeSceneSpec
    stage: object
    arms: dict[str, object]
    hands: dict[str, object]
    hand_base_local: dict[str, object] = field(default_factory=dict)
    hand_mount_offset: dict[str, object] = field(default_factory=dict)
    static_body_poses: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    hidden_gripper_links: dict[str, int] = field(default_factory=dict)
    root_delta_z: float = 0.0

    def assert_visual_contract(self) -> dict[str, object]:
        """Return the resolved scene paths and reject missing scene branches."""
        result = {
            "asset_mode": "native_panda_plus_separate_dh116",
            "panda_paths": dict(self.spec.panda_paths),
            "hand_paths": dict(self.spec.hand_paths),
            "hand_usd": {side: str(path) for side, path in self.spec.hand_usd.items()},
            "flange_link": self.spec.flange_link,
        }
        for side in SIDES:
            if side not in self.arms or side not in self.hands:
                raise ValueError(f"missing native scene side: {side}")
            panda_hand = self.stage.GetPrimAtPath(
                f"{self.spec.panda_paths[side]}/{self.spec.flange_link}"
            )
            if not panda_hand.IsValid():
                raise ValueError(f"missing Panda DH116 mount: {side}")
            if self.hidden_gripper_links.get(side, 0) < len(_PANDA_GRIPPER_FINGER_LINKS):
                raise ValueError(f"Panda stock gripper is still visible: {side}")
            if not self.stage.GetPrimAtPath(self.spec.hand_paths[side]).IsValid():
                raise ValueError(f"missing replacement DH116 hand: {side}")
        result["hidden_stock_gripper_links"] = dict(self.hidden_gripper_links)
        return result

    def apply_root_delta_z(self, delta_z: float) -> None:
        """Translate both native Panda roots by the same world-Z increment.

        The DH116 roots are mounted separately from the Panda articulations, so
        the hand roots are re-synchronised from their corresponding flange
        after the Panda root write.  Joint targets and left/right identity are
        left untouched.
        """
        if not np.isfinite(delta_z):
            raise ValueError("root delta z must be finite")
        if not delta_z:
            return
        for side in SIDES:
            arm = self.arms[side]
            root_state = arm.data.root_state_w[0].clone()
            root_state[2] += float(delta_z)
            arm.write_root_pose_to_sim(root_state.unsqueeze(0))
        self.root_delta_z += float(delta_z)
        sync_native_hand_roots(self)


def default_native_scene_spec(root: Path) -> NativeSceneSpec:
    """Return the project-local native Panda plus separate DH116 layout."""
    hand_root = Path(root) / "assets/fixed_dual_panda_dh116_v1/dependencies"
    spec = NativeSceneSpec(
        root_path="/World/DH116DualPandaScene",
        panda_paths={"left": "/World/LeftPanda", "right": "/World/RightPanda"},
        hand_paths={"left": "/World/LeftDH116", "right": "/World/RightDH116"},
        hand_usd={side: hand_root / name for side, name in _HAND_ASSET_NAMES.items()},
        flange_link="panda_hand",
        panda_base_pos={
            "left": (0.0747597133794231, 0.2491319181085123, -0.0116718834772184),
            "right": (0.0744969153918507, -0.2493211823662, -0.0123784326847),
        },
        hand_root_pose={
            "left": (0.38154750224737377, 0.2373296462947355, 0.3612084291104437,
                     -0.00524514, 0.99977878, -0.00643838, -0.01932409),
            "right": (0.379352202466893, -0.237214928805469, 0.36181091453662617,
                      0.00606007, 0.99966999, 0.00472094, -0.02451318),
        },
    )
    validate_scene_spec(spec)
    for side, path in spec.hand_usd.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing DH116 {side} USD: {path}")
    return spec


def _panda_config():
    """Load IsaacLab's native Franka config after SimulationApp startup."""
    try:
        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    except ModuleNotFoundError:
        import sys

        sys.path.insert(0, "/home/xk/miniconda3/envs/go2/lib/python3.11/site-packages/isaaclab/source/isaaclab_assets")
        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    return FRANKA_PANDA_CFG


def _hide_builtin_panda_fingers(stage, panda_path: str) -> int:
    from pxr import Usd, UsdGeom

    hidden = 0
    matched = []
    # The native Franka asset nests the gripper fingers below ``panda_hand``;
    # they are not direct children of the articulation root.  Match by leaf
    # name while traversing the complete Panda subtree so the replacement
    # DH116 is not visually occluded by the stock two-finger gripper.
    # Keep panda_hand as the DH116 mounting housing/flange. Only the stock
    # parallel fingers are replaced by the dexterous hand.
    names = _PANDA_GRIPPER_FINGER_LINKS
    root = stage.GetPrimAtPath(panda_path)
    # ``panda_instanceable.usd`` places the visual links behind an instance
    # proxy, so include instance proxies in the traversal as well.
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() in names and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden += 1
            matched.append(str(prim.GetPath()))
    return hidden



def _disable_gravity(stage, root_path: str) -> int:
    """Keep the standalone DH116 preview at its commanded pose."""
    from pxr import PhysxSchema, UsdPhysics

    changed = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root_path):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.CreateDisableGravityAttr().Set(True)
            changed += 1
    return changed


def _pose_matrix(pose):
    from pxr import Gf

    position = pose[:3]
    quaternion = Gf.Quatd(*[float(value) for value in pose[3:7]])
    matrix = Gf.Matrix4d().SetRotate(Gf.Matrix3d(quaternion))
    # SetTranslate() replaces the full matrix in this USD binding.  Preserve
    # the quaternion while authoring only the translation column.
    matrix.SetTranslateOnly(Gf.Vec3d(*[float(value) for value in position]))
    return matrix


# Candidate3's reviewed contact section: both hands face the same wire
# section, with the thumb/index pad centres separated along world X.  These
# are visual placement targets only; runtime grasp qualification remains
# delegated to the contact/MPC checks.
_CANDIDATE3_PAD_TARGETS = {
    "left": np.asarray(
        [[0.4405987122556929, 0.19978045446135564, 0.309395176551825],
         [0.4197235329376601, 0.1994373379573411, 0.3094012188098023]],
        dtype=float,
    ),
    "right": np.asarray(
        [[0.4392239960403026, -0.19937243259494003, 0.30963017459773173],
         [0.42052292629002086, -0.20033497348712415, 0.30927251415931256]],
        dtype=float,
    ),
}
_PAD_LOCAL_POINTS = {
    "left": {
        "thumb": np.asarray((0.00017948482980035264, -0.010452785059975968, -0.03394154097890579)),
        "index": np.asarray((-0.0038150536494581076, -0.015150306258402396, 0.040506412058763396)),
    },
    "right": {
        "thumb": np.asarray((-0.0008940861981015615, 0.03219328112399958, -0.015560792002874102)),
        "index": np.asarray((0.004119789543680456, -0.01512276321265944, 0.03972204958180129)),
    },
}


def _quat_to_matrix_wxyz(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    return np.asarray(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=float,
    )


def _rotation_to_wxyz(rotation):
    from scipy.spatial.transform import Rotation

    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]],
        dtype=float,
    )


def _pose_to_numpy_matrix(pose):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_matrix_wxyz(np.asarray(pose)[3:7])
    matrix[:3, 3] = np.asarray(pose, dtype=float)[:3]
    return matrix


def _numpy_matrix_to_pose(matrix):
    return np.asarray(
        [*np.asarray(matrix, dtype=float)[:3, 3].tolist(),
         *_rotation_to_wxyz(np.asarray(matrix, dtype=float)[:3, :3]).tolist()],
        dtype=float,
    )


def _align_vector_rotation(source, target):
    """Return the minimum world rotation that maps source onto target."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3)
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(source)))] = 1.0
        cross = np.cross(source, axis)
        cross /= np.linalg.norm(cross)
        return 2.0 * np.outer(cross, cross) - np.eye(3)
    skew = np.asarray(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _root_transform_for_pad_alignment(base_pose, current_pads, desired_pads):
    """Return a world root pose aligning pad centres around the current base."""
    base_pose = np.asarray(base_pose, dtype=float)
    current_pads = np.asarray(current_pads, dtype=float)
    desired_pads = np.asarray(desired_pads, dtype=float)
    base_rotation = _quat_to_matrix_wxyz(base_pose[3:7])
    correction = _align_vector_rotation(
        current_pads[1] - current_pads[0], desired_pads[1] - desired_pads[0]
    )
    new_rotation = correction @ base_rotation
    local_pads = (current_pads - base_pose[:3]) @ base_rotation
    new_translation = desired_pads.mean(axis=0) - local_pads.mean(axis=0) @ new_rotation.T
    return np.asarray(
        [*new_translation, *_rotation_to_wxyz(new_rotation)], dtype=float
    )


def _refine_hand_root_to_candidate3_pads(hand, side):
    """Rotate/translate one DH116 root so its actual pads face candidate3 wire."""
    import torch

    base_index = hand.body_names.index("base_link")
    thumb_index = hand.body_names.index("finger14_link")
    index_index = hand.body_names.index("finger23_link")
    poses = hand.data.body_link_pose_w[0].detach().cpu().numpy()
    base_pose = poses[base_index]

    def point(body_index, local):
        pose = poses[body_index]
        return _quat_to_matrix_wxyz(pose[3:7]) @ local + pose[:3]

    local_points = _PAD_LOCAL_POINTS[side]
    current = np.asarray(
        [point(thumb_index, local_points["thumb"]),
         point(index_index, local_points["index"])],
        dtype=float,
    )
    desired = _CANDIDATE3_PAD_TARGETS[side]
    root_pose = _root_transform_for_pad_alignment(base_pose, current, desired)
    new_translation = root_pose[:3]
    new_rotation = _quat_to_matrix_wxyz(root_pose[3:7])
    matrix_np = np.eye(4, dtype=float)
    matrix_np[:3, :3] = new_rotation
    matrix_np[:3, 3] = new_translation
    matrix = _pose_matrix(root_pose)
    root_state = torch.tensor(
        [root_pose.tolist()],
        device=hand.device,
        dtype=torch.float32,
    )
    hand.write_root_pose_to_sim(root_state)
    return matrix, matrix_np, current, desired


def _set_root_world_matrix(stage, hand_path: str, matrix) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Xformable(stage.GetPrimAtPath(hand_path)).MakeMatrixXform().Set(matrix)


def _set_root_world_pose(stage, hand_path: str, pose) -> None:
    _set_root_world_matrix(stage, hand_path, _pose_matrix(pose))


def _ordered_hand_state(hand, active_q, device: str):
    """Expand active DH116 joints and reorder them to Isaac's joint_names."""
    import torch

    from .linear_coupling import JOINTS, expand

    full = dict(zip(JOINTS, expand(active_q)))
    return torch.tensor(
        [[full[name] for name in hand.joint_names]],
        device=device,
        dtype=torch.float32,
    )


def _hold_hand_target(hand, state) -> None:
    """Keep the standalone hand at the commanded pose during physics steps."""
    # Only the six active joints are actuated.  Supplying them in ACTIVE order
    # avoids IsaacLab's actuator-order fallback; passive followers remain at
    # the explicitly written coupled state for this pose preview.
    from .linear_coupling import ACTIVE

    active_ids = [hand.joint_names.index(name) for name in ACTIVE]
    hand.set_joint_position_target(state[:, active_ids], joint_ids=active_ids)
    hand.write_data_to_sim()


def spawn_native_scene(sim, spec: NativeSceneSpec, pose: dict[str, np.ndarray], device: str):
    """Spawn native Panda articulations and separately referenced DH116 hands."""
    import torch
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from .linear_coupling import ACTIVE, JOINTS, expand

    validate_scene_spec(spec)
    panda_cfg = _panda_config()
    stage = omni.usd.get_context().get_stage()
    arms: dict[str, object] = {}
    hands: dict[str, object] = {}
    hidden_gripper_links: dict[str, int] = {}
    base_positions = spec.panda_base_pos or {
        "left": (0.0, 0.25, 0.0), "right": (0.0, -0.25, 0.0)
    }
    for side in SIDES:
        q = np.asarray(pose[side], dtype=np.float64)
        cfg = panda_cfg.copy()
        cfg.prim_path = spec.panda_paths[side]
        cfg.init_state.pos = base_positions[side]
        cfg.init_state.joint_pos = {
            f"panda_joint{index}": float(q[index - 1]) for index in range(1, 8)
        }
        cfg.actuators = {
            name: actuator for name, actuator in cfg.actuators.items() if name != "panda_hand"
        }
        cfg.actuators["panda_hand_hidden"] = ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"], stiffness=0.0, damping=0.0,
            effort_limit_sim=0.0, velocity_limit_sim=0.3,
        )
        arms[side] = Articulation(cfg)

    sim.reset()
    stage.Load()
    for side in SIDES:
        if not stage.GetPrimAtPath(f"{spec.panda_paths[side]}/{spec.flange_link}").IsValid():
            children = [str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).startswith(spec.panda_paths[side])]
            raise RuntimeError(f"native Panda flange is missing: {spec.panda_paths[side]}/{spec.flange_link}; children={children[:30]}")

    for side in SIDES:
        # Use this side's state.  Reusing the last arm-loop value made both
        # DH116 references receive the right-hand joint vector.
        q = np.asarray(pose[side], dtype=np.float64)
        hand_path = spec.hand_paths[side]
        hand_prim = stage.DefinePrim(hand_path, "Xform")
        hand_prim.GetReferences().AddReference(str(spec.hand_usd[side]))
        stage.Load()
        _disable_gravity(stage, hand_path)
        hand_root_path = hand_path
        if not stage.GetPrimAtPath(f"{hand_root_path}/base_link").IsValid():
            children = [str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).startswith(hand_path)]
            raise RuntimeError(f"DH116 reference did not compose at {hand_root_path}; children={children[:20]}")
        hand_cfg = ArticulationCfg(
            prim_path=hand_root_path,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(zip(JOINTS, map(float, expand(q[7:]))))
            ),
            actuators={
                "active": ImplicitActuatorCfg(
                    joint_names_expr=list(ACTIVE), stiffness=10.0, damping=.05,
                    effort_limit_sim=.15, velocity_limit_sim=.3,
                ),
                "passive": ImplicitActuatorCfg(
                    joint_names_expr=[name for name in JOINTS if name not in ACTIVE],
                    stiffness=0.0, damping=0.0, effort_limit_sim=0.0,
                    velocity_limit_sim=.3,
                ),
            },
        )
        hands[side] = Articulation(hand_cfg)
        # Isaac's stock Panda is an instanceable USD.  Break instancing on
        # this copy before hiding its stock fingers, otherwise USD rejects
        # edits to the finger visuals as instance-proxy authoring.
        sim_utils.make_uninstanceable(spec.panda_paths[side], stage)
        hidden = _hide_builtin_panda_fingers(stage, spec.panda_paths[side])
        hidden_gripper_links[side] = hidden

    runtime = NativeSceneRuntime(
        spec=spec, stage=stage, arms=arms, hands=hands,
        hidden_gripper_links=hidden_gripper_links,
    )
    runtime.assert_visual_contract()
    sim.reset()
    for side in SIDES:
        arm, hand = runtime.arms[side], runtime.hands[side]
        q = np.asarray(pose[side], dtype=np.float32)
        arm_state = arm.data.default_joint_pos.clone()
        for index, name in enumerate(arm.joint_names):
            if name.startswith("panda_joint") and name[-1:].isdigit():
                joint_index = int(name[-1])
                if 1 <= joint_index <= 7:
                    arm_state[0, arm.joint_names.index(name)] = float(q[joint_index - 1])
        hand_state = _ordered_hand_state(hand, q[7:], device)
        arm.write_joint_state_to_sim(arm_state, torch.zeros_like(arm_state))
        hand.write_joint_state_to_sim(hand_state, torch.zeros_like(hand_state))
        arm.reset()
        hand.reset()
        _hold_hand_target(hand, hand_state)
        target_root = (spec.hand_root_pose or {}).get(side)
        if target_root is not None:
            root_state = torch.tensor(
                [list(target_root)], device=device, dtype=torch.float32
            )
            hand.write_root_pose_to_sim(root_state)
    sim.step(render=False)
    # Preserve the reviewed candidate3 DH116 root poses.  The hands remain
    # separate articulations, with a fixed per-side transform from each Panda
    # flange; moving roots to chase pad points breaks the mechanical mount.
    target_roots = spec.hand_root_pose or {}
    flange_worlds = {}
    for side in SIDES:
        arm = runtime.arms[side]
        hand = runtime.hands[side]
        q = np.asarray(pose[side], dtype=np.float32)
        arm.update(1.0 / 240.0)
        try:
            flange_index = arm.body_names.index(spec.flange_link)
        except ValueError as exc:
            raise RuntimeError(f"native Panda flange body is missing: {spec.flange_link}") from exc
        flange_pose = arm.data.body_link_pose_w[0, flange_index].detach().cpu().numpy()
        flange_world = _pose_matrix(flange_pose)
        flange_worlds[side] = flange_world
        if side in target_roots:
            target_root = _pose_matrix(target_roots[side])
            _set_root_world_matrix(runtime.stage, spec.hand_paths[side], target_root)
            # The initialization step lets PhysX settle the newly composed
            # articulations and can move active joints toward their default
            # drive targets.  Re-teleport the reviewed candidate state after
            # that step, then refresh kinematic link transforms without a
            # second physics step so the static preview is exact.
            hand_state = _ordered_hand_state(hand, q[7:], device)
            hand.write_joint_state_to_sim(hand_state, torch.zeros_like(hand_state))
            _hold_hand_target(hand, hand_state)
        else:
            base_index = hand.body_names.index("base_link")
            base_pose = hand.data.body_link_pose_w[0, base_index].detach().cpu().numpy()
            base_world = _pose_matrix(base_pose)
            runtime.hand_base_local[side] = base_world
            _set_root_world_matrix(
                runtime.stage,
                spec.hand_paths[side],
                flange_world * base_world.GetInverse(),
            )
            hand_state = _ordered_hand_state(hand, q[7:], device)
            hand.write_joint_state_to_sim(hand_state, torch.zeros_like(hand_state))
            _hold_hand_target(hand, hand_state)
    # Refresh articulation views and retain the reviewed flange-to-hand mount.
    sim.physics_sim_view.update_articulations_kinematic()
    for side in SIDES:
        if side not in target_roots:
            continue
        hand = runtime.hands[side]
        hand.update(1.0 / 240.0)
        body_poses = hand.data.body_link_pose_w[0].detach().cpu().numpy().copy()
        body_by_name = dict(zip(hand.body_names, body_poses))
        old_base = _pose_to_numpy_matrix(body_by_name["base_link"])
        new_root = _pose_to_numpy_matrix(target_roots[side])
        root_delta = new_root @ np.linalg.inv(old_base)
        runtime.static_body_poses[side] = {
            name: _numpy_matrix_to_pose(root_delta @ _pose_to_numpy_matrix(body_pose))
            for name, body_pose in body_by_name.items()
        }
        runtime.hand_mount_offset[side] = (
            flange_worlds[side].GetInverse() * _pose_matrix(target_roots[side])
        )
    # ``sim.reset()`` above re-applies the referenced Panda visuals on some
    # Isaac builds; enforce the replacement-hand visibility after that reset.
    for side in SIDES:
        runtime.hidden_gripper_links[side] = _hide_builtin_panda_fingers(
            runtime.stage, spec.panda_paths[side]
        )
    runtime.assert_visual_contract()
    return runtime


def apply_native_targets(runtime: NativeSceneRuntime, pose: dict[str, np.ndarray], device: str) -> None:
    """Write one independent arm/hand target state for each side."""
    import torch

    from .linear_coupling import JOINTS, expand

    for side in SIDES:
        q = np.asarray(pose[side], dtype=np.float32)
        arm = runtime.arms[side]
        hand = runtime.hands[side]
        arm_state = arm.data.default_joint_pos.clone()
        for name in arm.joint_names:
            if name.startswith("panda_joint") and name[-1:].isdigit():
                joint_index = int(name[-1])
                if 1 <= joint_index <= 7:
                    arm_state[0, arm.joint_names.index(name)] = float(q[joint_index - 1])
        hand_state = _ordered_hand_state(hand, q[7:], device)
        arm.write_joint_state_to_sim(arm_state, torch.zeros_like(arm_state))
        hand.write_joint_state_to_sim(hand_state, torch.zeros_like(hand_state))
        _hold_hand_target(hand, hand_state)


def sync_native_hand_roots(runtime: NativeSceneRuntime) -> None:
    """Align each independent DH116 root prim to its Panda flange pose."""
    for side in SIDES:
        arm = runtime.arms[side]
        arm.update(1.0 / 240.0)
        flange_index = arm.body_names.index(runtime.spec.flange_link)
        flange_pose = arm.data.body_link_pose_w[0, flange_index].detach().cpu().numpy()
        mount_offset = runtime.hand_mount_offset.get(side)
        if mount_offset is not None:
            _set_root_world_matrix(
                runtime.stage,
                runtime.spec.hand_paths[side],
                _pose_matrix(flange_pose) * mount_offset,
            )
            continue
        local_base = runtime.hand_base_local.get(side)
        if local_base is None:
            raise RuntimeError(f"missing DH116 base-link mount transform for {side}")
        _set_root_world_matrix(runtime.stage, runtime.spec.hand_paths[side],
                               _pose_matrix(flange_pose) * local_base.GetInverse())


def candidate3_pose(path: Path) -> dict[str, np.ndarray]:
    """Load the reviewed candidate3 pose without changing its values."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "dh116-static-pose-v1":
        raise ValueError("candidate3 pose schema mismatch")
    states = payload.get("states")
    if not isinstance(states, dict) or set(states) != set(SIDES):
        raise ValueError("candidate3 pose must contain left and right states")
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        values = np.asarray(states[side], dtype=np.float64)
        if values.shape != (13,):
            raise ValueError(f"candidate3 {side} state must have shape (13,)")
        if not np.isfinite(values).all():
            raise ValueError(f"candidate3 {side} state contains non-finite values")
        result[side] = values
    return result


def validate_scene_spec(spec: NativeSceneSpec) -> None:
    """Validate explicit left/right scene identity before Isaac startup."""
    if set(spec.panda_paths) != set(SIDES) or set(spec.hand_paths) != set(SIDES):
        raise ValueError("scene spec must contain distinct left and right Panda/hand paths")
    if set(spec.hand_usd) != set(SIDES):
        raise ValueError("scene spec must contain left and right DH116 USD paths")
    if spec.panda_paths["left"] == spec.panda_paths["right"]:
        raise ValueError("left and right Panda prim paths must be distinct")
    if spec.hand_paths["left"] == spec.hand_paths["right"]:
        raise ValueError("left and right DH116 prim paths must be distinct")
    resolved = {side: Path(spec.hand_usd[side]).resolve() for side in SIDES}
    if resolved["left"] == resolved["right"]:
        raise ValueError("left and right DH116 USD assets must be distinct")
    if any(path.name == "fixed_dual_panda_dh116.usd" for path in resolved.values()):
        raise ValueError("combined fixed_dual_panda_dh116.usd is not allowed")
    if not spec.flange_link:
        raise ValueError("Panda flange link is required")
