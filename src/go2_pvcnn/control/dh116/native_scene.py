"""Asset and pose contracts for the native Panda + separate DH116 scene."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SIDES = ("left", "right")

_HAND_ASSET_NAMES = {
    "left": "d134680ab619_DH116_left.usd",
    "right": "3046ad2ed855_DH116_right.usd",
}


def wire_orientation_wxyz() -> tuple[float, float, float, float]:
    """Rotate the cylinder's local Z axis into the tabletop X direction."""
    return (0.70710678, 0.0, 0.70710678, 0.0)


@dataclass(frozen=True)
class NativeSceneSpec:
    root_path: str
    panda_paths: dict[str, str]
    hand_paths: dict[str, str]
    hand_usd: dict[str, Path]
    flange_link: str


@dataclass
class NativeSceneRuntime:
    spec: NativeSceneSpec
    stage: object
    arms: dict[str, object]
    hands: dict[str, object]

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
        return result


def default_native_scene_spec(root: Path) -> NativeSceneSpec:
    """Return the project-local native Panda plus separate DH116 layout."""
    hand_root = Path(root) / "assets/fixed_dual_panda_dh116_v1/dependencies"
    spec = NativeSceneSpec(
        root_path="/World/DH116DualPandaScene",
        panda_paths={"left": "/World/LeftPanda", "right": "/World/RightPanda"},
        hand_paths={"left": "/World/LeftDH116", "right": "/World/RightDH116"},
        hand_usd={side: hand_root / name for side, name in _HAND_ASSET_NAMES.items()},
        flange_link="panda_hand",
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
    from pxr import UsdGeom

    hidden = 0
    for name in ("panda_leftfinger", "panda_rightfinger"):
        prim = stage.GetPrimAtPath(f"{panda_path}/{name}")
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden += 1
    return hidden


def _set_root_world_pose(stage, hand_path: str, pose) -> None:
    from pxr import Gf, UsdGeom

    position = pose[:3]
    quaternion = Gf.Quatd(*[float(value) for value in pose[3:7]])
    matrix = Gf.Matrix4d().SetRotate(Gf.Matrix3d(quaternion))
    matrix.SetTranslate(Gf.Vec3d(*[float(value) for value in position]))
    UsdGeom.Xformable(stage.GetPrimAtPath(hand_path)).MakeMatrixXform().Set(matrix)


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
    for side, y in (("left", 0.25), ("right", -0.25)):
        q = np.asarray(pose[side], dtype=np.float64)
        cfg = panda_cfg.copy()
        cfg.prim_path = spec.panda_paths[side]
        cfg.init_state.pos = (0.0, y, 0.0)
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

    for side, _y in (("left", 0.25), ("right", -0.25)):
        hand_path = spec.hand_paths[side]
        hand_prim = stage.DefinePrim(hand_path, "Xform")
        hand_prim.GetReferences().AddReference(str(spec.hand_usd[side]))
        stage.Load()
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
        _hide_builtin_panda_fingers(stage, spec.panda_paths[side])

    runtime = NativeSceneRuntime(spec=spec, stage=stage, arms=arms, hands=hands)
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
        hand_state = torch.tensor(expand(q[7:]), device=device, dtype=torch.float32)[None]
        arm.write_joint_state_to_sim(arm_state, torch.zeros_like(arm_state))
        hand.write_joint_state_to_sim(hand_state, torch.zeros_like(hand_state))
        arm.reset()
        hand.reset()
    sim.step(render=False)
    for side in SIDES:
        arm = runtime.arms[side]
        arm.update(1.0 / 240.0)
        try:
            flange_index = arm.body_names.index(spec.flange_link)
        except ValueError as exc:
            raise RuntimeError(f"native Panda flange body is missing: {spec.flange_link}") from exc
        flange_pose = arm.data.body_link_pose_w[0, flange_index].detach().cpu().numpy()
        _set_root_world_pose(runtime.stage, spec.hand_paths[side], flange_pose)
    sim.physics_sim_view.update_articulations_kinematic()
    return runtime


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
