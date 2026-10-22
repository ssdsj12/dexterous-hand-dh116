"""Asset and pose contracts for the native Panda + separate DH116 scene."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SIDES = ("left", "right")


@dataclass(frozen=True)
class NativeSceneSpec:
    root_path: str
    panda_paths: dict[str, str]
    hand_paths: dict[str, str]
    hand_usd: dict[str, Path]
    flange_link: str


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
