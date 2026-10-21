"""Teacher/student replay for the decoupled DH116 hand task.

The teacher receives privileged wire progress and an action proposed by the
batched joint MPC.  The student is trained from the same hand-only record;
Panda state is deliberately rejected at the schema boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> list[float]:
    arr = np.asarray(value, dtype=np.float32)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr.reshape(-1).tolist()


@dataclass(frozen=True)
class HandOnlyTransition:
    obs_student: list[float]
    obs_teacher_privileged: list[float]
    previous_action: list[float]
    mpc_action: list[float]
    action: list[float]
    reward: float
    done: bool
    phase: str

    def __post_init__(self) -> None:
        for name in ("previous_action", "mpc_action", "action"):
            if len(getattr(self, name)) != 12:
                raise ValueError(f"{name} must contain 12 hand actuator values")
        if self.phase not in {"grasp", "lift"}:
            raise ValueError("phase must be grasp or lift")
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_transition(*, obs_student: Any, obs_teacher_privileged: Any,
                    previous_action: Any, mpc_action: Any, action: Any,
                    reward: float, done: bool, phase: str) -> HandOnlyTransition:
    """Validate and construct one hand-only replay transition."""
    for forbidden in ("arm", "panda", "robot_arm", "left_arm", "right_arm"):
        # This catches accidental dictionary flattening by callers.
        for candidate in (obs_student, obs_teacher_privileged):
            if isinstance(candidate, dict) and forbidden in candidate:
                raise ValueError("Panda/arm state is forbidden in hand-only replay")
    return HandOnlyTransition(
        obs_student=_array(obs_student, "obs_student"),
        obs_teacher_privileged=_array(obs_teacher_privileged, "obs_teacher_privileged"),
        previous_action=_array(previous_action, "previous_action", (12,)),
        mpc_action=_array(mpc_action, "mpc_action", (12,)),
        action=_array(action, "action", (12,)), reward=float(reward), done=bool(done), phase=phase,
    )


def write_replay(path: str | Path, transitions: Iterable[HandOnlyTransition], *, role: str) -> dict[str, Any]:
    """Write JSONL replay and return a hash-bearing manifest fragment."""
    if role not in {"teacher", "student"}:
        raise ValueError("role must be teacher or student")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as stream:
        for item in transitions:
            if not isinstance(item, HandOnlyTransition):
                raise TypeError("replay entries must be HandOnlyTransition")
            stream.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
            count += 1
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"role": role, "path": str(target), "transitions": count, "sha256": digest,
            "action_dim": 12, "arm_state_included": False}


__all__ = ["HandOnlyTransition", "make_transition", "write_replay"]
