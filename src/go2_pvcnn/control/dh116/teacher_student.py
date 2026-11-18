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
    contact_force_n: float = 0.0
    wire_lift_m: float = 0.0
    physics_valid: bool = False

    def __post_init__(self) -> None:
        for name in ("previous_action", "mpc_action", "action"):
            if len(getattr(self, name)) != 12:
                raise ValueError(f"{name} must contain 12 hand actuator values")
        if self.phase not in {"grasp", "lift"}:
            raise ValueError("phase must be grasp or lift")
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if not np.isfinite(float(self.contact_force_n)) or self.contact_force_n < 0:
            raise ValueError("contact_force_n must be finite and non-negative")
        if not np.isfinite(float(self.wire_lift_m)):
            raise ValueError("wire_lift_m must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_physics_outcome(*, contact_force_n: float, wire_lift_m: float,
                          max_contact_force_n: float = 30.0,
                          min_lift_m: float = 0.02) -> tuple[bool, float]:
    """Score measured rigid-proxy feedback; this is not a deformable-wire qualification.

    A lift is accepted only when measured contact is present, the measured
    displacement reaches ``min_lift_m``, and the peak force remains below the
    existing Isaac probe abort threshold.  Over-limit force is a negative
    example even if the proxy was displaced.
    """
    values = (contact_force_n, wire_lift_m, max_contact_force_n, min_lift_m)
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("physics outcome values must be finite")
    if contact_force_n < 0 or max_contact_force_n <= 0 or min_lift_m <= 0:
        raise ValueError("force and lift thresholds must be positive")
    if contact_force_n > max_contact_force_n:
        return False, -1.0
    if contact_force_n >= 1.0 and wire_lift_m >= min_lift_m:
        return True, 1.0
    return False, 0.0


def qualify_proxy_lift(*, contact_peak_n: float, wire_lift_m: float,
                       abort_reason: str | None, max_contact_force_n: float = 30.0,
                       min_lift_m: float = 0.02) -> bool:
    """Require an entire non-aborted, under-limit run to retain proxy lift."""
    values = np.asarray([contact_peak_n, wire_lift_m, max_contact_force_n, min_lift_m], dtype=np.float64)
    if not np.isfinite(values).all() or contact_peak_n < 0 or max_contact_force_n <= 0 or min_lift_m <= 0:
        raise ValueError("proxy lift qualification values must be finite and non-negative")
    return bool(abort_reason is None and contact_peak_n <= max_contact_force_n and wire_lift_m >= min_lift_m)


class LinearImitationPolicy:
    """Load and execute a hand-only linear or trajectory checkpoint.

    Linear checkpoints are least-squares fits, not neural networks or PPO
    policies. Trajectory checkpoints interpolate a preplanned MPC grasp path
    by grasp progress and hold its final grasp target throughout lift. Inference
    accepts one observation or a batch and clips targets to DH116 limits.
    """

    def __init__(self, *, role: str, input_dim: int, weights: Any = None,
                 progress_knots: Any = None, actions: Any = None,
                 previous_smoothing: float = 0.0) -> None:
        if role not in {"teacher", "student"}:
            raise ValueError("policy role must be teacher or student")
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.role = role
        self.input_dim = input_dim
        self.kind = "trajectory_imitation_checkpoint" if progress_knots is not None else "linear_imitation_checkpoint"
        self.weights = None
        self.progress_knots = None
        self.actions = None
        if self.kind == "linear_imitation_checkpoint":
            matrix = np.asarray(weights, dtype=np.float32)
            if matrix.shape != (input_dim + 1, 12) or not np.isfinite(matrix).all():
                raise ValueError("policy weights must be finite with shape (input_dim+1, 12)")
            self.weights = matrix
        else:
            knots = np.asarray(progress_knots, dtype=np.float32)
            path = np.asarray(actions, dtype=np.float32)
            if (input_dim != 25 or knots.ndim != 1 or len(knots) < 2
                    or path.shape != (len(knots), 12)
                    or not np.isfinite(knots).all() or not np.isfinite(path).all()
                    or np.any(np.diff(knots) <= 0) or knots[0] < 0 or knots[-1] > 1):
                raise ValueError("trajectory checkpoint needs sorted [0,1] progress knots and matching (N,12) actions")
            if (not np.isfinite(previous_smoothing)
                    or not 0.0 <= previous_smoothing <= 1.0):
                raise ValueError("previous_smoothing must be finite and in [0,1]")
            self.progress_knots = knots
            self.actions = path
            self.previous_smoothing = float(previous_smoothing)
        from .linear_coupling import active_limits
        lower, upper = active_limits()
        self.lower = np.tile(lower, 2).astype(np.float32)
        self.upper = np.tile(upper, 2).astype(np.float32)

    @classmethod
    def load(cls, path: str | Path, *, expected_role: str | None = None) -> "LinearImitationPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must contain a JSON object")
        digest = payload.get("sha256")
        unsigned = {key: value for key, value in payload.items() if key != "sha256"}
        actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        if not isinstance(digest, str) or digest != actual:
            raise ValueError("checkpoint SHA256 mismatch")
        role = payload.get("role")
        if expected_role is not None and role != expected_role:
            raise ValueError(f"expected {expected_role} checkpoint, got {role!r}")
        kind = payload.get("kind")
        if (kind not in {"linear_imitation_checkpoint", "trajectory_imitation_checkpoint"}
                or payload.get("output_dim") != 12
                or payload.get("arm_state_included") is not False):
            raise ValueError("checkpoint is not a hand-only 12D linear imitation policy")
        if kind == "trajectory_imitation_checkpoint":
            if role != "student":
                raise ValueError("trajectory checkpoints are student policies")
            return cls(role=role, input_dim=payload.get("input_dim"),
                       progress_knots=payload.get("progress_knots"),
                       actions=payload.get("actions"),
                       previous_smoothing=payload.get("previous_smoothing", 0.0))
        return cls(role=role, weights=payload.get("weights"), input_dim=payload.get("input_dim"))

    def predict(self, observation: Any) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if (values.ndim != 2 or values.shape[1] != self.input_dim
                or not np.isfinite(values).all()):
            raise ValueError(f"observation must be finite with shape ({self.input_dim},) or (batch,{self.input_dim})")
        if self.kind == "trajectory_imitation_checkpoint":
            progress = np.clip(values[:, -1], self.progress_knots[0], self.progress_knots[-1])
            target = np.stack([
                np.interp(progress, self.progress_knots, self.actions[:, joint])
                for joint in range(12)
            ], axis=1).astype(np.float32)
            previous = values[:, 12:24]
            action = ((1.0 - self.previous_smoothing) * target
                      + self.previous_smoothing * previous)
        else:
            augmented = np.concatenate((values, np.ones((len(values), 1), dtype=np.float32)), axis=1)
            action = augmented @ self.weights
        action = np.clip(action, self.lower[None, :], self.upper[None, :])
        return action[0] if single else action


def make_trajectory_checkpoint(*, progress_knots: Any, actions: Any,
                               previous_smoothing: float = 0.0) -> dict[str, Any]:
    """Build a hashed student checkpoint from an MPC grasp path.

    Progress is the hand's grasp phase. Once it reaches one, interpolation
    clamps to the last closure target so the hand keeps grasping while Panda
    executes the separate lift trajectory.
    """
    knots = np.asarray(progress_knots, dtype=np.float32)
    path = np.asarray(actions, dtype=np.float32)
    if (knots.ndim != 1 or len(knots) < 2 or path.shape != (len(knots), 12)
            or not np.isfinite(knots).all() or not np.isfinite(path).all()
            or np.any(np.diff(knots) <= 0) or knots[0] < 0 or knots[-1] > 1):
        raise ValueError("progress_knots and actions must be finite, sorted and shaped (N,) and (N,12)")
    if not np.isfinite(previous_smoothing) or not 0 <= previous_smoothing <= 1:
        raise ValueError("previous_smoothing must be finite and in [0,1]")
    payload: dict[str, Any] = {
        "role": "student", "kind": "trajectory_imitation_checkpoint",
        "input_dim": 25, "output_dim": 12,
        "progress_knots": knots.tolist(), "actions": path.tolist(),
        "previous_smoothing": float(previous_smoothing),
        "arm_state_included": False,
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload

def make_transition(*, obs_student: Any, obs_teacher_privileged: Any,
                    previous_action: Any, mpc_action: Any, action: Any,
                    reward: float, done: bool, phase: str,
                    contact_force_n: float = 0.0, wire_lift_m: float = 0.0,
                    physics_valid: bool = False) -> HandOnlyTransition:
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
        contact_force_n=float(contact_force_n), wire_lift_m=float(wire_lift_m),
        physics_valid=bool(physics_valid),
    )


def relabel_transitions(transitions: Iterable[HandOnlyTransition], policy: Any, *,
                        episode_size: int = 1) -> list[HandOnlyTransition]:
    """Create a student replay by executing a hand-only policy on each record.

    MPC actions stay in ``mpc_action`` as the expert reference; only the
    executed ``action`` field is replaced.  Physics observations and rewards
    are copied verbatim, so this helper cannot turn a synthetic rollout into a
    physical success claim.
    """
    if (getattr(policy, "role", None) != "student"
            or not callable(getattr(policy, "predict", None))):
        raise ValueError("a student policy with predict() is required")
    if isinstance(episode_size, bool) or not isinstance(episode_size, int) or episode_size < 1:
        raise ValueError("episode_size must be a positive integer")
    source = list(transitions)
    if len(source) % episode_size:
        raise ValueError("transitions must contain complete step-major episodes")
    previous_by_env: dict[int, np.ndarray] = {}
    output: list[HandOnlyTransition] = []
    for index, item in enumerate(source):
        if not isinstance(item, HandOnlyTransition):
            raise TypeError("transitions must contain HandOnlyTransition values")
        env = index % episode_size
        previous = (np.asarray(item.previous_action, dtype=np.float32)
                    if env not in previous_by_env else previous_by_env[env])
        observation = np.asarray(item.obs_student, dtype=np.float32).copy()
        # The synthetic hand-only observation layout is [active_joints(12),
        # previous_action(12), grasp_progress].  Roll both state slices with
        # the policy output after the first step; other feature layouts remain
        # untouched and are still validated by the policy.
        if env in previous_by_env and observation.shape == (25,):
            observation[:12] = previous
            observation[12:24] = previous
        action = np.asarray(policy.predict(observation), dtype=np.float32)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("student policy must return a finite 12D action")
        previous_by_env[env] = action.copy()
        output.append(make_transition(
            obs_student=observation,
            obs_teacher_privileged=item.obs_teacher_privileged,
            previous_action=previous,
            mpc_action=item.mpc_action,
            action=action,
            reward=item.reward,
            done=item.done,
            phase=item.phase,
            contact_force_n=item.contact_force_n,
            wire_lift_m=item.wire_lift_m,
            physics_valid=item.physics_valid,
        ))
    return output


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


__all__ = ["HandOnlyTransition", "LinearImitationPolicy", "make_transition", "relabel_transitions",
           "qualify_proxy_lift", "make_trajectory_checkpoint", "score_physics_outcome", "write_replay"]
