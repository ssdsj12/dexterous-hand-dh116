"""Optional per-policy-step PPO diagnostics for the visual Student.

The recorder is intentionally outside the policy and rollout contracts. It
copies already-produced tensors, writes JSONL rows, and keeps an online summary
so enabling the flag does not change action sampling, rewards, or optimizer
updates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _vector(value: Any, n: int, name: str, *, dtype=np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape ({n},)")
    return result


def _matrix(value: Any, n: int, width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n, width) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape ({n}, {width})")
    return result


class VisionPolicyDiagnosticRecorder:
    """Write one row per environment and policy step, with online summaries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        self.rows = 0
        self._action_sum = np.zeros((24,), dtype=np.float64)
        self._action_sq_sum = np.zeros((24,), dtype=np.float64)
        self._log_std_sum = np.zeros((24,), dtype=np.float64)
        self._reward_sum = 0.0
        self._value_sum = 0.0
        self._advantage_sum = 0.0

    def record(
        self,
        *,
        env_id: Any,
        episode_id: Any,
        iteration: int,
        policy_step: int,
        timestamp_s: Any,
        action24: Any,
        latent_actions: Any,
        latent_mean: Any,
        log_std: Any,
        log_prob: Any,
        value: Any,
        advantage: Any,
        returns: Any,
        reward: Any,
        done: Any,
        reward_terms: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        action = np.asarray(action24, dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 24 or not np.isfinite(action).all():
            raise ValueError("action24 must be finite with shape (N, 24)")
        n = action.shape[0]
        ids = np.asarray(env_id).reshape(-1)
        episodes = np.asarray(episode_id).reshape(-1)
        if ids.shape != (n,) or episodes.shape != (n,):
            raise ValueError("env_id and episode_id must have shape (N,)")
        if type(iteration) is not int or type(policy_step) is not int:
            raise ValueError("iteration and policy_step must be integers")
        times = _vector(timestamp_s, n, "timestamp_s")
        std = _matrix(log_std, n, 24, "log_std")
        logp = _vector(log_prob, n, "log_prob")
        values = _vector(value, n, "value")
        advantages = _vector(advantage, n, "advantage")
        returns_v = _vector(returns, n, "returns")
        rewards = _vector(reward, n, "reward")
        done_v = np.asarray(done, dtype=bool).reshape(-1)
        if done_v.shape != (n,):
            raise ValueError(f"done must have shape ({n},)")
        terms: dict[str, np.ndarray] = {}
        for name, term in reward_terms.items():
            terms[str(name)] = _vector(term, n, f"reward_terms[{name!r}]")
        latent = None
        if latent_actions is not None:
            latent = _matrix(latent_actions, n, 24, "latent_actions")
        mean = None if latent_mean is None else _matrix(latent_mean, n, 24, "latent_mean")

        rows = []
        for i in range(n):
            row = {
                "env_id": int(ids[i]),
                "episode_id": int(episodes[i]),
                "iteration": iteration,
                "policy_step": policy_step,
                "timestamp_s": float(times[i]),
                "action24": action[i].tolist(),
                "latent_actions": None if latent is None else latent[i].tolist(),
                "latent_mean": None if mean is None else mean[i].tolist(),
                "log_std": std[i].tolist(),
                "log_prob": float(logp[i]),
                "value": float(values[i]),
                "advantage": float(advantages[i]),
                "returns": float(returns_v[i]),
                "reward": float(rewards[i]),
                "done": bool(done_v[i]),
                "reward_terms": {name: float(values_[i]) for name, values_ in terms.items()},
            }
            self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows.append(row)
        self.stream.flush()
        self.rows += n
        self._action_sum += action.sum(0)
        self._action_sq_sum += np.square(action).sum(0)
        self._log_std_sum += std.sum(0)
        self._reward_sum += float(rewards.sum())
        self._value_sum += float(values.sum())
        self._advantage_sum += float(advantages.sum())
        return rows

    def summary(self) -> dict[str, Any]:
        if self.rows == 0:
            return {"rows": 0, "by_hand": {}}
        action_mean = self._action_sum / self.rows
        action_var = np.maximum(self._action_sq_sum / self.rows - np.square(action_mean), 0.)
        action_std = np.sqrt(action_var)
        log_std_mean = self._log_std_sum / self.rows
        return {
            "rows": self.rows,
            "mean_reward": self._reward_sum / self.rows,
            "mean_value": self._value_sum / self.rows,
            "mean_advantage": self._advantage_sum / self.rows,
            "log_std_mean": log_std_mean.tolist(),
            "by_hand": {
                "left": {
                    "action_mean": action_mean[:12].tolist(),
                    "action_std": action_std[:12].tolist(),
                },
                "right": {
                    "action_mean": action_mean[12:24].tolist(),
                    "action_std": action_std[12:24].tolist(),
                },
            },
        }

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()
