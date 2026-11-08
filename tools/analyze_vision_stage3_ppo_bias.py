"""Analyze existing visual Student policy/physics diagnostics without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LEFT = np.array([0, 1, 2, 3, 4, 5, 12, 13, 14, 18, 19, 20])
RIGHT = np.array([6, 7, 8, 9, 10, 11, 15, 16, 17, 21, 22, 23])
RIGHT_PALM_Z = 17
STYLE_NAMES = {"pinch_thumb_index": "pinch", "tripod_thumb_index_middle": "tripod"}


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)) if flat.size else None,
        "std": float(np.std(flat)) if flat.size else None,
        "p05": float(np.quantile(flat, .05)) if flat.size else None,
        "p50": float(np.quantile(flat, .50)) if flat.size else None,
        "p95": float(np.quantile(flat, .95)) if flat.size else None,
        "min": float(np.min(flat)) if flat.size else None,
        "max": float(np.max(flat)) if flat.size else None,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return None
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _style(env_ids: np.ndarray) -> np.ndarray:
    return np.where(env_ids < 512, "pinch_thumb_index", "tripod_thumb_index_middle")


def analyze(run: Path) -> dict[str, Any]:
    policy_rows = [json.loads(line) for line in (run / "policy_ppo_diagnostics.jsonl").open()]
    physics_rows = [json.loads(line) for line in (run / "action_diagnostics.jsonl").open()]
    if len(policy_rows) != len(physics_rows):
        raise RuntimeError(f"row count mismatch: policy={len(policy_rows)} action={len(physics_rows)}")

    env_ids = np.asarray([r["env_id"] for r in policy_rows], dtype=np.int64)
    style = _style(env_ids)
    actions = np.asarray([r["action24"] for r in policy_rows], dtype=np.float64)
    log_std = np.asarray([r["log_std"] for r in policy_rows], dtype=np.float64)
    advantage = np.asarray([r["advantage"] for r in policy_rows], dtype=np.float64)
    returns = np.asarray([r["returns"] for r in policy_rows], dtype=np.float64)
    values = np.asarray([r["value"] for r in policy_rows], dtype=np.float64)
    rewards = np.asarray([r["reward"] for r in policy_rows], dtype=np.float64)
    done = np.asarray([r["done"] for r in policy_rows], dtype=bool)
    loads = np.asarray([r["contact_loads_n"] for r in physics_rows], dtype=np.float64) if "contact_loads_n" in physics_rows[0] else None
    valid = np.asarray([r["contact_valid"] for r in physics_rows], dtype=bool) if "contact_valid" in physics_rows[0] else None
    actual_pose = np.asarray([r["actual_palm_pose_m"] for r in physics_rows], dtype=np.float64)
    terms = sorted({name for row in policy_rows for name in row.get("reward_terms", {})})
    term_arrays = {name: np.asarray([row.get("reward_terms", {}).get(name, 0.0) for row in policy_rows], dtype=np.float64)
                   for name in terms}

    result: dict[str, Any] = {
        "run": str(run),
        "rows": len(policy_rows),
        "done_count": int(done.sum()),
        "style_split": {name: int((style == name).sum()) for name in STYLE_NAMES},
        "action_contract": "finger12|palm_translation6|rotation6",
        "right_palm_z_action_index": RIGHT_PALM_Z,
        "all_rows_same_policy_step": len({r["policy_step"] for r in policy_rows}) == 32,
        "by_style": {},
        "correlation_caveat": "Pearson correlations are descriptive associations in one final-checkpoint rollout; they do not establish causality.",
    }
    for style_name in STYLE_NAMES:
        mask = style == style_name
        style_result: dict[str, Any] = {
            "rows": int(mask.sum()),
            "action": {
                "left_12d_mean": actions[mask][:, LEFT].mean(0).tolist(),
                "left_12d_std": actions[mask][:, LEFT].std(0).tolist(),
                "right_12d_mean": actions[mask][:, RIGHT].mean(0).tolist(),
                "right_12d_std": actions[mask][:, RIGHT].std(0).tolist(),
                "right_palm_z": _summary(actions[mask, RIGHT_PALM_Z]),
            },
            "log_std": {
                "left_12d_mean": log_std[mask][:, LEFT].mean(0).tolist(),
                "right_12d_mean": log_std[mask][:, RIGHT].mean(0).tolist(),
            },
            "ppo": {
                "advantage": _summary(advantage[mask]),
                "returns": _summary(returns[mask]),
                "value": _summary(values[mask]),
                "reward": _summary(rewards[mask]),
                "done_count": int(done[mask].sum()),
            },
            "reward_terms": {name: _summary(term_arrays[name][mask]) for name in terms},
            "correlations_with_right_palm_z_action": {
                "advantage": _corr(actions[mask, RIGHT_PALM_Z], advantage[mask]),
                "returns": _corr(actions[mask, RIGHT_PALM_Z], returns[mask]),
                "value": _corr(actions[mask, RIGHT_PALM_Z], values[mask]),
                "reward": _corr(actions[mask, RIGHT_PALM_Z], rewards[mask]),
                **{f"reward_term:{name}": _corr(actions[mask, RIGHT_PALM_Z], values[mask])
                   for name, values in term_arrays.items()},
            },
        }
        if loads is not None:
            style_result["contact"] = {
                "left_load_n": _summary(loads[mask, 0]),
                "right_load_n": _summary(loads[mask, 1]),
                "right_palm_z_corr_with_right_load": _corr(actions[mask, RIGHT_PALM_Z], loads[mask, 1]),
                "right_palm_z_corr_with_right_contact_valid": _corr(actions[mask, RIGHT_PALM_Z], valid[mask, 1].astype(float)) if valid is not None else None,
                "right_contact_valid_rate": float(valid[mask, 1].mean()) if valid is not None else None,
            }
        style_result["actual_palm_displacement"] = {
            "left_z_m": _summary(actual_pose[mask, 0, 2]),
            "right_z_m": _summary(actual_pose[mask, 1, 2]),
            "right_action_z_corr_with_right_actual_z": _corr(actions[mask, RIGHT_PALM_Z], actual_pose[mask, 1, 2]),
        }
        result["by_style"][style_name] = style_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analyze(args.run), indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
