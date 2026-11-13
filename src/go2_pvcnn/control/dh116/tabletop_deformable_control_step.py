"""CPU-only R0 control-step settlement using the unchanged physical task.

This reducer consumes supplied observations. It neither advances physics nor
issues actions. Diagnostic coefficients are explicit fixture inputs; the
frozen task's unknown coefficients and all native/PPO gates stay unchanged.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from . import tabletop_deformable_grasp_task, tabletop_deformable_task_budget

import numpy as np

from .tabletop_deformable_grasp_task import FlexibleGraspEpisode, R0_TERMS
from .tabletop_deformable_task_budget import SETUP_AWARE_VERSION, plan_flexible_experiment

VERSION = "dh116-flexible-r0-control-settlement-cpu-v1"
DECIMATION = 8


def _implementation_identity():
    paths = (Path(__file__), Path(tabletop_deformable_grasp_task.__file__),
             Path(tabletop_deformable_task_budget.__file__))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class FlexibleR0ControlStep:
    """Own one reset and settle each complete 480/60 Hz interval exactly once.

    An original success candidate or a known joint/instability/prohibited-support
    fault closes at its first measured sample; missing sources never mean zero.
    A rejected physical packet invalidates this reducer. Reuse requires a new
    independently budgeted instance, not replaying a partially consumed window.
    """

    def __init__(self, task, grasp, *, env_id, budget_config, diagnostic_coefficients):
        if (not isinstance(diagnostic_coefficients, dict)
                or set(diagnostic_coefficients) != set(R0_TERMS)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in diagnostic_coefficients.values())):
            raise ValueError("complete finite nonnegative CPU diagnostic coefficients required")
        plan = plan_flexible_experiment(deepcopy(budget_config))
        if (plan["budget_contract_version"] != SETUP_AWARE_VERSION
                or plan["phase"] != "frozen_eval" or plan["updates"] != 0):
            raise ValueError("only an independent frozen-eval CPU accounting budget is accepted")
        if plan["setup_physics_steps_per_env"] is None:
            raise ValueError("explicit setup budget required")
        if (type(env_id) is not int or not 0 <= env_id < plan["environments"]
                or plan["measured_solver_substeps_per_env"] is None):
            raise ValueError("one declared environment and known solver substeps required")
        self._plan = plan
        self._solver_stride = (plan["measured_solver_substeps_per_env"]
                               // plan["measured_physics_steps_per_env"])
        self._implementation = deepcopy(_implementation_identity())
        recipe = dict(implementation_sha256=self._implementation, version=VERSION, scope="cpu_diagnostic_only", task=deepcopy(task),
                      grasp=grasp, budget=deepcopy(budget_config),
                      coefficients=deepcopy(diagnostic_coefficients),
                      aggregation="sum_of_existing_physical_contributions_no_unknown_imputation",
                      early_terminal=("success_candidate", "joint_limit", "instability",
                                      "prohibited_external_support"))
        self._recipe_sha256 = hashlib.sha256(json.dumps(
            recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self._episode = FlexibleGraspEpisode(deepcopy(task), grasp, env_id=env_id,
                                            diagnostic_coefficients=deepcopy(diagnostic_coefficients))
        self._reset_attempted = self._initialized = self._invalid = self._closed = False
        self._control = 0
        self._consumed = 0
        self._rows = None
        self._ready = False
        self._reasons = []

    @property
    def recipe_sha256(self):
        return self._recipe_sha256

    @property
    def implementation_sha256(self):
        return deepcopy(self._implementation)

    def reset(self, reset_hooks, read_after_reset):
        if self._reset_attempted:
            raise RuntimeError("one reset per reducer budget; use a fresh independently budgeted reducer")
        self._reset_attempted = True
        self._episode.reset(reset_hooks, read_after_reset)
        if self._episode.clock[0] != self._plan["setup_physics_steps_per_env"]:
            raise ValueError("reset physics clock does not account for the declared setup steps")
        self._initialized = True

    def _state(self):
        if self._invalid:
            raise RuntimeError("invalid physical window; reducer cannot resume or settle")
        if not self._initialized:
            raise RuntimeError("reset required")
        if self._closed:
            raise RuntimeError("closed episode or exhausted control budget")

    def _binding(self, control_index, recipe_sha256):
        if recipe_sha256 != self.recipe_sha256:
            raise ValueError("recipe identity differs")
        if type(control_index) is not int or control_index != self._control:
            raise ValueError("control index repeated, skipped, or foreign")

    def begin(self, *, control_index, recipe_sha256, action):
        self._state()
        if self._rows is not None:
            raise RuntimeError("settle the existing control interval before beginning another")
        self._binding(control_index, recipe_sha256)
        if self._control >= self._plan["control_steps_per_env"]:
            raise RuntimeError("closed control budget")
        action = np.asarray(action, dtype=float)
        if action.shape != (24,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("finite normalized action24 required")
        self._action = action.copy().tolist()
        self._start_clock = tuple(self._episode.clock)
        self._rows = []
        self._ready = False
        self._reasons = []

    def push(self, snapshot, measurements, *, control_index, recipe_sha256):
        self._state()
        self._binding(control_index, recipe_sha256)
        if self._rows is None:
            raise RuntimeError("begin a control interval before supplying physics")
        if self._ready:
            raise RuntimeError("settle the completed interval before consuming more physics")
        if self._consumed >= self._plan["measured_physics_steps_per_env"]:
            raise RuntimeError("closed physical sample budget")
        try:
            if snapshot.get("rod_tick", -1) - self._episode.clock[1] != self._solver_stride:
                raise ValueError("rod stride differs from the recipe's solver substep budget")
            row = self._episode.step(snapshot, measurements)
            for term in row["terms"].values():
                for field in ("raw", "contribution"):
                    if term[field] is not None and not math.isfinite(term[field]):
                        raise ValueError("nonfinite original physical reward term")
        except Exception:
            self._invalid = True
            raise
        self._rows.append(deepcopy(row))
        self._consumed += 1
        if row["terms"]["joint_limit"]["raw"] == 1:
            self._reasons.append("joint_limit")
        if row["terms"]["instability"]["raw"] == 1:
            self._reasons.append("instability")
        if row["prohibited_support_latched"]:
            self._reasons.append("prohibited_external_support")
        if row["success_candidate"]:
            self._reasons.append("success_candidate")
        self._ready = bool(self._reasons) or len(self._rows) == DECIMATION
        return self._ready

    def settle(self):
        if self._invalid:
            raise RuntimeError("invalid physical window cannot settle")
        if self._rows is None:
            raise RuntimeError("no control interval to settle, or already settled")
        if not self._ready:
            raise RuntimeError("incomplete physical interval cannot settle")
        rows, last = self._rows, self._rows[-1]
        count = len(rows)
        terms, blocked = {}, []
        try:
            for name in R0_TERMS:
                entries = [row["terms"][name] for row in rows]
                known = [entry for entry in entries if entry["contribution"] is not None]
                if len(known) != count:
                    blocked.append(name)
                terms[name] = dict(
                    unit=entries[0]["unit"], coefficient=entries[0]["coefficient"],
                    valid_samples=len(known), expected_samples=count,
                    raw_sum=math.fsum(e["raw"] for e in entries) if len(known) == count else None,
                    contribution_sum=math.fsum(e["contribution"] for e in entries) if len(known) == count else None,
                    sources=sorted({e["source"] for e in entries if e["source"] is not None}))
            reward = None if blocked else math.fsum(t["contribution_sum"] for t in terms.values())
            if reward is not None and not math.isfinite(reward):
                raise ValueError("nonfinite control reward sum")
        except (ValueError, OverflowError):
            self._invalid = True
            raise
        terminated = bool(self._reasons)
        exhausted = self._control + 1 == self._plan["control_steps_per_env"]
        receipt = dict(
            version=VERSION, scope="cpu_diagnostic_only", recipe_sha256=self.recipe_sha256,
            implementation_sha256=self.implementation_sha256,
            env_id=last["env_id"], episode_uid=last["episode_uid"], reset_id=last["reset_id"],
            rod_uid=last["rod_uid"], grasp=last["grasp"], control_index=self._control,
            action24=self._action, action_status="declared_not_issued_by_this_reducer",
            physics_step_start=self._start_clock[0], physics_step_end=last["physics_step"],
            rod_tick_start=self._start_clock[1], rod_tick_end=last["rod_tick"],
            timestamp_start_s=self._start_clock[2], timestamp_end_s=last["timestamp_s"],
            physical_samples=count, expected_full_interval_samples=DECIMATION,
            partial_interval_due_to_terminal=terminated and count < DECIMATION,
            terms=terms, blocked_terms=blocked, diagnostic_reward=reward, training_reward=None,
            terminated=terminated, truncated=exhausted and not terminated,
            terminal_reasons=list(self._reasons), success_candidate=last["success_candidate"],
            native_qualification=False, new_physics_steps=0, optimizer_updates=0,
            consumed_physics_samples=self._consumed,
            accounted_setup_physics_steps=self._plan["setup_physics_steps_per_env"],
            total_accounted_physics_steps=self._consumed + self._plan["setup_physics_steps_per_env"])
        self._rows = None
        self._ready = False
        self._control += 1
        self._closed = terminated or exhausted
        return receipt
