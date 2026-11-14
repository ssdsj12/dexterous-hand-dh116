"""Opt-in diagnostic R0 domain; legacy task/control/reward files remain unchanged.

No physics calls, new weights or success qualification. Missing raw measurements
remain missing. A proven pre-grasp domain has no fixed two-hand slip anchor;
its *contribution*, not its measurement, is explicitly inapplicable. Any observed
bilateral contact permanently ends that domain until an actual episode reset.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .tabletop_deformable_grasp_task import (
    FlexibleGraspEpisode, R0_TERMS, _exact_keys, _sample, _finite_number,
)
from .tabletop_deformable_control_step import FlexibleR0ControlStep, DECIMATION

VERSION = "dh116-flexible-r0-applicability-diagnostic-v2"
CONFIG = Path(__file__).resolve().parents[4] / "configs/dh116_tasks/flexible_r0_applicability_diagnostic_v2.json"
_EXPECTED = {"version":"dh116-flexible-r0-applicability-diagnostic-v2","scope":"offline_diagnostic_only","enabled_by_default":False,"native_qualification":False,"training_reward_enabled":False,"coefficient_source":"explicit_diagnostic_only","reset_approach":"same_reset_endpoint_measured_distance_or_unknown","slip_applicability":"known_pre_bilateral_contact_only_then_episode_latched","missing_raw_policy":"keep_null_invalid_even_when_contribution_inapplicable","success":{"net_lift_m":0.03,"hold_s":5,"max_slip_m":0.0005}}


def _contract():
    raw = CONFIG.read_bytes()
    observed = json.loads(raw)
    if observed != _EXPECTED:
        raise ValueError("independent diagnostic R0 applicability contract differs")
    return dict(reward_contract_version=VERSION,
                reward_config_sha256=hashlib.sha256(raw).hexdigest(),
                reward_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class ApplicableR0Episode(FlexibleGraspEpisode):
    """Original task semantics with explicit, auditable reward applicability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = _contract()
        if (self.lift_target_m, self.hold_target_s, self.slip_limit_m) != (.03, 5., .0005):
            raise ValueError("original success standards required")
        self.bilateral_contact_seen = False
        self.contact_history_unknown = False

    def reset(self, reset_hooks, read_after_reset, *, reset_approach=None):
        super().reset(reset_hooks, read_after_reset)
        try:
            self.bilateral_contact_seen = False
            self.contact_history_unknown = False
            if reset_approach is not None:
                _exact_keys(reset_approach, ("sample_identity", "approach_distance_m"), "reset approach")
                self._measurement_identity(reset_approach, self.clock)
                value, source = _sample(reset_approach, "approach_distance_m")
                if value is not None:
                    if not isinstance(value, (tuple, list)) or len(value) != 2:
                        raise ValueError("two measured reset approach distances required")
                    self.approach = (math.fsum(_finite_number(x, "reset approach") for x in value)/2., source)
        except Exception:
            self.active = False
            raise

    def step(self, snapshot, measurements):
        try:
            row = super().step(snapshot, measurements)
            row["diagnostic_reward_without_applicability"] = row["diagnostic_reward"]
            coverage = row["terms"]["required_contact"]["raw"]
            geometry = row["terms"]["grasp_geometry"]["raw"]
            known = coverage is not None and geometry is not None
            bilateral = known and coverage == 1. and geometry == 1.
            self.bilateral_contact_seen |= bilateral
            self.contact_history_unknown |= not known
            for name, term in row["terms"].items():
                term.update(applicable=True, applicability_source="original_term_domain",
                            applicability_reason="measurement_required")
                if name == "material_anchor_slip" and term["raw"] is None:
                    if not self.bilateral_contact_seen and not self.contact_history_unknown and known and not bilateral:
                        term.update(applicable=False,
                            applicability_source=row["terms"]["required_contact"]["source"],
                            applicability_reason="known_pre_bilateral_contact_no_fixed_anchor",
                            contribution=None if term["coefficient"] is None else 0.)
                    elif not self.bilateral_contact_seen and self.contact_history_unknown:
                        term.update(applicable=None, applicability_source=None,
                                    applicability_reason="contact_history_unknown_until_reset")
                    else:
                        term.update(applicable=True,
                                    applicability_reason="bilateral_contact_latched_measurement_required")
                term["contribution_valid"] = term["contribution"] is not None
            row["blocked_terms"] = [k for k,v in row["terms"].items() if not v["contribution_valid"]]
            row["diagnostic_reward"] = None if row["blocked_terms"] else math.fsum(
                v["contribution"] for v in row["terms"].values())
            if row["diagnostic_reward"] is not None and not math.isfinite(row["diagnostic_reward"]):
                raise ValueError("nonfinite diagnostic reward")
            row.update(self.contract, version=VERSION, bilateral_contact_seen=self.bilateral_contact_seen,
                       contact_history_unknown=self.contact_history_unknown)
            # training_reward=None and native_qualification=False are inherited.
            return row
        except Exception:
            self.active = False
            raise


class ApplicableR0ControlStep(FlexibleR0ControlStep):
    """Reuse original clock/action/budget/terminal guards; aggregate NA explicitly.

    Only reset baseline and settlement differ. Raw sums never count
    inapplicable samples as measured zero. This class cannot enable native PPO.
    """

    def __init__(self, task, grasp, *, env_id, budget_config, diagnostic_coefficients):
        super().__init__(task, grasp, env_id=env_id, budget_config=budget_config,
                         diagnostic_coefficients=diagnostic_coefficients)
        self.contract = _contract()
        self._implementation[Path(__file__).name] = self.contract["reward_source_sha256"]
        self._recipe_sha256 = hashlib.sha256(json.dumps(
            dict(legacy_recipe_sha256=self._recipe_sha256, **self.contract),
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self._episode = ApplicableR0Episode(deepcopy(task), grasp, env_id=env_id,
                                            diagnostic_coefficients=deepcopy(diagnostic_coefficients))

    def reset(self, reset_hooks, read_after_reset, *, reset_approach=None):
        if self._reset_attempted:
            raise RuntimeError("one reset per reducer budget; use a fresh independently budgeted reducer")
        self._reset_attempted = True
        self._episode.reset(reset_hooks, read_after_reset, reset_approach=reset_approach)
        if self._episode.clock[0] != self._plan["setup_physics_steps_per_env"]:
            self._episode.active = False
            raise ValueError("reset physics clock does not account for the declared setup steps")
        self._initialized = True

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
                defined = [e for e in entries if e["contribution"] is not None]
                measured = [e for e in entries if e["valid"]]
                if len(defined) != count:
                    blocked.append(name)
                terms[name] = dict(
                    unit=entries[0]["unit"], coefficient=entries[0]["coefficient"],
                    valid_samples=len(measured), expected_samples=count,
                    defined_contribution_samples=len(defined),
                    not_applicable_samples=sum(e["applicable"] is False for e in entries),
                    applicability_unknown_samples=sum(e["applicable"] is None for e in entries),
                    raw_sum=math.fsum(e["raw"] for e in entries) if len(measured)==count else None,
                    contribution_sum=math.fsum(e["contribution"] for e in entries) if len(defined)==count else None,
                    sources=sorted({e["source"] for e in entries if e["source"] is not None}),
                    applicability_sources=sorted({e["applicability_source"] for e in entries
                                                  if e["applicability_source"] is not None}))
            reward = None if blocked else math.fsum(t["contribution_sum"] for t in terms.values())
            if reward is not None and not math.isfinite(reward):
                raise ValueError("nonfinite control reward sum")
        except (ValueError, OverflowError):
            self._invalid = True
            raise
        terminated = bool(self._reasons)
        exhausted = self._control+1 == self._plan["control_steps_per_env"]
        receipt = dict(
            version=VERSION, scope="offline_diagnostic_only", recipe_sha256=self.recipe_sha256,
            implementation_sha256=self.implementation_sha256, **self.contract,
            env_id=last["env_id"], episode_uid=last["episode_uid"], reset_id=last["reset_id"],
            rod_uid=last["rod_uid"], grasp=last["grasp"], control_index=self._control,
            action24=self._action, action_status="declared_not_issued_by_this_reducer",
            physics_step_start=self._start_clock[0], physics_step_end=last["physics_step"],
            rod_tick_start=self._start_clock[1], rod_tick_end=last["rod_tick"],
            timestamp_start_s=self._start_clock[2], timestamp_end_s=last["timestamp_s"],
            physical_samples=count, expected_full_interval_samples=DECIMATION,
            partial_interval_due_to_terminal=terminated and count<DECIMATION,
            terms=terms, blocked_terms=blocked, diagnostic_reward=reward, training_reward=None,
            terminated=terminated, truncated=exhausted and not terminated,
            terminal_reasons=list(self._reasons), success_candidate=last["success_candidate"],
            native_qualification=False, new_physics_steps=0, optimizer_updates=0,
            consumed_physics_samples=self._consumed,
            accounted_setup_physics_steps=self._plan["setup_physics_steps_per_env"],
            total_accounted_physics_steps=self._consumed+self._plan["setup_physics_steps_per_env"])
        self._rows = None
        self._ready = False
        self._control += 1
        self._closed = terminated or exhausted
        return receipt
