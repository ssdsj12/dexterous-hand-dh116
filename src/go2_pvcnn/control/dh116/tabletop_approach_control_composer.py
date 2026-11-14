"""Explicit arithmetic-only approach/control candidate, never a training reward.

Existing raw R0 measurements and original control-cost formulas remain their
owners. This composer pays no contact/hold/support term and creates no sources.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from . import tabletop_readonly_control_cost as costs
from .tabletop_deformable_r0_replay import decode_owned, _digest

VERSION = "dh116-approach-control-composer-candidate-v1"
ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
PROOF = EVIDENCE/"r0-three-source-replay-20261114-sE6xRE/result-final.json"
PROOF_SHA = "70cb68f5382fdb015595e30d3a58191034ee751dee883c2900bdad0fd6ce0b31"
ARCHIVE = EVIDENCE/"approved-approach-r0-20261114-jfd3i7wj/native-runs/flex.f7-approach-r0-native/attempt-wy27jh8g/native"
ARCHIVE_SHA = {
    "physical.jsonl": "4e9b42b1fc96119888cce34d4977d2fd8720b4385409dda45a10f59fcbb8b83b",
    "live_r0.jsonl": "d2615f42439583fc90c10f9d937aeba652209828b77c94d14fd2510c511034a7",
    "director_reset.json": "c524f8c8a142da5e3909464146d991230ead8474aad428bb67f12a94552631ab",
}
IDENTITY = ("env_id", "episode_uid", "reset_id", "rod_uid")
COMPONENTS = ("approach_progress", "time", "action_rate", "velocity_excess", "joint_limit_terminal")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "finite explicit number required")
    return float(value)


def candidate_spec(approach_points_per_m):
    """The sole new coefficient has no default and is not a rigid inheritance."""
    coefficient = _number(approach_points_per_m)
    _require(coefficient > 0., "positive explicitly preregistered candidate coefficient required")
    return dict(version=VERSION, scope="arithmetic_only_not_training_profile",
        approach_points_per_m=coefficient, approach_coefficient_unit="points/m",
        approach_raw_unit="m", approach_time_rule="signed_difference_no_dt",
        cost_contract_sha256=costs.CONTRACT_SHA256,
        joint_limit_points=-25., joint_time_rule="single_terminal_event_no_dt",
        joint_source="original_TensorTask_failure_event_not_per_substep_penalty",
        training_reward=None, native_qualification=False, teacher_admission=False)


def _equal(a, b):
    return _digest(a) == _digest(b)


def _sum(values):
    return None if any(v is None for v in values) else math.fsum(values)


class ApproachControlComposer:
    """One ordered reset epoch, consumed exactly once; failure poisons the instance.

    Pure arithmetic inputs require matching original physical events, live R0
    rows and cost rows. The archive adapter additionally pins their bytes and
    the previously completed full R0 audit. No native owner is accepted here.
    """
    def __init__(self, spec):
        _require(isinstance(spec, dict), "explicit candidate specification required")
        _require(spec == candidate_spec(spec.get("approach_points_per_m")),
                 "exact candidate units/time ownership required")
        self.spec = deepcopy(spec)
        self._spec_sha256 = _digest(spec)
        self._next = 0
        self._epoch = self._clock = self._action = None
        self._closed = self._invalid = False

    def compose(self, window):
        if self._closed or self._invalid:
            raise ValueError("composer closed or invalid")
        try:
            return self._compose(window)
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
            self._invalid = True
            raise ValueError("invalid candidate window: "+str(error)) from error

    def _compose(self, window):
        _require(_digest(self.spec) == self._spec_sha256, "candidate specification changed after construction")
        _require(set(window) == {"r0_end", "live_rows", "physical_rows", "physical_result", "cost_control"},
                 "exact independent window owners required")
        end, c = window["r0_end"], window["cost_control"]
        r = end["control_receipt"]
        result = window["physical_result"]
        n = r["physical_samples"]
        _require(type(n) is int and 1 <= n <= 8 and r["expected_full_interval_samples"] == 8,
                 "one bounded original control window required")
        _require(r["control_index"] == c["control_index"] == result["control_step"] == self._next,
                 "control index reordered, skipped or already consumed")
        _require(all(len(v) == n for v in (window["live_rows"], window["physical_rows"],
                     end["physical_rows"], c["rows"]))
                 and c["physical_samples"] == result["physical_samples"] == n,
                 "every original physical row must appear once")
        _require(_equal(result, end["control_result"]), "R0 and actual terminal result differ")
        reason = result["termination_reason"]
        _require(c["terminal_reason"] == reason and r["terminated"] is (reason is not None)
                 and c["complete_eight_sample_window"] is (n == 8)
                 and r["partial_interval_due_to_terminal"] is (reason is not None and n < 8)
                 and (n == 8 or reason is not None), "terminal/partial interval differs")
        _require(all(x["training_reward"] is None for x in (r, end, c, result)),
                 "only independent untrained arithmetic inputs required")
        epoch = {k:r[k] for k in IDENTITY}
        _require(r["grasp"] == "pinch" and (self._epoch is None or epoch == self._epoch),
                 "one independently evidenced Pinch reset epoch required")
        start = (r["physics_step_start"], r["rod_tick_start"], r["timestamp_start_s"])
        _require(self._clock is None or start == self._clock, "adjacent control clocks differ")
        action = costs._array(c["action"], (1, 24), "actual normalized action")
        _require(np.all(abs(action) <= 1.) and np.array_equal(action[0], r["action24"]),
                 "R0 declaration and actual cost action differ")
        previous = c["previous_action"]
        _require((self._action is None and previous is None) or (
            self._action is not None and previous is not None and np.array_equal(previous, self._action)),
            "previous action must be unknown initially, then prior same-epoch action")
        rows = []
        raw = {"approach_progress": [], "joint_limit": []}
        for i, (live, physical, stored, cost) in enumerate(zip(
                window["live_rows"], window["physical_rows"], end["physical_rows"], c["rows"])):
            _require(live["actual_event_sha256"] == _digest(physical)
                     and _equal(live["r0_row"], stored), "actual physical/R0 row identity differs")
            x = physical["exchange"]
            identity, = x["action_readback"]["post"]["identity"]
            _require(_equal(identity, live["identity"]) and identity["clock_valid"] is True,
                     "same actual completed clock required")
            rr = stored["r0"]
            _require(all(rr[k] == identity[k] == epoch[k] for k in IDENTITY)
                     and rr["grasp"] == "pinch", "row belongs to another episode/reset/grasp")
            step = start[0]+i+1
            tick = start[1]+12*(i+1)
            _require(rr["physics_step"] == identity["physics_step"] == cost["physics_step"] == x["physics_step"] == step
                     and rr["rod_tick"] == identity["rod_tick"] == cost["rod_tick"] == tick
                     and identity["policy_step"] == cost["policy_step"] == x["policy_step"] == self._next
                     and identity["substep"] == cost["substep"] == i
                     and rr["timestamp_s"] == identity["timestamp_s"]
                     and math.isclose(identity["timestamp_s"]-start[2], (i+1)/480, rel_tol=0, abs_tol=2e-7),
                     "row order or actual physical clock differs")
            guard = reason if i == n-1 else None
            _require(x["guard_failure"] == stored["physical_guard_failure"] == cost["guard_failure"] == guard,
                     "terminal must belong to the final actual row")
            _require(rr["training_reward"] is None and cost["training_reward"] is None,
                     "historical diagnostic inputs cannot contain a training reward")
            for name, unit in (("approach_progress", "m"), ("joint_limit", "event")):
                term = rr["terms"][name]
                _require(term["unit"] == unit, "raw measurement unit changed")
                value = term["raw"]
                _require(term["valid"] is (value is not None)
                         and ((value is None and term["source"] is None) or (
                             value is not None and isinstance(term["source"], str) and bool(term["source"]))),
                         "explicit raw validity and measured source required")
                raw[name].append(None if value is None else _number(value))
            joint = raw["joint_limit"][-1]
            expected_joint = guard in ("left:joint_limit", "right:joint_limit")
            _require(joint is None or joint == float(expected_joint),
                     "joint event differs from the single actual terminal")
            hands = x["action_readback"]["post"]["hands"]
            qd = np.concatenate([costs._array(hands[s]["qd"], (1,17), "actual full qd")
                                 for s in ("left", "right")], axis=0)
            expected_cost = costs.formula_components(action, previous, qd)
            _require(set(cost["components"]) == set(costs.TERMS)
                     and cost["actual_max_abs_joint_velocity_numeric"] == float(abs(qd).max())
                     and cost["components"] == expected_cost,
                     "cost formula changed or duplicate reward owner added")
            contributions = dict(approach_progress=None if raw["approach_progress"][-1] is None
                else self.spec["approach_points_per_m"]*raw["approach_progress"][-1],
                **expected_cost, joint_limit_terminal=None if joint is None else -25.*joint)
            _require(all(v is None or math.isfinite(v) for v in contributions.values()),
                     "candidate contribution overflow")
            rows.append(dict(identity=deepcopy(identity), actual_event_sha256=live["actual_event_sha256"],
                raw_approach_m=raw["approach_progress"][-1], raw_joint_limit=joint,
                components=contributions, training_reward=None))
        endpoint = (r["physics_step_end"], r["rod_tick_end"], r["timestamp_end_s"])
        _require(endpoint == (step, tick, identity["timestamp_s"]), "receipt endpoint differs")
        for name, unit in (("approach_progress","m"), ("joint_limit","event")):
            term = r["terms"][name]
            _require(term["unit"] == unit and term["expected_samples"] == n
                     and term["valid_samples"] == sum(v is not None for v in raw[name])
                     and term["raw_sum"] == _sum(raw[name]), "raw row/control sums differ")
        totals = {k:_sum([row["components"][k] for row in rows]) for k in COMPONENTS}
        missing = [k for k,v in totals.items() if v is None]
        if reason is not None and reason not in ("left:joint_limit", "right:joint_limit"):
            missing.append("other_terminal_reward_semantics_unregistered")
        reward = None if missing else math.fsum(totals.values())
        _require(reward is None or math.isfinite(reward), "candidate total overflow")
        self._epoch, self._clock, self._action = epoch, endpoint, action.copy()
        self._next += 1
        self._closed = reason is not None
        return dict(version=VERSION, control_index=r["control_index"], identity=epoch,
            physical_samples=n, terminal_reason=reason, rows=rows, components=totals,
            candidate_reward=reward, missing_terms=missing, spec=deepcopy(self.spec),
            spec_sha256=_digest(self.spec), training_reward=None, full_R0_reward=False,
            native_qualification=False, teacher_admission=False)


def _pinned(path, sha):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), "regular pinned evidence required")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == sha, "evidence SHA differs: "+str(path))
    return raw


def evaluate_archive(*, approach_points_per_m, project=ROOT):
    """Replay the exact already-audited historical Pinch inputs, no geometry rerun."""
    spec = candidate_spec(approach_points_per_m)
    original = {name:_pinned(ARCHIVE/name, sha) for name,sha in ARCHIVE_SHA.items()}
    proof = costs._json(_pinned(PROOF, PROOF_SHA))
    audit = proof["full_existing_audit"]
    _require(audit["all_live_rows_recomputed"] is True
             and audit["artifact_sha256"] == ARCHIVE_SHA
             and audit["physical_rows"] == 232 and audit["control_windows"] == 29
             and proof["originals_unchanged"] is True and proof["sources_unchanged"] is True,
             "completed existing full R0 audit required")
    evaluated = costs.evaluate_bytes(original["physical.jsonl"],
        expected_sha256=ARCHIVE_SHA["physical.jsonl"], grasp="pinch", project=project)
    physical = [decode_owned(entry) for line in original["physical.jsonl"].splitlines()
                if (entry := costs._json(line)).get("operation") in ("reset", "begin", "physical", "end")]
    live = [decode_owned(costs._json(line)) for line in original["live_r0.jsonl"].splitlines()]
    _require([r["operation"] for r in physical] == [r["operation"] for r in live],
             "actual and R0 journal operations differ")
    composer = ApproachControlComposer(spec)
    controls = []
    group_physical, group_live = [], []
    for p, l in zip(physical, live):
        op = p["operation"]
        if op == "begin":
            _require(l["payload"]["pre_action_sha256"] == _digest(p["payload"]["pre_action_state"])
                     and l["payload"]["action_sha256"] == _digest(p["payload"]["action"]),
                     "same pre-action/action source required")
        elif op == "physical":
            group_physical.append(p["payload"]);group_live.append(l["payload"])
        elif op == "end":
            window = dict(r0_end=l["payload"], live_rows=group_live, physical_rows=group_physical,
                physical_result=p["payload"], cost_control=evaluated["controls"][len(controls)])
            controls.append(composer.compose(window))
            group_physical, group_live = [], []
    _require(len(controls) == 29 and not group_physical and not group_live
             and sum(r["physical_samples"] for r in controls) == 232, "full historical cohort required")
    known = [r["candidate_reward"] for r in controls if r["candidate_reward"] is not None]
    totals = {k:dict(known_window_sum=math.fsum(r["components"][k] for r in controls
                    if r["components"][k] is not None),
                    complete_sum=_sum([r["components"][k] for r in controls]))
              for k in COMPONENTS}
    return dict(version=VERSION, scope="pinned_historical_arithmetic_candidate_only", grasp="pinch",
        spec=spec, spec_sha256=_digest(spec), physical_samples=232, controls=controls,
        known_reward_windows=len(known), expected_reward_windows=29,
        known_reward_sum=math.fsum(known), complete_candidate_reward=_sum([r["candidate_reward"] for r in controls]),
        totals=totals, archive_sha256=deepcopy(ARCHIVE_SHA), full_audit_proof_sha256=PROOF_SHA,
        original_cost_contract_sha256=costs.CONTRACT_SHA256,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_reward=None, full_R0_reward=False, native_qualification=False, teacher_admission=False,
        mesh_reset_qualified=False, health_qualified=False, positive_contact_qualified=False,
        coefficient_status="explicit_candidate_not_rigid_inherited_not_tuned",
        native_steps=0, optimizer_updates=0, geometry_recomputed=False)
