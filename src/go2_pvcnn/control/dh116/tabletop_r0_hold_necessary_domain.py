"""Opt-in R0 hold contribution domain; unknown stability remains unknown.

Lack of a necessary bilateral-contact condition proves no hold increment.
This changes neither measured hold/stability values nor success/termination.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from .tabletop_deformable_r0_applicability import ApplicableR0Episode, ApplicableR0ControlStep

VERSION = "dh116-r0-hold-necessary-domain-diagnostic-v1"
CONFIG = Path(__file__).resolve().parents[4] / "configs/dh116_tasks/flexible_r0_hold_necessary_domain_v1.json"
EXPECTED = dict(version=VERSION, enabled_by_default=False, scope="offline_diagnostic_only",
    hold_domain="known_missing_bilateral_contact_has_no_hold_increment",
    missing_raw_policy="retain_null_invalid_do_not_measure_stability",
    instability_terminal_unchanged=True, training_reward_enabled=False,
    success=dict(net_lift_m=.03, hold_s=5., max_slip_m=.0005))

def contract():
    data = CONFIG.read_bytes()
    if json.loads(data) != EXPECTED:
        raise ValueError("exact isolated hold-domain configuration required")
    return dict(reward_contract_version=VERSION,
        reward_config_sha256=hashlib.sha256(data).hexdigest(),
        reward_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

class NecessaryHoldR0Episode(ApplicableR0Episode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = contract()

    def step(self, snapshot, measurements):
        try:
            row = super().step(snapshot, measurements)
            row["diagnostic_reward_without_hold_applicability"] = row["diagnostic_reward"]
            coverage = row["terms"]["required_contact"]
            geometry = row["terms"]["grasp_geometry"]
            known = coverage["raw"] is not None and geometry["raw"] is not None
            no_bilateral = known and (coverage["raw"] < 1. or geometry["raw"] < 1.)
            hold = row["terms"]["hold_progress"]
            if no_bilateral and hold["raw"] is None:
                hold.update(applicable=False, applicability_source=coverage["source"],
                    applicability_reason="known_missing_bilateral_hold_precondition",
                    contribution=None if hold["coefficient"] is None else 0.)
                hold["contribution_valid"] = hold["contribution"] is not None
            row["blocked_terms"] = [k for k,t in row["terms"].items() if not t["contribution_valid"]]
            row["diagnostic_reward"] = None if row["blocked_terms"] else math.fsum(
                t["contribution"] for t in row["terms"].values())
            row.update(version=VERSION, **self.contract)
            return row
        except Exception:
            self.active = False
            raise

class NecessaryHoldR0ControlStep(ApplicableR0ControlStep):
    def __init__(self, task, grasp, *, env_id, budget_config, diagnostic_coefficients):
        super().__init__(task, grasp, env_id=env_id, budget_config=budget_config,
                         diagnostic_coefficients=diagnostic_coefficients)
        self.contract = contract()
        self._implementation[Path(__file__).name] = self.contract["reward_source_sha256"]
        self._recipe_sha256 = hashlib.sha256(json.dumps(
            dict(parent_recipe_sha256=self._recipe_sha256, **self.contract),
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self._episode = NecessaryHoldR0Episode(deepcopy(task), grasp, env_id=env_id,
            diagnostic_coefficients=deepcopy(diagnostic_coefficients))

    def settle(self):
        result = super().settle()
        result["version"] = VERSION
        return result
