"""Initialize R0 *after* the coupled physical transaction, without fake hooks.

Pure initialization reuses original snapshot validation and task state semantics.
The native caller separately proves that real joint/rod/owner reset completed
before this method. Calling this function on historical data is not native proof.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .tabletop_deformable_grasp_task import (
    R0_TERMS, _TermStats, _sample, _finite_number, _exact_keys,
)
from .tabletop_deformable_r0_replay import _digest
from .tabletop_deformable_control_step import FlexibleR0ControlStep

VERSION = "dh116-r0-post-coupled-physical-reset-v1"
CLOCK = ("run_uuid","env_id","episode_uid","reset_id","rod_uid",
         "physics_step","rod_tick","timestamp_s")

def reset_clock_matches(before, after):
    if any(k not in before or k not in after or before[k] != after[k] for k in CLOCK):
        raise ValueError("actual SDK/solver/reset boundary changed during R0 initialization")

def initialize_episode(episode, snapshot, radius_sample, *, reset_approach=None):
    """Only original state initialization; no SDK writes, callbacks or physics."""
    e = episode
    e.active = False
    identity, clock, z, lengths, source = e._read_rod(snapshot)
    uid, reset_id, _ = identity
    if reset_id <= e.last_reset_id or uid in e.seen_episodes:
        raise ValueError("fresh episode UID and increasing reset ID required")
    radius, radius_source = _sample({"radius": radius_sample}, "radius")
    if radius is not None:
        radius = _finite_number(radius, "rod radius", minimum=np.finfo(float).eps)
    e.identity, e.clock, e.rod_source = identity, clock, source
    e.last_reset_id = reset_id
    e.seen_episodes.add(uid)
    e.rest_lengths = lengths.copy()
    e.radius, e.radius_source = radius, radius_source
    e.initial_median_z, e.initial_min_z = float(np.median(z)), float(np.min(z))
    e.approach = e.material_anchor = e.rod_stride = None
    e.slip_references = {side: {} for side in ("left", "right")}
    e.hold_start = None
    e.hold_steps, e.hold_s, e.best_lift = 0, 0., 0.
    e.phase = "approach"
    e.previous_bilateral = False
    e.prohibited_support_latched = False
    e.stats = {name: _TermStats() for name in R0_TERMS}
    if reset_approach is not None:
        _exact_keys(reset_approach, ("sample_identity","approach_distance_m"), "reset approach")
        e._measurement_identity(reset_approach, clock)
        value, approach_source = _sample(reset_approach, "approach_distance_m")
        if value is not None:
            if not isinstance(value, (tuple,list)) or len(value) != 2:
                raise ValueError("two measured reset approach distances required")
            e.approach = (math.fsum(_finite_number(x, "reset approach") for x in value)/2., approach_source)
    e.active = True

class PostPhysicalResetR0ControlStep(FlexibleR0ControlStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._implementation[Path(__file__).name] = source
        self._recipe_sha256 = _digest(dict(parent_recipe_sha256=self._recipe_sha256,
            version=VERSION, implementation_sha256=source))
        self.reset_initialization = None

    def reset(self, *args, **kwargs):
        raise RuntimeError("use completed physical boundary; do not supply replay/reset callbacks")

    def initialize_completed_boundary(self, snapshot, radius_sample, *, reset_approach=None):
        if self._reset_attempted:
            raise RuntimeError("one initialization per physical reset and reducer budget")
        self._reset_attempted = True
        try:
            initialize_episode(self._episode, snapshot, radius_sample,
                               reset_approach=deepcopy(reset_approach))
            if self._episode.clock[0] != self._plan["setup_physics_steps_per_env"]:
                raise ValueError("actual completed reset must match declared setup clock")
            self._initialized = True
            self.reset_initialization = dict(version=VERSION, status="initialized",
                snapshot_sha256=_digest(snapshot), radius_sha256=_digest(radius_sample),
                reset_approach_sha256=_digest(reset_approach),
                task_clock=list(self._episode.clock), task_identity=list(self._episode.identity),
                recipe_sha256=self.recipe_sha256, reset_callbacks_executed=0,
                actual_physical_reset_executed_by_initializer=False,
                new_physics_steps=0, native_qualification=False, teacher_admission=False)
            return deepcopy(self.reset_initialization)
        except Exception:
            self._initialized = False
            self._episode.active = False
            self._invalid = True
            raise
