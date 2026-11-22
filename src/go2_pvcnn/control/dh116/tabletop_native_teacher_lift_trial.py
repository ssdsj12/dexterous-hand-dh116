"""File-backed configuration and independent replay for the native Tripod Lift trial."""
from dataclasses import dataclass
import json
import math

from .tabletop_native_teacher_profile import checked_file
from .tabletop_training_reward import CONTACT, RewardConfig


@dataclass(frozen=True, slots=True)
class LiftTrial:
    config_sha256: str
    prelift_stable_s: float


def read_lift_trial(config):
    """The original contact config selects the legacy reward by default."""
    if type(config) is not RewardConfig or config._source_path is None:
        raise ValueError("file-backed reward config required")
    document = json.loads(checked_file(config._source_path, config.config_sha256))
    if "lift_diagnostic" not in document:
        return None
    raw = document["lift_diagnostic"]
    if (config.profile != CONTACT or type(raw) is not dict
            or set(raw) != {"profile", "grasp", "prelift_stable_s"}
            or raw["profile"] != "supported_lift_diagnostic_v2" or raw["grasp"] != "tripod"
            or type(raw["prelift_stable_s"]) is not float or raw["prelift_stable_s"] != .25):
        raise ValueError("invalid versioned Tripod Lift trial configuration")
    return LiftTrial(config.config_sha256, raw["prelift_stable_s"])


def lift_reference_from_reset(event, geometries, parameters):
    """Rebuild the reference from the authentic Reset records in scope."""
    import numpy as np

    from .tabletop_native_teacher_policy_pinch_distance import _radius
    from .tabletop_supported_lift_observation import FixedLiftReference

    reset = event["reset"]
    rod = reset["rod_reset"]
    snapshot = event["rod_boundary"]["rod_snapshot"]
    if (reset["status"] != "complete" or event["rod_boundary"]["boundary"] != "reset"
            or any(rod.get(key) != val for key, val in reset["after_clock"].items())
            or not np.array_equal(rod["node_positions"], snapshot["node_positions"])):
        raise ValueError("measured rod Reset must match the audited boundary")
    radius = _radius({"rod": rod, "material_director": parameters})["value"]
    tables = {}
    for row in geometries:
        evidence = row["evidence"]
        identity = evidence["identity"]
        side = evidence["side"]
        if (row["kind"] != "initial_geometry" or row["side"] != side or side in tables
                or identity["sdk_step"] != rod["physics_step"]
                or not math.isclose(identity["sdk_time_s"], rod["timestamp_s"], rel_tol=0., abs_tol=2e-7)
                or not math.isclose(evidence["rod_radius_m"], radius, rel_tol=0., abs_tol=1e-9)):
            raise ValueError("same SDK Reset table geometry and native radius required")
        tables[side] = (tuple(evidence["table_center_m"]), tuple(evidence["table_size_m"]))
    if set(tables) != {"left", "right"} or tables["left"] != tables["right"]:
        raise ValueError("two matching measured Reset table records required")
    center, size = tables["left"]
    return FixedLiftReference.from_rod_reset(rod, radius_m=radius, table_top_m=center[2] + size[2] / 2)


def _same_number(observed, expected):
    if expected is None:
        return observed is None
    return type(observed) in (int, float) and math.isclose(
        observed, expected, rel_tol=0., abs_tol=1e-12)


def verify_lift_settlement(reward, pre, post, config, trial):
    """Recompute the recorded Lift terms from physical endpoints and budget."""
    from .tabletop_teacher_lift_progress import lift_reward_delta
    from .tabletop_teacher_lift_reward import PROFILE, SCHEMA
    from .tabletop_training_reward import control_total

    if (reward["profile"] != PROFILE or type(reward["profile_version"]) is not int
            or reward["profile_version"] != 2 or reward["schema_sha256"] != SCHEMA
            or reward["config_sha256"] != trial.config_sha256 or reward["gamma"] != config.gamma):
        raise ValueError("frozen measured Lift reward profile differs")
    for point, terms, context in ((pre, reward["pre_terms"], reward["pre_context"]),
                                  (post, reward["terms"], reward["post_context"])):
        term, hold = terms["lift"], terms["hold"]
        height = point.observation.off_support_lift_m
        if (term["source"] != "measured_supported_rod_fixed_reset" or term["unit"] != "m"
                or term["applicable"] is not (height is not None) or term["valid"] is not True
                or not _same_number(term["raw"], height)
                or not _same_number(term["contribution"], point.lift_potential)
                or hold["applicable"] is not False or hold["raw"] is not None
                or not _same_number(hold["contribution"], 0.)
                or context["config_sha256"] != trial.config_sha256 or context["schema_sha256"] != SCHEMA
                or context["lift_phase"] != point.phase.phase
                or not _same_number(context["prelift_stable_s"], point.phase.prelift_stable_s)
                or context["lift_reset_rod_sha256"] != point.observation.reset_rod_sha256):
            raise ValueError("measured Lift term, phase or Reset binding differs")
    phi_pre, phi_post = (math.fsum(t["contribution"] for t in terms.values())
                         for terms in (reward["pre_terms"], reward["terms"]))
    terminated = reward["terminated"]
    costs = reward["cost_per_second"]
    delta = lift_reward_delta(pre, post, gamma=config.gamma, terminated=terminated)
    if (len(costs) != post.phase.physics_tick - pre.phase.physics_tick
            or "success" in reward["event_reasons"]
            or reward["event"] == 100.
            or not _same_number(reward["phi_pre"], phi_pre)
            or not _same_number(reward["phi_post_raw"], phi_post)
            or not _same_number(reward["endpoint_phi"], 0. if terminated else phi_post)
            or not _same_number(reward["post_context"]["lift_shaping_delta"], delta)
            or not _same_number(reward["reward"], control_total(
                phi_pre, phi_post, costs, config.gamma, reward["event"], terminated))):
        raise ValueError("Lift settlement arithmetic differs from measured replay")


def audit_recorded_lift(rows, config, trial, *, backend_profile):
    """Replay only the frozen physical journal. Never infer an unobserved Hold."""
    from collections import Counter

    from .tabletop_deformable_r0_replay import decode_owned
    from .tabletop_native_teacher_profile import profile_reader_scope
    from .tabletop_teacher_lift_progress import SupportedLiftProgress

    def events(op, kind=None):
        return [decode_owned(row["payload"]) for row in rows if row["operation"] == op
                and (kind is None or row["payload"].get("kind") == kind)]

    try:
        [reset] = events("owner_event", "reset")
        [prime] = events("owner_event", "priming_physical")
        [sources] = events("reset_sources")
        with profile_reader_scope(backend_profile):
            reference = lift_reference_from_reset(reset, events("initial_geometry"), sources["parameters"])
        progress = SupportedLiftProgress(reference, grasp="tripod",
                                         prelift_stable_s=trial.prelift_stable_s)
        current = previous = progress.observe(prime["exchange"])
        phases = Counter()
        bilateral_steps = off_support_steps = settlements = 0
        peak_potential = 0.
        best_off_support_m = None
        for row in rows:
            if row["operation"] == "physical":
                current = progress.observe(decode_owned(row["payload"])["exchange"])
                phases[current.phase.phase] += 1
                bilateral_steps += current.observation.bilateral
                height = current.observation.off_support_lift_m
                if height is not None:
                    off_support_steps += 1
                    best_off_support_m = height if best_off_support_m is None else max(height, best_off_support_m)
                peak_potential = max(peak_potential, current.lift_potential)
            elif row["operation"] == "settlement":
                verify_lift_settlement(decode_owned(row["payload"]["reward"]),
                                       previous, current, config, trial)
                previous = current
                settlements += 1
        if not settlements or current is not previous:
            raise ValueError("each measured Lift control window needs its recorded settlement")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("incomplete measured Lift journal") from error
    return dict(config_sha256=trial.config_sha256, reset_rod_sha256=reference.reset_rod_sha256,
                settlements=settlements, physical_steps=sum(phases.values()), phases=dict(phases),
                bilateral_steps=bilateral_steps, off_registered_support_steps=off_support_steps,
                best_off_registered_support_m=best_off_support_m, peak_lift_potential=peak_potential,
                hold_qualified=False)
