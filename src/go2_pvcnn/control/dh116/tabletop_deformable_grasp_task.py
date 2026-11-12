"""Opt-in CPU task contract for an original DH116 pair and a DeformX rod."""

import hashlib
import json
import math
from pathlib import Path
from numbers import Real

import numpy as np

from .deformx_dh116_colliders import DISTAL_PAD_LINKS, collider_plan
from .tabletop_material import hand_material_contract
from .tabletop_stereo_rod_state import VERSION as ROD_READBACK_VERSION


VERSION = "dh116-flexible-dual-hand-grasp-v1"
SIDES = ("left", "right")
GRASPS = ("pinch", "tripod", "power")
RESET_HOOKS = ("rod", "joint", "controller", "filter", "contact", "camera_history")
R0_TERMS = {
    "approach_progress": ("m", 1),
    "required_contact": ("fraction", 1),
    "contact_persistence": ("s", 1),
    "grasp_geometry": ("fraction", 1),
    "lift_progress": ("m", 1),
    "hold_progress": ("s", 1),
    "material_anchor_slip": ("m", -1),
    "joint_limit": ("event", -1),
    "instability": ("event", -1),
    "external_support_violation": ("event", -1),
}
_SUCCESS = {"net_lift_m": .03, "hold_s": 5., "max_slip_m": .0005, "external_support_allowed": False}
_SOURCE_PATHS = {
    "action": "docs/contracts/stereo_action_contract.json",
    "state": "docs/contracts/stereo_privileged_state_contract.json",
    "pose_reference": "configs/dh116_training/pose_reference.json",
    "rigid_task_reference": "configs/dh116_training/stereo_native_task_v1.json",
    "hand_material": "src/go2_pvcnn/control/dh116/tabletop_material.py",
    "original_asset_manifest": "assets/fixed_dual_panda_dh116_v1/manifest.json",
    "backend_lock": "docs/contracts/dh116_deformx_backend_lock.json",
    "pad_lock": "docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json",
}
_ROD_NA = (
    "gravity_world_m_s2", "length_m", "radius_m", "mass_kg", "density_kg_m3",
    "young_modulus_pa", "shear_modulus_pa", "bend_stiffness_n_m2", "damping", "friction_static",
    "friction_dynamic", "node_count", "solver_substeps_per_physics_step", "native_parameter_source_sha256",
)
_SCENE_NA = (
    "base_dofs", "self_collision_enabled", "rod_self_collision_enabled", "table_collision_enabled",
    "rod_table_collision_enabled", "table_material",
)
_SDF_REGISTRATION_SHA = "db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab"


def _exact_keys(record, names, label):
    if not isinstance(record, dict) or set(record) != set(names):
        raise ValueError(f"{label} fields differ from the task contract")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_sources(config, project):
    _exact_keys(config, _SOURCE_PATHS, "locked sources")
    sources = {}
    for name, relative in _SOURCE_PATHS.items():
        binding = config[name]
        _exact_keys(binding, ("path", "sha256"), name)
        path = project / relative
        if binding["path"] != relative or _sha256(path) != binding["sha256"]:
            raise ValueError(f"{name} source SHA256 or path differs")
        if name != "hand_material":
            sources[name] = json.loads(path.read_text())
    return sources


def load_flexible_grasp_task(path, project_root):
    """Read original contracts and authored geometry. Unknown runtime fields remain NA."""
    config = json.loads(Path(path).read_text())
    _exact_keys(config, ("version", "task_kind", "scope", "native_execution_enabled", "sources",
                         "contracts", "backend", "scene", "rod", "grasps", "success", "reward"), "task")
    if (config["version"], config["task_kind"], config["scope"]) != (
            VERSION, "deformable_rod", "no_camera_cpu_task_contract") or config["native_execution_enabled"] is not False:
        raise ValueError("expected the independent, native-disabled flexible task")
    project = Path(project_root).resolve()
    source = _locked_sources(config["sources"], project)
    action = source["action"]["action_contract"]
    state = source["state"]
    expected_contracts = {
        "action": {"size": 24, "version": action["version"], "contract_sha256": source["action"]["action_contract_sha256"]},
        "state": {"size": 371, "version": state["task_observation_version"]},
    }
    if action["size"] != 24 or state["size"] != 371 or config["contracts"] != expected_contracts:
        raise ValueError("original 24D action and 371D state contracts are required")
    backend = next(row for row in source["backend_lock"]["repositories"]
                   if row["path"] == source["backend_lock"]["cuda_backend_path"])
    pad_lock = source["pad_lock"]
    if backend["commit"] != pad_lock["backend_commit"] or config["backend"] != {
            "name": "DeformX", "commit": backend["commit"],
            "extension_sha256": pad_lock["backend_files"]["_ropesim.cpython-310-x86_64-linux-gnu.so"],
            "standalone_sdf_registration_sha256": _SDF_REGISTRATION_SHA}:
        raise ValueError("locked DeformX CUDA backend identity differs")

    scene = config["scene"]
    pose = source["pose_reference"]["reference_layout"]
    args = source["rigid_task_reference"]["arguments"]
    hands = {}
    for row in source["original_asset_manifest"]["sources"]:
        if row["kind"] == "dh116":
            path = project / _SOURCE_PATHS["original_asset_manifest"]
            original = path.parent / "dependencies" / Path(row["path"]).name
            if _sha256(original) != row["sha256"]:
                raise ValueError("original DH116 asset SHA256 differs")
            hands[row["side"]] = row["sha256"]
    material_profile = args[args.index("--hand-material") + 1]
    authored = {
        "physics_hz": source["action"]["physics_hz"], "control_hz": source["action"]["control_hz"],
        "table_center_m": pose["table_center"], "table_size_m": pose["table_size"],
        "table_top_m": pose["table_center"][2] + pose["table_size"][2] / 2,
        "original_palm_transport": args[args.index("--palm-transport") + 1],
        "original_hand_material_preset": material_profile,
        "original_hand_material_contract": hand_material_contract(material_profile)["hand_contact_material"],
        "original_hand_usd_sha256": hands,
    }
    _exact_keys(scene, (*authored, *_SCENE_NA), "scene")
    if any(scene[name] != value for name, value in authored.items()) or set(hands) != set(SIDES):
        raise ValueError("authored scene parameters differ")
    if any(scene[name] is not None for name in _SCENE_NA):
        raise ValueError("unconfirmed flexible scene fields must remain NA")

    rod = config["rod"]
    _exact_keys(rod, ("rods_per_env", "pinned_node_ids", *_ROD_NA), "rod")
    if type(rod["rods_per_env"]) is not int or rod["rods_per_env"] != 1 or rod["pinned_node_ids"] != []:
        raise ValueError("one continuous free rod with no pinned material nodes is required")
    gravity = rod["gravity_world_m_s2"]
    if gravity is not None and (not isinstance(gravity, list) or len(gravity) != 3
            or not all(type(x) in (float, int) and math.isfinite(x) for x in gravity) or gravity[2] >= 0):
        raise ValueError("the flexible grasp needs downward gravity")
    if any(rod[name] is not None for name in _ROD_NA):
        raise ValueError("unconfirmed flexible rod parameters must remain NA")
    _exact_keys(config["grasps"], GRASPS, "grasp profiles")
    for style, profile in config["grasps"].items():
        _exact_keys(profile, ("required_pads", "native_reset_source_sha256", "material_anchor_rest_m"), style)
        expected = dict(zip(SIDES, (spec["bodies"] for spec in collider_plan(style)["specs"])))
        if profile["required_pads"] != expected:
            raise ValueError(f"{style} pad identity differs from original DH116 collider plan")
        if profile["native_reset_source_sha256"] is not None or profile["material_anchor_rest_m"] is not None:
            raise ValueError("unconfirmed grasp reset and material anchors must remain NA")
    if config["success"] != _SUCCESS or config["success"]["external_support_allowed"] is not False:
        raise ValueError("30mm/5s/0.5mm and no external support required")
    if config["reward"] != {"profile": "flexible_r0", "r2_enabled": False, "coefficient_source": None,
                            "coefficients": dict.fromkeys(R0_TERMS)}:
        raise ValueError("flexible R0 coefficients must remain NA and R2 disabled pending qualification")
    config["native_blockers"] = [
        *(f"scene.{name}" for name in _SCENE_NA), *(f"rod.{name}" for name in _ROD_NA),
        *(f"grasps.{style}.native_reset_source_sha256" for style in GRASPS),
        "native_reset_and_371d_binding", "material_anchor_and_support_source", "reward.coefficients",
    ]
    config["native_qualification"] = False
    return config


def _finite_number(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} requires a finite number >= {minimum}")
    return float(value)


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} requires an explicit boolean")
    return value


def _sample(measurements, name):
    item = measurements.get(name)
    if item is None:
        return None, None
    _exact_keys(item, ("value", "source"), name)
    if item["value"] is not None and (not isinstance(item["source"], str) or not item["source"]):
        raise ValueError(f"{name} requires explicit measurement source")
    return item["value"], item["source"]


def _source(*sources):
    return "|".join(dict.fromkeys(item for item in sources if item)) or None


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _contact(measurements, side, rod_uid, material_length):
    value, source = _sample(measurements, side)
    if value is None:
        return None, source
    _exact_keys(value, ("rod_uid", "pads", "opposed", "material_s_m", "material_uncertainty_m",
                        "tangential_slip_m", "tangential_uncertainty_m"), f"{side} contact")
    if value["rod_uid"] != rod_uid:
        raise ValueError("both hands must measure the same rod UID")
    pads = value["pads"]
    if (not isinstance(pads, list) or any(name not in DISTAL_PAD_LINKS for name in pads)
            or len(set(pads)) != len(pads)):
        raise ValueError(f"{side} contact must identify distinct original pads")
    opposed = _boolean(value["opposed"], f"{side} opposed contact")
    if opposed and len(pads) < 2:
        raise ValueError(f"{side} opposed contact needs at least two measured pads")
    measurements = {}
    for field in ("material_s_m", "material_uncertainty_m", "tangential_slip_m", "tangential_uncertainty_m"):
        val = value[field]
        measurements[field] = None if val is None else _finite_number(val, f"{side}.{field}")
    if measurements["material_s_m"] is not None and measurements["material_s_m"] > material_length:
        raise ValueError(f"{side} material station is outside the rest rod")
    return dict(pads=pads, opposed=opposed, **measurements), source


class _TermStats:
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = self.total = 0.
        self.low = self.high = None
        self.sources = set()

    def update(self, raw, source):
        if raw is not None:
            self.count += 1
            delta = raw - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (raw - self.mean)
            self.total += raw
            self.low = raw if self.low is None else min(self.low, raw)
            self.high = raw if self.high is None else max(self.high, raw)
            self.sources.add(source)
        return {
            "validity_count": self.count, "mean": self.mean if self.count else None,
            "std": math.sqrt(max(self.m2 / self.count, 0.)) if self.count else None,
            "min": self.low, "max": self.high, "episode_sum": self.total if self.count else None,
            "source_identities": sorted(self.sources),
        }


class FlexibleGraspEpisode:
    """Consume one environment's fixed-clock rod samples and reward terms once."""

    def __init__(self, task, grasp, *, env_id, diagnostic_coefficients=None):
        if grasp not in GRASPS or type(env_id) is not int or env_id < 0 or task.get("native_qualification") is not False:
            raise ValueError("a loaded CPU task, grasp profile and one environment are required")
        self.env_id = env_id
        self.required = {side: frozenset(task["grasps"][grasp]["required_pads"][side]) for side in SIDES}
        self.grasp = grasp
        self.backend_sha256 = task["backend"]["extension_sha256"]
        self.dt = 1. / task["scene"]["physics_hz"]
        self.table_top_m = task["scene"]["table_top_m"]
        self.lift_target_m = task["success"]["net_lift_m"]
        self.hold_target_steps = round(task["success"]["hold_s"] / self.dt)
        self.slip_limit_m = task["success"]["max_slip_m"]
        self.coefficients = dict(task["reward"]["coefficients"])
        if diagnostic_coefficients is not None:
            _exact_keys(diagnostic_coefficients, R0_TERMS, "CPU diagnostic coefficients")
            self.coefficients = {name: _finite_number(value, name) for name, value in diagnostic_coefficients.items()}
        self.last_reset_id = -1
        self.seen_episodes = set()
        self.active = False

    def _read_rod(self, sample):
        packet = sample["rod_snapshot"]
        rod_uid = sample["rod_uid"]
        rod_tick = sample["rod_tick"]
        if (packet.get("version") != ROD_READBACK_VERSION or packet.get("task_kind") != "deformable_rod"
                or packet.get("coordinate_frame") != "world" or packet.get("env_ids") != [self.env_id]
                or packet.get("backend_sha256") != self.backend_sha256
                or packet.get("source_scope") not in ("solver_readback_unqualified", "cpu_fixture")
                or not isinstance(rod_uid, str) or not rod_uid or type(rod_tick) is not int or rod_tick < 0):
            raise ValueError("expected a same-environment, identified deformable rod readback")
        uids, ids = packet["episode_uids"], packet["reset_ids"]
        step = packet["physics_step"]
        if (len(uids) != 1 or not isinstance(uids[0], str) or not uids[0] or len(ids) != 1
                or type(ids[0]) is not int or ids[0] < 0 or type(step) is not int or step < 0):
            raise ValueError("rod readback requires episode, reset and physics-step identity")
        time = _finite_number(packet["timestamp_s"], "rod timestamp")
        positions = _array(packet["node_positions"])
        lengths = _array(packet["rest_lengths_m"])
        if (positions.ndim != 3 or positions.shape[0] != 1 or positions.shape[1] < 3
                or positions.shape[2] != 3 or lengths.shape != (1, positions.shape[1] - 1)
                or not np.isfinite(positions).all() or not np.isfinite(lengths).all() or np.any(lengths <= 0)):
            raise ValueError("finite material node positions and positive rest lengths are required")
        identity = (uids[0], ids[0], rod_uid)
        source = f"{packet['source_scope']}:{self.backend_sha256}"
        return identity, (step, rod_tick, time), positions[0, :, 2], lengths[0], source

    def reset(self, reset_hooks, read_after_reset):
        """Run all registered owners before capturing the first contact-free task snapshot."""
        self.active = False
        _exact_keys(reset_hooks, RESET_HOOKS, "reset hooks")
        for name in RESET_HOOKS:
            reset_hooks[name]()
        snapshot, radius_sample = read_after_reset()
        identity, clock, z, lengths, source = self._read_rod(snapshot)
        uid, reset_id, _ = identity
        if reset_id <= self.last_reset_id or uid in self.seen_episodes:
            raise ValueError("reset must bind a fresh episode UID and increasing reset ID")
        radius, radius_source = _sample({"radius": radius_sample}, "radius")
        if radius is not None:
            radius = _finite_number(radius, "rod radius", minimum=np.finfo(float).eps)
        self.identity, self.clock, self.rod_source = identity, clock, source
        self.last_reset_id = reset_id
        self.seen_episodes.add(uid)
        self.rest_lengths = lengths.copy()
        self.radius, self.radius_source = radius, radius_source
        self.initial_median_z, self.initial_min_z = float(np.median(z)), float(np.min(z))
        self.approach = self.material_anchor = self.rod_stride = None
        self.hold_steps, self.best_lift = 0, 0.
        self.phase = "approach"
        self.previous_bilateral = False
        self.stats = {name: _TermStats() for name in R0_TERMS}
        self.active = True

    def _clock_and_rod(self, snapshot):
        identity, clock, z, lengths, source = self._read_rod(snapshot)
        if identity != self.identity:
            raise ValueError("stale or foreign episode/reset/rod sample")
        if source != self.rod_source or not np.array_equal(lengths, self.rest_lengths):
            raise ValueError("episode material coordinates or rod source changed")
        step, rod_tick, time = clock
        last_step, last_rod_tick, last_time = self.clock
        stride = rod_tick - last_rod_tick
        if (step != last_step + 1 or stride <= 0 or (self.rod_stride is not None and stride != self.rod_stride)
                or not math.isclose(time - last_time, self.dt, abs_tol=2e-7, rel_tol=0.)):
            raise ValueError("physical or rod clock skipped, repeated or changed stride")
        return clock, z, stride

    def _measurement_identity(self, measurements, clock):
        uid, reset_id, rod_uid = self.identity
        step, rod_tick, time = clock
        expected = dict(env_id=self.env_id, episode_uid=uid, reset_id=reset_id, rod_uid=rod_uid,
                        physics_step=step, rod_tick=rod_tick)
        packet = measurements.get("sample_identity")
        _exact_keys(packet, (*expected, "timestamp_s"), "measurement identity")
        if (any(type(packet[key]) is not type(value) or packet[key] != value for key, value in expected.items())
                or not math.isclose(_finite_number(packet["timestamp_s"], "measurement timestamp"),
                                    time, abs_tol=1e-9, rel_tol=0.)):
            raise ValueError("measurement identity differs from this rod physics sample")

    def _slip(self, contacts, bilateral):
        """Use fixed rest arclength and independently reported pad tangential slip."""
        anchor = self.material_anchor
        if not bilateral or any(contact[field] is None for contact in contacts
                for field in ("material_s_m", "material_uncertainty_m")):
            return anchor, None, None, None
        stations = tuple(contact["material_s_m"] for contact in contacts)
        uncertainty = tuple(contact["material_uncertainty_m"] for contact in contacts)
        if (abs(stations[0] - stations[1]) <= sum(uncertainty)
                or any(value > self.slip_limit_m for value in uncertainty)):
            return anchor, None, None, None
        if anchor is None:
            if 2. * max(uncertainty) > self.slip_limit_m:
                return anchor, None, None, None
            anchor = stations, uncertainty
        fixed_stations, fixed_uncertainty = anchor
        material_pairs = tuple((abs(a - b), u + v) for a, b, u, v in
                               zip(stations, fixed_stations, uncertainty, fixed_uncertainty))
        if any(u > self.slip_limit_m for _, u in material_pairs):
            return anchor, None, None, None
        material = max(drift for drift, _ in material_pairs)
        tangent_pairs = tuple((contact["tangential_slip_m"], contact["tangential_uncertainty_m"])
                              for contact in contacts)
        if any(drift is None or u is None or u > self.slip_limit_m for drift, u in tangent_pairs):
            return anchor, material, None, None
        tangent = max(drift for drift, _ in tangent_pairs)
        bounds = material_pairs + tangent_pairs
        known_violation = any(drift - u > self.slip_limit_m for drift, u in bounds)
        known_compliance = all(drift + u <= self.slip_limit_m for drift, u in bounds)
        status = True if known_compliance else False if known_violation else None
        return anchor, material, tangent, status

    def step(self, snapshot, measurements):
        if not self.active:
            raise RuntimeError("episode must be reset before stepping")
        clock, z, stride = self._clock_and_rod(snapshot)
        self._measurement_identity(measurements, clock)
        contacts_with_sources = [_contact(measurements, side, self.identity[2], float(self.rest_lengths.sum()))
                                 for side in SIDES]
        contacts, sources = zip(*contacts_with_sources)
        contact_source = _source(*sources)
        contact_known = all(contact is not None for contact in contacts)
        coverage = geometry = None
        if contact_known:
            coverage = sum(len(self.required[side].intersection(contact["pads"])) / len(self.required[side])
                           for side, contact in zip(SIDES, contacts)) / len(SIDES)
            geometry = sum(contact["opposed"] for contact in contacts) / len(SIDES)
        bilateral = coverage == 1. and geometry == 1.
        anchor, material_slip, tangential_slip, slip_ok = self._slip(contacts, bilateral)
        approach, approach_source = _sample(measurements, "approach_distance_m")
        approach_progress = None
        if approach is not None:
            if not isinstance(approach, (tuple, list)) or len(approach) != len(SIDES):
                raise ValueError("approach distances must name both hands")
            approach = sum(_finite_number(x, "approach distance") for x in approach) / len(SIDES)
            if self.approach is not None:
                approach_progress = self.approach[0] - approach
        flags = {name: _sample(measurements, name) for name in
                 ("table_contact", "external_support", "authored_force", "attachment", "joint_limit", "stable")}
        for name, (value, _) in flags.items():
            if value is not None:
                _boolean(value, name)
        support_flags = [flags[name] for name in ("table_contact", "external_support", "authored_force", "attachment")]
        support_values, support_sources = zip(*support_flags)
        support_source = _source(*support_sources)
        joint_limit, joint_source = flags["joint_limit"]
        stable, stability_source = flags["stable"]
        support_known = all(value is not None for value in support_values)
        support_violation = None if not support_known else any(support_values)
        body_lift = float(np.median(z)) - self.initial_median_z
        min_lift = float(np.min(z)) - self.initial_min_z
        min_surface = None if self.radius is None else float(np.min(z)) - self.radius
        eligible_lift = contact_known and support_known and min_surface is not None
        lift_progress = None
        best = self.best_lift
        if eligible_lift:
            if bilateral and not support_violation and min_surface > self.table_top_m:
                best = max(best, min(self.lift_target_m, body_lift, min_lift))
            lift_progress = best - self.best_lift
        hold_known = (eligible_lift and joint_limit is not None and stable is not None
                      and (not bilateral or slip_ok is not None))
        holds = (hold_known and bilateral and not support_violation and not joint_limit and stable
                 and slip_ok is True and min_surface > self.table_top_m
                 and min(body_lift, min_lift) >= self.lift_target_m)
        # The first good endpoint starts the window. Only a consecutive pair
        # establishes one complete physics interval of observed hold.
        hold_steps = min(self.hold_steps + 1, self.hold_target_steps) if holds and self.phase == "hold" else 0
        hold_progress = None if not hold_known else float(hold_steps > self.hold_steps) * self.dt
        lift_source = _source(self.rod_source, self.radius_source, contact_source, support_source)
        raw = {
            "approach_progress": (approach_progress, _source(approach_source, self.approach and self.approach[1])),
            "required_contact": (coverage, contact_source),
            "contact_persistence": (float(bilateral and self.previous_bilateral) * self.dt
                                    if contact_known else None, contact_source),
            "grasp_geometry": (geometry, contact_source),
            "lift_progress": (lift_progress, lift_source),
            "hold_progress": (hold_progress, _source(lift_source, joint_source, stability_source)),
            "material_anchor_slip": (material_slip, contact_source),
            "joint_limit": (None if joint_limit is None else float(joint_limit), joint_source),
            "instability": (None if stable is None else float(not stable), stability_source),
            "external_support_violation": (None if support_violation is None else float(support_violation), support_source),
        }
        self.clock, self.rod_stride, self.material_anchor = clock, stride, anchor
        previous_phase = self.phase
        self.phase = "hold" if holds else "lift" if bilateral else "grasp" if coverage else "approach"
        self.approach = (approach, approach_source) if approach is not None else None
        self.hold_steps, self.best_lift = hold_steps, best
        self.previous_bilateral = bilateral
        terms, term_stats = {}, {}
        for name, (value, source) in raw.items():
            unit, sign = R0_TERMS[name]
            coefficient = self.coefficients[name]
            contribution = None if coefficient is None or value is None else coefficient * value * sign
            terms[name] = dict(raw=value, unit=unit, valid=value is not None, source=source,
                               coefficient=coefficient, contribution=contribution)
            term_stats[name] = self.stats[name].update(value, source)
        contributions = [entry["contribution"] for entry in terms.values()]
        diagnostic_reward = None if any(v is None for v in contributions) else math.fsum(contributions)
        uid, reset_id, rod_uid = self.identity
        return dict(env_id=self.env_id, episode_uid=uid, reset_id=reset_id, rod_uid=rod_uid,
                    grasp=self.grasp, physics_step=clock[0], rod_tick=clock[1], timestamp_s=clock[2],
                    phase=self.phase, previous_phase=previous_phase, terms=terms, term_stats=term_stats, diagnostic_reward=diagnostic_reward, training_reward=None,
                    blocked_terms=[name for name, row in terms.items() if row["contribution"] is None],
                    body_net_lift_m=body_lift, min_node_net_lift_m=min_lift, min_surface_z_m=min_surface,
                    material_anchor_slip_m=material_slip, tangential_slip_m=tangential_slip,
                    hold_steps=hold_steps, hold_s=hold_steps * self.dt, hold_valid=hold_known,
                    success_candidate=hold_steps == self.hold_target_steps, native_qualification=False)
