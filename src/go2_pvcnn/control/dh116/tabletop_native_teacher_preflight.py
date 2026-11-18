"""CPU-only static gate and post-owner Reset source verification.

Order: static -> create sole owner -> actual Reset/live verification -> policy.
A static pass is never a native run, PPO update, or physical success claim.
"""
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import re
from pathlib import Path
import sys

import numpy as np

from .tabletop_native_teacher_profile import (
    BackendProfile, ROOT, MODEL_SCHEMA_VERSION, INPUT_SCHEMA_VERSION,
    CONTEXT_SCHEMA_VERSION, TARGET_SCHEMA_VERSION, checked_file, digest, profile_reader_scope)

CONFIG_SCHEMA = "dh116-native-teacher-preflight-config-v1"
RESET_MANIFEST_SCHEMA = "dh116-native-teacher-resolved-reset-v1"
EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
R08_REPORT = EVIDENCE / "pinch-v3-independent-reaudit-portable-20261115-a639k8su/reaudit.json"
R08_SHA = "fcf8f92c0b2838e72a6fe6d026cea61c54759b30350b607c0da6b26db6db6a15"
ORIGINAL_RECEIPT_SHA = "6bc124165c6cd907e84ac013b07c9f9df45b2ee1cd4f9499253b42d924adc4ef"
SMOKE = frozenset(("backend", "resolved_reset", "common66", "rod41x13", "current_parameters",
    "same_clock_distance", "accepted_request24", "sdk_target_readback", "original_guards",
    "native_health", "task_context", "eight_sdk_per_control", "finite_budget"))
FULL = SMOKE | frozenset(("required_pad_contact", "opposition", "bilateral_support",
    "off_table", "no_external_support", "fixed_material_slip", "tangential_slip",
    "hold_motion", "continuous_history"))
BUDGET_KEYS = frozenset(("max_sdk_steps", "max_resets", "max_solver_calls",
                         "max_episode_count", "wall_timeout_s"))


def required_capabilities(profile):
    if profile == "approach_wiring_smoke_v1":
        return SMOKE
    if profile == "flexible_train_r1_full":
        return FULL
    raise ValueError("unknown training profile")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_config(*, style="pinch", profile="approach_wiring_smoke_v1"):
    required_capabilities(profile)
    return dict(schema=CONFIG_SCHEMA, profile=profile, style=style,
        model_schema=MODEL_SCHEMA_VERSION, input_schema=INPUT_SCHEMA_VERSION,
        context_schema=CONTEXT_SCHEMA_VERSION, action_dim=24, physics_hz=480, control_hz=60,
        sdk_steps_per_control=8, original_guards_enabled=True, r2_enabled=False,
        force_proxy_enabled=False, ppo={"gamma": .995 if profile == "approach_wiring_smoke_v1" else .999},
        reward_config=None, resolved_reset_manifest=None, budget=None, authorization=None,
        resource_lock=None, capability_sources={})


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration schema object required")
    baseline = candidate_config()
    if set(config) != set(baseline):
        raise ValueError("unknown or missing configuration schema fields")
    required_capabilities(config["profile"])
    if config["style"] not in ("pinch", "tripod", "power"):
        raise ValueError("independent style contract required")
    for key in ("schema", "model_schema", "input_schema", "context_schema",
                "action_dim", "physics_hz", "control_hz", "sdk_steps_per_control",
                "original_guards_enabled", "r2_enabled", "force_proxy_enabled"):
        if type(config[key]) is not type(baseline[key]) or config[key] != baseline[key]:
            raise ValueError("new Teacher schema/action/frequency/guard contract differs: " + key)
    ppo = config["ppo"]
    if (not isinstance(ppo, dict) or type(ppo.get("gamma")) not in (float, int)
            or not math.isfinite(ppo["gamma"]) or not 0 < ppo["gamma"] < 1):
        raise ValueError("finite PPO gamma configuration schema required")
    if not isinstance(config["capability_sources"], dict):
        raise ValueError("capability source schema required")
    if not set(config["capability_sources"]) <= required_capabilities(config["profile"]):
        raise ValueError("unknown capability source contract")
    for key in ("reward_config", "resolved_reset_manifest", "authorization", "resource_lock"):
        value = config[key]
        if value is not None and (not isinstance(value, dict) or set(value) != {"path", "sha256"}
                or not isinstance(value["path"], str) or not value["path"]
                or not isinstance(value["sha256"], str)
                or len(value["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in value["sha256"])):
            raise ValueError("pinned file binding configuration schema differs: " + key)
    if config["budget"] is not None and (not isinstance(config["budget"], dict)
                                        or set(config["budget"]) != BUDGET_KEYS):
        raise ValueError("budget configuration schema differs")
    return config


def _binding(binding, project):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("pinned file source binding required")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = Path(project) / path
    return path, checked_file(path, binding["sha256"])


@dataclass(frozen=True, slots=True, init=False)
class ResolvedResetManifest:
    path: str
    sha256: str
    raw: bytes

    def __init__(self, *args, **kwargs):
        raise TypeError("resolved Reset manifest must be loaded from actual pinned source files")

    def data(self):
        return json.loads(self.raw)

    def verify(self, *, project=ROOT):
        config = candidate_config(style=self.data()["style"])
        config["resolved_reset_manifest"] = dict(path=self.path, sha256=self.sha256)
        if load_reset_manifest(config, project=project) != self:
            raise ValueError("current resolved Reset manifest bytes/provenance differ")
        return self


def load_reset_manifest(config, *, project=ROOT):
    try:
        path, raw = _binding(config["resolved_reset_manifest"], project)
        value = json.loads(raw)
        if (value.get("schema") != RESET_MANIFEST_SCHEMA or value.get("style") != config["style"]
                or value.get("pose_preview_only") is not False
                or value.get("target_schema") != TARGET_SCHEMA_VERSION
                or value.get("target_material_ids") != list(range(41))):
            raise ValueError("actual resolved reset/style/target manifest required; preview cannot qualify")
        if set(value.get("source_refs", {})) != {"original_receipt", "reset_readback", "numeric_reset"}:
            raise ValueError("resolved reset original raw source bindings required")
        refs = value["source_refs"]
        _, original_raw = _binding(refs["original_receipt"], project)
        original = json.loads(original_raw)
        retained = refs["original_receipt"]["sha256"] == ORIGINAL_RECEIPT_SHA
        if not retained:
            if original.get("style") != config["style"]:
                raise ValueError("independent original Reset style provenance required")
            exit_code = original.get("exit_code")
            if type(exit_code) is not int or exit_code != 0:
                raise ValueError("natural exit 0 from the independent native Reset run required")
        seed = value["reset_seed"]
        from .tabletop_dual_hand_reset import _seed
        _seed(seed)
        for name, shape in (("positions", (1, 41, 3)), ("velocities", (1, 41, 3)),
                            ("quaternions", (1, 41, 4))):
            array = np.asarray(value["rod_state"][name])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("actual resolved rod Reset arrays required")
        if not isinstance(value["mesh_bindings_sha256"], str) or len(value["mesh_bindings_sha256"]) != 64:
            raise ValueError("resolved collision mesh binding source required")
        # Derive state from the pinned original raw artifacts, rather than
        # letting a digest bless invented finite values in a manifest.
        from .tabletop_deformable_r0_replay import decode_owned, encode_owned
        _, reset_raw = _binding(refs["reset_readback"], project)
        saved = decode_owned(json.loads(reset_raw))
        saved = saved.get("readback", saved)
        receipt = saved.get("completed_reset_receipt", saved)
        if "completed_reset_receipt" in saved:
            receipt = dict(receipt, mesh_state_readback={k: v for k, v in saved.items()
                            if k != "completed_reset_receipt"})
        _, numeric_raw = _binding(refs["numeric_reset"], project)
        audit = decode_owned(json.loads(numeric_raw))
        from . import tabletop_mesh_observation_source as source
        from .tabletop_mesh_reset_numeric_state import validate_numeric_reset_audit
        if retained:
            if config["style"] != "pinch":
                raise ValueError("Pinch retained Reset proof cannot transfer to Tripod/Power")
            retained_reset_source(project=project)
            from scripts.verify_retained_dh116_pinch_v3 import producer_profile
            with producer_profile(project, Path(original["output"])), source.source_scope():
                validate_numeric_reset_audit(audit, reset_receipt=receipt)
        else:
            with source.source_scope():
                validate_numeric_reset_audit(audit, reset_receipt=receipt)
        if receipt["seed"] != value["reset_seed"]:
            raise ValueError("resolved reset seed differs from original raw source")
        for name in ("positions", "velocities", "quaternions"):
            if not np.array_equal(np.asarray(value["rod_state"][name], np.float32),
                                  audit["samples"][1]["state"][name]):
                raise ValueError("resolved rod state differs from original raw source")
        if digest(encode_owned(receipt["rod_reset"]["mesh_bindings"])) != value["mesh_bindings_sha256"]:
            raise ValueError("resolved mesh bindings differ from original raw source")
        for key in ("reset_readback", "numeric_reset"):
            ref = refs[key]
            file = Path(ref["path"])
            if (not file.is_absolute() or file.parent != Path(original["output"])
                    or original["artifact_sha256"].get(file.name) != ref["sha256"]):
                raise ValueError("resolved source is not bound to original receipt artifact inventory")
        result = object.__new__(ResolvedResetManifest)
        for key, item in dict(path=str(path.absolute()), sha256=hashlib.sha256(raw).hexdigest(), raw=raw).items():
            object.__setattr__(result, key, item)
        return result
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError("resolved_reset_manifest_source_missing: " + str(error)) from error


def validate_budget(config, *, project=ROOT):
    budget = config["budget"]
    if (not isinstance(budget, dict) or set(budget) != BUDGET_KEYS
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in budget.values())
            or any(type(budget[k]) is not int for k in BUDGET_KEYS - {"wall_timeout_s"})):
        raise ValueError("finite explicit positive native budget source required")
    try:
        _, raw = _binding(config["authorization"], project)
        auth = json.loads(raw)
        # File provenance and exact recipe/budget/resource correspondence,
        # never a ready/approved Boolean passed to this function.
        if (auth.get("schema") != "dh116-native-teacher-budget-authorization-v1"
                or auth.get("scope") != "new_native_teacher_ppo"
                or auth.get("profile") != config["profile"] or auth.get("style") != config["style"]
                or auth.get("budget") != budget or auth.get("max_ppo_updates", 0) < 1
                or auth.get("config_sha256") != authorization_config_digest(config)
                or not isinstance(auth.get("session_uid"), str) or not auth["session_uid"]):
            raise ValueError("corresponding new native budget authorization required")
        _, lock_raw = _binding(config["resource_lock"], project)
        lock = json.loads(lock_raw)
        if (lock.get("schema") != "dh116-native-resource-reservation-v1"
                or lock.get("session_uid") != auth["session_uid"]
                or lock.get("resource_class") != "isaac_native_singleton"
                or lock.get("authorization_sha256") != config["authorization"]["sha256"]
                or lock.get("pid") != auth.get("owner_pid") or type(lock.get("pid")) is not int):
            raise ValueError("corresponding singleton resource lock required")
        import os
        os.kill(lock["pid"], 0)
        from scripts.run_dh116_work_packages import NATIVE_LOCK
        info = NATIVE_LOCK.stat()
        observed = False
        for row in Path("/proc/locks").read_text().splitlines():
            fields = row.split()
            if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
                continue
            major, minor, inode = fields[5].split(":")
            observed |= (int(fields[4]) == lock["pid"] and
                (int(major, 16), int(minor, 16), int(inode)) ==
                (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino))
        if not observed:
            raise ValueError("actual corresponding kernel singleton resource lock not held")
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError("native budget authorization/resource lock source missing: " + str(error)) from error
    return dict(budget)


def authorization_config_digest(config):
    return digest({k: v for k, v in config.items() if k not in {"authorization", "resource_lock"}})


def validate_reward_gamma(reward, ppo):
    if type(ppo.get("gamma")) not in (float, int) or reward.gamma != ppo["gamma"]:
        raise ValueError("reward and PPO gamma must come from the same numeric config")


def retained_reset_source(*, project=ROOT):
    """Reuse the sealed R08 review; never reparse the 621MB physical stream."""
    report = json.loads(checked_file(R08_REPORT, R08_SHA))
    if (report["original_receipt_sha256"] != ORIGINAL_RECEIPT_SHA
            or report["original_scheduler_status"] != "failed"
            or report["native_execution_performed"] is not False
            or report["native_qualification"] is not False
            or report["new_physics_steps"] != 0 or report["ppo_updates"] != 0
            or report["verified_saved_physical_sources"] is not True
            or report["original_receipt_unchanged"] is not True
            or report["original_artifacts_unchanged"] is not True
            or report["numeric_reset"]["actual_samples"] != 2
            or report["numeric_reset"]["numeric_state_unchanged"] is not True):
        raise ValueError("sealed R08 retained-source scope differs")
    for path, sha in report["replay_source_sha256"].items():
        p = Path(path)
        checked_file(p if p.is_absolute() else Path(project) / p, sha)
    return report


def _component(name, valid, *, reader, source_sha256=None, applicable=True,
               reason=None, release=""):
    return dict(name=name, source_complete=bool(valid), applicable=applicable,
        valid=bool(valid) if applicable else False, reader=reader, source_sha256=source_sha256,
        blocked_reason=reason if applicable and not valid else None, release_condition=release)


def _reader_source_sha256(value, reader):
    for name in ("digest", "sha256"):
        candidate = getattr(value, name, None)
        if type(candidate) is str and re.fullmatch(r"[0-9a-fA-F]{64}", candidate):
            return candidate
    path = Path(reader)
    return sha256_file(path) if path.is_file() else None


def static_preflight(config, *, project=ROOT, backend_profile=None):
    validate_config(config)
    project = Path(project).resolve()
    components = []
    def check(name, call, reader, release, *, applicable=True):
        if not applicable:
            components.append(_component(name, False, reader=reader, applicable=False, release=release))
            return None
        try:
            value = call()
            components.append(_component(name, True, reader=reader,
                source_sha256=_reader_source_sha256(value, project / reader),
                release=release))
            return value
        except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
            components.append(_component(name, False, reader=reader, reason=str(error), release=release))
            return None
    def backend():
        value = backend_profile if backend_profile is not None else BackendProfile.load(project)
        if type(value) is not BackendProfile:
            raise ValueError("actual BackendProfile required")
        return value.verify(project)
    profile = check("backend", backend, "tabletop_mesh_observation_source.source_basis",
                    "restore exact reviewed observation binary and all current source bytes")
    manifest = check("resolved_reset_manifest", lambda: load_reset_manifest(config, project=project),
          "resolved_reset_manifest", "pin independently reviewed raw Reset sources and exact numeric state")
    check("finite_budget", lambda: validate_budget(config, project=project),
          "budget_authorization_and_resource_lock",
          "freeze separate finite native PPO budget, matching authorization and singleton reservation")
    # A validated independent Reset does not borrow historical R08 provenance.
    fresh_reset = (manifest is not None and
                   manifest.data()["source_refs"]["original_receipt"]["sha256"] != ORIGINAL_RECEIPT_SHA)
    check("retained_reset", lambda: retained_reset_source(project=project), str(R08_REPORT),
          "restore sealed R08 report/current replay sources when using historical evidence",
          applicable=config["style"] == "pinch" and not fresh_reset)
    modules = (("model", "tabletop_native_teacher_model.py", "MODEL_SCHEMA_VERSION", MODEL_SCHEMA_VERSION),
               ("context", "tabletop_native_teacher_context.py", "CONTEXT_SCHEMA_VERSION", CONTEXT_SCHEMA_VERSION))
    for name, filename, constant, expected in modules:
        path = project / "src/go2_pvcnn/control/dh116" / filename
        if not path.is_file():
            components.append(_component(name, False, reader=str(path),
                reason=name + "_implementation_pending", release="implement and verify new independent " + name + " schema"))
        else:
            def schema(path=path, constant=constant, expected=expected):
                module = importlib.import_module("go2_pvcnn.control.dh116." + path.stem)
                if getattr(module, constant, None) != expected:
                    raise ValueError("independent model/context schema differs")
                return module
            check(name, schema, str(path), "supply current new schema CPU implementation")
    reward_module = project / "src/go2_pvcnn/control/dh116/tabletop_training_reward.py"
    if not reward_module.is_file():
        components.append(_component("reward", False, reader=str(reward_module),
            reason="reward_implementation_pending", release="integrate C03 numeric reward loader and pinned profile config"))
    else:
        def reward():
            path, _ = _binding(config["reward_config"], project)
            module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_training_reward")
            value = module.load_config(path, ppo_gamma=config["ppo"]["gamma"])
            validate_reward_gamma(value, config["ppo"])
            if value.profile != config["profile"]:
                raise ValueError("reward profile differs")
            return value
        check("reward", reward, str(reward_module), "pin actual C03 config and matching PPO gamma")
    sources = {
        "common66": "tabletop_teacher_mounted_wrist_source.py",
        "rod41x13": "tabletop_deformable_privileged_teacher.py",
        "current_parameters": "tabletop_native_parameter_endpoints.py",
        "same_clock_distance": "tabletop_bilateral_r0_approach.py",
        "native_health": "tabletop_r0_health_separation.py",
        "accepted_request24": "tabletop_action24_history_owner.py",
        "sdk_target_readback": "tabletop_joint_target_readback.py",
        "original_guards": "tabletop_deformable_physical_exchange.py",
        "eight_sdk_per_control": "tabletop_deformable_control_step.py"}
    for capability in sorted(required_capabilities(config["profile"]) - {"backend", "resolved_reset", "finite_budget", "task_context"}):
        filename = sources.get(capability)
        if capability == "same_clock_distance" and config["style"] != "pinch":
            components.append(_component(capability, False, reader="BilateralMeasuredPinchApproach",
                reason=config["style"] + "_distance_reader_implementation_pending",
                release="independent style-specific same-clock fixed-material41 reader; Pinch proof cannot transfer"))
        elif filename:
            path = project / "src/go2_pvcnn/control/dh116" / filename
            check(capability, lambda p=path: p.read_bytes(), str(path),
                  "current CPU reader bytes; actual same-owner Reset boundary still required")
        else:
            # Full-only sources require versioned, hashed implementation exports,
            # not caller qualification flags. No full gate is added to smoke.
            def declared(capability=capability):
                binding = config["capability_sources"].get(capability)
                if binding is None:
                    raise ValueError(capability + "_reader_implementation_pending")
                path, raw = _binding(binding, project)
                contract = json.loads(raw)
                if (contract.get("schema") != "dh116-native-teacher-capability-reader-v1"
                        or contract.get("capability") != capability or contract.get("style") != config["style"]):
                    raise ValueError(capability + "_reader_contract_missing")
                for binding in contract["implementation_sources"]:
                    _binding(binding, project)
                reader = contract.get("reader")
                if not isinstance(reader, str) or ":" not in reader:
                    raise ValueError(capability + "_reader_implementation_pending")
                module_name, function_name = reader.split(":", 1)
                if not module_name.startswith("go2_pvcnn.control.dh116."):
                    raise ValueError("source-controlled capability reader module required")
                module = importlib.import_module(module_name)
                if (not callable(getattr(module, function_name, None))
                        or getattr(module, "CAPABILITY_SCHEMA", None) != contract["schema"]
                        or capability not in getattr(module, "SUPPORTED_CAPABILITIES", ())
                        or sha256_file(module.__file__) not in
                            {row["sha256"] for row in contract["implementation_sources"]}):
                    raise ValueError(capability + "_reader_contract_missing")
                if not contract["implementation_sources"]:
                    raise ValueError(capability + "_reader_implementation_pending")
                return contract
            check(capability, declared, capability, "implement and independently verify applicable real reader")
    blocked = [c["blocked_reason"] for c in components if c["applicable"] and not c["valid"]]
    return dict(profile=config["profile"], style=config["style"], model_schema=config["model_schema"],
        backend_profile_digest=None if profile is None else profile.digest,
        static_ready=not blocked, live_reset_verified=False, components=components,
        blocked_reasons=blocked, required_capabilities=sorted(required_capabilities(config["profile"])),
        ledger=dict(width=371, measured_selected=66, unknown_nan_false=305,
                    unknown_slots_block_new_schema=False),
        new_native_steps=0, ppo_updates=0, native_qualification=False, teacher_admission=False)


def validate_live_numeric(audit, *, reset_receipt, rope, expected_clock):
    from .tabletop_mesh_reset_numeric_state import validate_numeric_reset_audit
    from .tabletop_deformable_r0_replay import decode_owned
    result = validate_numeric_reset_audit(audit, reset_receipt=reset_receipt)
    receipt = decode_owned(reset_receipt)
    if (result["source_instance_id"] != id(rope)
            or receipt["after_clock"] != expected_clock):
        raise ValueError("current live owner/reset clock differs")
    return result


def validate_live_mesh(sim, rod, *, reset_receipt, mesh_plan):
    from .tabletop_mesh_reset_readback import (
        validate_reset_readback, capture_mesh_state, validate_capture)
    from .tabletop_deformable_r0_replay import decode_owned, encode_owned
    receipt = decode_owned(reset_receipt)
    status = validate_reset_readback(receipt["mesh_state_readback"],
                                    reset_receipt=receipt, mesh_plan=mesh_plan)
    current = capture_mesh_state(sim, rod)
    validate_capture(current)
    saved = receipt["mesh_state_readback"]["after"]
    if (current["source_instance_id"] != saved["source_instance_id"]
            or current["source_instance_id"] != id(rod.rope)
            or current["clock_before"] != saved["clock_after"]
            or digest(encode_owned(current["snapshot"])) != digest(encode_owned(saved["snapshot"]))):
        raise ValueError("current collision mesh owner/clock/current/target state differs from actual Reset")
    return status


def validate_live_mounted(bindings, observation, *, read_identity):
    from .tabletop_teacher_mounted_wrist_source import (
        capture_mounted_hand, adapt_mounted_hands, validate_mounted_observation)
    from .tabletop_palm_mount import MountView
    before = read_identity()
    validate_mounted_observation(observation)
    if observation["identity"] != before or tuple(b.side for b in bindings) != ("left", "right"):
        raise ValueError("current SDK mounted source identity/reset differs")
    packets = {p["side"]: p for p in observation["source_packets"]}
    fresh = []
    for binding in bindings:
        p = packets[binding.side]
        path = binding.env_paths[before["env_id"]]
        fresh.append(capture_mounted_hand(binding.hand, binding.mount or MountView(binding.hand),
            side=binding.side, read_identity=read_identity, data_env_index=0,
            expected_articulation_path=path, expected_hand_container_path=p["hand_container_path"],
            reset_seed=p["reset_seed"]))
    rebuilt = adapt_mounted_hands(fresh, expected_identity=before)
    if (not np.array_equal(rebuilt["valid"], observation["valid"])
            or not np.array_equal(rebuilt["values"], observation["values"], equal_nan=True)):
        raise ValueError("current SDK mounted common values differ from saved source")
    if read_identity() != before:
        raise ValueError("SDK clock advanced during mounted readback")
    return validate_mounted_observation(rebuilt)


@dataclass(frozen=True, slots=True)
class LiveResetContract:
    """Borrowed real owner handles plus owned receipts from this Reset.

    C04 supplies actual mounted capture through its owner; its implementation
    must be in profile inventory. This contract never accepts qualification bits.
    """
    owner: object
    sim: object
    rod: object
    bindings: tuple
    reset_receipt: object
    numeric_audit: object
    mounted_observation: object
    parameter_packet: object
    distance_reader: object
    mesh_plan: object


def live_preflight(owner, reset_contract, *, backend_profile, manifest):
    if (type(reset_contract) is not LiveResetContract or reset_contract.owner is not owner
            or isinstance(owner, (bool, dict))):
        raise ValueError("real same-instance owner and LiveResetContract required")
    if type(manifest) is not ResolvedResetManifest or type(backend_profile) is not BackendProfile:
        raise ValueError("immutable source-loaded profile and resolved Reset manifest required")
    from . import tabletop_mesh_observation_source as source
    from . import tabletop_dual_hand_reset as reset
    from . import tabletop_mesh_reset_numeric_state as numeric
    from .tabletop_deformable_r0_replay import decode_owned
    from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation
    from .tabletop_bilateral_r0_approach import BilateralMeasuredPinchApproach
    from .tabletop_r0_health_separation import capture_solver_health
    from .tabletop_deformx_parameter_snapshot import validate_snapshot
    c = reset_contract
    if (getattr(owner, "rod", None) is not c.rod or getattr(owner, "sim", None) is not c.sim
            or getattr(owner, "bindings", None) is not c.bindings):
        raise ValueError("live owner SDK/rod handle differs")
    if getattr(owner, "backend_profile", None) is not backend_profile:
        raise ValueError("owner must share the same immutable BackendProfile instance")
    backend_profile.verify()
    source.verify_runtime(c.rod)
    manifest.verify()
    data = manifest.data()
    receipt = decode_owned(c.reset_receipt)
    audit = decode_owned(c.numeric_audit)
    with profile_reader_scope(backend_profile):
        clock = reset._clock(c.sim, c.rod)
        if receipt["after_actual_clocks"] != clock or receipt["after_clock"] != clock["rod"]:
            raise ValueError("live SDK/solver/reset UID clock differs")
        validate_live_numeric(audit, reset_receipt=receipt, rope=c.rod.rope, expected_clock=clock["rod"])
        # Read q/qd and SDK targets afresh; never compare only cached receipts.
        hands = reset._hands(c.bindings, c.rod, finite=True)
        from .tabletop_deformable_r0_replay import encode_owned
        if digest(encode_owned(hands)) != digest(encode_owned(receipt["hands"])):
            raise ValueError("current SDK q/qd/targets/body poses differ from actual Reset")
        if receipt["seed"] != data["reset_seed"]:
            raise ValueError("actual Reset differs from immutable resolved seed")
        sample = {}
        current = numeric._capture(c.rod.rope, c.sim, c.rod, sample)
        for name in ("positions", "velocities", "quaternions"):
            if (not np.array_equal(current[name], audit["samples"][1]["state"][name])
                    or not np.array_equal(current[name], np.asarray(data["rod_state"][name], dtype=np.float32))):
                raise ValueError("current rod position/velocity/director differs from resolved Reset")
        if digest(encode_owned(receipt["rod_reset"]["mesh_bindings"])) != data["mesh_bindings_sha256"]:
            raise ValueError("current/target mesh binding identity differs from resolved Reset")
        # Existing actual mesh readback validator verifies before/after target and
        # current poses. It is intentionally not replaced by a Boolean.
        validate_live_mesh(c.sim, c.rod, reset_receipt=receipt, mesh_plan=c.mesh_plan)
        def preaction_clock():
            actual = c.rod.clock()
            actual.update(policy_step=c.mounted_observation["identity"]["policy_step"],
                          substep=0, phase="pre_action_before_target_write")
            return actual
        common = validate_live_mounted(c.bindings, c.mounted_observation, read_identity=preaction_clock)
        if common["valid_channels"] != 66 or common["unknown_channels"] != 305:
            raise ValueError("actual mounted common66 ledger source differs")
        if any(c.mounted_observation["identity"].get(k) != clock["rod"][k]
               for k in ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "timestamp_s")):
            raise ValueError("mounted common source is from a different live Reset")
        validate_snapshot(current["parameters"], expected_num_envs=1, expected_nodes=41,
                          expected_time_s=clock["rod"]["solver_time_s"])
        parameters = decode_owned(c.parameter_packet)
        if (parameters["source_instance_id"] != id(c.rod.rope) or parameters["identity"] != clock["rod"]
                or digest(encode_owned(parameters["raw_native_parameters"])) != digest(encode_owned(current["parameters"]))
                or not np.array_equal(parameters["node_positions_m"], current["positions"])
                or not np.array_equal(parameters["segment_quaternions_xyzw"], current["quaternions"][:, :40])):
            raise ValueError("current live parameters/directors/owner differ from actual parameter packet")
        for packet in c.mounted_observation["source_packets"]:
            if packet["reset_seed"]["root_seed_w"] != receipt["seed"]["authored_root_seed_w"]:
                raise ValueError("mounted authored reference differs from resolved raw Reset")
        if type(c.distance_reader) is not BilateralMeasuredPinchApproach or data["style"] != "pinch":
            raise ValueError("independent style-specific real distance reader required")
        distance = c.distance_reader.capture_reset(receipt, c.parameter_packet, grasp=data["style"])
        if distance["missing_reason"] is not None:
            raise ValueError("actual bilateral Reset distance unavailable")
        health = capture_solver_health(c.rod, backend=sys.modules["ropesim._ropesim"], sim=c.sim,
                                       expected_clock=clock["rod"])
        if health["last_error"] != 0:
            raise ValueError("actual native numerical health is not zero")
        if reset._clock(c.sim, c.rod) != clock:
            raise ValueError("clock advanced during live Reset verification")
    return dict(live_reset_verified=True, owner_instance_id=id(owner), rod_instance_id=id(c.rod.rope),
        reset_identity=clock["rod"], backend_profile_digest=backend_profile.digest,
        resolved_reset_manifest_sha256=manifest.sha256, distance_source_sha256=distance["source_sha256"],
        new_native_steps=0, ppo_updates=0, native_qualification=False, teacher_admission=False)
