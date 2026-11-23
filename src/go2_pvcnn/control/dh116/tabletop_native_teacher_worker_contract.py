"""Finite C06 preparation contracts; structural validity grants no execution lease.

No App, CUDA or training imports occur here. V1 native recipes retain their own
240-step gate. This protocol permits independent Pinch/Tripod sources at sealed smoke horizons.
"""
from copy import deepcopy
import math
from pathlib import Path
import re

SHA256 = re.compile(r"[0-9a-f]{64}")
ENTRYPOINT = "scripts/train_dh116_native_teacher_worker.py"
RECIPE_SCHEMA = "dh116-native-teacher-recipe-v2"
RESULT_SCHEMA = "dh116-native-teacher-worker-result-v1"
SOURCE_FILES = frozenset({
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_worker_contract.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_worker_lease.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_worker_artifacts.py",
    # Exact qualified C04/C05 owned dependency files, plus the CPU reference
    # readers/model/optimizer whose actual bytes establish expected identity.
    "docs/superpowers/plans/2026-11-15-c04-policy-owned-distance-addendum.md",
    "docs/superpowers/plans/2026-11-15-c04-reset-bound-history-addendum.md",
    "docs/superpowers/reports/2026-11-15-c04-native-teacher-session-collector.md",
    "docs/superpowers/reports/2026-11-15-c05-native-teacher-training-checkpoint.md",
    "docs/superpowers/reports/2026-11-15-c05-checkpoint-origin-audit.md",
    "scripts/train_dh116_native_teacher_ppo.py",
    "tests/test_dh116_no_camera_collector.py",
    "tests/test_native_teacher_collector.py",
    "tests/test_native_teacher_policy_pinch_distance.py",
    "tests/test_native_teacher_session.py",
    "tests/test_native_teacher_checkpoint.py",
    "tests/test_native_teacher_training.py",
    *("src/go2_pvcnn/control/dh116/"+name for name in (
        "tabletop_cpu_fixture_owners.py", "tabletop_native_teacher_action_history.py",
        "tabletop_native_teacher_checkpoint.py", "tabletop_native_teacher_collector.py",
        "tabletop_native_teacher_policy_pinch_distance.py", "tabletop_native_teacher_session.py",
        "tabletop_native_teacher_sources.py", "tabletop_native_teacher_training.py",
        "tabletop_no_camera_collector.py", "tabletop_native_teacher_profile.py",
        "tabletop_native_teacher_preflight.py", "tabletop_native_teacher_model.py",
        "tabletop_native_teacher_context.py", "tabletop_training_reward.py",
        "tabletop_deformable_privileged_ppo.py", "tabletop_privileged_ppo.py",
        "tabletop_privileged_checkpoint.py", "tabletop_deformable_privileged_checkpoint.py",
        "tabletop_flexible_measured_checkpoint.py", "tabletop_deformable_r0_replay.py")),
    *("src/go2_pvcnn/control/dh116/"+name for name in (
        "tabletop_deformable_dual_evidence.py", "tabletop_deformable_bimanual_step.py",
        "tabletop_deformable_solver_clock.py", "tabletop_action24_history_owner.py",
        "tabletop_stereo_rod_state.py")),
    "configs/dh116_tasks/flexible_pinch_candidate_v1.json",
})
TRIPOD_SOURCE_FILES = frozenset("src/go2_pvcnn/control/dh116/"+name for name in (
    "tabletop_tripod_reset_manifest.py", "tabletop_tripod_reset_distance.py",
    "tabletop_native_teacher_policy_tripod_distance.py", "tabletop_fresh_supported_builder.py",
    "tabletop_supported_mesh_reset.py", "tabletop_fresh_supported_exchange.py",
    "tabletop_power_shared_rod.py", "tabletop_power_supported_task.py",
    "tabletop_deformable_dual_hand_episode.py", "tabletop_deformable_physical_exchange.py",
    "tabletop_deformable_live_geometry.py"))
SMOKE_PROFILE = "approach_wiring_smoke_v1"
CONTACT_PROFILE = "approach_contact_prelift_v1"
POWER_SOURCE_FILES = frozenset({
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_power_distance.py",
    "src/go2_pvcnn/control/dh116/tabletop_tripod_reset_manifest.py",
    "src/go2_pvcnn/control/dh116/tabletop_fresh_supported_builder.py",
    "src/go2_pvcnn/control/dh116/tabletop_supported_mesh_reset.py",
    "configs/dh116_tasks/flexible_power_near_open_settle_v1.json",
    "configs/dh116_tasks/flexible_power_supported_training_reset_v1.json",
})
CONTACT_SOURCE_FILES = frozenset({
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_source_priming.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_contact_context.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_contact_measurement.py",
    "src/go2_pvcnn/control/dh116/tabletop_power_contact_evidence.py",
    "configs/dh116_training/approach_contact_prelift_v1.json",
    "tests/test_native_teacher_contact_worker.py",
})
FIXED_LIMITS = dict(controls=32, sdk_steps_per_control=8, setup_sdk_steps=2,
    measured_sdk_steps=256, warmup_sdk_steps=0, other_sdk_steps=0,
    max_resets=1, max_episode_count=1, max_solver_calls=256,
    internal_substeps_per_solver_call=12, max_internal_solver_substeps=3072,
    max_processes=1)
JSONL_LIMITS = {"physics.jsonl":256, "transitions.jsonl":32,
    "ownership.jsonl":256, "training_events.jsonl":2048}
ARTIFACTS = {"training_result.json", "physics.jsonl", "transitions.jsonl",
    "ownership.jsonl", "training_events.jsonl", "shutdown_complete.json", "teacher_clean.pt"}


JOURNAL_CONTRACT = dict(raw_file="training_events.jsonl",
    raw_preservation="all_native_journal_rows_original_sequence",
    derived_files={"physics.jsonl":"physical", "transitions.jsonl":"transition_recorded",
        "ownership.jsonl":"owner_and_lifecycle_events"}, retain_raw_sequence_references=True)


def teacher_journal_row_budget():
    """Actual C04 event multiplicity plus a finite lifecycle/failure reserve.

    This budgets the COMPLETE raw journal. Derived ledgers are additional
    projections with raw sequence references, never a replacement or lossy
    reduction of the original durable stream. Provider admission must prove its
    complete event routing fits; overflow requires a new reviewed finite schema.
    """
    sdk = {name:FIXED_LIMITS["measured_sdk_steps"] for name in
        ("delivered_exchange", "physical", "budget_consumed", "policy_pinch_distance")}
    control = {name:FIXED_LIMITS["controls"] for name in
        ("decision", "settlement", "transition_recorded")}
    measured = sum(sdk.values()) + sum(control.values())
    maximum = JSONL_LIMITS["training_events.jsonl"]
    return dict(per_sdk_operations=sdk, per_control_operations=control,
        required_measurement_rows=measured,
        setup_reset_origin_optimizer_cleanup_failure_allowance=maximum-measured,
        max_raw_journal_rows=maximum)


def fresh_smoke_limits(controls):
    """Exactly two bounded diagnostic horizons on the same task and reward."""
    if type(controls) is not int or controls not in (8, 32):
        raise ValueError("Pinch smoke requires exactly 8 or 32 controls")
    measured = controls * FIXED_LIMITS["sdk_steps_per_control"]
    limits = dict(FIXED_LIMITS, controls=controls, measured_sdk_steps=measured,
                  max_solver_calls=measured,
                  max_internal_solver_substeps=measured * FIXED_LIMITS["internal_substeps_per_solver_call"])
    rows = dict(JSONL_LIMITS)
    if controls == 8:
        rows.update({"physics.jsonl": measured, "transitions.jsonl": controls,
                     "ownership.jsonl": 128, "training_events.jsonl": 768})
    return limits, rows


def validate_fresh_smoke_rollout(controls, budget, approved_reward_controls):
    """A short collector horizon must have its own exact SDK/solver budget."""
    if controls == approved_reward_controls:
        return
    limits, _ = fresh_smoke_limits(controls)
    if (controls != 8 or approved_reward_controls != 32
            or budget["max_sdk_steps"] != limits["setup_sdk_steps"] + limits["measured_sdk_steps"]
            or budget["max_solver_calls"] != limits["max_solver_calls"]
            or budget["max_episode_count"] != 1 or budget["max_resets"] != 1):
        raise ValueError("rollout length differs from approved reward/PPO config or exact short budget")


def contact_limits(controls):
    """One separately charged physical prime, outside all PPO control rows."""
    limits, rows = fresh_smoke_limits(controls)
    limits.update(warmup_sdk_steps=1, max_solver_calls=limits["max_solver_calls"]+1,
                  max_internal_solver_substeps=limits["max_internal_solver_substeps"]+12)
    return limits, rows


def teacher_profile_limits(profile, controls, grasp):
    if profile == CONTACT_PROFILE and grasp == "tripod":
        return contact_limits(controls)
    if profile == SMOKE_PROFILE and grasp in ("pinch", "tripod", "power"):
        return fresh_smoke_limits(controls)
    raise ValueError("independent grasp/profile worker source required")


def validate_profile_rollout(profile, grasp, controls, budget, approved_controls):
    teacher_profile_limits(profile, controls, grasp)
    if profile == CONTACT_PROFILE:
        from .tabletop_native_teacher_session import validate_contact_rollout
        return validate_contact_rollout(controls, budget, approved_controls)
    return validate_fresh_smoke_rollout(controls, budget, approved_controls)


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _exact_int(value, expected):
    return type(value) is int and value == expected


def validate_teacher_recipe_v2(spec):
    """Validate a finite structural profile, without probing resources.

    All declared source paths and inputs are hashed by the shared scheduler
    identity helper. The actual worker entrypoint must exist there; this module
    neither creates it nor asserts it is ready.
    """
    fields = {"version", "entrypoint", "argv", "source_files", "budget", "validator",
        "inputs", "teacher_contract"}
    if (not isinstance(spec, dict) or set(spec) not in (fields, fields | {"environment"})
            or not _exact_int(spec["version"], 2) or spec["entrypoint"] != ENTRYPOINT):
        raise ValueError("exact C06 v2 worker recipe schema required")
    if "environment" in spec:
        environment = spec["environment"]
        threads = environment.get("DH116_NUM_THREADS") if isinstance(environment, dict) else None
        if (not isinstance(environment, dict) or set(environment) != {"DH116_NUM_THREADS"}
                or type(threads) is not str or threads not in {str(n) for n in range(1, 65)}):
            raise ValueError("bounded DH116_NUM_THREADS is the only Teacher environment override")
    contract = spec["teacher_contract"]
    keys = {"schema", "profile", "style", "training_config", "semantic_sha256", "source_sha256",
        "backend_sha256", "target_sha256", "reset_sha256", "reward_sha256", "limits", "journal_contract"}
    if (not isinstance(contract, dict) or set(contract) != keys
            or contract["schema"] != RECIPE_SCHEMA
            or contract["profile"] not in (SMOKE_PROFILE, CONTACT_PROFILE)
            or contract["style"] not in ("pinch","tripod","power")
            or contract["profile"] == CONTACT_PROFILE and contract["style"] != "tripod"
            or any(not _sha(contract[k]) for k in keys if k.endswith("_sha256"))):
        raise ValueError("independent pinned grasp smoke Teacher contract required")
    if (contract["journal_contract"] != JOURNAL_CONTRACT
            or contract["journal_contract"].get("retain_raw_sequence_references") is not True):
        raise ValueError("complete original NativeJournal and lossless derived views required")
    binding = contract["training_config"]
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute()
            or not _sha(binding["sha256"])):
        raise ValueError("pinned absolute training configuration binding required")
    if spec["argv"] != ["--config", binding["path"], "--config-sha256", binding["sha256"],
            "--additional-updates", "1", "--output", "{output}"]:
        raise ValueError("exact bounded config/update/output command required")
    if spec["inputs"] != {binding["path"]: binding["sha256"]}:
        raise ValueError("training config must correspond to exact pinned scheduler input")
    sources = spec["source_files"]
    if (not isinstance(sources, list) or any(not isinstance(x, str) for x in sources)
            or len(set(sources)) != len(sources) or not SOURCE_FILES <= set(sources)):
        raise ValueError("C06 source closure must include recipe and lease implementation")
    if contract["style"] == "tripod" and not TRIPOD_SOURCE_FILES <= set(sources):
        raise ValueError("complete independent Tripod source closure required")
    if contract["style"] == "power" and not POWER_SOURCE_FILES <= set(sources):
        raise ValueError("complete independent Power source closure required")
    if contract["profile"] == CONTACT_PROFILE and not CONTACT_SOURCE_FILES <= set(sources):
        raise ValueError("complete independent contact source closure required")
    limits = contract["limits"]
    if not isinstance(limits, dict):
        raise ValueError("explicit grasp smoke limits required")
    fixed, row_budget = teacher_profile_limits(contract["profile"], limits.get("controls"), contract["style"])
    end_step = fixed["setup_sdk_steps"] + fixed["warmup_sdk_steps"] + fixed["measured_sdk_steps"]
    budget = spec["budget"]
    bounds = {"gpu_index":(0,63), "timeout_seconds":(31,900), "min_gpu_free_mib":(1,1000000),
        "min_disk_free_bytes":(1,10**13)}
    if (not isinstance(budget, dict) or set(budget) != set(bounds)|{"physics_steps", "environments", "updates"}
            or any(type(budget[k]) is not int or not lo <= budget[k] <= hi for k,(lo,hi) in bounds.items())
            or not _exact_int(budget["physics_steps"],end_step) or not _exact_int(budget["environments"],1)
            or not _exact_int(budget["updates"],1)):
        raise ValueError("finite v2 budget requires the exact SDK cap, one environment and one update")
    if (set(limits) != set(fixed)|{"inner_wall_timeout_s", "jsonl_limits"}
            or any(not _exact_int(limits[k],v) for k,v in fixed.items())):
        raise ValueError("complete exact finite control/Reset/solver/process limits required")
    wall = limits["inner_wall_timeout_s"]
    rows = limits["jsonl_limits"]
    if (type(wall) not in (int,float) or not math.isfinite(wall)
            or not 0 < wall <= budget["timeout_seconds"]-30
            or not isinstance(rows, dict) or set(rows) != set(row_budget)
            or any(not _exact_int(rows[k],v) for k,v in row_budget.items())):
        raise ValueError("inner wall and complete finite outer JSONL limits required")
    validator = spec["validator"]
    if (not isinstance(validator, dict)
            or set(validator) != {"result_file", "equals", "artifacts", "line_counts"}
            or validator["result_file"] != "training_result.json"
            or validator["equals"] != {"schema":RESULT_SCHEMA}
            or not isinstance(validator["artifacts"], list)
            or len(validator["artifacts"]) != len(ARTIFACTS) or set(validator["artifacts"]) != ARTIFACTS
            or validator["line_counts"] != {name: row_budget[name] for name in ("physics.jsonl", "transitions.jsonl")}
            or any(type(v) is not int for v in validator["line_counts"].values())):
        raise ValueError("exact independent Teacher transition/SDK/checkpoint artifacts required")
    return deepcopy(spec)


SEMANTIC_SCHEMA = "dh116-native-teacher-training-semantic-v1"
BACKEND_FIELDS = {"profile_sha256", "binary_sha256", "inventory_sha256", "source_tree_sha256", "patch_sha256"}
PHYSICS_FIELDS = {"physics_hz", "control_hz", "solver_substeps", "physics_config_sha256",
    "action_contract_sha256", "target_schema"}
MODEL_FIELDS = {"version", "task_kind", "grasp", "task_config_sha256", "checkpoint_namespace", "model",
    "reward_profile", "use_cameras", "expert_initialization", "native_ready", "native_prerequisites"}
MODEL_SHAPE_FIELDS = {"common_size", "action_size", "hidden", "rod_feature_size", "rod_hidden", "initial_std"}
PPO_FIELDS = {"gamma", "lam", "clip", "lr", "epochs", "minibatches", "entropy_coef", "value_coef",
    "max_grad_norm", "min_std", "max_std", "target_kl", "kl_backtrack_steps"}


def _canonical(value):
    import hashlib
    import json
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("finite canonical JSON semantic object required") from error
    return hashlib.sha256(raw).hexdigest()


def _checked_binding(binding, *, project):
    from .tabletop_native_teacher_profile import checked_file
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str) or not binding["path"] or not _sha(binding["sha256"])):
        raise ValueError("full content SHA and file binding required")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = Path(project) / path
    return path, checked_file(path, binding["sha256"])


def training_semantic_config(training_config, *, backend_identity, source_identity, physics_identity, project=None):
    """Bind training behavior to content; process authorization stays separate.

    This is an additional strict identity, never a replacement for C04's live
    configuration digest, model identity, or the original checkpoint contracts.
    Budget and process paths are audited by the lease/recipe and omitted only
    from this resumable training identity. Every remaining config field enters
    the returned object, including complete model and PPO configuration.
    """
    from .tabletop_native_teacher_preflight import validate_config
    from .tabletop_native_teacher_profile import ROOT
    from .tabletop_training_reward import load_config
    import json
    project = ROOT if project is None else Path(project)
    expected = {"schema", "preflight", "model_config", "episode_cap_s", "rollout_steps"}
    if (not isinstance(training_config, dict) or set(training_config) != expected
            or training_config["schema"] != "dh116-native-teacher-training-cpu-v1"
            or type(training_config["rollout_steps"]) is not int):
        raise ValueError("exact C05 training configuration schema required")
    pre = deepcopy(validate_config(training_config["preflight"]))
    validate_profile_rollout(pre["profile"], pre["style"], training_config["rollout_steps"], pre["budget"], 32)
    cap = training_config["episode_cap_s"]
    if type(cap) not in (float,int) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("finite explicit episode cap required")
    model = training_config["model_config"]
    if (not isinstance(model,dict) or set(model) != MODEL_FIELDS
            or not isinstance(model["model"],dict) or set(model["model"]) != MODEL_SHAPE_FIELDS
            or model["grasp"] != pre["style"] or model["task_kind"] != "deformable_rod"
            or model["use_cameras"] is not False or model["native_ready"] is not False
            or not _sha(model["task_config_sha256"])):
        raise ValueError("complete independent model semantic configuration required")
    shapes = dict(common_size=371, action_size=24, hidden=[256,256], rod_feature_size=13, rod_hidden=64)
    if (any(model["model"][k] != v for k,v in shapes.items())
            or type(model["model"]["initial_std"]) not in (int,float)
            or not 0 < model["model"]["initial_std"] <= .3):
        raise ValueError("new common66/rod41/context50 model topology required")
    ppo = pre["ppo"]
    if (not set(ppo) <= PPO_FIELDS or any(type(v) not in (int,float) or not math.isfinite(v)
            for k,v in ppo.items() if not (k == "target_kl" and v is None))):
        raise ValueError("known finite original PPO behavior fields required")
    if (not isinstance(backend_identity,dict) or set(backend_identity) != BACKEND_FIELDS
            or any(not _sha(v) for v in backend_identity.values())
            or not isinstance(source_identity,dict) or not source_identity
            or any(not isinstance(k,str) or not k or not _sha(v) for k,v in source_identity.items())):
        raise ValueError("complete backend and source SHA identities required")
    if (not isinstance(physics_identity,dict) or set(physics_identity) != PHYSICS_FIELDS
            or not _exact_int(physics_identity["physics_hz"],480)
            or not _exact_int(physics_identity["control_hz"],60)
            or not _exact_int(physics_identity["solver_substeps"],12)
            or physics_identity["target_schema"] != "fixed_material_nodes41_set_distance_v1"
            or any(not _sha(physics_identity[k]) for k in ("physics_config_sha256","action_contract_sha256"))):
        raise ValueError("explicit physical/action/target content identities required")
    for key in ("reward_config", "resolved_reset_manifest"):
        path, raw = _checked_binding(pre[key],project=project)
        try:
            content = json.loads(raw)
        except (ValueError,UnicodeDecodeError) as error:
            raise ValueError("pinned semantic configuration must contain JSON") from error
        if not isinstance(content,dict):
            raise ValueError("pinned semantic configuration must contain an object")
        if key == "reward_config":
            reward = load_config(path,ppo_gamma=ppo["gamma"])
            if reward.config_sha256 != pre[key]["sha256"] or reward.profile != pre["profile"]:
                raise ValueError("reward bytes or profile changed during semantic validation")
        if key == "resolved_reset_manifest" and pre["style"] in ("tripod","power"):
            from .tabletop_tripod_reset_manifest import SUPPORTED_SCHEMAS
            if content.get("schema") != SUPPORTED_SCHEMAS[pre["style"]] or content.get("style") != pre["style"]:
                raise ValueError("independent supported manifest style/schema required")
        pre[key] = {"sha256":pre[key]["sha256"]}
    for key in ("authorization", "resource_lock", "budget"):
        pre.pop(key)
    semantic = dict(schema=SEMANTIC_SCHEMA, training_schema=training_config["schema"], preflight=pre,
        model_config=deepcopy(model), episode_cap_s=cap, rollout_steps=training_config["rollout_steps"],
        backend_identity=deepcopy(backend_identity), source_identity=deepcopy(source_identity),
        physics_identity=deepcopy(physics_identity),
        policy_execution=dict(device="cpu",optimizer_device="cpu",rng_schema="torch_cpu_rng_v1"))
    _canonical(semantic)
    return semantic


def training_semantic_digest(training_config, **identities):
    return _canonical(training_semantic_config(training_config, **identities))


def checkpoint_from_a_receipt(repo, package_name, receipt_sha256, *, expected_semantic_sha256):
    """Resolve B's exact bytes from the existing independently verified A receipt.

    The scheduler owns intent/approval/log/outer-exit qualification. This reader
    adds the new training/checkpoint contract; it does not deserialize weights,
    create proof files, or grant fresh resource and budget authorization.
    """
    from .tabletop_native_teacher_profile import checked_file
    import json
    if not _sha(receipt_sha256) or not _sha(expected_semantic_sha256):
        raise ValueError("complete A receipt and semantic SHA required")
    package = repo.registry()["packages"][package_name]
    if package["executor"] != "native_bounded" or package["status"] != "passed":
        raise ValueError("qualified A scheduler package required")
    recipe = repo.native_recipe(package)
    spec = recipe["spec"]
    if (spec.get("version") != 2
            or spec["teacher_contract"]["semantic_sha256"] != expected_semantic_sha256):
        raise ValueError("matching v2 A training semantics required")
    path = repo._verified_native_receipt(package_name, package["evidence_scope"], recipe, receipt_sha256)
    receipt = json.loads(checked_file(path,receipt_sha256))
    if (receipt.get("native_qualification") is not True or receipt.get("native_execution_performed") is not True
            or not _exact_int(receipt.get("outer_exit_code"),0)):
        raise ValueError("A requires qualified natural outer exit zero")
    output = Path(receipt["output"])
    if not output.is_absolute() or output.resolve(strict=True) != output:
        raise ValueError("actual regular A native output required")
    result_path = output / spec["validator"]["result_file"]
    result = json.loads(checked_file(result_path,receipt["artifact_sha256"][result_path.name]))
    checkpoint = validate_teacher_training_result(result,
        expected_semantic_sha256=expected_semantic_sha256, artifact_sha256=receipt["artifact_sha256"])
    checkpoint_path = output / checkpoint["path"]
    checked_file(checkpoint_path,checkpoint["sha256"])
    return dict(path=str(checkpoint_path),sha256=checkpoint["sha256"],loaded_updates=1,
        semantic_sha256=expected_semantic_sha256)


def validate_teacher_training_result(result, *, expected_semantic_sha256, artifact_sha256):
    """Shared A-result structure audit; a SHA is not full checkpoint qualification."""
    if (not isinstance(result,dict) or not {"schema","semantic_sha256","training","clean_checkpoint"} <= result.keys()
            or not isinstance(result["training"],dict)):
        raise ValueError("complete A training result object and counters required")
    if result.get("schema") != RESULT_SCHEMA or result.get("semantic_sha256") != expected_semantic_sha256:
        raise ValueError("A training result semantic identity differs")
    training = result["training"]
    if (any(not _exact_int(training.get(k),v) for k,v in {"loaded_updates":0,"target_updates":1,"updates":1}.items())
            or type(training.get("accepted_minibatch_updates")) is not int or training["accepted_minibatch_updates"] < 1
            or type(training.get("parameter_delta_l2")) not in (float,int)
            or not math.isfinite(training["parameter_delta_l2"]) or training["parameter_delta_l2"] <= 0):
        raise ValueError("A requires one genuinely accepted finite changed update")
    checkpoint = result["clean_checkpoint"]
    if (not isinstance(checkpoint,dict) or set(checkpoint) != {"path","sha256"}
            or checkpoint["path"] != "teacher_clean.pt" or not _sha(checkpoint["sha256"])
            or artifact_sha256.get(checkpoint["path"]) != checkpoint["sha256"]):
        raise ValueError("clean checkpoint must bind the actual scoped A bytes")
    return deepcopy(checkpoint)


def worker_training_source_contract(repository, package, spec):
    """Derive the shared checkpoint source contract from actual scheduler files.

    This pure builder is not qualification. The v2 auditor separately requires
    the real current registry package and exact frozen recipe context.
    """
    import hashlib
    import json
    from scripts.run_dh116_work_packages import native_recipe_sources
    inventory=native_recipe_sources(repository,package,spec)
    actual=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    if spec["teacher_contract"]["source_sha256"]!=actual:
        raise ValueError("training source contract differs from actual complete worker source closure")
    return {"worker_source_closure":actual}


def worker_registered_native_package(repository,package):
    """Resolve one actual registry package for both pre-App lease and audit."""
    from scripts.run_dh116_work_packages import Repository
    if type(repository) is not Repository:
        raise ValueError("v2 checkpoint/lease audit requires actual Repository context")
    matches=[(name,value) for name,value in repository.registry()["packages"].items()
        if value.get("executor")=="native_bounded" and value==package]
    if len(matches)!=1:
        raise ValueError("v2 audit requires one actual current registry native package")
    name,current=matches[0]
    return name,deepcopy(current)


def worker_training_target_contract(backend_profile,manifest):
    """One fixed material-node target map for both provider and CPU auditor."""
    from .tabletop_native_teacher_profile import BackendProfile,TARGET_SCHEMA_VERSION
    from .tabletop_native_teacher_preflight import ResolvedResetManifest
    from .tabletop_tripod_reset_manifest import PowerResetManifest, TripodResetManifest
    expected = {"pinch":ResolvedResetManifest,"tripod":TripodResetManifest,"power":PowerResetManifest}
    if (type(backend_profile) is not BackendProfile or type(manifest) not in expected.values()
            or type(manifest) is not expected.get(manifest.data().get("style"))):
        raise ValueError("actual source-loaded backend profile and independent grasp Reset manifest required")
    target=dict(target_schema=backend_profile.target_schema,target_material_ids=list(range(41)))
    data=manifest.data()
    if (target["target_schema"]!=TARGET_SCHEMA_VERSION
            or any(data.get(key)!=value for key,value in target.items())
            or any(not _exact_int(item,index) for index,item in enumerate(data["target_material_ids"]))):
        raise ValueError("actual fixed material41 target source differs")
    return target
