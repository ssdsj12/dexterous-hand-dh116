#!/usr/bin/env python3
"""Freeze a private, unsigned A-fresh Pinch smoke32 recipe on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts import run_dh116_work_packages as scheduler
from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_launch as actual
from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker


PACKAGE = "teacher.pinch-ppo-fresh-smoke32-v1"
EXECUTION = "configs/dh116_execution"
CORE = f"{EXECUTION}/pinch-actual-smoke32-core-v1.json"
COMPANION = f"{EXECUTION}/pinch-actual-smoke32-launch-v1.json"
CORE_IDENTITY = ("command_spec_sha256", "code_config_sha256")


def _encoded(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def window_limits(*, inner_wall_s):
    return dict(worker.FIXED_LIMITS, inner_wall_timeout_s=inner_wall_s, jsonl_limits=dict(worker.JSONL_LIMITS))


def prepare_training(template, expected_sha256, *, inner_wall_s):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import validate_config
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    from go2_pvcnn.control.dh116.tabletop_native_teacher_training import VERSION

    data = json.loads(checked_file(template, expected_sha256))
    if (not isinstance(data, dict)
            or set(data) != {"schema", "preflight", "model_config", "episode_cap_s", "rollout_steps"}
            or data["schema"] != VERSION or type(data["rollout_steps"]) is not int
            or data["rollout_steps"] != worker.FIXED_LIMITS["controls"]):
        raise ValueError("exact Pinch smoke32 template required")
    preflight = validate_config(data["preflight"])
    if preflight["style"] != "pinch" or preflight["profile"] != "approach_wiring_smoke_v1":
        raise ValueError("independently sourced Pinch approach smoke template required")
    if any(preflight[key] is not None for key in ("budget", "authorization", "resource_lock")):
        raise ValueError("unleased template with no previous execution budget required")
    limits = worker.FIXED_LIMITS
    preflight["budget"] = dict(max_sdk_steps=258, max_resets=limits["max_resets"],
        max_solver_calls=limits["max_solver_calls"], max_episode_count=limits["max_episode_count"],
        wall_timeout_s=inner_wall_s)
    return data


def freeze_recipe(repository, *, training, environment, limits, pins_for_source, resource_budget):
    package = dict(title="Pinch fresh PPO smoke32 diagnostic", status="pending", executor="native_bounded",
        dependencies=[], requires_approval=True, resource_class=scheduler.NATIVE_SCOPE,
        evidence_scope="pinch_fresh_ppo_smoke32_diagnostic_v1", retry_policy=scheduler.NATIVE_RETRY,
        command_spec=CORE, tests=[])
    roots = actual.SOURCE_FILES | {actual.PROVIDER, worker.ENTRYPOINT}
    files = sorted(worker.SOURCE_FILES | {actual.ENVIRONMENT} | roots
                   | actual._local_import_closure(repository.root, roots))
    training_sha = scheduler.file_hash(training)
    spec = dict(version=2, entrypoint=worker.ENTRYPOINT,
        argv=["--config", str(training), "--config-sha256", training_sha,
              "--additional-updates", "1", "--output", "{output}"],
        source_files=files, inputs={str(training): training_sha},
        budget=dict(resource_budget, gpu_index=3, physics_steps=258, environments=1, updates=1),
        validator=dict(result_file="training_result.json", equals={"schema": worker.RESULT_SCHEMA},
            artifacts=sorted(worker.ARTIFACTS), line_counts={"physics.jsonl": 256, "transitions.jsonl": 32}))
    inventory = scheduler.native_recipe_sources(repository, package, spec)
    source = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    spec["teacher_contract"] = dict(pins_for_source(source), schema=worker.RECIPE_SCHEMA,
        profile="approach_wiring_smoke_v1", style="pinch",
        training_config=dict(path=str(training), sha256=training_sha), source_sha256=source,
        limits=limits, journal_contract=worker.JOURNAL_CONTRACT)
    worker.validate_teacher_recipe_v2(spec)
    spec_bytes = _encoded(spec)
    recipe = scheduler.native_recipe_identity(repository, package, spec_bytes, spec)
    experience = environment["experience"]
    launch = dict(schema=actual.SCHEMA, core_recipe_identity={key: recipe[key] for key in CORE_IDENTITY},
        source_closure_sha256=source, role="a_fresh", device_plan=dict(physical_gpu_index=3,
            cuda_visible_devices="3", logical_device="cuda:0", experience_path=experience["path"],
            experience_sha256=experience["sha256"]),
        debugger="gdb_on_segv", provider=dict(path=actual.PROVIDER, entrypoint="run_a",
            sha256=inventory[actual.PROVIDER]), checkpoint_origin=None)
    launch_bytes = _encoded(launch)
    package["actual_launch"] = dict(path=COMPANION, sha256=hashlib.sha256(launch_bytes).hexdigest())
    actual.validate_actual_launch(launch, repository=repository, package=package, recipe=recipe)
    return dict(package=package, spec=spec, spec_bytes=spec_bytes, launch=launch, launch_bytes=launch_bytes)


def _static_pins(root, config, source_sha256):
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as physics_source
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import load_reset_manifest, static_preflight
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, checked_file, digest
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config

    pre = config["preflight"]
    profile = BackendProfile.load(root)
    preflight = static_preflight(pre, project=root, backend_profile=profile)
    if (preflight["static_ready"] or
            [item["name"] for item in preflight["components"] if item["applicable"] and not item["valid"]]
            != ["finite_budget"]):
        raise ValueError("draft source preflight failed: " + ", ".join(preflight["blocked_reasons"]))
    reset = load_reset_manifest(pre, project=root)
    target = worker.worker_training_target_contract(profile, reset)
    physical = json.loads(checked_file(root / physics_source.CONFIG, physics_source.CONFIG_SHA))
    backend = dict(profile_sha256=profile.digest, binary_sha256=profile.binary_sha256,
        inventory_sha256=profile.inventory_sha256, source_tree_sha256=profile.source_tree_sha256,
        patch_sha256=profile.patch_sha256)
    physics = dict(physics_hz=physical["physics_hz"], control_hz=physical["control_hz"],
        solver_substeps=physical["rod"]["solver_substeps"], physics_config_sha256=physics_source.CONFIG_SHA,
        action_contract_sha256=digest(ACTION_CONTRACT), target_schema=target["target_schema"])
    reward = load_config(root / pre["reward_config"]["path"], ppo_gamma=pre["ppo"]["gamma"])
    semantic = worker.training_semantic_digest(config, backend_identity=backend,
        source_identity={"worker_source_closure": source_sha256}, physics_identity=physics, project=root)
    return dict(backend_sha256=profile.binary_sha256, reset_sha256=reset.sha256,
        target_sha256=digest(target), reward_sha256=reward.config_sha256, semantic_sha256=semantic)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private directory outside the source tree")
    parser.add_argument("--inner-wall-s", type=int, required=True)
    parser.add_argument("--outer-wall-s", type=int, required=True)
    parser.add_argument("--min-gpu-free-mib", type=int, required=True)
    parser.add_argument("--min-disk-free-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output.absolute()
    if output.parent.resolve(strict=True) != output.parent or output.resolve() != output or output.is_relative_to(root):
        raise ValueError("fresh private nonsymlink draft directory required")
    repo = scheduler.Repository(root, evidence_dir=output.parent)
    config = prepare_training(args.template, args.template_sha256, inner_wall_s=args.inner_wall_s)
    environment = json.loads((root / actual.ENVIRONMENT).read_bytes())
    output.mkdir(mode=0o700)
    training = output / "training.json"
    training.write_bytes(_encoded(config))
    frozen = freeze_recipe(repo, training=training, environment=environment,
        limits=window_limits(inner_wall_s=args.inner_wall_s),
        pins_for_source=lambda source: _static_pins(root, config, source),
        resource_budget=dict(timeout_seconds=args.outer_wall_s, min_gpu_free_mib=args.min_gpu_free_mib,
                             min_disk_free_bytes=args.min_disk_free_bytes))
    from go2_pvcnn.control.dh116.tabletop_native_teacher_worker_artifacts import _cpu_reference

    spec, package = frozen["spec"], frozen["package"]
    context = dict(package=package, source_contract=worker.worker_training_source_contract(repo, package, spec))
    with _cpu_reference(spec, repo, context):
        pass
    core, companion = output / Path(CORE).name, output / Path(COMPANION).name
    core.write_bytes(frozen["spec_bytes"])
    companion.write_bytes(frozen["launch_bytes"])
    identity = actual.actual_launch_identity(frozen["launch"], companion_sha256=package["actual_launch"]["sha256"])
    core_identity = frozen["launch"]["core_recipe_identity"]
    packet = dict(schema="dh116-pinch-actual-recipe-cpu-draft-v1", package_name=PACKAGE,
        registry_proposal=package, source_head=repo.head(), cpu_draft_verified=True, registered=False,
        native_execution_performed=False, native_ready=False, native_qualification=False, teacher_admission=False,
        core_identity=core_identity, actual_identity=identity,
        attempt_fingerprint=actual._digest(dict(core_identity=core_identity, actual_identity=identity)),
        sha256={p.name: scheduler.file_hash(p) for p in (core, companion, training)})
    (output / "draft.json").write_bytes(_encoded(packet))
    print(json.dumps(packet, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
