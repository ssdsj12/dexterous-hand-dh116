"""Owned Pinch PPO worker. The scheduler verifies eventual outer-process exit."""
from contextlib import closing
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from weakref import ref

from . import tabletop_native_teacher_actual_process as process


@dataclass(frozen=True)
class _Attempt:
    repository: object
    package: dict
    spec: dict
    output: Path
    device: dict
    session_uid: str

    @property
    def project(self):
        return self.repository.root


def _verify_attempt(config, lease, output, *, role="a_fresh"):
    if type(lease) is not process.ActualWorkerLease:
        raise TypeError("verified actual worker lease required")
    lease.revalidate()
    if lease.role != role or ((role == "a_fresh") != (lease.resume_origin is None)):
        raise ValueError("actual worker role and its authorized resume origin must agree")
    if json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(
            lease.process_config, sort_keys=True, allow_nan=False):
        raise ValueError("actual worker config differs from the verified process overlay")
    repository, package = lease._repository, lease._package
    _, recipe, _, _, intent = process._registered(repository, package, lease._envelope.registered_intent)
    if config["rollout_steps"] != recipe["spec"]["teacher_contract"]["limits"]["controls"]:
        raise ValueError("actual worker rollout config differs from the approved controls")
    output = Path(output)
    if (not output.is_absolute() or str(output) != intent["output"]
            or output.parent.resolve(strict=True) != output.parent):
        raise ValueError("actual worker output differs from the approved canonical directory")
    if os.path.lexists(output):
        raise FileExistsError("actual worker output must be new: " + str(output))
    return _Attempt(repository, package, recipe["spec"], output, lease.device_plan, lease._envelope.nonce)


def _fresh_reset_owners():
    sources = {
        "controller": "fresh policy is constructed after Reset; C04 clears issued-action history",
        "filter": "Pinch approach smoke has no standalone filter at its first Reset",
        "contact": "C04 opens the live smoke source after the first physical Reset",
        "camera_history": "camera-free worker has no camera history owner",
    }

    def absent(name, context):
        return dict(owner=name, status="not_present", episode_uid=context["episode_uid"], source=sources[name])

    return {name: partial(absent, name) for name in sources}


def _prepare_source_pack(owner, contract, *, device):
    from .tabletop_native_parameter_endpoints import NativeParameterEndpoints
    from .tabletop_native_teacher_sources import NativePinchSources

    sources = owner.sources
    reader = sources.reader
    rope = owner.rod.rope
    backend = sys.modules.get("ropesim._ropesim")
    clock = owner.rod.clock()
    if (type(sources) is not NativePinchSources or sources.owner is not owner
            or sources.scope != "owner_source_unqualified"
            or type(reader) is not NativeParameterEndpoints or reader.device != device
            or backend is None or type(rope) is not backend.RopeBatch
            or reader._rope_ref is None or reader._rope_ref() is not rope
            or sources.parameters.get("source_instance_id") != id(rope)
            or len(owner.bindings) != 2
            or any(str(binding.hand.data.joint_pos.device) != device for binding in owner.bindings)
            or str(owner.sim.device) != device or owner.sim.current_time_step_index != 2
            or clock["physics_step"] != 2 or clock["rod_tick"] != 0
            or clock["episode_uid"] != owner.episode_uid):
        raise ValueError("live source pack requires the same Pinch Reset, native rod and both SDK hands")
    profile, mode = contract.backend_profile, contract.context_mode

    def bind_model(model):
        if model._backend_profile is not profile or model.context_mode != mode:
            raise ValueError("policy differs from the verified live source pack")

    return bind_model


def _checkpoint_contract(collector, attempt):
    return _make_checkpoint_contract(collector.trainer, collector.live.static, collector.sources.scope, attempt)


def _make_checkpoint_contract(trainer, static, scope, attempt):
    from scripts.diagnose_dh116_teacher_initialization import validate_loaded_project_modules
    from .tabletop_mesh_observation_source import CONFIG, CONFIG_SHA
    from .tabletop_native_teacher_actual_launch import audit_loaded_actual_sources
    from .tabletop_native_teacher_checkpoint import build_training_contract
    from .tabletop_native_teacher_profile import checked_file, digest
    from .tabletop_native_teacher_training import VERSION
    from .tabletop_native_teacher_worker_contract import (
        training_semantic_digest, worker_training_source_contract, worker_training_target_contract,
    )
    from .tabletop_surface_contract import ACTION_CONTRACT

    profile, manifest = static.backend_profile, static.reset_manifest
    source = worker_training_source_contract(attempt.repository, attempt.package, attempt.spec)
    target = digest(worker_training_target_contract(profile, manifest))
    physical = json.loads(checked_file(attempt.project / CONFIG, CONFIG_SHA))
    config = static.config

    training = dict(schema=VERSION, preflight=config.preflight, model_config=config.model_config,
                    episode_cap_s=config.episode_cap_s, rollout_steps=config.rollout_steps)
    semantic = training_semantic_digest(training, source_identity=source, project=attempt.project,
        backend_identity=dict(profile_sha256=profile.digest, binary_sha256=profile.binary_sha256,
            inventory_sha256=profile.inventory_sha256, source_tree_sha256=profile.source_tree_sha256,
            patch_sha256=profile.patch_sha256),
        physics_identity=dict(physics_hz=physical["physics_hz"], control_hz=physical["control_hz"],
            solver_substeps=physical["rod"]["solver_substeps"], physics_config_sha256=CONFIG_SHA,
            action_contract_sha256=digest(ACTION_CONTRACT), target_schema=profile.target_schema))
    expected = dict(backend_sha256=profile.binary_sha256, reset_sha256=manifest.sha256,
                    target_sha256=target, reward_sha256=static.reward_config.config_sha256,
                    semantic_sha256=semantic)
    if any(attempt.spec["teacher_contract"][key] != value for key, value in expected.items()):
        raise ValueError("live Teacher semantic identity differs from the frozen recipe")
    validate_loaded_project_modules(attempt.project)
    audit_loaded_actual_sources(attempt.repository, attempt.package)
    return build_training_contract(trainer, scope=scope,
        reward_sha256=expected["reward_sha256"], target_sha256=target, reset_sha256=manifest.sha256,
        source_sha256=source, config_sha256=semantic)



def _record_actual_wall_budget(collector, lease, phase, *, operation="fresh_update_wall_budget"):
    owner = collector.owner
    runtime = owner.contract.runtime
    sample = runtime.wall_readback()
    if (sample["parent_deadline_s"] != lease.deadline_monotonic_s or sample["remaining_s"] <= 0.
            or owner.context.external_cap_s != sample["local_cap_s"]
            or owner.context.episode_cap_s != owner.contract.config.episode_cap_s):
        raise ValueError("live actual Teacher wall budget or context cap differs")
    collector.journal.emit(operation, dict(sample, phase=phase,
        episode_uid=owner.episode_uid, sdk_step=owner.sim.current_time_step_index,
        episode_cap_s=owner.context.episode_cap_s))


def _acquire(sim, owners, trace, *, attempt, config, lease, resume=None):
    import torch

    from .tabletop_native_teacher_actual_factories import (
        FRESH_POLICY_EXPLORATION_SEED,
        FRESH_POLICY_INITIALIZATION_SEED,
        build_actual_factories,
    )
    from .tabletop_fast_owned_codec import codec_scope
    from .tabletop_native_teacher_actual_update import run_fresh_update, run_resumed_update
    from .tabletop_native_teacher_collector import NativeJournal
    from .tabletop_native_teacher_session import (
        NativeSessionConfig, build_fresh_pinch_owner, create_native_collector,
    )

    session = NativeSessionConfig(preflight=config["preflight"], model_config=config["model_config"],
        journal_path=attempt.output / "training_events.jsonl", episode_cap_s=config["episode_cap_s"],
        rollout_steps=config["rollout_steps"], project=attempt.project)
    with closing(NativeJournal(session.journal_path)) as journal:
        def build(static):
            return build_fresh_pinch_owner(static, sim=sim, owners=owners, prepared=None,
                ownership=trace, ledger=journal, run_uuid=attempt.session_uid,
                episode_uid=attempt.session_uid + ":0", owner_resets=_fresh_reset_owners(), mesh_plan=None)

        factory_options = {}
        restored_contract = None

        def make_restore_contract(trainer, owner):
            nonlocal restored_contract
            restored_contract = _make_checkpoint_contract(trainer, owner.contract, owner.sources.scope, attempt)
            return restored_contract

        if resume is not None:
            factory_options["resume_contract"] = make_restore_contract
        factories = build_actual_factories(lease=lease, episode_factory=build,
            prepare_source_pack=partial(_prepare_source_pack, device=attempt.device["logical_device"]),
            **factory_options)
        collector = create_native_collector(session, factories)
        record_budget = _record_actual_wall_budget if resume is None else partial(
            _record_actual_wall_budget, operation="resume_update_wall_budget")
        try:
            checkpoint_contract = (_checkpoint_contract(collector, attempt) if resume is None
                                   else restored_contract)
            record_budget(collector, lease, "before_update")
            if resume is not None:
                with codec_scope(), collector.trace_wall_windows(trace.event):
                    result = run_resumed_update(collector, checkpoint_contract=checkpoint_contract,
                                                controls=config["rollout_steps"], origin=resume)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(FRESH_POLICY_EXPLORATION_SEED)
                    rng_bytes = torch.get_rng_state().numpy().tobytes()
                    journal.emit("fresh_policy_rng", {
                        "version": "dh116-fresh-policy-cpu-rng-v1",
                        "device_scope": "cpu",
                        "initialization_seed": FRESH_POLICY_INITIALIZATION_SEED,
                        "exploration_seed": FRESH_POLICY_EXPLORATION_SEED,
                        "initial_weights_sha256": collector._epoch["weights"],
                        "exploration_state_sha256": hashlib.sha256(rng_bytes).hexdigest(),
                        "deterministic_physics_claimed": False,
                        "native_qualification": False,
                        "teacher_admission": False,
                    })
                    # The one owned worker can accelerate the legacy encoder during
                    # collection; the unchanged replay schema remains authoritative.
                    with codec_scope(), collector.trace_wall_windows(trace.event):
                        result = run_fresh_update(collector, checkpoint_contract=checkpoint_contract,
                                                  controls=config["rollout_steps"])
            record_budget(collector, lease, "after_update")
        except BaseException:
            try:
                collector.close()
            except BaseException:
                # The collector records cleanup faults in the same journal.
                # R06 needs the original acquisition/update failure for teardown.
                pass
            raise
        collector.close()
        return result


def _run_isaac(attempt, config, lease, *, resume=None):
    from scripts import diagnose_dh116_teacher_initialization as native
    from scripts.diagnose_dh116_teacher_stat_cache_release import make_before_collect
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
    from .tabletop_native_teacher_profile import BackendProfile, checked_file
    from .tabletop_stat_cache_release import stat_cache_source_contract

    if os.environ.get("CUDA_VISIBLE_DEVICES") != attempt.device["cuda_visible_devices"]:
        raise ValueError("actual Worker CUDA visibility differs from its verified device plan")
    device = attempt.device["logical_device"]
    experience = attempt.device["experience_path"]
    checked_file(experience, attempt.device["experience_sha256"])
    profile = BackendProfile.load(attempt.project)
    binary = Path(profile.binary_path).resolve(strict=True)
    stat_cache = stat_cache_source_contract()
    sys.path.insert(0, str(binary.parents[1]))
    from ropesim import _ropesim

    native.validate_native_module(_ropesim, binary, {str(binary): profile.binary_sha256})
    assert_clean_import_boundary()
    native.validate_loaded_project_modules(attempt.project)
    from isaaclab.app import AppLauncher

    argv = sys.argv
    try:
        sys.argv = argv[:1]
        app = AppLauncher(headless=True, device=device, enable_cameras=False,
                          fast_shutdown=False, experience=experience).app
    finally:
        sys.argv = argv
    entered = False
    try:
        import carb.settings
        from isaacsim.core.simulation_manager import SimulationManager
        import isaaclab.sim as sim_utils
        import omni.timeline
        import torch

        from scripts import dh116_pinch_acquisition_v2 as lifecycle_bindings
        from scripts.dh116_fault_safe_lifecycle_v3 import run_lifecycle
        from scripts.dh116_pinch_startup_v3 import StartupTrace, snapshot_for_startup
        from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
        from .tabletop_preview_launcher import commit_preview_stop
        from .tabletop_simulation_owner_cleanup import (
            clear_known_simulation_instance, simulation_source_contract,
        )

        lifecycle = attempt.output / "lifecycle"
        lifecycle.mkdir()
        settings = carb.settings.get_settings()
        cleanup_contract = simulation_source_contract()
        cache_release = make_before_collect(dict(output=str(lifecycle),
            stat_cache_release=dict(source_contract=stat_cache, max_reset_calls=1)))
        sim_reference = None

        def freeze_player():
            settings.set_bool("/app/player/playSimulations", False)
            return settings.get("/app/player/playSimulations")

        def make_sim():
            nonlocal sim_reference
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
                dt=1/480, device=device, render_interval=1, gravity=(0., 0., -9.81),
                physx=sim_utils.PhysxCfg(solver_type=0)))
            sim_reference = ref(sim)
            return sim

        def acquire(sim, owners, trace, cleanup):
            result = _acquire(sim, owners, trace, attempt=attempt, config=config, lease=lease,
                              **({"resume": resume} if resume is not None else {}))
            clock = result["physical"]
            cache_release(sim, expected_sdk_step=clock["end_sdk_step"])
            return dict(result, physics_rows=clock["end_sdk_step"] - clock["start_sdk_step"])

        def read_clock():
            observed = sim_reference() if sim_reference is not None else None
            return native.clock(observed) if observed is not None else None

        def stop():
            result = commit_preview_stop(omni.timeline.get_timeline_interface(),
                                         freeze=freeze_player, read_clock=read_clock)
            native.write_new(lifecycle / "stop_commit.json", result)

        names = tuple(side + "_" + suffix for side in ("left", "right") for suffix in
                      ("hand", "view", "asset_sim_view", "permanent_composer", "instantaneous_composer"))
        with bindings((lifecycle_bindings, "OWNED_NAMES", names + ("runtime", "rope", "episode")),
                      (lifecycle_bindings, "LifetimeTrace", StartupTrace),
                      (lifecycle_bindings, "snapshot_weakrefs", snapshot_for_startup)):
            entered = True
            return run_lifecycle(SimpleNamespace(update=app.update,
                                 close=lambda: app.close(wait_for_replicator=False)),
                make_sim=make_sim, acquire=acquire, synchronize=lambda: torch.cuda.synchronize(device),
                clear_sim=lambda sim: clear_known_simulation_instance(sim, contract=cleanup_contract),
                stop=stop, manager_is_none=lambda: SimulationManager.get_physics_sim_view() is None,
                freeze=freeze_player, output=lifecycle, cleanup="explicit")
    except BaseException as error:
        if not entered:
            try:
                app.close(wait_for_replicator=False)
            except BaseException as cleanup:
                raise RuntimeError(f"Worker setup failed: {error}; App.close failed: {cleanup}") from None
        raise


def run_a(*, config, lease, output):
    """Run one bounded fresh smoke update with the approved process lease."""
    return _run_owned_update(config, lease, output, role="a_fresh")


def run_b(*, config, lease, output):
    """Restore A only after a new authorised Reset, then make one update."""
    return _run_owned_update(config, lease, output, role="b_resume")


def _run_owned_update(config, lease, output, *, role):
    from scripts.diagnose_dh116_teacher_initialization import write_new
    from .tabletop_native_teacher_actual_artifacts import derive_actual_artifacts, derive_actual_resume_artifacts

    resume = None
    if role == "a_fresh":
        attempt = _verify_attempt(config, lease, output)
    else:
        attempt = _verify_attempt(config, lease, output, role=role)
        resume = lease.resume_origin
    attempt.output.mkdir()
    lifecycle = (_run_isaac(attempt, config, lease) if resume is None
                 else _run_isaac(attempt, config, lease, resume=resume))
    if lifecycle.get("close_mode") != "normal" or lifecycle.get("failure") is not None:
        raise RuntimeError("R06 normal in-process close required before recording the update")
    if resume is None:
        derive_actual_artifacts(spec=attempt.spec, output=attempt.output)
    else:
        derive_actual_resume_artifacts(spec=attempt.spec, output=attempt.output, origin=resume)
    result = dict(lifecycle["acquisition"], process_exit_verified=False,
                  native_qualification=False, teacher_admission=False)
    write_new(attempt.output / "owned_update.json", result)
    return result
