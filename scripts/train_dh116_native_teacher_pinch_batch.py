#!/usr/bin/env python3
"""GPU-local native Isaac Pinch Teacher PPO run with independent rods."""
import argparse
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from scripts.audit_dh116_isaac_batch_scene import _gpu_memory, _run, load_grasp_archive
from scripts.smoke_dh116_teacher_batch_dynamic import lane_bindings, reset_lane_exchange


def _check_smoke(path, digest):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file

    outer = json.loads(checked_file(path, digest))
    if (outer["outer_gdb_exit_code"] != 0 or outer["forced_exit"]
            or outer["debugger_inferior_status"] != dict(exit_code=0, signal=None)):
        raise ValueError("natural native two-lane smoke exit 0 required")
    directory = path.parent / "native"
    result = json.loads(checked_file(directory / "dynamic_result.json", outer["artifacts"]["dynamic_result.json"]))
    lanes = result["budget"]["lanes"]
    if (result["environments"] != 2 or result["ppo_updates"] != 0
            or result["nonzero_routed_hand_steps"]["0"] <= 0 or result["nonzero_routed_hand_steps"]["1"] != 0
            or lanes["0"]["resets"] != 2 or lanes["1"]["resets"] != 1
            or lanes["0"]["rod_tick"] != 96 or result["sdk_step"] != result["budget"]["sdk_step"]):
        raise ValueError("actual separated hand reaction, local Reset and a subsequent complete window required")


def _exercise(sim, scenes, seed, prepared, archive, nodes, simulator_owners, output, gpu, end, *, contract):
    from contextlib import ExitStack
    import math

    from go2_pvcnn.control.dh116.tabletop_native_teacher_batch_episode import NativeBatchPhysicalEpisode
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_provider import _fresh_reset_owners
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_update import _seal_checkpoint
    from go2_pvcnn.control.dh116.tabletop_native_teacher_batch import BatchBudget, NativeTeacherBatch
    from go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint import build_training_contract
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    from go2_pvcnn.control.dh116.tabletop_native_teacher_model import NativeContextTeacher
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest, profile_reader_scope
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import NativeEpisodeOwner
    from scripts.audit_dh116_batch_rod_exchange import _build_rods
    from scripts.diagnose_dh116_teacher_initialization import write_new

    rods = _build_rods(scenes, archive, nodes, simulator_owners)
    profile = contract.backend_profile
    budget = BatchBudget(range(len(rods)), start_sdk_step=sim.current_time_step_index, max_sdk_step=end)
    journals = [NativeJournal(output / f"lane_{i:03d}.jsonl") for i in range(len(rods))]
    group_log = NativeJournal(output / "training.jsonl")
    model = NativeContextTeacher(contract.config.model_config, backend_profile=profile)
    trainer = DeformablePPOTrainer(model, contract.ppo_config, rollout_steps=contract.config.rollout_steps,
                                   num_envs=len(scenes))
    with ExitStack() as cleanup:
        cleanup.callback(group_log.close)
        for journal in journals:
            cleanup.callback(journal.close)

        def reset(env_id, old):
            rod, scene, journal = rods[env_id], scenes[env_id], journals[env_id]
            previous = rod.clock() if old is not None else None
            if old is not None:
                old.context.close()
                old.close()
            bindings = lane_bindings(rod, scene)
            ep = NativeBatchPhysicalEpisode(sim, rod, bindings, backend_artifact=Path(profile.binary_path),
                backend_sha256=profile.binary_sha256, inspect_geometry=lambda: {}, physics_budget_end=end,
                record=lambda _row: None)
            events = () if env_id != 0 or old is not None else (
                dict(kind="setup", phase="before_reset", sdk_step=0, sdk_time_s=0.),
                dict(kind="setup", phase="after_reset", sdk_step=sim.current_time_step_index,
                     sdk_time_s=float(sim.current_time)),
            )
            uid = f"{rod.run_uuid}:{env_id}:{rod.clock()['reset_id'] + 1}"
            owner = NativeEpisodeOwner(ep, contract, episode_uid=uid, owner_resets=_fresh_reset_owners(),
                mesh_plan=archive.mesh_plan, ledger=journal, setup_events=events,
                initial_geometry=lambda: (), previous_reset=previous, batch_lane=True)
            cleanup.callback(owner.close)
            exchange, receipt = reset_lane_exchange(sim, rod, scene, seed, archive, prepared,
                end=end, owner_resets=owner.owner_resets, bindings=bindings)
            event = ep.install_batch_reset(exchange, receipt)
            measured = exchange._read_geometry()
            for side, clearance in zip(("left", "right"), measured["clearance_m"][0], strict=True):
                row = dict(kind="initial_geometry", side=side, evidence=dict(side=side,
                    identity=dict(sdk_step=sim.current_time_step_index, sdk_time_s=float(sim.current_time)),
                    clearance_m=float(clearance), source=measured["source"]))
                owner.ledger.emit("initial_geometry", row)
                owner.history.record(row)
            with profile_reader_scope(profile):
                owner.sources.after_reset(event)
                owner.initialize_context_and_reward_source(event)
            return owner

        batch = NativeTeacherBatch([reset(i, None) for i in range(len(rods))], trainer,
                                   reset_lane=reset, journal=group_log, budget=budget)
        peak = _gpu_memory(gpu)
        times = []
        for _ in range(contract.config.rollout_steps):
            window = batch.collect()
            times.append(window["window_seconds"])
            peak = max(peak, _gpu_memory(gpu))
            message = dict(grasp="pinch", gpu=gpu, environments=len(rods), sdk_step=budget.sdk_step,
                           window_seconds=times[-1], peak_observed_gpu_mib=peak)
            print("DH116_TRAINING_WINDOW", json.dumps(message), flush=True)
        before = [p.detach().double().clone() for p in model.parameters()]
        metrics = batch.update()
        delta = sum(float((p.detach().double() - old).square().sum())
                    for p, old in zip(model.parameters(), before, strict=True))
        if (trainer.counters != dict(updates=1, environment_steps=len(rods) * contract.config.rollout_steps)
                or metrics["minibatch_updates"] <= 0 or not math.isfinite(delta) or delta <= 0.):
            raise RuntimeError("complete accepted original PPO optimizer update required")
        trainer._native_completed_update = trainer.counters["updates"]
        trainer._native_accepted_minibatches = metrics["minibatch_updates"]
        ck_contract = build_training_contract(trainer, scope="owner_source_unqualified",
            reward_sha256=contract.reward_config.config_sha256, reset_sha256=contract.reset_manifest.sha256,
            target_sha256=digest(contract.reset_manifest.data()["target_material_ids"]),
            source_sha256=contract.source_sha256,
            config_sha256=digest(dict(ppo=asdict(trainer.config), environments=len(rods),
                                      rollout_steps=trainer.rollout_steps)))
        checkpoint = _seal_checkpoint(output / "teacher_pinch_batch64_update_1.pt", trainer, ck_contract)
        result = dict(sdk_step=budget.sdk_step, budget=budget.snapshot(), environments=len(rods),
                      ppo_updates=trainer.counters["updates"], metrics=metrics,
                      parameter_delta_l2=math.sqrt(delta), peak_observed_gpu_mib=peak,
                      window_seconds=times, checkpoint_sha256=checkpoint["checkpoint_sha256"])
        write_new(output / "ppo_result.json", result)
        print("DH116_PPO_UPDATE", json.dumps(result), flush=True)
        return result


def main():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_mesh_observation_source import BACKEND, BACKEND_SHA
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import load_reset_manifest
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, checked_file
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import SessionBudget
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    from scripts.run_dh116_work_packages import NATIVE_LOCK, native_file_lock
    from scripts.diagnose_dh116_teacher_initialization import write_new

    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("receipt", "smoke", "training-config", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    for flag in ("receipt-sha256", "smoke-sha256", "training-config-sha256"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("exclusive visible physical GPU required")
    project = Path(__file__).resolve().parents[1]
    _check_smoke(args.smoke, args.smoke_sha256)
    archive = load_grasp_archive("pinch", args.receipt, args.receipt_sha256)
    cfg = json.loads(checked_file(args.training_config, args.training_config_sha256))
    manifest = load_reset_manifest(cfg["preflight"], project=project)
    if manifest.data()["reset_seed"] != archive.pair["reset_seed"]:
        raise ValueError("Pinch physical archive and training Reset do not share a seed")
    cfg["project"] = project
    cfg["rollout_steps"] = 32
    reward = load_config(project / "configs/dh116_training/pinch_finger_goal_reward_v1.json", ppo_gamma=.995)
    if reward.finger_goal.grasp != "pinch" or any(
            reward.ppo[key] != value for key, value in (
             ("learning_rate", .0001), ("gamma", .995), ("target_kl", .02), ("kl_backtrack_steps", 3))):
        raise ValueError("unchanged Pinch finger-goal PPO recipe required")
    p = reward.ppo
    ppo = PPOConfig(lr=p["learning_rate"], gamma=p["gamma"], lam=p["gae_lambda"], clip=p["clip"],
        epochs=p["epochs"], minibatches=p["minibatches"], value_coef=p["value"], entropy_coef=p["entropy"],
        max_grad_norm=p["max_grad_norm"], target_kl=p["target_kl"], kl_backtrack_steps=p["kl_backtrack_steps"])
    end = 2 + 8 * cfg["rollout_steps"]
    limits = dict(max_sdk_steps=end, max_solver_calls=end - 2, wall_timeout_s=3600.)
    tracked_sources = [Path(__file__), Path(__file__).with_name("smoke_dh116_teacher_batch_dynamic.py"),
         project / "src/go2_pvcnn/control/dh116/tabletop_native_teacher_batch.py",
         project / "src/go2_pvcnn/control/dh116/tabletop_native_teacher_batch_episode.py"]
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in tracked_sources}
    contract = SimpleNamespace(config=SimpleNamespace(**cfg), backend_profile=BackendProfile.load(project),
        reset_manifest=manifest, reward_config=reward, ppo_config=ppo, budget=limits, runtime=SessionBudget(limits),
        source_sha256=hashes)
    checked_file(BACKEND, BACKEND_SHA)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu) > 512:
            raise RuntimeError("physical GPU is already occupied before serial PPO")
        args.output.mkdir(exist_ok=False)
        write_new(args.output / "run_plan.json", dict(grasp="pinch", environments=64, gpu=args.gpu,
            sdk_budget=end, rollout_steps=32, additional_updates=1, ppo=asdict(ppo), source_sha256=hashes,
            smoke_sha256=args.smoke_sha256))
        sys.path.insert(0, str(BACKEND.parents[1]))
        from ropesim import _ropesim
        if Path(_ropesim.__file__).resolve() != BACKEND:
            raise ValueError("native training loaded a different DeformX binary")
        _run(project, archive, args.gpu, 64, args.output, coupled=True, max_sdk_step=end,
             extra_exercise=partial(_exercise, contract=contract))


if __name__ == "__main__":
    main()
