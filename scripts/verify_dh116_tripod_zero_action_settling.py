#!/usr/bin/env python3
"""Bounded, no-Actor settling counterfactual over the existing supported Tripod scene.

Zero commands retain controller targets; bodies and rod remain dynamic. This is
never a policy rollout, an expert trajectory, or a grasp qualification.
"""
import argparse
from copy import deepcopy
from functools import partial
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from scripts import record_dh116_tripod_supported_reset as reset_entry

ROOT = reset_entry.ROOT
native = reset_entry.native
VERSION = "dh116-tripod-zero-action-settling-v1"
SCOPE = "zero_action_settling_diagnostic"


def budget(*, controls, wall_seconds):
    if type(controls) is not int or not 1 <= controls <= 30:
        raise ValueError("diagnostic controls must be an integer in [1,30]")
    if type(wall_seconds) not in (float, int) or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 360:
        raise ValueError("finite bounded diagnostic wall seconds required")
    return dict(gpu=2, num_envs=1, mode="tripod_zero_action_settling", sdk_setup_steps=2, priming_sdk_steps=1,
        sdk_steps=3+8*controls, solver_steps=1+8*controls,
        solver_substeps=12*(1+8*controls), diagnostic_control_windows=controls,
        control_steps=controls, policy_windows=0, teacher_queries=0, model_loads=0,
        ppo_updates=0, episode_resets=1, soft_wall_seconds=wall_seconds,
        explicit_app_updates_for_stop=1, auto_retries=0, auto_kill=False)


def run_diagnostic(episode, *, controls, raw_record, enrich, physical_record, remaining_s):
    """Retain every advanced raw row before source/boundary validation can fail."""
    import torch
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode

    limits = budget(controls=controls, wall_seconds=360)
    step = None
    failure = None
    completed = 0
    returned_rows = persisted_rows = source_valid_rows = 0
    try:
        if (type(episode) is not DualHandPhysicalEpisode or episode._state != "active"
                or episode._control_index != 0 or getattr(episode, "_settling_attempted", False)
                or episode._sim.current_time_step_index != 2 or episode._rod.clock()["rod_tick"] != 0
                or episode._end < limits["sdk_steps"]):
            raise ValueError("fresh Reset2 episode and complete independent settling budget required")
        if not all(callable(x) for x in (raw_record, enrich, physical_record, remaining_s)):
            raise ValueError("durable raw/source recorders and wall clock required")
        episode._settling_attempted = True
        step = episode._exchange._step
        def measured(action, substep):
            nonlocal returned_rows, persisted_rows, source_valid_rows
            remaining = remaining_s()
            if not math.isfinite(remaining) or remaining <= 0:
                raise RuntimeError("diagnostic wall budget exhausted before next SDK step")
            if bool(torch.count_nonzero(action)):
                raise ValueError("diagnostic may issue only original zero 24D commands")
            before = deepcopy(episode._rod.clock())
            if before["physics_step"] >= limits["sdk_steps"]:
                raise RuntimeError("independent diagnostic SDK budget exhausted")
            row = step(action, substep)
            returned_rows += 1
            # Raw returned buffers survive getter, enrichment, boundary, and guard errors.
            raw_record(deepcopy(row))
            issued = torch.as_tensor(row["action_readback"]["request"]["action24"])
            if (issued.shape != (1, 24) or not issued.is_floating_point()
                    or not bool(torch.isfinite(issued).all()) or bool(torch.count_nonzero(issued))):
                raise ValueError("actual issued request must remain finite float zero 24D")
            clock = episode._rod.clock()
            if (clock["physics_step"] != before["physics_step"]+1
                    or clock["rod_tick"] != before["rod_tick"]+12
                    or row["physics_step"] != clock["physics_step"]):
                raise ValueError("diagnostic SDK/solver advancement differs")
            source_error = None
            try:
                enrich(row)
            except Exception as error:
                source_error = type(error).__name__ + ": " + str(error)
                row["diagnostic_source_error"] = source_error
                error.__traceback__ = None
            physical_record(deepcopy(row))
            persisted_rows += 1
            if source_error is not None:
                raise RuntimeError(source_error)
            source_valid_rows += 1
            return row
        episode._exchange._step = measured
        episode._state = "failed"
        row = measured(torch.zeros(1, 24, dtype=torch.float64), 0)
        packet = episode._boundary.capture(row["rod"])
        episode._record(dict(kind="diagnostic_priming_physical", exchange=row,
            rod_boundary=deepcopy(packet), on_policy=False, teacher_admission=False))
        episode._rod_packet = packet
        if row["guard_failure"]:
            raise RuntimeError(row["guard_failure"])
        episode._state = "active"
        for _ in range(controls):
            result = episode.control(torch.zeros(1, 24, dtype=torch.float64))
            if result["termination_reason"]:
                raise RuntimeError(result["termination_reason"])
            completed += 1
    except Exception as error:
        failure = type(error).__name__ + ": " + str(error)
        error.__traceback__ = None
        episode._state = "failed"
    finally:
        if step is not None:
            episode._exchange._step = step
    # Actual clocks include partial progress even when the native call never returned.
    clock = episode._rod.clock()
    ticks = clock["rod_tick"]
    result = dict(version=VERSION, diagnostic_scope=SCOPE, on_policy=False,
        sdk_steps=episode._sim.current_time_step_index, solver_steps=(ticks+11)//12,
        solver_substeps=ticks, diagnostic_control_windows=completed,
        returned_solver_rows=returned_rows, persisted_physics_rows=persisted_rows,
        source_valid_rows=source_valid_rows,
        solver_completion_verified=(ticks == 12*returned_rows),
        policy_windows=0, model_loads=0, teacher_queries=0, ppo_updates=0,
        failure=failure, native_qualification=False, teacher_admission=False, grasp_qualified=False)
    if failure:
        result["guarded_stop_sdk_step"] = result["sdk_steps"]
    return result


class _PhysicsSink:
    """Exclusive fsynced raw schema understood by the existing contact auditor."""
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")
        self.sequence = 0

    def write(self, exchange):
        from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import encode_owned
        row = dict(raw_sequence=self.sequence, payload=dict(exchange=exchange,
            diagnostic_scope=SCOPE, on_policy=False, teacher_admission=False))
        self.stream.write(json.dumps(encode_owned(row), allow_nan=False)+"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1

    def close(self):
        self.stream.close()


def freeze_plan(output, *, manifest_path, manifest_sha256, controls, wall_seconds):
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import load_tripod_reset_manifest
    manifest = load_tripod_reset_manifest(dict(style="tripod",
        resolved_reset_manifest=dict(path=str(manifest_path), sha256=manifest_sha256)), project=ROOT)
    data = manifest.data()
    plan = reset_entry.freeze_plan(output, task_config=Path(data["task_config"]["path"]),
        task_config_sha256=data["task_config"]["sha256"], extra_sources=data["extra_sources"])
    plan.update(version=VERSION, diagnostic_scope=SCOPE, on_policy=False,
        reset_manifest=dict(path=manifest.path, sha256=manifest.sha256),
        preserved_scope="same supported17 Reset2/S0; no reward, Actor, PPO, camera, or task_start change")
    plan["budget"].update(budget(controls=controls, wall_seconds=wall_seconds))
    plan["frozen_files"][manifest.path] = manifest.sha256
    plan["frozen_files"][str(Path(__file__).resolve())] = native.sha(Path(__file__))
    return plan


def acquire(sim, owners, trace, output, *, plan, deadline):
    from go2_pvcnn.control.dh116.tabletop_fresh_supported_builder import build_fresh_supported_owned
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_numeric_state import numeric_reset_state_scope
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, profile_reader_scope
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import load_tripod_reset_manifest
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import NativeParameterEndpoints
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_tripod_distance import PolicyOwnedTripodGeometry
    from go2_pvcnn.control.dh116.tabletop_native_teacher_solver_health import capture_teacher_solver_health
    from go2_pvcnn.control.dh116.tabletop_native_teacher_contact_measurement import capture_supported_contact
    from go2_pvcnn.control.dh116.tabletop_mesh_observation_evidence import _write
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as episode_module
    from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import ClearanceGuardedExchange
    import numpy as np

    fresh = episode = reader = geometry = rod = raw = physical = record = enrich = None
    try:
        expected = budget(controls=plan["budget"]["diagnostic_control_windows"],
                          wall_seconds=plan["budget"]["soft_wall_seconds"])
        if any(plan["budget"].get(k) != v for k, v in expected.items()):
            raise ValueError("exact independent diagnostic budget required")
        profile = BackendProfile.load(ROOT)
        if profile.digest != plan["backend_profile_sha256"]:
            raise ValueError("current native backend profile differs")
        manifest = load_tripod_reset_manifest(dict(style="tripod",
            resolved_reset_manifest=plan["reset_manifest"]), project=ROOT)
        data = manifest.data()
        run = "tripod-zero-settling-"+uuid4().hex
        episode_uid = run+":0"
        def record(row):
            trace.event("zero_action_episode", kind=row["kind"])
            if row["kind"] == "reset":
                _write(output/"episode_reset.json", row)
        fresh = build_fresh_supported_owned(sim, owners, project=ROOT, prepared=None,
            ownership=trace, record=record, backend_profile=profile,
            task_config=Path(plan["task_config"]["path"]),
            task_config_sha256=plan["task_config"]["sha256"], extra_sources=plan["extra_sources"],
            run_uuid=run, episode_uid=episode_uid, physics_budget_end=expected["sdk_steps"])
        episode = fresh.episode
        rod = episode._rod
        if fresh.reset_seed_sha256 != data["reset_seed"]["sha256"]:
            raise ValueError("existing qualified Reset seed differs")
        with profile_reader_scope(profile), numeric_reset_state_scope(sim=sim, rod=rod,
                record=lambda row: _write(output/"mesh_reset_numeric_state.json", row)):
            with bindings((episode_module, "FlexiblePhysicalExchange", ClearanceGuardedExchange)):
                result = episode.reset(fresh.reset_seed, episode_uid=episode_uid,
                    owner_resets=diagnostic_reset_owners(), supported_mesh_plan=fresh.mesh_plan,
                    record_reset_readback=lambda row: _write(output/"mesh_reset_state.json",
                        dict(readback=row, mesh_plan=fresh.mesh_plan)))
        _write(output/"reset_geometry.json", list(fresh.initial_geometry()))
        _write(output/"pre_action_common.json", episode.pre_action_state()["common"])
        with profile_reader_scope(profile):
            reader = NativeParameterEndpoints(np.asarray(data["rod_state"]["positions"], np.float32)[0],
                device=str(episode._bindings[0].hand.data.joint_pos.device))
            reader.capture_reset(result["reset"]["rod_reset"], material_reference_uid=episode_uid+":material41",
                rope=rod.rope, backend=sys.modules["ropesim._ropesim"], sim=sim, read_clock=rod.clock)
            geometry = PolicyOwnedTripodGeometry(ROOT, scope="owner_source_unqualified",
                task_config=Path(plan["task_config"]["path"]),
                task_config_sha256=plan["task_config"]["sha256"], mesh_bindings=rod.bindings)
        def enrich(row):
            with profile_reader_scope(profile):
                row["material_director"] = reader.capture(row["rod"], rope=rod.rope,
                    backend=sys.modules["ropesim._ropesim"], sim=sim, read_clock=rod.clock)
                row["original_guard_failure"] = row["guard_failure"]
                row["closure_geometry"] = geometry.capture(row, sim=sim,
                    bindings=episode._bindings, read_clock=rod.clock)
                health = capture_teacher_solver_health(rod, backend=sys.modules["ropesim._ropesim"],
                    sim=sim, expected_clock=rod.clock(), grasp="tripod")
                row["native_teacher_health"] = health
                contact = capture_supported_contact(row["rod"], grasp="tripod")
                row["diagnostic_contact"] = contact
                reasons = row["closure_geometry"]["failure_reasons"]
                if not row["closure_geometry"]["source_complete"]:
                    raise ValueError("complete real Tripod pad geometry unavailable")
                if reasons and row["guard_failure"] is None:
                    row["guard_failure"] = row["closure_geometry"]["terminal_reason"]
                if health["last_error"] != 0 and row["guard_failure"] is None:
                    row["guard_failure"] = "native_numerical_health"
                if contact["table_contact_failure"] and row["guard_failure"] is None:
                    row["guard_failure"] = "native_teacher_table_contact"
        raw = _PhysicsSink(output/"raw_physics.jsonl")
        physical = _PhysicsSink(output/"physics.jsonl")
        outcome = run_diagnostic(episode, controls=expected["diagnostic_control_windows"],
            raw_record=raw.write, enrich=enrich, physical_record=physical.write,
            remaining_s=lambda: deadline-time.monotonic())
        outcome["reset_seed_sha256"] = fresh.reset_seed_sha256
        outcome["reset_manifest"] = plan["reset_manifest"]
        native.write_new(output/"settling_capture.json", outcome)
        return outcome
    finally:
        if raw is not None:
            raw.close()
        if physical is not None:
            physical.close()
        sim = owners = trace = fresh = episode = reader = geometry = rod = raw = physical = record = enrich = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "plan", "reset-manifest"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reset-manifest-sha256", required=True)
    parser.add_argument("--controls", type=int, default=30)
    parser.add_argument("--wall-seconds", type=float, default=360)
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--verify-existing-plan", action="store_true")
    args = parser.parse_args(argv)
    native.validate_execution_context(ROOT)
    params = dict(manifest_path=args.reset_manifest, manifest_sha256=args.reset_manifest_sha256,
                  controls=args.controls, wall_seconds=args.wall_seconds)
    if args.verify_existing_plan or args.approved_frozen_sha256 is None:
        if args.verify_existing_plan and args.approved_frozen_sha256 is not None:
            parser.error("CPU verifier cannot execute an approved plan")
        plan = freeze_plan(args.output, **params)
        if args.verify_existing_plan:
            if json.loads(args.plan.read_bytes()) != plan or args.output.exists():
                raise ValueError("independently rebuilt plan or exclusive output differs")
        else:
            native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=native.digest(plan), budget=plan["budget"])))
        return 0
    plan = json.loads(args.plan.read_bytes())
    if native.digest(plan) != args.approved_frozen_sha256 or args.output.exists():
        raise ValueError("exact independent frozen plan and exclusive output required")
    command = [sys.executable, "-B", "-m", "scripts.verify_dh116_tripod_zero_action_settling",
        "--verify-existing-plan", "--output", str(args.output), "--plan", str(args.plan),
        "--reset-manifest", str(args.reset_manifest), "--reset-manifest-sha256", args.reset_manifest_sha256,
        "--controls", str(args.controls), "--wall-seconds", str(args.wall_seconds)]
    subprocess.run(command, check=True)
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    for path, digest in plan["frozen_files"].items():
        checked_file(path, digest)
    if json.loads(args.plan.read_bytes()) != plan or args.output.exists():
        raise ValueError("frozen plan changed after CPU verification")
    deadline = time.monotonic()+plan["budget"]["soft_wall_seconds"]
    reset_entry.lifecycle.execute(plan, acquire_operation=partial(acquire, plan=plan, deadline=deadline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
