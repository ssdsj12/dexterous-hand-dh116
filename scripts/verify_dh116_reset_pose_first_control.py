#!/usr/bin/env python3
"""Verify one original Pinch control window on the opt-in Reset-pose backend."""
import argparse
from functools import partial
import json
from pathlib import Path
from uuid import uuid4

from scripts import record_dh116_teacher_numeric_reset as reset

ROOT = reset.ROOT
NATIVE = reset.native
BACKEND_CONTRACT = ROOT / "docs/contracts/dh116_deformx_zero_time_reset_pose_v1.json"
SDK_END = 10


def bind_candidate(plan, binary, source_root):
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as baseline

    candidate = json.loads(BACKEND_CONTRACT.read_bytes())
    binary = binary.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    if candidate["baseline_native_backend_sha256"] != baseline.BACKEND_SHA:
        raise ValueError("Reset-pose patch must extend the exact locked Teacher observation backend")
    if binary.name != baseline.BACKEND.name or NATIVE.sha(binary) != candidate["candidate_backend_sha256"]:
        raise ValueError("exact independent Reset-pose CUDA backend required")
    baseline_sources = json.loads(baseline.PREPARATION.read_bytes())["prepared_source_sha256"]
    deltas = candidate["exact_source_deltas"]
    changed = dict(baseline_sources)
    for name, row in deltas.items():
        if changed[name] != row["baseline"]:
            raise ValueError("candidate solver source does not descend from the Teacher build")
        changed[name] = row["candidate"]
    inventory = {source_root/name: digest for name, digest in changed.items()}
    inventory.update({binary: candidate["candidate_backend_sha256"],
                      BACKEND_CONTRACT: NATIVE.sha(BACKEND_CONTRACT),
                      ROOT/candidate["patch"]: candidate["patch_sha256"]})
    for name in ("__init__.py", "isaac_bridge.py", "sdf.py"):
        path = source_root/"ropesim"/name
        inventory[binary.parent/name] = inventory[path]
    for path, expected in inventory.items():
        if path.is_symlink() or NATIVE.sha(path) != expected:
            raise ValueError(f"independent solver artifact differs: {path}")
        plan["frozen_files"][str(path.resolve(strict=True))] = expected
    return [str(binary), candidate["candidate_backend_sha256"]]


def acquire(sim, owners, trace, output, *, plan, commit):
    import traceback
    import torch
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from scripts import audit_dh116_pinch_initial_pose as geometry
    from scripts.audit_dh116_certified_segment_distance import segment_hull_distance
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners

    fresh = episode = rod = None
    count = 0

    def record(row):
        nonlocal count
        count += 1
        NATIVE.write_new(output/f"event_{count:03}.json", encode_owned(row))

    try:
        pair = plan["pair_proposal"]
        seed = builder.load_bound_pair_proposal(pair)["reset_seed"]
        run = "reset-pose-first-control-" + uuid4().hex
        with bindings((geometry, "segment_hull_distance", segment_hull_distance)):
            fresh = builder.build_fresh_owned(sim, owners, project=ROOT, prepared=None,
                ownership=trace, record=record, run_uuid=run, episode_uid=run+":0",
                physics_budget_end=SDK_END, expected_reset_seed=seed, pair_proposal=pair,
                diagnostic_backend=plan["diagnostic_reset_pose_backend"])
            episode = fresh.episode
            episode.reset(seed, episode_uid=run+":0", owner_resets=diagnostic_reset_owners())
            for item in fresh.initial_geometry():
                record(item)
        rod = owners.resources["runtime"]
        clock = rod.clock()
        if NATIVE.clock(sim)["sdk_step"] != 2 or clock["rod_tick"] != 0:
            raise ValueError("colliders must be bound at the actual zero-step Reset")
        receipt = rod.commit_reset_mesh_poses(expected_clock=clock) if commit else None
        NATIVE.write_new(output/"reset_pose_commit.json", dict(enabled=commit, clock=clock, result=receipt))
        if NATIVE.clock(sim)["sdk_step"] != 2 or rod.clock() != clock:
            raise ValueError("collider commit unexpectedly advanced the clocks")
        action = torch.tensor(plan["action24"], dtype=torch.float64)[None]
        result = episode.control(action)
        NATIVE.write_new(output/"first_control.json", encode_owned(dict(action=action, result=result, clock=rod.clock())))
        if result["termination_reason"] or result["physical_samples"] != 8 or rod.clock()["rod_tick"] != 96:
            raise RuntimeError(f"original first control terminated: {result['termination_reason']}")
        return dict(version=plan["version"], sdk_steps=10, solver_steps=96,
                    control_steps=1, ppo_updates=0, commit=commit, native_qualification=False)
    except Exception:
        step = NATIVE.clock(sim)["sdk_step"]
        if not 2 <= step <= SDK_END:
            raise
        evidence = dict(failure=traceback.format_exc(), guarded_stop_sdk_step=step, ppo_updates=0)
        NATIVE.write_new(output/"first_control_failure.json", evidence)
        return evidence
    finally:
        sim = owners = trace = fresh = episode = rod = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pair-proposal", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--treatment", choices=("commit", "legacy"), required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    NATIVE.validate_execution_context(ROOT)
    pair = dict(path=str(args.pair_proposal.resolve(strict=True)), sha256=args.pair_sha256)
    plan = reset.freeze_plan(args.output, pair_proposal=pair)
    plan["diagnostic_reset_pose_backend"] = bind_candidate(plan, args.candidate_binary, args.candidate_source)
    plan.update(version="dh116-pinch-reset-pose-first-control-v1", treatment=args.treatment,
                action24=[.1 if i in (12, 15) else 0. for i in range(24)])
    plan["budget"].update(sdk_steps=10, solver_steps=96, control_steps=1,
                          teacher_queries=0, ppo_updates=0, soft_wall_seconds=180,
                          auto_retries=0, auto_kill=False)
    digest = NATIVE.digest(plan)
    if args.approved_frozen_sha256 is None:
        NATIVE.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
        return 0
    if (args.approved_frozen_sha256 != digest
            or NATIVE.digest(json.loads(args.plan.read_bytes())) != digest):
        raise ValueError("frozen first-control recipe differs")
    reset.execute(plan, acquire_operation=partial(acquire, plan=plan, commit=args.treatment=="commit"),
                  diagnostic_backend=plan["diagnostic_reset_pose_backend"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
