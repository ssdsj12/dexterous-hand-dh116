"""Freeze and supervise one no-camera right Tripod tracking diagnostic."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from scripts import record_dh116_teacher_numeric_reset as reset
from scripts.verify_dh116_tripod_right_tracking import (
    BUDGET, TrackingStopped, run_owned_tracking, validate_schedule,
)


ROOT = reset.ROOT
native = reset.native
SCHEMA = "dh116-right-tripod-owned-native-v1"
TRACKING_SHA = "e419ed68ba6cff54e5ecdf0b679a2b72550851519d918a785c293677c496b112"
WALL_SECONDS = 180
CLOSE_RESERVE_SECONDS = 10


def freeze_plan(output, tracking):
    tracking = Path(tracking).resolve(strict=True)
    if native.sha(tracking) != TRACKING_SHA:
        raise ValueError("frozen right Tripod reference bytes differ")
    validate_schedule(json.loads(tracking.read_bytes()))
    plan = reset.freeze_plan(output)
    path = str(tracking)
    plan["frozen_files"][path] = TRACKING_SHA
    plan["budget"].update(mode="right_tripod_noncontact_tracking", sdk_steps=BUDGET["sdk_steps"],
                          control_steps=22, solver_steps=2112, soft_wall_seconds=WALL_SECONDS,
                          close_reserve_seconds=CLOSE_RESERVE_SECONDS)
    plan.update(version=SCHEMA, tracking_plan=dict(path=path, sha256=TRACKING_SHA),
                preserved_scope="free grounded rod with left parked and right-only tracking")
    return plan


def claim_attempt(output, fingerprint, command):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    path = output.with_name(output.name + "-attempt.json")
    native.write_new(path, dict(schema=SCHEMA, fingerprint=fingerprint, argv=command,
        started_monotonic=time.monotonic(), supervisor_pid=os.getpid(), auto_retry=False, auto_kill=False))
    return path


def validate_outer(output, *, exit_code, wall_seconds):
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence

    if exit_code != 0:
        raise ValueError("right Tripod worker failed")
    if not 0 <= wall_seconds <= WALL_SECONDS:
        raise ValueError("right Tripod exceeded the wall budget")
    output = Path(output)
    names = ("result.json", "cleanup_stages.json", "app_close_returned.json",
             "stat_cache_release.json", "physical.jsonl")
    if not all((output / name).is_file() for name in names):
        raise ValueError("complete natural-close and physical receipts required")
    result, cleanup, closed, cache = [json.loads((output / name).read_bytes()) for name in names[:-1]]
    acquisition = result["acquisition"]
    if (result["failure"] is not None or cleanup["failure"] is not None
            or any(packet["close_mode"] != "normal" for packet in (result, cleanup, closed))
            or closed["app_close_returned"] is not True
            or result["lifetime_released_before_close"] is not True
            or acquisition["physics_rows"] != 176 or acquisition["control_steps"] != 22
            or acquisition["ppo_updates"] != 0
            or cache["clock_before"]["sdk_step"] != BUDGET["sdk_steps"]):
        raise ValueError("natural-close or right Tripod acquisition receipt differs")
    ledger = inspect_evidence(output / "physical.jsonl")
    if (ledger["physics_budget_end"] != 178 or ledger["last_sdk_step"] != 178
            or ledger["measured_physics_rows"] != 176 or ledger["settled_control_windows"] != 22
            or ledger["terminal_reason"] is not None or ledger["unsettled_control_window"]):
        raise ValueError("right Tripod physical receipt is incomplete")
    return {name: native.sha(output / name) for name in names}


def _worker(plan, fingerprint, claim_path):
    output = Path(plan["output"])
    claim = json.loads(Path(claim_path).read_bytes())
    if (claim_path != output.with_name(output.name + "-attempt.json")
            or claim["schema"] != SCHEMA or claim["fingerprint"] != fingerprint
            or claim["supervisor_pid"] != os.getppid() or output.exists()):
        raise ValueError("live one-shot right Tripod supervisor claim required")
    native.write_new(output.with_name(output.name + "-worker.json"),
        dict(pid=os.getpid(), fingerprint=fingerprint))
    deadline = claim["started_monotonic"] + WALL_SECONDS
    if time.monotonic() >= deadline:
        raise TimeoutError("right Tripod wall budget reached before App startup")
    seed = plan["tracking_plan"]
    run_uuid = "tripod-right-tracking-" + uuid4().hex

    def acquire(sim, owners, trace, destination):
        try:
            return run_owned_tracking(sim, owners, trace, project=ROOT,
                plan_path=seed["path"], plan_sha256=seed["sha256"], output=destination,
                run_uuid=run_uuid, episode_uid=run_uuid + ":0",
                soft_deadline=deadline - CLOSE_RESERVE_SECONDS)
        except TrackingStopped as error:
            return dict(failure=str(error), guarded_stop_sdk_step=native.clock(sim)["sdk_step"],
                        teacher_admission=False, grasp_qualified=False)
        finally:
            sim = owners = trace = None

    reset.execute(plan, acquire_operation=acquire)


def _supervise(args, plan, fingerprint):
    output = Path(plan["output"])
    claim = output.with_name(output.name + "-attempt.json")
    command = [sys.executable, "-B", "-m", "scripts.launch_dh116_tripod_right_tracking",
               "--output", str(output), "--tracking-plan", plan["tracking_plan"]["path"],
               "--plan", str(args.plan.resolve(strict=True)), "--approved-frozen-sha256", fingerprint,
               "--worker-claim", str(claim)]
    claim_attempt(output, fingerprint, command)
    started = json.loads(claim.read_bytes())["started_monotonic"]
    log = output.with_name(output.name + "-native.log")
    with log.open("xb") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.flush()
        os.fsync(stream.fileno())
    elapsed = time.monotonic() - started
    try:
        artifacts = validate_outer(output, exit_code=completed.returncode, wall_seconds=elapsed)
        failure = None
    except (ValueError, OSError, KeyError) as error:
        artifacts = {name: native.sha(output / name)
                     for name in ("result.json", "cleanup_stages.json", "physical.jsonl",
                                  "app_close_returned.json", "fault_app_close_returned.json")
                     if (output / name).is_file()}
        failure = str(error)
    receipt = dict(schema=SCHEMA, fingerprint=fingerprint, output=str(output),
                   exit_code=completed.returncode, wall_seconds=elapsed, failure=failure,
                   log_sha256=native.sha(log), artifact_sha256=artifacts,
                   tracking_verified=failure is None, teacher_admission=False, grasp_qualified=False,
                   ppo_updates=0, auto_retry=False, auto_kill=False)
    native.write_new(output.with_name(output.name + "-outer.json"), receipt)
    print(json.dumps(receipt), flush=True)
    return int(failure is not None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "tracking-plan", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--worker-claim", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    native.validate_execution_context(ROOT)
    plan = freeze_plan(args.output, args.tracking_plan)
    fingerprint = native.digest(plan)
    if not args.approved_frozen_sha256:
        if args.worker_claim:
            parser.error("worker requires a frozen launch fingerprint")
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=fingerprint, budget=plan.get("budget"))))
        return 0
    if (args.approved_frozen_sha256 != fingerprint or json.loads(args.plan.read_bytes()) != plan
            or Path(plan["output"]).exists()):
        raise ValueError("exact frozen Tripod plan and a fresh output are required")
    if args.worker_claim:
        _worker(plan, fingerprint, args.worker_claim)
        return 0
    return _supervise(args, plan, fingerprint)


if __name__ == "__main__":
    raise SystemExit(main())
