#!/usr/bin/env python3
"""Collect a small DH116 hand-only teacher/student replay.

This is an offline smoke/training data collector, not a hardware driver.  It
starts at grasp (there is intentionally no approach phase) and then lift.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

from go2_pvcnn.control.dh116.hand_only_env import HandOnlySpec
from go2_pvcnn.control.dh116.teacher_student import (
    make_trajectory_checkpoint, make_transition, write_replay,
)
from go2_pvcnn.control.dh116.parallel_mpc import ParallelJointMPC


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--steps-per-episode", type=int, default=24)
    p.add_argument("--initial-action", type=float, nargs=12, default=[0.0] * 12,
                   metavar=("J0", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11"),
                   help="previous 12D active-joint command used as the first MPC reference")
    p.add_argument("--teacher-noise", type=float, default=0.0,
                   help="optional Gaussian perturbation around planned MPC actions (rad)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/dh116_hand_only_teacher_student"))
    p.add_argument("--teacher-only", action="store_true")
    p.add_argument("--student-only", action="store_true")
    p.add_argument("--rod-backend", choices=("cpu", "torch_cuda"), default="cpu")
    args = p.parse_args()
    if args.teacher_only and args.student_only:
        p.error("--teacher-only and --student-only are mutually exclusive")
    if args.episodes <= 0 or args.steps_per_episode < 2:
        p.error("episodes must be positive and steps-per-episode must be at least 2 to include grasp and lift")
    if not np.isfinite(args.initial_action).all():
        p.error("initial-action must contain 12 finite joint values")
    if not np.isfinite(args.teacher_noise) or args.teacher_noise < 0:
        p.error("teacher-noise must be finite and non-negative")
    if args.device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
        except ImportError as exc:
            raise RuntimeError("CUDA requested but torch is unavailable") from exc
    rng = np.random.default_rng(args.seed)
    spec = HandOnlySpec()
    mpc = ParallelJointMPC(device=args.device)
    transitions = []
    # Explicit neutral pose; only hand features are exposed.
    poses = np.zeros((args.episodes, 2, 7), dtype=np.float32)
    poses[..., 3] = 1.0
    joints = np.zeros((args.episodes, 2, 6), dtype=np.float32)
    previous = np.broadcast_to(np.asarray(args.initial_action, dtype=np.float32),
                               (args.episodes, 12)).copy()
    joints = previous.reshape(args.episodes, 2, 6).copy()
    grasp_steps = args.steps_per_episode // 2
    lift_steps = args.steps_per_episode - grasp_steps
    initial_wire = np.zeros((args.episodes, 2), dtype=np.float32)
    initial_state = {"hand_poses": poses, "active_joints": joints,
                     "wire_features": initial_wire, "previous_action": previous}
    planned, plan_diagnostics = mpc.plan_trajectory(
        initial_state, grasp_steps=grasp_steps, lift_steps=lift_steps,
        previous_action=previous,
    )
    # Transfer the finished CUDA trajectory once, not once per MPC step.
    if hasattr(planned, "detach"):
        planned = planned.detach().cpu().numpy()
    plan_diagnostics = {
        key: (value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value))
        for key, value in plan_diagnostics.items()
    }
    for step in range(args.steps_per_episode):
        grasp = float(plan_diagnostics["grasp_progress"][0, step])
        lift = float(plan_diagnostics["lift_progress"][0, step])
        planned_action = np.asarray(planned[:, step], dtype=np.float32)
        # The teacher label defaults to the precomputed MPC trajectory exactly;
        # optional noise is only for deliberate robustness augmentation.
        teacher = planned_action + rng.normal(0.0, args.teacher_noise, planned_action.shape).astype(np.float32)
        teacher = np.clip(teacher, np.tile(mpc.lower, 2), np.tile(mpc.upper, 2))
        phase = "grasp" if step < grasp_steps else "lift"
        for env in range(args.episodes):
            obs_student = np.concatenate((joints[env].reshape(-1), previous[env], [grasp]))
            obs_priv = np.concatenate((obs_student, [lift]))
            transitions.append(make_transition(
                obs_student=obs_student, obs_teacher_privileged=obs_priv,
                previous_action=previous[env], mpc_action=planned_action[env],
                action=teacher[env], reward=float(grasp + lift),
                done=step == args.steps_per_episode - 1, phase=phase))
        previous = teacher
        joints = teacher.reshape(args.episodes, 2, 6)
    args.out.mkdir(parents=True, exist_ok=True)
    teacher_manifest = write_replay(args.out / "teacher.jsonl", transitions, role="teacher")
    student_manifest = write_replay(args.out / "student.jsonl", transitions, role="student")
    def fit(role: str) -> dict[str, object]:
        if role == "student":
            # Student follows the precomputed grasp segment and then holds its
            # last grasp target while Panda owns the separate lift motion.
            knots = np.concatenate((
                np.asarray([0.0], dtype=np.float32),
                np.asarray(plan_diagnostics["grasp_progress"][0, :grasp_steps], dtype=np.float32),
            ))
            actions = np.concatenate((
                np.asarray(args.initial_action, dtype=np.float32)[None, :],
                np.asarray(planned[0, :grasp_steps], dtype=np.float32),
            ), axis=0)
            payload = make_trajectory_checkpoint(
                progress_knots=knots, actions=actions, previous_smoothing=0.15,
            )
            (args.out / "student_checkpoint.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return {"path": str(args.out / "student_checkpoint.json"),
                    "sha256": payload["sha256"], "input_dim": payload["input_dim"],
                    "output_dim": 12, "kind": payload["kind"],
                    "grasp_knots": int(len(knots)), "lift_behavior": "hold_final_grasp_target"}
        key = "obs_teacher_privileged" if role == "teacher" else "obs_student"
        x = np.asarray([getattr(t, key) for t in transitions], dtype=np.float32)
        y = np.asarray([t.action for t in transitions], dtype=np.float32)
        coef, *_ = np.linalg.lstsq(np.concatenate((x, np.ones((len(x), 1), dtype=np.float32)), axis=1), y, rcond=None)
        payload = {"role": role, "kind": "linear_imitation_checkpoint", "input_dim": int(x.shape[1]),
                   "output_dim": 12, "weights": coef.tolist(), "arm_state_included": False}
        raw = json.dumps(payload, sort_keys=True).encode()
        payload["sha256"] = hashlib.sha256(raw).hexdigest()
        (args.out / f"{role}_checkpoint.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return {"path": str(args.out / f"{role}_checkpoint.json"), "sha256": payload["sha256"],
                "input_dim": payload["input_dim"], "output_dim": 12}
    teacher_ckpt = fit("teacher")
    student_ckpt = fit("student")
    diagnostics_summary = {
        "planner": "batched_complete_grasp_lift_reference_trajectory",
        "student": "grasp_trajectory_imitation_then_hold_during_panda_lift",
        "trajectory_shape": list(np.asarray(planned).shape),
        "phase_counts": {"grasp": args.episodes * grasp_steps,
                          "lift": args.episodes * lift_steps},
        "mean_tracking_cost": float(np.mean(plan_diagnostics["tracking_cost"])),
        "initial_action": np.asarray(args.initial_action, dtype=np.float64).tolist(),
        "teacher_noise_rad": args.teacher_noise,
    }
    manifest = {"task": "dh116_hand_only_grasp_lift", "entry_phase": "grasp",
                "action_dim": spec.action_dim, "previous_action_dim": spec.previous_action_dim,
                "arm_state_included": False, "episodes": args.episodes,
                "steps_per_episode": args.steps_per_episode, "device": args.device,
                "rod_backend": args.rod_backend,
                "mpc": diagnostics_summary,
                "teacher": teacher_manifest, "student": student_manifest}
    manifest["teacher_checkpoint"] = teacher_ckpt
    manifest["student_checkpoint"] = student_ckpt
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "episodes": args.episodes,
                      "transitions": len(transitions), "action_dim": 12,
                      "device": args.device, "rod_backend": args.rod_backend,
                      "planner": "preplanned_batched_grasp_lift"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
