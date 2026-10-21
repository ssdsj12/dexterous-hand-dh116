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
from go2_pvcnn.control.dh116.teacher_student import make_transition, write_replay
from go2_pvcnn.control.dh116.parallel_mpc import ParallelJointMPC


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--steps-per-episode", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/dh116_hand_only_teacher_student"))
    p.add_argument("--teacher-only", action="store_true")
    p.add_argument("--student-only", action="store_true")
    p.add_argument("--rod-backend", choices=("cpu", "torch_cuda"), default="cpu")
    args = p.parse_args()
    if args.teacher_only and args.student_only:
        p.error("--teacher-only and --student-only are mutually exclusive")
    if args.episodes <= 0 or args.steps_per_episode <= 0:
        p.error("episodes and steps-per-episode must be positive")
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
    previous = np.zeros((args.episodes, 12), dtype=np.float32)
    for step in range(args.steps_per_episode):
        grasp = min(1.0, (step + 1) / max(1, args.steps_per_episode // 2))
        lift = 0.0 if grasp < 1.0 else min(1.0, (step + 1 - args.steps_per_episode // 2) /
                                             max(1, args.steps_per_episode // 2))
        wire = np.tile(np.array([[grasp, lift]], dtype=np.float32), (args.episodes, 1))
        state = {"hand_poses": poses, "active_joints": joints,
                 "wire_features": wire, "previous_action": previous}
        planned, diagnostics = mpc.plan(state)
        # Teacher follows the parallel-MPC target with bounded noise; student
        # receives only the non-privileged observation and imitates the target.
        teacher = planned + rng.normal(0.0, 0.005, planned.shape).astype(np.float32)
        teacher = np.clip(teacher, -1.0, 1.0)
        phase = "grasp" if grasp < 1.0 else "lift"
        for env in range(args.episodes):
            obs_student = np.concatenate((joints[env].reshape(-1), previous[env], [grasp]))
            obs_priv = np.concatenate((obs_student, [lift]))
            transitions.append(make_transition(
                obs_student=obs_student, obs_teacher_privileged=obs_priv,
                previous_action=previous[env], mpc_action=planned[env],
                action=teacher[env], reward=float(grasp + lift),
                done=step == args.steps_per_episode - 1, phase=phase))
        previous = teacher
        joints = teacher.reshape(args.episodes, 2, 6)
    args.out.mkdir(parents=True, exist_ok=True)
    teacher_manifest = write_replay(args.out / "teacher.jsonl", transitions, role="teacher")
    student_manifest = write_replay(args.out / "student.jsonl", transitions, role="student")
    def fit(role: str) -> dict[str, object]:
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
    manifest = {"task": "dh116_hand_only_grasp_lift", "entry_phase": "grasp",
                "action_dim": spec.action_dim, "previous_action_dim": spec.previous_action_dim,
                "arm_state_included": False, "episodes": args.episodes,
                "steps_per_episode": args.steps_per_episode, "device": args.device,
                "rod_backend": args.rod_backend,
                "mpc": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in diagnostics.items()},
                "teacher": teacher_manifest, "student": student_manifest}
    manifest["teacher_checkpoint"] = teacher_ckpt
    manifest["student_checkpoint"] = student_ckpt
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
