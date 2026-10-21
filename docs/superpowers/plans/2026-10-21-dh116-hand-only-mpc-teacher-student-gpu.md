# DH116 Hand-Only Parallel MPC Teacher-Student GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Train only the DH116 hand on grasp/lift using parallel joint-space MPC references, privileged teacher data, student observations, and a selectable GPU Cosserat rod backend while Panda joints remain frozen and out of the action/learning loop.

**Architecture:** A hand-only environment exposes six active DH116 actuator coordinates per hand (12 total) and derives passive joints through the existing coupling map. A batched MPC planner produces reference actions from wire state and the previous joint action; a privileged teacher learns from those references and a student distills to non-privileged observations. The Cosserat backend is selected explicitly (`cpu` or `torch_cuda`), with parity tests on a small rod before it is used for training.

**Tech Stack:** Python, NumPy, PyTorch CUDA, existing DH116 coupling/Jacobian modules, PyElastica/DeformX CPU backend, pytest.

## Global Constraints

- Panda arm joints are frozen and never appear in the hand policy action vector.
- Training scope is grasp and lift only; approach is excluded from the training episode.
- Previous joint action is part of the planner input, dataset record, and policy observation.
- Six active joints per hand are authoritative; passive DH116 joints remain affine followers.
- GPU Cosserat is opt-in and must fail clearly when CUDA is unavailable; CPU remains the reference backend.
- Do not claim hardware or physical success from offline teacher/student metrics.
- Do not add or track `1332`, `runs/`, caches, or archived evidence.

### Task 1: Hand-only state/action contract and parallel MPC reference generator

**Files:**
- Create: `src/go2_pvcnn/control/dh116/hand_only_env.py`
- Create: `src/go2_pvcnn/control/dh116/parallel_mpc.py`
- Create: `tests/test_dh116_hand_only_env.py`
- Create: `tests/test_dh116_parallel_mpc.py`

Implement `HandOnlySpec(action_dim=12, previous_action_dim=12)` and `HandOnlyState` with two hand poses, six active joints per hand, wire grasp/lift features, and the previous 12D action. Implement `ParallelJointMPC.plan(batch_state, previous_action)` returning `(batch,12)` bounded active-joint targets and diagnostics; use batched torch/NumPy math, no Panda inputs, and a deterministic grasp-then-lift phase target. Reject wrong shapes, NaN, and CUDA requests without CUDA.

### Task 2: Teacher/student trajectory collection and PPO training entry points

**Files:**
- Create: `scripts/train_dh116_teacher_student.py`
- Create: `src/go2_pvcnn/control/dh116/teacher_student.py`
- Create: `tests/test_dh116_teacher_student.py`

Implement a replay schema containing `obs_student`, `obs_teacher_privileged`, `previous_action`, `mpc_action`, `action`, `reward`, `done`, and `phase` for grasp/lift only. Add deterministic teacher collection from `ParallelJointMPC`, a PPO update for the privileged teacher, and student distillation against teacher actions plus MPC references. The CLI must support `--device cuda:0`, `--episodes`, `--seed`, `--out`, `--teacher-only`, and `--student-only`; it must emit `manifest.json`, `teacher.jsonl`, `student.jsonl`, and checkpoints with SHA256 metadata. Arm state is omitted and a fixed arm context is rejected if supplied.

### Task 3: GPU Cosserat rod backend and backend selection

**Files:**
- Create: `vendor/DeformX/co_sim/gpu_cosserat.py`
- Modify: `vendor/DeformX/co_sim/free_rod.py`
- Modify: `vendor/DeformX/co_sim/table_rod.py`
- Modify: `vendor/DeformX/co_sim/moving_contact.py`
- Create: `tests/test_gpu_cosserat_backend.py`

Add a torch CUDA Cosserat rod state/integrator with the same snapshot fields (`rod_position`, `rod_velocity`, `rod_director`, `rod_omega`) and explicit `backend='torch_cuda'` selection. Keep CPU behavior unchanged. Implement a small batched explicit update and table/contact force path sufficient for hand-only training; validate finite tensors, CUDA availability, and deterministic CPU/GPU parity on one short step within documented tolerance. No silent CPU fallback when `torch_cuda` is requested.

### Task 4: Integrate hand-only training with GPU rod and document execution

**Files:**
- Modify: `scripts/train_dh116_teacher_student.py`
- Create: `scripts/verify_dh116_hand_only_training.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `.superpowers/sdd/progress.md`

Add `--rod-backend {cpu,torch_cuda}` to the training CLI, run a short CUDA smoke training with no Panda action channels, verify teacher/student records and previous-action continuity, and record the exact command/results and limitations. Add a verification command that checks manifest hashes and confirms `action_dim=12`, no arm keys, and GPU backend selection.
