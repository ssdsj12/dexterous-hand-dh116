# Grasp Control Probe Implementation Plan

> Execute the approved design in this session, preserving the active trainer until offline checks pass.

**Goal:** Compare current PPO with fixed closure commands under unchanged physics and guards.

**Architecture:** An inference-only branch in `run_tensor_training` reuses TensorWorld/TensorTask/TensorEnv. A default-disabled substep observer captures pre-reset evidence. A separate probe module creates bounded actions and summarizes physical outcomes.

**Tech Stack:** Python, PyTorch, NumPy, Isaac Lab.

## Global Constraints

One Isaac process; 480 Hz physics / 60 Hz control; original rewards and guards; 8 probe envs; 3 episodes each; targets 4/8/12 N with z 0/−1 mm; archive checkpoint before stopping trainer. No reward edits in this experiment.

## Task 1: Probe actions and evidence

- [ ] Create `tests/test_dh116_control_probe.py` with failing tests for `joint_actions(target,lower,center,upper)`, force mapping and strict `check_probe_contract(saved,current)`.
- [ ] Create `src/go2_pvcnn/control/dh116/tabletop_control_probe.py`; map joints piecewise back to [-1,1], reject out-of-envelope targets, encode force as `2*(force-.1)/(maximum-.1)-1`.
- [ ] Add per-substep capture before reset, including terminal row; derive continuous dwell within one episode only. Verify episode boundaries cannot share dwell.
- [ ] Run `CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest tests/test_dh116_control_probe.py -q`.

## Task 2: Inference integration

- [ ] Add `--control-probe-checkpoint` to `scripts/train_dh116_tabletop_gpu_ppo.py`, requiring tensor mode and 8 envs, excluding resume/play/initialize-from.
- [ ] Branch after task/contract construction in `tabletop_tensor_training.py`, load weights only, and return probe report without any PPO update.
- [ ] Add optional callback to `tabletop_tensor_env.py` after guard/phase transition checks and before immediate reset; baseline execution has no callback.
- [ ] Test existing TensorEnv reset behavior and compile changed files. Read-only review of the integration.

## Task 3: Actual simulation

- [ ] Copy atomic checkpoint and active launch metadata to a new diagnostic directory; verify finite policy/optimizer and captured iteration.
- [ ] Stop only identified observer/trainer processes; confirm Isaac trainer exited.
- [ ] Launch original command with output replaced, num-envs=8, episode budget removed, and control-probe checkpoint supplied.
- [ ] Inspect progress at least each minute, diagnose any execution failure, preserve all outputs.
- [ ] Summarize 24 measured episodes, gate blockers and force directions; compare against the design's actual-height criteria.
- [ ] Restore training from preserved checkpoint if no justified control change is required; otherwise retain diagnosis and solve the demonstrated cause before further PPO.

Self-review: no missing dependencies, no reward change, no forced lift or overwritten measured force, no success assertion from commanded motion.
