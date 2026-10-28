# Immediate PPO Force Target Implementation Plan

**Goal:** Execute the user's confirmed choice: PPO chooses each hand's 0.1–12 N target, applied immediately without a target slew.

**Architecture:** Use an explicit null `force_rate_n_s` to disable only the force-setpoint slew in scalar, NumPy batch, and GPU tensor controllers. Finite positive rates preserve legacy behavior and serialized contracts. A separate profile selects this mode; exact native checkpoint migration preserves policy, Adam and episode counters.

**Tech Stack:** Python, NumPy, PyTorch, Isaac Sim, RSL-RL.

## Constraints and acceptance

- Do not prescribe a fixed 12 N request; retain both policy force actions and their 0.1–12 N mapping.
- Physical velocity, torque, offset, geometry and measured-force protection remain unchanged. Immediate targets do change the target-relative unloading threshold immediately.
- Keep close-start one-second contact reward, five-point cap, measured grip gating and real lift success.
- Actual force is a physics result; neither a force reading nor object motion may be synthesized to match the target.
- Keep one Isaac instance. Resume a complete checkpoint into a new run, retain source artifacts.

## Task 1: Explicit target mode and parity

- [x] Add tests in `tests/test_dh116_instant_force_target.py` for immediate upward/downward target changes at 480 Hz, unchanged limited closure, inactive rows, scalar/NumPy/CUDA parity, and no readiness from requested force alone.
- [x] Confirm failing tests before implementation.
- [x] In `tabletop_load_parameters.py`, allow None only for `force_rate_n_s`; keep all other validation. In `tabletop_force_feedback.py`, `tabletop_batched_task.py`, and `tabletop_tensor_task.py`, use `requested.copy()`/`requested.clone()` when rate is None; otherwise retain existing slew computation.
- [x] Add `v2_pressure_12n_instant_candidate.json` with `max_force_n=12`, `force_rate_n_s=null`, unchanged remaining drive parameters.

## Task 2: Resume and actual validation

- [x] Test migration: only 12 N / 2 N/s to identical 12 N / null rate is accepted; reject unrelated changes and reverse migration.
- [x] Add the exact migration in `tabletop_tensor_training.py`, requiring both matching load and feedback fields. Update default launcher to the separate instant profile.
- [x] Run instant-target, load, force-feedback, tensor, batch and pressure-relief regressions. Check diff and shell syntax.
- [x] Preserve current checkpoint, stop owned trainer, launch one 4096-environment run with the instant profile and existing close-start reward.
- [x] Confirm actual runtime contract, simulator drive readback, finite optimizer update, reward expiry and failure records. Record measured results without claiming instantaneous measured force or successful lift.
- [x] Update README/CHANGELOG with verified results and limitations. No commit/push requested.

Validation: 297 tests passed; one expected warning from the existing intentional overflow-termination test. Independent reviewer: 4 instant-target tests passed, no blocking findings.

- v97从第225轮恢复至第229轮/22511616样本；运行合同确认目标速率为null，全部4096环境驱动读回通过，策略/优化器有限。已出现1个超力失败回合并按原规则复位，训练继续；尚无hold/lift或成功。297项回归通过（1个既有溢出保护测试预期警告），独立复核4项通过。TensorBoard选择 `v97_instant_force4096`。
