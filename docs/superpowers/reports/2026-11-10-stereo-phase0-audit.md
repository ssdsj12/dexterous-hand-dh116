# DH116 stereo Phase 0 source audit

Baseline: `11-10@6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b`. Independent branch: `codex/dh116-stereo-privileged-dagger`. Instruction document: 930 lines, SHA256 `a09551b72c4540c3fb88f308461a2b0176cc5c625cbc974c8a138b765f90a2e6`.

This report records source and CPU arithmetic evidence only. No Isaac process, physics experiment, training run, historical checkpoint rewrite or environment upgrade is part of Phase 0. The user's subsequent instruction authorizes continuing Phase 1–7 implementation, while preserving native promotion gates.

## Audit deliverables

- [371D per-dimension contract](../../stereo_privileged_state_contract.md): actual source expressions, units, frames, normalization, sideband validity, deployment observability and allowed supervision for all indices 0–370.
- [Canonical 24D action audit](../../stereo_action_contract.md): all dimensions, physical scales, vector-norm limits, actual controller/SDK paths and SHA enforcement boundaries.
- [Reward audit](../../privileged_teacher_reward_audit.md): formulas, coefficients, dt, profile enablement, overwrite/overlap and actual formal success gates.
- Machine-readable counterparts under `docs/contracts/`; [critical source SHA freeze](../../contracts/stereo_phase0_source_manifest.json).
- [Phase 0–7 implementation plan](../plans/2026-11-10-stereo-privileged-dagger-implementation.md).

## Findings and implementation decisions

1. The audited configuration produces **153+106+92+20=371**. It is a conditional layout, not every possible `TensorTask` configuration. Legacy phase one-hot and GraspShaping are absent on this route. Task timers, reward-controller state and unqualified feedback are still present and cannot all become valid RGB regression labels.
2. Existing visual input is **4×961=3844**, from actual RGB-D plus proprioception. It is separate from the privileged Critic. The new stereo RGB-only path needs a new version and cannot silently reinterpret 3844D.
3. Canonical actions are left/right fingers 0–11, left/right world translation 12–17 and intrinsic XYZ parameters 18–23. Scales are 0.3 rad/s, 0.01 m/s and 0.5 rad/s. Each wrist 3-vector is norm-limited; nominal adapter decoding is not itself that projection. Target, issued command and measured motion remain separate.
4. Canonical action SHA is `498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9`. Some existing validators enforce the full dictionary; one redundant shadow digest is not independently checked. New artifacts must explicitly verify both semantic dictionary and digest without changing the old path.
5. Legacy diagnostic Teacher qualification includes **261D/26D and short holds**. It is neither a 371D/24D PPO Teacher nor qualification for the new expert pool. Historical success claims are not imported into new-model qualification.
6. CLI reward default is `baseline`; actual enabled terms depend on flags. Surface V4 may retain old hold/dwell while adding early hold/milestones. Free-wrist disables old GraspShaping, so its inactive potential must not be reported as an active duplicate. New R0 passes through existing task reward once; new R1/R2 cannot blindly copy and add coefficients.
7. Formal surface success uses 30 mm, 5 s, fixed-anchor drift ≤0.5 mm, opposed bilateral accepted pads, ordinary-load/depth/speed/finite gates. The surface branch does not consume the earlier generic `stable` Fz gate. Diagnostic 3 mm slip tracking and formal 30 mm anchor tracking differ. No success rule is changed to resolve this audit discrepancy.
8. Feedback zeroing after reset/invalidity does not provide valid supervision by itself. The 371D vector lacks a complete embedded validity mask. New datasets require explicit sideband masks; source-unknown or invalid values stay excluded. Qualified signed support, friction attribution and hardware feedback remain NA until independently established.

## Phase boundaries

The layout, action semantics and existing reward arithmetic are sufficiently identified to implement an **isolated** privileged Teacher R0. Source evidence does **not** qualify support shaping, successful new-Teacher data, formal DAgger or final deployment. Those promotion gates remain closed. Later isolated software can be implemented and CPU-tested while native verification is pending, as the user explicitly requests.

Old controllers, Teacher/Student/PPO modules, material/actuator parameters, 60/480 Hz timing and historical checkpoints remain untouched. The original `11-10` ref is not advanced. All new experiments require explicit opt-in and exclusive output directories. No native or task success is claimed by audit tests.

## Verification record

CPU environment: existing Windows Python 3.12.13, PyTorch 2.14.0, NumPy 2.5.3, pytest 9.1.1. This is not the server's Python 3.10.12/PyTorch 2.7.0+cu128 environment. SciPy is read from the already present `artifacts/stage7_lifecycle_cpu_deps_v01`; no packages are installed. CUDA is disabled and tests do not import Isaac.

Integrated validation: **178 passed in 11.03s, exit 0** (80 new audit cases and 98 existing surface-success/reward/feedback cases). Early red tests intentionally rejected missing manifests; those results are not used as completed validation.

```powershell
$env:PYTHONPATH='src;artifacts/stage7_lifecycle_cpu_deps_v01'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_stereo_privileged_state_contract_audit.py tests/test_stereo_action_contract_audit.py tests/test_privileged_teacher_reward_audit.py tests/test_stereo_phase0_source_freeze.py tests/test_surface_success.py tests/test_surface_reward_ablation.py tests/test_surface_feedback.py
```

Private log: `artifacts/stereo-phase0-20261110/integrated-cpu-02.log`. `git diff --check` passes. The Phase 0 commit stages only documentation, machine-readable audit artifacts and read-only tests; future-phase red tests remain uncommitted until their implementation phase.

Protected baseline: all **528** originally tracked `src/`, `scripts/`, `configs/` files are compared by exact SHA256 against a private pre-task manifest. Critical source/report bytes are also publicly bound in the source-freeze JSON. No old checkpoint is loaded or written.
