# Source-bound Kit stat-cache release Implementation Plan

> **For agentic workers:** Use executing-plans task by task; independent read-only SDK/API review runs in parallel. No native execution is authorized by this plan.

**Goal:** Add an isolated, source-qualified use of Kit's existing reset_stat_cache before the original owned-resource GC, while leaving legacy execution unchanged.
**Architecture:** A pure-Python adapter validates the installed source and active importlib hook, records scalar exception/frame ownership and unchanged SDK/App clocks, then invokes the existing reset API once. A new diagnostic CLI reuses the original AppLauncher/lock/runtime/capture/teardown; only that entry supplies an optional before_collect callback.
**Tech Stack:** Python3.10, current Torch/Isaac install, pytest CPU source models, existing Git relay.

## 1. API contract and CPU reproduction
Files: src/go2_pvcnn/control/dh116/tabletop_stat_cache_release.py; tests/test_teacher_stat_cache_release.py.
- [x] Write missing-API assertions and tests using the actual installed stat-cache Python AST plus a fake Kit App. A real temporary Python package forces the same importlib missing-extension checks.
- [x] Record red output outside Git.
- [x] Implement stat_cache_source_contract, snapshot_stat_cache and release_owned_stat_cache. Validate exact module/file/function provenance, active hook and enabled cache before mutation. Read bounded exception traceback/f_back metadata without locals or object retention.
- [x] Confirm the actual-source cache retains a Scene before reset, the reset API removes that retention after CPU GC, observers cannot block the API on ordinary IO errors, and SDK/App clock changes are rejected.
Run: CUDA_VISIBLE_DEVICES='' DH116_INSTALLED_SDK_AUDIT=1 python -B -m pytest -q -p no:cacheprovider tests/test_teacher_stat_cache_release.py.

## 2. Reuse the original GC boundary
Files: src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py; scripts/diagnose_dh116_teacher_initialization.py.
- [x] Add before_collect=None to run_owned_call after synchronize, failed-traceback detachment and resource-dictionary clearing, immediately before its existing gc.collect.
- [x] Test real run_owned_call ordering sync -> cache reset -> original GC -> ownership checkpoint, including reset/observer failure. Keep the original GC count at one.
- [x] Add optional before_owner_collect=None seams to the legacy diagnostic functions. Default None follows the old path; only the new entry passes a callback.
Implemented seam: ordinary hook exceptions become cleanup error text, KeyboardInterrupt/SystemExit are deferred; both paths still run the original GC/checkpoint, then interruption is propagated. No new GC.
No SDK edits, direct frame clearing, physics/control changes or cache-disable setting.

## 3. Independent diagnostic entry
File: scripts/diagnose_dh116_teacher_stat_cache_release.py.
- [x] New versioned freeze/verify/main reuses legacy freeze inputs, budget and execute_native; bind stat_cache.py, ext_settings.py and _internal.py source SHA.
- [x] The callback resolves the already-loaded cache module, uses actual SDK2 and App update readers, writes before/after/error JSON with fsync, and drops the temporary simulation alias in finally.
- [x] Test distinct plan version, exact approval fingerprint, fresh output, one-call budget and failure propagation. No alternative AppLauncher or copy of the runtime.
- [x] Preserve strict geometry errors and nonzero process exit. Removing SIGSEGV alone is not native exit0 or Teacher qualification.

## 4. Regression, protection and publication
- [x] New26 and related522 CPU regressions verified; Registry55 passed after metadata update (overlaps related suite).
- [x] Report/readiness/Registry/README/CHANGELOG updated;528 protected hashes and3 original models verified.
- [ ] Commit explicit files on11-17; Windows imports the same commit and normally pushes; fetch and verify SHA/tree/file list. PR2 remainsDraft.
- [ ] Freeze the final published repair plan outside Git. Request new bounded native authorization only after all code/tests/report are concrete. Previous budget is spent.

## Independent review closure
- [x] Compile-only actual function code/defaults/globals comparison and real settings predicate provenance.
- [x] Read-only post-observer and post-receipt checks, including a direct live-dict check after the last clock reader; no repeat reset.
- [x] Record all five review failures before repair; final targeted26 passed.
