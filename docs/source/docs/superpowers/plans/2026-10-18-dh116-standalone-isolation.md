# DH116 Standalone Isolation Implementation Plan

> **For agentic workers:** Use executing-plans inline, following the user's existing Inline Execution preference. No subagents or automatic commits.

**Goal:** An independently runnable DH116 project in `/home/xk/coding/dexterous hand`, preserving original sources and evidence.

**Architecture:** Copy a minimal Python dependency closure, required simulation assets/backend, manufacturer resources and historical evidence. Keep immutable source copies separate from newly adapted runtime resources. Track all copies with SHA and explicitly defer live output synchronization.

**Tech Stack:** Python 3.11, pathlib/shutil/hashlib, pytest, USD/Isaac Lab, NumPy, DeformX.

## Global Constraints

- No deletion or modification of original runtime sources; core0/single-thread computation.
- Do not interrupt PID6780 or start a second Isaac instance.
- No links back to original project; external installed software environment is allowed.
- Historical state pins remain unchanged. New assets/configurations need new validation.
- Directory names with spaces must work.

## Task 1: Copy manifest and immutable inventory

Files: new `tools/migration_copy.py`, `tests/test_migration_copy.py` under target directory.

- [ ] Test first: source hash equals target hash; reject existing different target; skip symlinks rather than follow unchecked external paths; reject destinations outside project; record source/destination/hash/classification.
- [ ] Run `python -m pytest tests/test_migration_copy.py -q`, observe missing implementation, then implement `copy_file(source, root, relative, category)` returning a JSON-compatible entry.
- [ ] Use helper to copy DH116 modules/scripts/tests, required contact bridge/visual modules, relevant docs, manufacturer tree and backend without environments/git/caches.
- [ ] Copy old DH116 evidence to archive, excluding live trial directory/log; record live_pending source separately.
- [ ] Verify manifest hashes and source non-mutation; no unrelated O6/Go2 tasks.

## Task 2: Runtime asset isolation

Files: target `tools/relocate_assets.py`, `tests/test_asset_isolation.py`, `assets/`, `configs/`.

- [ ] Test that all runtime USD external references resolve beneath target root and that missing references fail rather than fall back.
- [ ] Recursively collect USD layers/mesh/texture dependencies, preserve original archive bytes, create rewritten runtime copies with relative references.
- [ ] Rebuild source manifests/config layer pins against new assets, rerun geometric checks; never rewrite old stage pins.
- [ ] Validate both mounted and standalone DH116 assets and required Panda source dependency closure.

## Task 3: Standalone code and verification

Files: target `scripts/`, `src/`, `tests/`, `README.md`, `CODEX_RESUME.txt`.

- [ ] Test import origins with old PYTHONPATH removed, test path helpers from directory containing spaces.
- [ ] Replace active hardcoded source roots in copied code with explicit root arguments/default project-relative resources; preserve algorithms and gates.
- [ ] Run all copied DH116 tests with core0/one numerical thread and document passes/skips/failures honestly.
- [ ] When old Isaac process is terminal, finalize immutable archive sync and run one new-root Isaac smoke check; no concurrent simulator.
- [ ] Document commands, software dependencies, current grasp qualification and archive-only states. Update migration_manifest status only after actual checks.

## Review

Spec coverage: Task1 covers preservation/provenance/live handling; Task2 covers USD/assets/pins; Task3 covers code independence/runtime/docs. Existing dirty source worktree remains unchanged except planning/evidence notes. Never mark Task3 complete from copy/hash evidence alone.
