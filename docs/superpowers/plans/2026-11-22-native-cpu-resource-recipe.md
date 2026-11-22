# Native CPU Resource Recipe Implementation Plan

> Execute the existing approved repair chain in Ubuntu; use TDD and independent readonly review.

**Goal:** make the existing scheduler CPU thread setting usable by a bounded Teacher recipe.
**Architecture:** optional environment in validate_teacher_recipe_v2 accepts only canonical DH116_NUM_THREADS=1..64; no dependency or library environment changes. Existing recipe hashing binds its bytes.
**Tech Stack:** current Python3.10/PyTorch2.7, bash scripts/python.sh.

- [x] Reproduce Teacher schema refusal: 3 positive/fingerprint cases fail; 6 invalid environment cases reject.
- [x] Extend only the strict v2 field/schema validation; old no-environment, v1 fingerprint, native budgets and source contracts stay unchanged.
- [x] Run worker/actual-launch/scheduler CPU suites; demonstrate actual scripts/python.sh sees configured threads with CUDA masked.
- [x] Freeze distinct Tripod v10 core/launch plus exact source closure, 8 controls/66SDK/768substeps/1update/180-240s; leave v9 spent.
- [x] Update report/readiness/Registry, protect528/3 models/fixed RGB, commit Ubuntu and publish same commit from Windows.
No long training, physics modification, unqualified checkpoint resume or automatic retry.
