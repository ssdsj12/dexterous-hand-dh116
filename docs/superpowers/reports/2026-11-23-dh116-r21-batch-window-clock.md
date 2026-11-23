# DH116 R21 batch-window and current clock audit

This package implements the first CPU-verifiable part of the R21 agreement. `TeacherPhysicalGroup` uses the original shared exchange and tracks its window number separately from each episode's control count. A full eight-tick window with trustworthy lane endings can be committed, then `reset_done` accepts only a fresh exchange for the selected lane and retains the actual lane's hand, rod, clock and physics budget. An incomplete window or original physical guard closes the batch before the commit callback. The legacy diagnostic keeps its original all-lane stop behavior.

The two-lane regression uses the existing PPOTrainer and original exchange with CPU PhysX and solver doubles. It checks distinct policy-issued actions, record-before-Reset order, same-endpoint bootstrap on timeout, no bootstrap on genuine termination and no cross-episode GAE. Another lane continues through its own rod tick and reaction chain while the done lane starts a new episode. The suite also rejects early and window-end clearance faults and a Reset that tries to extend its SDK budget. These fixtures do not represent a native hand–wire grasp.

The current-clock v3 probe pins the checked source closure separately from the historical v2 gate. It reuses the same reader probe on SDK 240, 258, 2402 and 9600 for both CPU accumulation hypotheses. All 32 primary reader rows passed, with actual SDK and nominal rod timestamp fields kept distinct in the synthetic packets. The historical v2 suite still deliberately rejects the four changed source files. Its original fingerprints and evidence remain intact. CPU input probes establish neither SDK callback precision nor live long-clock readiness.

## Reproduction

The selected CPU regressions completed with 211 passed and exit code 0.

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q --tb=short \
  tests/test_teacher_batch_window.py tests/test_native_current_clock_consumers.py \
  tests/test_teacher_physical_group.py tests/test_dh116_deformable_rollout_admission.py \
  tests/test_teacher_batch_inputs.py tests/test_dh116_isaac_batch_scene.py \
  tests/test_dh116_flexible_physical_exchange.py tests/test_dh116_poststep_clearance_guard.py \
  tests/test_dh116_privileged_teacher.py tests/test_native_training_clock.py
```

The independent historical run of `tests/test_native_long_clock_consumers_v2.py` returned exit 1 with 16 failed and 6 passed. The failures report `clock_consumer_source_sha256 differs`. Log and table SHA-256 values are in the readiness record.

Native progress remains at the earlier 32/64 held-pose capacity stage. The batched launcher must bind the real per-lane source, reward and Reset transaction before attempting the two-lane dynamic run. The adverse test must show a nonzero hand–rod reaction on the owning lane, unchanged neighboring hand and rod state under Reset, and natural process exit. Fresh native clock evidence uses the original 480 Hz interval and current installed SDK callback readings.

All 528 protected source hashes passed. The same two historical Checkpoint paths under `/tmp` remain absent, with their expected digests retained.
