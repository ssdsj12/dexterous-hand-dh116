# GPU vector expansion implementation plan

**Goal:** Increase parallel training toward 512 environments on the RTX 5070.
**Architecture:** Clone GPU physics; keep the existing host controller and task.
**Constraints:** One Isaac instance, one CPU thread; no relaxed physical guards,
no invented contact, no overwrite of existing runs/checkpoints.

- [x] Save the current four-environment checkpoint and verify process exit.
- [x] Run two-environment GPU compatibility experiment using unchanged task.
- [x] Implement and test batched collision geometry against independent SciPy.
- [x] Implement clone-based batched articulation/object/contact I/O in
  `src/go2_pvcnn/control/dh116/tabletop_gpu_isaac.py`.
- [x] Discover zero-contact GPU failure; normalize thin-hull cooking coordinates
  while preserving exact world vertices, then verify actual reaction forces.
- [x] Add summary-only episode tracing with return/minimum-clearance regression.
- [x] Review all view ordering and partial-reset data transfers.
- [x] Verify >=12-slot asymmetric object reset and sensor indexing in Isaac.
- [ ] Measure 512-environment capacity, isolation, contact and throughput.
- [ ] Start PPO with validated geometry/control semantics, preserve old weights
  through the existing explicit weights-only transfer if contracts match.
- [ ] Verify optimizer updates, finite observations, checkpoint output and report
  the actual environment count, throughput and any remaining host bottleneck.

Validation commands: `DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q
 tests/test_dh116_batch_geometry.py tests/test_dh116_tabletop_env.py
 tests/test_dh116_force_feedback.py tests/test_dh116_tabletop_vec_training.py`.
Runtime entry point: `scripts/train_dh116_tabletop_gpu_ppo.py`, adaptive grasp,
force feedback, 480 Hz physics, mimic frequency 400, hollow straw/high friction,
zero reset jitter. Every experiment gets a new output directory.

Current execution: v46 completed all96 admission episodes; v47 has loaded512
and detected all1024 hands contacting the object. Full512 admission and new
optimizer/checkpoint verification are pending in the running pipeline. Software
regression208 passed. Capacity fits (~3.7GB); measured preflight ~107 transitions/s.
