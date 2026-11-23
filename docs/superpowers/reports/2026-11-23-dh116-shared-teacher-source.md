# Shared parent/row Teacher source and GPU policy

SharedTeacherSource explicitly owns one actual SharedRodOwner, native binary,
N environment rows and each row's run/rod/episode/Reset/tick identity. Full
parameter_snapshot.num_envs remains N. There is no fake per-lane RopeBatch
instance or retagged one-env native snapshot. An old packet is rejected after
Reset even when global SDK/solver time stays unchanged. Native scope verifies
the already loaded actual RopeBatch class and binary; CPU doubles are rejected.
Complete source inventory, LICENSE, build logs and candidate binary are checked
against dh116_shared_backend_source_v1.json. Original source/binaries are intact.

One capture reads positions, velocities and the complete native parameter batch.
It validates all 41 nodes, EA/EI decoding, free root, masses, gravity, radius,
friction and same-clock native health. Mounted source packets retain the same
66 measured fields and 305 raw NaN/false fields. Environment origins are checked
against both hands' authored seed packets. Only rod node XYZ is translated for
policy features; physics, velocity, tangent, curvature and material identity keep
their existing semantics. Arrays are owned. Profiles are checked on construction;
per-window reads do not rehash all native source files.

SharedNativeTeacher reuses one extracted initialize_native_network with the
existing NativeContextTeacher. Old model identity checks stay strict. New
checkpoints bind source family, backend/source hashes, grasp, reward SHA and N,
and reject historical checkpoints. The original Tanh distribution and original
PPO update/GAE/KL/backtracking code are reused. DeformablePPOTrainer now accepts
an explicit device (default remains cpu); real GPU policy inference and optimizer
update passed. Rollout ownership and GAE still use the original CPU ledger.
This is not a claim that the entire physical/rollout pipeline is GPU zero-copy.

## Verification

Fail-first tests exposed both a missing source implementation and an initially
unbound rod_uid. They now reject duplicate/swapped rows, wrong origins, foreign
context/rod epochs, changed Reset/tick/SDK/solver clocks, native errors, and CPU
solver doubles under native scope. CPU source/model tests: 11 passed, one CUDA
skip. Combined legacy model, batched inputs, deformable PPO and new source
regression on physical GPU 3: 57 passed in 49.04 s, no failures. This includes
an actual CUDA inference/update using explicitly synthetic CPU-scoped state;
it is not new Isaac data or native Teacher training.

Command:
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q
-p no:cacheprovider tests/test_dh116_shared_teacher_source.py
tests/test_native_teacher_model.py tests/test_teacher_batch_inputs.py
tests/test_dh116_deformable_privileged_ppo.py

Binary-exact relative positions translated by [3,-6,0] give matching complete
inputs and deterministic actions at 1e-6. A non-binary 0.025 m line stored in
World FP32 instead produced a 5.9604644775390625e-6 adjacent-strain difference.
That failed result remains recorded. A separate regression preserves this raw
quantization instead of snapping nodes, rounding strain or relaxing guards.
Antigravity R5 initially suggested one tolerance for both cases, then accepted
the separation after this measured counterexample. Its MCP is still an older
tree; review used current excerpts, and Codex independently verified the code.

## Remaining work

No live Isaac PPO update ran in this package. Connect the shared capture to
actual physical windows, same-clock original geometry/R0 endpoints, control
history, bootstrap-before-Reset settlement and checkpoint recovery. Then verify
real two-lane PPO and scale to 128/512 for all three grasps. All original physical
guards, success thresholds and expert admission remain unchanged. The existing
Pinch non-target clearance failure is not downgraded. GPUs 0/1 were occupied;
only 2/3 were idle at the last read, so three concurrent jobs were not started.
All 528 protected files remain unchanged. Historical /tmp checkpoints remain
absent rather than recreated or claimed verified.
