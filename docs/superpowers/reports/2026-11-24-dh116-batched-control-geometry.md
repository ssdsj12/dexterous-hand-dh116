# Batched native Teacher control and geometry

## Scope and implementation
Based on 07547ba91b81afd5b34003d5d519c88f5b458f32, branch 11-17.
The user explicitly requested clock-only submit in the owned single-writer training path.
The original Bimanual audited begin/submit/finish remains available; its no-tick external
target-change rejection remains tested. The fast path requires exclusive SDK writes.
Reset/episode/rod identities invalidate cached post; initialization and Reset recapture
full topology, while each SDK finish checks the actual float32 targets.

tabletop_batched_control batches original finger kernels, wrist projection, and
world mesh-origin to hand-COM reaction transforms. Batch arrays remain lane-separated.
Original 30 N guard remains. A review found and fixed an alternate-bindings submit hole
and restored all-link finite/unit-quaternion checks including Power noncontact links.

tabletop_batched_geometry_metrics keeps complete original hull vertices on GPU
and computes float64 table bounds, AABBs and all41-node support bounds.
Negative coarse bounds call original exact certificates. Near a threshold (1e-7 m)
the original CPU math decides; this distance chooses the algorithm, never the threshold.

tabletop_batched_pad_sdf retains original FP32 SciPy/NumPy local coordinates and
explicit trilinear interpolation on resident Torch grids. No grid_sample rescaling.
OwnedGeometryMode.capture_from_post reuses the exact same completed finish post,
validates actual link/source/clock identity, and reads velocities without re-reading q.
Memory packets are explicitly not durable replay certificates. Full audit mode remains.
Fixed expected radius is cached; actual shared solver radius and source epoch remain checked.

## Verification
CPU integration: 91 passed, 4 CUDA-skipped, 38.06s.
Separate real CUDA parity tests cover 128-lane control/reactions/geometry and SDF caches.
Original target math and wrist projections agree within 1e-10/1e-12 where applicable;
guard failures, stale clocks, Reset, changed view, NaN and cross-lane submission are tested.
SDF cache CUDA regression: 36 passed, 18.80s.
Original 528 protected source/config/asset hashes unchanged. Two historical /tmp checkpoints
were absent before and after this work, so their contents could not be verified.

## Native N2 proof
GPU3 Pinch: 2 environments, 64 control windows, 522 SDK steps, two actual PPO updates.
Natural inferior exit 0; no forced exit; 103.8735s including initialization/cleanup.
Iterations 24.1034 / 24.4773s. Actual capture count 1028:
512 measured SDK steps * 2 lanes + 2 startup + 2 local Reset captures.
Checkpoint policy/Adam/RNG roundtrip verified. This is finger-goal/wiring smoke,
not demonstrated contact/Lift/Hold success.

## Current limits
The actual 128-lane two-window timing probe has passed with natural exit 0.
A separate full 128-lane two-update run has completed with natural exit 0. Full three-grasp 128-lane training remains unverified.
Tripod finger33 remains blocked by its unchanged source penetration guard.
Original full24D, unknown371D mask, 480/60Hz, physical thresholds and success standards remain.

Antigravity R9 reviewed supplied current excerpts only; its MCP is pinned to an older
worktree. Its comments are supplementary, not proof that it read the current source.
It endorsed retaining Reset/no-step-write invalidation and raised numerical boundary risks.
Codex independently checked the actual predicates; claimed pad penetration thresholds
are not substituted for the existing required_sdf_valid predicate.

## Evidence
Raw evidence remains outside Git under
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119:
- shared-ppo2-pinch-batch-v1-gpu3-hzv_h5sj (N2 two updates)
- shared-128-pinch-batch-probe-gpu3-0d33sve1 (N128 probe)
- batch-acceleration-cpu-regression.log
- batch-acceleration-tests.json
- batch-acceleration-review.json
Each native directory contains frozen argv, source SHA inventory, budget, GDB log and
outer inferior-exit receipt. No raw models or large logs are published.

## Actual N128 result
GPU3, 128 environments, 26 SDK steps, 2 real policy windows, zero updates.
Natural exit 0; 289.6794s total including scene build and cleanup.
Windows: 16.34795 / 16.30529s vs original 41.46975 / 44.65229s.
Average control-window speedup: 2.64x; wall time falls about 62%.
Sampled full GPU peak: 2882 MiB (2.81 GiB); peak RSS 4,845,984 KiB.
Capture count: 2176 = 128 startup + 128*16 SDK finish. No Reset in this probe.

|First-window phase|Original seconds|Batched seconds|
|---|---:|---:|
|Action projection/reaction submit|17.9984|5.6811|
|PhysX dispatch|0.0721|0.0726|
|SDK guards/mesh targets|0.9695|0.9775|
|Shared solver/compact|0.1716|0.1623|
|Post readback|2.2152|0.9943|
|Geometry and R0|18.6562|6.5638|

Batched hull/SDF work inside geometry totals 0.3738s for the first window.
Remaining Python source validation, per-articulation SDK writes and exact guard/R0
work still dominate. Do not claim 1–2s/windows or zero-copy.
Original/source-equivalence tests and numeric boundaries do not constitute grasp qualification.
Full 32-window rollout + PPO timing must come from the separate active run:
shared-128-pinch-batch-ppo2-gpu3-tv8ry9vr.
Native probe outer SHA: 90973ff12531854aa357b11632948f60bf06ec10c7a447b0ba638c603fb84171.
Registry regression: 60 passed, 14.11s.

## Full N128 update baseline (completed)
The separate N128 run completed 64 windows, 522 SDK steps, two actual PPO updates,
including local Reset and continued sampling. Inferior exit0, no forced exit.
Iteration totals: 537.8467 / 542.5839s; optimizer: 0.2050 / 0.1009s.
Both checkpoint policy/Adam/RNG roundtrips passed. Sampled peak GPU use 3440MiB.
This confirms optimization is no longer the timing bottleneck. The rollout still
takes about nine minutes/update and does not meet the newly requested 6–8s target.
No contact/Lift/Hold success or three-grasp qualification is granted.
Outer receipt SHA ca20e5ad88634f750f0d9e0b86594dca889dd5f7a4346c0867ae192e1482d653.
The next independent implementation replaces per-environment Articulations and
keeps flat contact outputs, observations, geometry, reward and rollout on device.
