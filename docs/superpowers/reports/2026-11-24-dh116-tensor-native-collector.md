# Actual tensor native collector integration

Foundation published by Windows as 7454f5a332672b72262696b384422c5a43933fbf.
Original physics/24D/371D, 480Hz/60Hz,12 solver substeps, source geometry,
1um precision and all original failure predicates are retained. R2 remains disabled.
This is finger-goal/R0 wiring, not qualified grasp/Lift/Hold evidence.

## Actual code
NativeTensorCollector owns exactly two real regex Articulations, TensorControl,
TensorMeshBridge, TensorRodOwner, TensorSourceOwner, full original Pinch geometry,
TensorTrainingReward, device inputs/rollout/GAE and original guarded PPO.
8 substeps abort on the first hard guard; sparse original exact geometry is resolved
before the next SDK. Pre-Reset critic is recorded before selected Reset.
N2 alone saves raw actual cold snapshots for independent original-path replay;
N128 disables these snapshots. Timings add CPU wall sections, no new GPU barrier.

## Failure and correction
tensor-n2-pinch-gpu3-7q7qhccz stopped at SDK2 with CPU get_coms / CUDA index mismatch,
natural inferior exit1, no controls/solver/PPO, no forced exit, app.close normal.
Static local COM is now copied once at Startup before actual row permutation, then cached.
A real GPU2 CPU-buffer/CUDA-index test passes; no per-substep COM transfer is introduced.

## Actual N2 result
tensor-n2-pinch-comfix-gpu3-dn1hpaa2: natural inferior exit0, 51.92127s total,
18 SDK (2 initialization +16 coupled), 2 windows, 0 PPO updates, 2 actual Articulations.
Selected lane1 Reset at SDK10 preserves peer hand/rod/mesh, both composer buffers,
reaction, source/controller/reward references and epoch identities, then continues.
Window wall times0.9311205/0.5190035s INCLUDE N2 raw snapshot I/O and local Reset.
Torch allocated peak16,313,344bytes is not total driver VRAM.

Independent CPU replay reads all20 SHA-bound snapshots and re-evaluates original
BatchedControl/projection/actual24D, original raw table/F4/SDF geometry, hard guards,
COM reactions and scalar TrainingReward, without a second physics rollout.
Maximum: issued24D/actual targets/raw table=0; wrist roots1.11e-16m;
SDF distance2.38763e-9m; same-input scalar reward0; original-SDF-recomputed reward2.59602e-7.
Discrete guard decisions match. The original5mm stereo or grasp criteria are not involved.

15,744 actual pairs are TABLE ONLY. Hand pair count and hand force/torque are zero.
This validates interface, identity, arithmetic and selected Reset, NOT nonzero hand-reaction
ownership or grasp qualification. Those remain an explicit independent native gap.
Replay SHA181007ad5a559c28a31d72af7e56ae28ac173be45a39dfff9069b25988221074.

## Continued acceptance
GPU3 N128 two-window throughput-only probe completed with natural exit0 and no PPO update.
Target~0.2s/window and6–8s per128x32 update are still unverified.
Tripod raw source/cooked support guard adaptation and true checkpoint continuation
are independently being tested; neither is inferred from Pinch.

## Actual N128 throughput
tensor-128-pinch-probe-gpu3-nttlibse, 18SDK/two windows, natural inferior exit0,
no forced exit. Full cold process251.394s includes original128-scene startup and close.
Window0.7847092/0.5811083s, versus prior16.34795/16.30529s: second-window28.06x speedup.
Sampled whole-GPU peak2980MiB; Torch allocator peak135,144,448bytes.
Second window CPU-wall phase attribution (no added per-phase CUDA barriers):
- observation/policy sample0.027141s, reward begin0.009997s;
- projection/reaction submission0.064547s;
- PhysX0.068028s, readback0.008328s, raw source/mesh0.020381s;
- CUDA rod/flat export/reduction0.025815s;
- geometry/R0/postguards0.341435s;
- value/settlement0.014771s.
The0.2s/window and6–8s/32-window-update target is NOT met.
32 times the steady probe window is18.60s, an estimate, not a measured full update.
No new128 PPO update is counted. Next optimization targets the measured geometry
and command-kernel launch overhead; guard threshold and abort timing stay fixed.

Package CPU regression:87 passed/1 skipped in18.73s; separate real GPU2 startup COM cache test:8 passed. Initial checkpoint serialization is tested; trainer continuation qualification remains separate until the new loader is independently verified.
