# Dual Pinch builder to one complete control interval: CPU integration

Date: 2026-11-13
Base: 93c923a940c43c76c8461ccbc449e46a9af496ba
Branch: codex/dual-pinch-interval-93c923a

## Result and scope

The published build_reset_owned now has a realistic CPU integration regression
through DualHandPhysicalEpisode.pre_action_state and its complete eight-tick
control interval. The actual builder, reset, source geometry reader, 24D action
adapter, wrist safety projection, physical exchange, dual-hand rod runtime,
pair validator, boundary provider and mounted state reconstruction are reused.
No production integration defect was reproduced; no production changes are
justified by this fixture.

This package adds one test file and this report. It changes no frozen source,
native recipe, physics parameter, registry, journal, authorization or launcher.
No Isaac App, CUDA solver, GPU allocation, native lock, training or PPO ran.
The test imports offline USD/source geometry with scripts/python.sh and uses
CPU SDK/solver doubles. This is not a dual-hand native or contact qualification.

## Exact sources and composition

The builder loads the existing private paired result:
- Evidence: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/paired-pinch-reset-BaP0hx/result.json
- Evidence SHA256: 996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37
- Reset seed SHA256: 647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d

The original builder fixture supplies actual source USD/FK, both locked pad
geometries and the published 41-node same-rod reset proposal at Z=0.305 m.
The new test extends its SDK-shaped doubles only after the real reset:
- One distinct mounted articulation per hand, all original 17 named DOFs and
  23 actual-shaped link buffers. DOF types, native targets, measured positions,
  velocities and cache fields use the existing capture/validation paths.
- A CPU SDK step consumes the submitted targets, applies the original passive
  mimic mapping, updates source FK, and copies finite-difference linear/angular
  link velocities into both SDK-shaped readback and cache.
- A small nonzero finger action (channels 1 and 7, each 0.001) exercises the
  original rate kernels. It changes measured state while leaving wrist action
  zero and retaining the original 0.0021 m clearance guard.
- The fake solver advances one existing rod step per SDK step. Its explicit
  synthetic pair fixture has 12 substeps, left mesh slot 0/right slot 6, 1/2 N
  respectively, and exactly matching aggregate node/mesh reaction. The table
  retains slot 5; this fixture does not simulate deformation or discover contact.

No direct action-to-force path is introduced. The test verifies the real
one-step reaction delay: initial composers receive zero, subsequent steps
receive the preceding rod reaction, independently -1 N left and -2 N right.

## Assertions and failure boundaries

The successful path asserts two setup steps followed by eight SDK steps and
eight solver calls: final SDK step 10, rod tick 96, solver time 8/480 seconds.
Every physical record and rod boundary shares the actual copied step identity.
A ninth-through-sixteenth interval is rejected at the budget gate before any
further SDK step. This check uses the existing record callback in memory; there
is no duplicate evidence journal.

The original state contract remains 66 valid channels out of 371, with all 305
unknown channels NaN. The strict Teacher consumer rejects the observation.
Training reward is None and Teacher admission remains false. Hand-table PhysX
contact source is unavailable (pair packet physx_contact_pairs is None).
Synthetic rod/hand mesh pairs do not upgrade that source.

Two negative cases assert precise partial-progress boundaries:
1. Corrupt a right-hand original DOF name: rejection at pre-action capture,
   SDK stays at setup step 2, solver calls 0, physical records 0.
2. Duplicate a solver pair identity: rejection after exactly one SDK/solver
   step (SDK 3, solver 1), physical records 0. A repeated control call is
   rejected because the episode remains failed; no rollback or replay occurs.

The fixture releases its explicit hand-list and SDK-step closure owners after
run_owned_call and then uses test-only Python cycle collection. Weakrefs to
both hands, their views and the fake rope are dead on success and failure.
This proves narrow CPU ownership cleanup; it does not prove arbitrary native
callback/SimulationContext lifetime or remove any SDK callback.

## Validation

New focused command:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_pinch_interval.py -x

Result: 3 passed in 87.86 seconds. The earlier initial zero-action version also
passed 3 tests in 81.06 seconds; it is not counted twice.

Existing immediate composition regressions:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_hand_episode.py tests/test_dh116_flexible_physical_exchange.py tests/test_bimanual_physical_action_adapter.py

Result: 23 passed in 3.56 seconds. Final unique total: 26 passed, 0 failed.
Repository.verify_protection completed with exit 0 for the existing 528+2
protected entries. No tracked original file differs from base.

Evidence directory:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-interval-cpu-i10t2tko

The directory contains focused.json (exact source SHA and terminal result),
regression.json, regression.log, first-pass.json and protection.json. The
regression uses real subprocess capture; focused terminal output is recorded
from completed tool session 35734, not represented as native output.

Final test source SHA256: b619e22bcf1c62b9007ffdc9166633a698b1bc0b220707eced139199ad142764

## Remaining native scope

A future separately approved dual-hand native package still needs real SDK
motion, measured geometry at every boundary, actual solver pair acquisition,
independent contact provenance and actual release/outer-exit evidence.
No result here proves force tracking, physical grasp, full 371D observations
or Teacher PPO readiness. The 2.1 mm policy floor is unchanged. No native recipe
or budget is frozen by this test-only package.
