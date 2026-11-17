# Dual Pinch guarded closure post-exit audit (CPU only)

Date: 2026-11-13. Base: `d31b88c6f71e168a7b8f5f7ba6afd616ccdb3535`.
This package adds no native entry, recipe, resource acquisition, SDK step, or authorization. Original frozen files are unchanged. Native execution and Teacher qualification remain untested by this package.

## API and immutable checks

`audit_outcome(output, outer_exit, cleanup, *, profile_sha256=None)` is exported by `scripts/validate_dh116_dual_pinch_closure_interval.py`. The caller writes its return value as `dual-pinch-closure-post-exit-audit.json`. Only explicit cleanup is accepted. The independently frozen worker must supply the expected profile SHA; omitted or mismatching expectation fails closed. The currently selected profile is:

`67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498`.

Qualification requires exactly two setup steps plus sixteen measured steps in two complete eight-step control windows, no guard termination, zero PPO, matching result/acquisition code identity, the frozen 2x24 action schedule, and producer journal equal to independent disk replay. The original 2.1 mm wrist floor is required. Frozen `_outer` and `_geometry` are called directly: outer/inferior zero exit and no signal, complete normal GDB stack/register/maps artifacts, 180 s limit, lock evidence, original left/right source identities, 17-DOF reset, original paired seed/float32 rod nodes and independently recomputed source/locked-pad clearances are retained.

`dh116_dual_pinch_lifetime_evidence.lifetime_evidence(..., expected_steps=18)` retains the original six lifetime phases, process maps, manager STOP release, all non-Simulation owned references false after STOP, honest full-release boolean, actual App close return, clean pre-App USD import boundary, exact PHYSICS_READY unsubscribe/original clear receipt, and frozen post-measurement clocks. An AST regression proves its original function body is unchanged except replacing the fixed integer 10 by `expected_steps`; a positive-integer argument check is added. Simulation liveness is reported as observed and is not silently converted to release.

Every `sdk_clock.jsonl` row must exactly equal `operation.sdk_clock_sample(physical_row, ordinal)`, including SDK/rod/post timestamps, policy step, and substep. All four integer counters retain strict integer types. Standard physics budget and clock are independently reconstructed from durable physical rows. Missing or tampered time fields cannot pass using only the SDK counter.

A successful audit grants only `exact_2_setup_plus_16_measured_guarded_closure_execution_only`. Same-rod contact, grasp and Teacher flags remain false; static hand-table contact remains NA, with no absence proof. Failed or incomplete output preserves available result counters, original journal termination reason, valid terminal prefix, observed Simulation liveness and artifact hashes without qualification.

## CPU evidence and dependency provenance

The CPU operation fixtures use the actual builder/reset/guarded episode/journal path with SDK-shaped doubles and source geometry. Complete execution produced 18 total/16 measured rows. The clearance-failure fixture produced 11 total/9 measured rows and preserved `post_step_source_clearance:right`. Outer/GDB/liveness evidence in these fixtures is explicitly synthetic CPU test data, not Isaac evidence.

The original full fixture run was 33 passed / 5 failed in 81.20 s; all five failures exposed a missing `BOUNDARY_VERSION` import in the extracted helper. The operation fixtures themselves completed before these audit failures. After correcting the import, their unchanged physical rows were replayed: 38 passed in 34.07 s. Independent review then identified the expanded scheduler clock contract; final replay added five timestamp/counter/type tamper cases and passed **43 tests (26 new + 17 original) in 34.66 s**.

The final run executed the two pytest modules through `scripts/python.sh` with `/mnt/mydisk/xk/envs/dh116/bin/python`, `DH116_NUM_THREADS=2`, empty `CUDA_VISIBLE_DEVICES`, and `PYTHONDONTWRITEBYTECODE=1`. Its explicit pytest fixture hook reused the archived CPU operation rows to avoid rerunning setup while upstream preparation changed. It derived the expanded clock only from those already recorded SDK/rod/post fields, including the original `post_step_clearance.sdk_time_s`. No physical row or missing time was fabricated.

Private evidence:

- Initial full run: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-closure-postexit-cpu-6c82e54g`.
- Corrected first replay and durable complete/terminal CPU audit copies: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-closure-postexit-replay-14sg78m8`. Template manifest SHA: `1df071d5dc1011821a3aeca5724615846228c9dc6ed6773b3c7c63a5a1104b28`.
- Final expanded-clock replay: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-closure-postexit-clock-i9zc9mfh`, containing exact replay script, metadata, test log, and projected CPU ledger copies.
- Actual archived v4 10-step attempt `attempt-fz50kogx` under `flex.f7-dual-pinch-zero-interval-v4-native/9e95d33e8b0565133c05d8fae5eb5c5eb68de269806ae561e4c5bd022bd9e202` is read only as a negative test. Its clean zero-interval result cannot acquire closure scope. No native attempt was created.

The final tests used temporary, uncommitted copies of the parent's preparation and fixture. Preparation SHA: `28a19068df7c34c13fa30d95f59e509e349e93182dab5f2964cb47442943c7eb`; fixture SHA: `b7905e09f6c73e84953ec92a53352d09b8c60d9e2b02dc4c701db95d59eff513`. The initial operation rows were generated with preparation SHA `443064ae4c79f56a330d5bc29d6f80922b8e1582193effc6addddca8f28b19aa`. These dependency copies are excluded from this commit. Final upstream profile-geometry additions and native wrapper require the parent's integration regression; this report does not claim their later bytes were tested here.

The normal integrated regression is:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_pinch_closure_postexit.py tests/test_dh116_dual_pinch_postexit.py
```

## Source identity and boundaries

- Lifetime helper SHA: `525cd89ecf80547fd9083e1862aea4a2e9c19346e5a24475efcd78d80544ea5a`.
- New audit SHA: `3590e3ed7c17dd5c6e8d29b23ad043e0e43348d5902e125d7b513ada993aa3d1`.
- New tests SHA: `fd8c9ebea3bbe47701133ae8306de98d13c1150d036f25f7272322a700732ede`.
- Original validator SHA remains `29c3480591c3a822d795867eeb961e134a33cf2795dc1104f924c7893fbc0cb9`; original preparation and clearance guard were also byte-compared against the base. `Repository.verify_protection()` passed for all 530 protected files.

Read-only peer review confirmed the lifetime extraction and original outer/geometry reuse, and requested the expanded clock comparison now covered by negative tests. This is a CPU audit implementation, not evidence of actual closure, full 371-state readiness, static-table contact absence, or native owner release. No registry/readiness/README/CHANGELOG edits or push are included.
