# R0 actual pad/rod approach source — 2026-11-13

Base: `57d5ae744a05288160c10ffc25f9a318fac7c7ee`. This is a separate offline/readback package following applicability v2, not new native physics or PPO training.

## Actual change

`src/go2_pvcnn/control/dh116/tabletop_r0_measured_approach.py::MeasuredPinchApproach.capture`:
1. Reuses `MeasuredClosureGeometry.validate` to recompute the saved geometry from actual 23-link PhysX poses, actual solver nodes and locked side-specific SDF sources. Cached flags/distances alone are not trusted.
2. Revalidates the native getter radius and raw/derived/source/clock identity using the existing packet check.
3. Re-evaluates each required Pinch pad's SDF with that radius, retaining signed gaps, node IDs, lattice coverage and raw per-pad samples.
4. Computes the mean of nonnegative minimum **sampled-node** gaps over the two required pads in each hand. Missing lattice coverage is NA, never an average over only available pads or an interpolated hidden point.
5. Supplies only `approach_distance_m` to R0. It does not supply contact, stability, support or slip. `contact_observed=null`, `contact_inferred_from_geometry=false`; geometric penetration does not establish real contact.
6. Pins implementation/source hashes and same environment/episode/reset/physics/solver clock. Units m, frame World-to-actual-pad-local, uncertainty remains unknown.
7. Refuses Tripod/Power substitution for the historical Pinch recipe.

`scripts/replay_dh116_native_parameter_r0.py` adds an opt-in `--measured-approach` mode. Default original/v2 comparison remains unchanged. The new mode uses exactly the same pinned receipt, checkpoint-independent actual actions and two complete historical controls. No reset metric is borrowed from a post-step sample.

## Tests and failures

**101 CPU tests passed, 0 failures, 47.07s**:
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_r0_measured_approach.py tests/test_dh116_native_parameter_r0_replay.py tests/test_dh116_r0_applicability.py tests/test_dh116_r0_applicability_history.py tests/test_dh116_measured_closure_geometry.py tests/test_deformable_r0_replay.py
```
Log: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-approach-regression-KHOk3j/pytest.log`.
The new source contributes 15 tests including tampered pose/clock/nodes/SDF/radius provenance, missing coverage, unknown contact, grasp-specific refusal, isolation and actual-cohort replay. Missing-module red log retained at `evidence/r0-approach-red-dbsiPH/pytest.log`.
The independent source-owner and historical R0 modules are unchanged.

## Same-cohort offline result

Historical native receipt SHA: `f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901`.
Ledger SHA: `1981d68067f6554e63a27b190b9eb55f8663501ebdc9daca4700c73929e4ba2a`.
Final comparison SHA: `a856c2927ca4101ccaac2bf58e81f3ab3b57b42e77b16a3184cdb923d9d035a3`.
Final output: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-measured-approach-final-rDfrsA/comparison.json`.
Native imports: []. New physical steps: **0**. PPO updates: **0**.

| Actual physics step | Left approach gap mean (mm) | Right approach gap mean (mm) |
|---|---:|---:|
| 3 | 1.833953 | 2.160681 |
| 10 | 1.833953 | 2.160681 |
| 11 | 1.833569 | 2.160378 |
| 18 | 1.822415 | 2.150581 |

These are positive sampled-node SDF approach distances, **not palm lift, object lift or contact measurements**.
The second control's original difference calculation now returns 0.00001081894151866436 m, i.e. 0.0108189415 mm average approach progress. No action or physics changes produced this comparison.

| Mode | Control 0 approach valid samples | Control 1 approach valid samples | Remaining blocked terms in control 1 |
|---|---:|---:|---|
| Original R0, radius available | 0/8 | 0/8 | approach, hold, material slip, instability |
| Applicability v2 only | 0/8 | 0/8 | approach, hold, instability |
| v2 + actual geometry source | 7/8 | 8/8 | hold, instability |

The first post-reset approach sample remains unknown as a *difference*: the exact metric was not captured at reset. All 16 post-step distances exist, giving 15 legitimate differences. No retroactive reset fill.
The real pair ledger still has **0 hand–rod pairs** and **7872 table pairs**. Total diagnostic reward remains unknown because stability/hold sources are unqualified; training_reward stays null. The coefficient vector is explicitly the old CPU fixture, not a physical PPO reward.

Reproduction (choose a new output path):
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/replay_dh116_native_parameter_r0.py --measured-approach --output /path/to/new/comparison.json
```

## Remaining dependencies

- Capture this exact metric at a genuine reset endpoint in a newly frozen recipe.
- Bind a defined measured stability predicate; do not label absent errors as stable grasp.
- Obtain actual bilateral contact, material slip and external-support/hand-table provenance.
- Qualify the remaining 371D channels, reset owner order and production R0 coefficients.
- Tripod/Power retain independent collision/contact prerequisites.

Teacher admission remains **0/3**. No native budget reused, no checkpoint/model/asset or physics changes, no raw data uploaded. R2 stays disabled and PR #2 remains Draft.
