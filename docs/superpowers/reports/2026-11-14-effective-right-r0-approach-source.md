# Current effective-right Pinch approach source — 2026-11-14

## Scope and root cause
CPU / read-only historical native-data work. The old `SegmentMeasuredPinchApproach` correctly rejects the current `EffectiveRightGeometry` version. An explicit `EffectiveRightMeasuredApproach` now reuses the original distance algorithm with the matching original geometry validator; neither legacy readers nor default native entry are replaced.

The metric is the mean positive sampled-material-node gap to the two required locked pad SDFs on each hand. It uses actual SDK link transforms, actual solver nodes and native radius, in metres. It is not an exact continuous surface distance, contact sensor, fixed material anchor, hardware sensor or success signal. Missing lattice coverage remains unknown.

Actual post-Reset body getters define the baseline. No authored pose and no first stepped frame substitutes for Reset. Tripod/Power remain rejected by this Pinch-only adapter.

## Tests
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_effective_right_r0_approach.py tests/test_dh116_r0_measured_approach.py tests/test_dh116_r0_reset_approach.py tests/test_dh116_segment_r0_approach.py tests/test_dh116_bilateral_r0_approach.py -q --tb=short
```
69 passed, 72.11 s. Before implementation: 7 expected missing-adapter failures and 1 passing legacy-rejection case.

The new test also feeds the same actual Reset and first eight physical samples through the unchanged original R0 reducer. Approach has 8/8 valid samples and its first raw reward equals mean(reset distance) minus mean(first measured distance), tolerance 1e-15. Hold, material-anchor slip and instability remain blocked; training reward is null. No default training path was enabled.

## Read-only full ledger
Source receipt `208b6c6d16f7f7e23c8e72a721343bc74276d242fcc4ef0dd71b6d16182df508` remains native-budget-failed. Source physical SHA:
`4d2b457e7e0c581c0aed139b25e2531f1554bada54ef7b836825c833b0536298`.

All 232 complete measurements produced valid geometric distances. Values below are **sampled positive SDF gaps**, not lift or effective contact:

| Boundary | Left (mm) | Right (mm) |
|---|---:|---:|
| Actual Reset | 1.149197342 | 0.561854336 |
| First physical sample | 1.149220625 | 0.561890891 |
| Last physical sample | 1.147792442 | 0.000000000 |
| Minimum over 232 | 1.147374045 | 0.000000000 |
| Maximum over 232 | 1.149220625 | 0.561890891 |

Zero means the positive-gap metric reaches zero; it does not establish original inner-pad contact, opposition, hold or success. In the same source run the right raw contacts were rejected by the original inner ellipse and left raw contact was absent.

Private results:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/effective-right-r0-approach-20261114/`.
All-sample SHA: `fb1e30e5a14e85e0e6c7be0bf12728f400955b4382fbadf3c16d046eab74443b`.
Summary SHA: `fbcad74267fe13f2e8abb766803ee01937b83307506533348720817a5b3a70ca`.

## Remaining gates
This package adds a usable measured approach source and proves first-window R0 wiring on historical samples. It creates no native recipe or new physics, no Teacher model or training success. Future live integration requires its own frozen source/Reset binding and bounded native evidence. Hold/slip/stability and full training-state/reward admission remain unresolved; no thresholds, geometry, actuator limits or historical checkpoints changed.
