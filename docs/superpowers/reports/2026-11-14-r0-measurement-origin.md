# R0 measurement origin / sample evidence separation — 2026-11-14

## Scope
Independent opt-in observer based on `f4945ad6e08255f2ef8412b6b75927c534b75b5c`. No original reward, control, physics, asset, actuator, model, success standard or legacy observer edited. No new Isaac launch, PPO update or training admission.

## Root cause and changes
The consumed complete-pair run naturally exited 0 but used 196.211877563 s, exceeding its 180 s allowance. Raw per-frame evidence SHA was embedded in each measurement source and repeatedly accumulated in R0 source sets.

`tabletop_r0_measurement_origin.MeasurementOrigin` binds four independent source origins (left, right, table contact, joint protection) to actual reset identity, immutable mesh/source bindings, SDK joint/view identities and implementation hashes. Original per-frame source strings, atomic packet SHA, effective-pad sample SHA and complete clock remain in each record. Unknown contact/slip/stability stays unknown. `restore_measurements` verifies the provenance digest and reverses this mapping. A changed origin/epoch/source is rejected.

`OriginLiveR0Observer` is an explicit subclass; the original observer remains default. Source labels are the only reducer inputs changed. Native objects are neither stored nor queried.

## Verification
CPU command:
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_r0_measurement_origin.py tests/test_r0_origin_replay.py tests/test_r0_live_observer.py tests/test_r0_after_physical_reset.py tests/test_live_effective_pad_source.py tests/test_live_effective_pad_epoch.py tests/test_live_effective_pad_accounting.py tests/test_complete_pair_capture.py tests/test_fast_owned_codec.py -q --tb=short
```
162 passed, 41.41 s. New tests first failed for missing implementation (14 and 8 cases); an earlier test helper import error was corrected before the valid red run. All logs retained privately.

Offline exact replay:
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.replay_dh116_measurement_origin --output <new-private-output>
```
The baseline was recomputed from original physical data, then compared exactly with the original saved live R0. The candidate was compared after only explicit, reversible provenance mapping. All reward values, coefficients, units, validity, blockers, phases, guard results, sample counts, clocks and actual actions agreed: 232 physical rows and 29 control windows. Tamper tests reject altered reward/validity/clock/blocker/guard/value/contact data.

| Metric | Original replay | Origin replay |
|---|---:|---:|
| Serialized replay bytes | 92,882,700 | 11,078,335 |
| Physical R0 compact bytes | 42,425,591 | 1,482,231 |
| Cumulative source occurrences | 162,168 | 1,392 |
| Unique composite sources | 928 | 4 |
| Sequential CPU replay wall time | 28.961 s | 30.348 s |

Storage improved; CPU wall time did **not** improve in this measurement. This does not prove the native budget issue resolved. Full atomic raw capture and per-frame provenance remain available. No sparse sampling, omitted physical steps, altered reward or weaker protection was used.

## Evidence
Private directory: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-measurement-origin-20261114/`.

- Source failed receipt SHA: `208b6c6d16f7f7e23c8e72a721343bc74276d242fcc4ef0dd71b6d16182df508`.
- Original physical SHA: `4d2b457e7e0c581c0aed139b25e2531f1554bada54ef7b836825c833b0536298`.
- Original live R0 SHA: `6581685565bdea48ef66f2b35fd670a1c8790856ab889da737ac17d1630f389d`.
- Exact replay result SHA: `5ce2de3c29aa8699b0f832617949aa202663fe292f1c802020c3d0b82b5f15f0`.

Raw ledgers remain private and read-only. The failed source does not become qualified through offline replay.

## Remaining gates / next package
Build a distinct bounded entry, independent same-clock provenance validator and owner regression. Freeze and publish that recipe before a once-only authorized native run. Do not repeat the consumed recipe. Actual approach/hold/material slip/instability sources, effective opposed contact, full 371D and training coefficients still block Teacher admission. Pinch/Tripod/Power and the original 30 mm / 5 s / 0.5 mm criteria remain separate and unchanged.
