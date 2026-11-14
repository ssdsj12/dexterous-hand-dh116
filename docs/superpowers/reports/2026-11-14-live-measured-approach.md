# 2026-11-14: actual reset and substep approach source connection

## Scope and verified source

This package connects the already validated effective-right measured approach
reader to an opt-in live R0 session. It adds no SDK step, command, control
boundary, reward coefficient, physics parameter, grasp success or PPO update.
The original LiveSession/LiveR0Observer defaults remain unchanged.

Source native run tested HEAD `a3dbd5f25136e8ffc131329c44a5d306a4bd4fbf`:
receipt SHA `5481fb005ccc3d59197c7489a795f7623b1fbe36dc0bc8978a3760d48636b41b`.
Its 234 total SDK steps (2 setup + 232 measurement), 29 complete controls,
natural exit 0 and 159.2748216369655 s are **prior native evidence**.
The separate complete post-exit audit took about 84.28 s; 159.27 s is not an
end-to-end audit runtime. Source scope is sealed-owner/full-post-exit execution,
not full R0, positive contact or Teacher admission.

Physical JSONL SHA:
`f00c545d2464a4ba0776025a8a3fe6c3c1e2187ee81c6c46a8ef4fb67c0e32cc`.
Original live R0 SHA:
`0f8e814fd80adeaa2fc3c2ee4309d70cdf7c8dbc0260d9618879a267ae3c58dd`.
Raw evidence remains private and read-only.

## Actual defect and changes

The previous live initializer explicitly passed `reset_approach=None`, and
its record path supplied no auxiliary approach measurement. Valid geometry
stored in the same physical journal therefore could not become an R0 approach
measurement.

- `scripts/prepare_dh116_live_measured_approach.py`:
  `MeasuredApproachObserver.complete_physical_reset` verifies actual completed
  reset, native parameter packet and locked SDF geometry via the existing
  reader; stores the real reset baseline and provenance.
  `record` recomputes each same-clock sample, feeds only the measured approach
  auxiliary to the unchanged reducer, and retains full measurement evidence.
  Caller overrides, mismatched geometry/radius/clock fail closed.
  `end` keeps returned and cumulative baseline-availability metadata equal.
  `MeasuredApproachSession.initialize` temporarily binds the new observer to
  the existing initializer and restores the binding on success or exception.
  No native handles are retained by the connector.
- `scripts/replay_dh116_live_measured_approach.py`: pinned-receipt/physical-SHA
  CPU replay; compares every other physical and R0 term history to the saved
  original row, rejects missing/extra rows and verifies original files unchanged.
- `tests/test_live_measured_approach.py`: actual native first-window replay,
  old default regression, source corruption/caller injection, exception-safe
  owner binding, report consistency and non-approach comparison tampering.

## Read-only full-cohort result

The same completed source cohort produced 232/232 valid measured approach
samples and 29/29 complete control windows. Every non-approach physical fact,
guard, R0 term and term history compared exactly equal to the source.

| Distance, mm | Left | Right |
|---|---:|---:|
| Actual post-reset | 1.149197342 | 0.561854336 |
| First measurement | 1.149220625 | 0.561890891 |
| Last measurement | 1.147792442 | 0 |

This is a positive discrete node-to-pad SDF gap, not exact continuous cable
distance, effective pad contact, load-bearing evidence or calibrated uncertainty.
The source has zero effective-region candidates on both hands, even though the
right raw pair count is 2129. Right distance zero is not grasp success.

All 29 controls still block `hold_progress`, `material_anchor_slip` and
`instability`. Training reward stays null; Teacher admission stays false.
232 per-frame source identities are preserved. No historical source evidence
was merged into a fake new native run.

CPU replay: 42.80747943802271 s; output JSONL SHA
`33bdbd1c2a0cd5409a5987d288f58e48122430afc1959297425cbe9cb6fbd6f3`.
The full replay emits returned control receipts; the subsequent cumulative
report metadata fix changes no measured or reward values.

## Tests and failure evidence

Initial red: eight tests failed because the live connector did not exist.
Initial green: eight passed. Broad old/new regression: 43 passed in 202.17 s.
An additional report consistency assertion then reproduced one failure:
the returned control was corrected but the observer's retained report still
said the reset baseline was unavailable. Both are now updated together.

Final validation: **19 passed in 181.38 s**, zero failures. It is recorded in the registry CPU result and private
`live-approach-20261114/regression-02.log`; the publication transaction reruns
the explicit new/reader regressions before committing.
Commands (Ubuntu existing environment, no Isaac launch):

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_live_measured_approach.py tests/test_effective_right_r0_approach.py tests/test_fast_r0_native_session.py
```

Replay uses `python -m scripts.replay_dh116_live_measured_approach` with explicit
`--receipt`, `--expected-receipt-sha256`, `--expected-physical-sha256`,
`--output` (new directory) and `--project`.
Logs, original receipts and 32 MiB replay ledger are private, not committed.

## Remaining work

Freeze an independent recipe and post-exit equality validator for the new live
source, then use only its standing-authorized bounded scope. CPU replay is not
proof it fits the native 180 s wall budget. Do not retry the consumed original
or sealed recipe. Hold/slip/stability, full 371D source availability, valid
bilateral load-bearing and separate Tripod/Power admission remain unresolved.
Do not start formal PPO, claim full R0, or change the 30 mm / 5 s / 0.5 mm rules.
