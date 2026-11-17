# Same-clock measured approach: bounded native preparation (2026-11-14)

## Change and scope

Independent entry `scripts/verify_dh116_approach_r0.py` binds only
`MeasuredApproachSession` around the already qualified sealed owner.
It preserves its atomic contact capture, live safety guards, actions, physical
configuration, actual reset, resource-release order and natural app.close.
All temporary bindings restore on normal completion and exception.

`scripts/validate_dh116_approach_r0.py:audit` reconstructs the observer from
the actual reset and native parameter packet, then independently recomputes
every pre-action identity, 480 Hz row and settled R0 window from physical.jsonl.
It compares whole live packets/receipts, not just reward sums or row counts.
Original physical/contact/lifecycle validators still run unchanged.

`scripts/run_dh116_approach_work_packages.py` refuses qualification unless
actual reset approach is available AND all 232 measured samples are valid.
Complete-but-unknown records do not pass this measured-source scope.
Full R0, positive grasp and Teacher admission remain false.

## Exact frozen recipe

- Recipe: `configs/dh116_execution/approach_r0_native_v1.json`.
- Recipe SHA256: `4e3b631cdde215a27bd0ea16881521f4e6fd48d4259ad17ff81b378d41f89838`.
- Experiment code/config SHA256: `d5251c8f71bf7eb470cc9f482dad8013726ed4ff577c6990b60204ebf4108ad7`.
- Single Isaac on GPU3, one environment, 0 PPO updates, 234 total SDK steps
  (2 initialization + 232 measurements), 29 controls, maximum 180 s, one attempt.
- Original physical inputs and control/physics limits are unchanged.
- Reuses the qualified native parameter receipt and requires the actual sealed
  execution receipt through scheduler dependencies. No invented output path.
- New scope: `actual_reset_and_same_clock_measured_approach_R0_234_steps_only`.
- No retry of consumed original/origin/sealed configurations.
- Complete CPU audit runs after natural exit and is reported separately.

The user's standing authorization covers a new frozen diagnostic in this
bounded scope. The scheduler still checks the actual fingerprint, prerequisite
receipts, protected files, resource availability and single-instance lock.
No long PPO, physics expansion, reward changes or success qualification follow.

## CPU evidence

Initial red: 9 tests failed due to the missing independent native entry.
The first implementation/regression passed 32 tests.
Three added tests then reproduced an admission bug: complete records with
missing/partial approach coverage were not explicitly rejected. The wrapper now
requires reset validity and 232/232 source coverage. Final validation: **35 passed in 153.96 s**, zero failures, recorded in the
registry and private `approach-native-20261114/regression-02.log`.

The real Python acquisition owner is tested with CPU SDK doubles:
all 232 physical samples and 29 controls, actual reset source, pending ledger
seal and full R0 recomputation. This is not new Isaac execution.
Tampered live reward data is rejected. Budget expansion and failed-owner binding
leaks are rejected. Existing source/ledger regressions are also run.

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_approach_r0_native.py tests/test_live_measured_approach.py tests/test_deferred_physical_audit.py
```

## Readiness and remaining work

Native execution has not occurred in this preparation package. Its runtime
cannot be inferred from CPU replay. The predecessor's 159.27 s natural owner
exit does not guarantee this additional source fits 180 s.
Run once only after publication and exact fingerprint binding, retaining
timeout/failure evidence without forced process exit or budget expansion.

Hold, slip, stability, full 371D sources, actual bilateral support and separate
Tripod/Power still block Teacher. Geometry gap zero does not imply contact.
R2 stays disabled; Lift 30 mm, Hold 5 s and Slip 0.5 mm remain unchanged.
