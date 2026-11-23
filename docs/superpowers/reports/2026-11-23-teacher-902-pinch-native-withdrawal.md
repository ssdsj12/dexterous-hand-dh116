# First native-verifier leg for the staged 902 Pinch route

The first executable leg reuses the original Pinch acquisition, supported Reset, and exact 24-window approach through SDK 432. It then submits only the eight R6 index-withdrawal windows, through SDK 496. Their original action vectors keep translation at zero and preserve the recorded float64 input, including the tiny wrist arithmetic residuals. The plan excludes wrist alignment, finger seating, contact, and lift phases.

The replay gate verifies the actual SDK 240 parked state, the original action request in each completed row, and the sealed SDK 425–432 control, geometry, rod and force history. The 64 withdrawal transitions must retain both end supports with positive load on each original solver substep. The shared hands-clear geometry checks remain in the control path. Each observed withdrawal row must return the expected left/right finger and carriage targets as identical binary32 bytes before another command window can begin. Both native q and qd remain present in the ordinary readback, separate from the target comparison.

The standalone CPU screen reproduced all eight original archived approach transitions and all 64 expected withdrawal target buffers. The minimum across all 22 source/cooked rod envelopes per hand was **0.763687 mm** on the archived rod frames under ideal commanded tracking. Cross-hand Y-axis separation was at least **320.417 mm**. Final index targets were 0.6834389567 rad left and 0.6381822824 rad right. The sealed CPU proof SHA-256 is `e2448612841d9f1b18e6f703ada649be737a5630bc186f75c9a9b149c6e8db76`. This is a planned-target and archived-geometry check. Native tracking, loaded contact, hold, and lift remain unverified.

The new launcher calls the generic shared replay hook when an exact approved plan is supplied. The isolated CPU plan pins the current shared source, so the final plan must be regenerated after integration with the generic hook before any approved native launch. No launch occurred in this work package.

## Fixed nonparticipating digits

The 64 withdrawal rows leave local active slots 3, 4, and 5 on both hands at their bit-identical SDK 432 backend values. All six input channels are exactly zero in the eight new command windows. The unmodified 24-window archived prefix contains a normalized left-middle command of (3.3038970864707196×10^{-10}) and left-middle backend drift of (3.944027396340921×10^{-11}) rad from the first controlled sample. The frozen recipe records this distinction. The previous full R6 aim and seat plan also includes residual nonparticipating inputs up to (4.0978193283081055×10^{-6}). A strict fixed-digit prefix or later phase needs separate re-certification from the applicable original state.

## Freeze and preflight

The isolated plan caps the run at SDK 496, with 32 command windows and 5,928 solver substeps. Its canonical fingerprint is `81207a97585ecbfd51a6eb451957dcd9441fa86b5c119ee4a1424de501657d3e`. A GDB run of the verifier with only `--verify-existing-plan` reproduced that fingerprint, verified 2,246 pinned files, and exited naturally with code 0. The output directory assigned for a future approved native run stayed absent.

Evidence and exact command/environment JSON files live in `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119`. The CPU freeze uses `python -B -m scripts.verify_dh116_902_pinch_withdrawal` with the original acquisition inputs and the new archive/CPU-proof arguments recorded in its argv JSON. The GDB argv prepends `gdb --batch --nx --return-child-result` and includes `--verify-existing-plan`. After integrating the generic runner, use the integrated worktree paths and fresh plan/output paths for both commands.

| Evidence | SHA-256 |
|---|---|
| `sync-pinch-round7-withdrawal-cpu-v2.json` | `e2448612841d9f1b18e6f703ada649be737a5630bc186f75c9a9b149c6e8db76` |
| `sync-pinch-round7-withdrawal-frozen-v2.json` | `ca709bf583cd09e1dde145e91edc7dbdf0af0c95a101d91c09248b8cce52c8dc` |
| `sync-pinch-round7-withdrawal-freeze-v2-argv.json` | `a14fb93df4fbabec4b9ed256468314b3e706aca24ca210cf6206abba7db681a0` |
| `sync-pinch-round7-withdrawal-gdb-v2-argv.json` | `da19b531581a5e8e517087870cef8d3bb4a35b79f7c52ba4916d28f02c65f3a9` |
| `sync-pinch-round7-withdrawal-gdb-v2.log` | `e2808e26e73440e54e5093869d088132df942647bda1f2ac482b32fa16e79f5a` |
| `sync-pinch-round7-withdrawal-regression-v2.log` | `fb70579587208f1b89ddba0d70389cb747da63f76b42fcf25a4856b8f43ffce3` |

The final regression passed **190 tests**. Its gate test replays the authenticated original 192-row approach ledger and rejects a one-bit change to an expected float32 withdrawal target in a synthetic next row.
