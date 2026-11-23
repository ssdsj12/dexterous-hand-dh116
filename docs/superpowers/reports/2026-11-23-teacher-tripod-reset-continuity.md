# Tripod measured Reset continuity and archived wire motion

Two independent native records were compared through their natural-exit receipts. The historical source is the 30-window zero-action settling run at `1c3668340457c8a25de2125c0248ba9cacdfaa6e`. The current source is the newly verified Tripod Reset at `d415dd7610b6614daaa60801cd275c994ba33886`. The source epochs and project paths remain distinct.

The comparison recovered 59 byte-identical physical arrays, covering rod node positions, velocities and directors, both hands' joint state, targets and body poses, the public observation validity and valid values, and all 44 source/cooked vertex arrays. The recorded task digest, Reset seed, common contract and solver parameters also matched. This confirms the same measured initial condition and makes the historical zero-action motion useful for approach planning. The current Reset is still the authority for current execution.

## Historical wire displacement (no issued action)

The audit reads all 241 SHA-verified rows and uses the completed priming tick as the first decision anchor. Each maximum is over every sample in the following eight SDK steps, so a return to the starting point at the last step cannot hide intervening motion.

| Material nodes | Largest within-window displacement across 30 control windows | Within-window displacement in the final control window |
| --- | ---: | ---: |
| All 41 nodes | 1.184 mm | 0.219 mm |
| Right candidates 2, 3, 4 | 0.298 mm | 0.056 mm |
| Left candidates 33, 35, 36 | 0.544 mm | 0.105 mm |

The final window still had a maximum node speed of 13.804 mm/s. A measured initial-state match does not certify an active approach's shape, a motion bound under hand loading, or equilibrium. Use local reserves from the correct interval and check them on any actual continuation. The candidate replay must measure tracking and bilateral reaction as well as the current wire, before reaching the original Lift and Hold gates.

The current-source 30-window diagnostic was frozen through `scripts/python.sh` and reproduced by a CPU-only GDB rehearsal. Its exact SHA is `097f0da9dc214fd490f3fb3f23b91a86f6a802f05e4d525e07ceb1879620329d`. It remains unexecuted because the archived control answers the zero-action question for the identical measured starting state. No new native data or PPO updates were produced.

## Reproduce

```bash
python -B -m scripts.audit_dh116_tripod_reset_continuity \
  --historical-receipt "$SETTLING_OUTER" --historical-sha256 "$SETTLING_SHA" \
  --current-receipt "$CURRENT_RESET_OUTER" --current-sha256 "$CURRENT_RESET_SHA" \
  --output "$FRESH_JSON"
```

The private evidence directory `tripod-reset-continuity-r5-ux55arn4` contains the command and result SHA-256 `11c8b053e5223313a9c405dca134c045f2291b8b12675bab480a56cec9936aac`. The 72 focused CPU tests passed. The test log SHA-256 is `37e8c9c82922d6dd2ffc47f7837f4be63d758df2680e60e65e428eeec49ac590`.
