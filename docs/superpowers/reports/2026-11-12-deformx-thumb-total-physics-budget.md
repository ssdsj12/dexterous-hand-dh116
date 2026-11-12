# DH116 thumb calibration total PhysX budget

Base `3120e87cceb5b57c9058d58b4fc055f610deae94`.

## Historical evidence and budget discrepancy

The approved thumb batch executed one left and one right calibration serially. Both outer processes and GDB inferiors exited naturally with code 0. Each measured ledger contains 240 continuous rows and 68 contact rows. The measured SDK counter runs from 3 through 242 on both sides.

In the installed Isaac SDK, Timeline PLAY dispatches `SimulationManager._warm_start()` and then `_create_simulation_view()`. Each invokes `update_simulation(get_physics_dt(), 0.0)`. The SDK increments the counter from subscribed physics-step events. The probe called `sim.reset()` before measuring either thumb. Actual use was 242 PhysX steps per side and **484 in total**, four above the authorized 480. The older recipe bound the measured ledger length and omitted the two SDK setup steps per side.

The original contact ledgers and natural-exit records remain historical diagnostic evidence. The batch failed the total-budget requirement. The original attempts, approval, code fingerprint and receipt bytes remain available for audit.

## Corrected calibration recipe

- `--steps 240` sets the total PhysX budget per side. Setup consumes two ticks and leaves 238 measured ticks. The probe executes the original command phases 1 through 238, preserving the velocity, position endpoint and timing of the approach. This removes two stationary tail samples.
- The probe checks the actual SDK counter before and after reset. It checks the current timeline and remaining budget before each measured step, then requires an increment of exactly one. The result records setup, measured and total counts and the SDK counter boundaries.
- The optional scheduler `validator.physics_clock` ties setup and measured ledger length to the total in the frozen recipe. Independently it checks the result counters and the phase and SDK indices of every measured row. It stores the validated counters in the receipt. Existing recipes without this option use their existing artifact-validation schema.
- The right probe accepts a qualified left scheduler receipt with a verified budget and matching ledger. The legacy direct evidence option for this probe has been removed.
- The two corrected thumb recipes have new command and code/config fingerprints. The old approval and results confer no launch or qualification for them. A future native batch requires separately frozen approval. This change does not run any new experiment. The scheduler's full-source fingerprint also changes for other native recipes, preserving their recorded evidence under its original identity.

Future budget examples for Teacher and Hold are CPU planning calculations. An example with 512 measured environment rows does not include SDK initialization and does not authorize 512 total physical ticks.

## Verification

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_work_packages.py
```

The initial focused regression failed in all three intended places. The combined CPU regression then passed 61 tests in 17.75 seconds. Tests cover 2 + 238, rejecting 2 + 240, early or missing setup ticks, extra measurement ticks, paused timeline, ledger gaps, changed final counter, the preserved approach curve, historical-left rejection and stale approval. The protection inventory matched. Read-only inspection of the actual old approval showed no launch or qualification for either corrected recipe. No Isaac instance was launched for this work package.

## Final SDK tick before callback release

The installed SDK disables `/app/player/playSimulations` around the STOP update. The thumb probe now sets and verifies that flag before reading the final counter and releasing its owned SimulationContext. The probe compares that counter with the measured ledger and records the observed value in the ownership log. It preserves the existing ownership order and reasserts the flag before the STOP update and App close. A late tick fails the process after guarded cleanup. If the flag cannot be verified, the probe avoids calling App update or close.

Four CPU fakes exercised the actual probe lifecycle for normal completion, a failed run, a late tick, and a rejected freeze. The combined CPU regression passed 65 tests in 16.72 seconds. This guard requires native validation under a new authorization and fingerprint.
