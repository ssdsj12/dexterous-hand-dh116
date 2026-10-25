# Direct-close PPO mimic recovery, 2026-10-23

Scope: fix the measured right index coupling instability before continuing PPO.
Rigid proxy, 12 direct finger position targets, controller palms, object-based
success without four-pad gates remain as approved. All Isaac runs are serial,
CPU0/one numerical thread. No source/assets/logs uploaded.

## Evidence and isolated comparisons

The diagnostic wrapper `/tmp/dh116_replay_failure.py` replays recorded 60 Hz
policy actions through the unchanged real backend and normal 12-step settling
reset. It uses the first row's shared root offset; no contacts, rewards or joint
states are restored from replay rows. The seed remains the original measured
close snapshot. Substep measurements are saved at 240 Hz. Although the failure
matched exactly, this is not restoration of a PhysX solver checkpoint.

| Run | Change relative to failing setup | Outcome |
| --- | --- | --- |
| v22 episode 5 replay | none | same error -0.0201752962 at 2.575 s |
| v22 passive-speed diagnostic | follower limit .3 -> 3 rad/s only | prefix peak .0188142, no hard error |
| v22 recovery diagnostic | follower-consistent master target only | prefix peak .00844743, no hard error |
| v22 recovery extended | recorded prefix then zero actions | full 15 s timeout, peak .00844743, no hard error |
| v23 PPO, recovery only | fresh policy, original velocity limits | stopped at 3754 steps, right finger22 -.0202644 |
| v23 episode 7 replay | none | same error -.0202644424 at 2.495833 s |
| v23 passive-speed diagnostic | follower limit .3 -> 3 in addition to recovery | prefix peak .0170326, no hard error |
| v24 full replay | both changes, recorded prefix then zero actions | 901 control frames, full timeout, no hard error; control-frame peak .015791 |

Prefix-only diagnostic output has no success qualification. Full timeout means
physical guards stayed within bounds on that trajectory, not object pickup.
The wrapper `replay_rows` field in early runs was written after flushing and is
zero; the actual row/substep counts are in `episode_0000.json`/`substeps.json`.
Later wrapper versions report physical sample count instead. Use the new CLI's
explicit `frames_used`/`source_frames_used` for unambiguous future comparisons.

## Repair

The old warning target was `q_master`. With error
`e = q_follower - ratio*q_master - offset`, this leaves the displaced coupling
constraint in place. Recovery now targets
`q_master + e/ratio = (q_follower-offset)/ratio`, clips to coupled joint limits,
and retains the .05 rad/s command limiter. Other fingers can still explore;
palms pause on the existing .004 rad warning, hard mimic limit remains .02 rad.
The safety recovery target may exceed the policy's local ±.08 envelope, but not
the coupled joint range. No measured joint state is overwritten.

V23 showed this control repair alone was insufficient: on a different action
sequence, follower velocity saturated while its master moved oppositely. The
passive joint solver limit is now 3 rad/s only in direct mode; active drives stay
.3 rad/s, active effort .15 Nm, gains 10/.2, command speed .05 rad/s. Passive
joints still have zero drive gains/effort. Legacy residual settings are retained.
This is a solver constraint change, not a proven hardware speed specification.

The installed IsaacLab actuator configuration documents that a tight solver
velocity bound can hinder convergence. NVIDIA's [stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/guides/articulation_stability_guide.html)
also discusses competing contact/drive/mimic constraints. These references support
the test hypothesis; the local controlled comparisons provide task evidence.

Checkpoint contract now records `follower_consistent_recovery_v2` and both active
and passive velocity limits. V22/v23 checkpoints cannot be silently resumed into
the changed semantics; v24 starts fresh.

## Reproduce the latest full replay

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --output runs/NEW_REPLAY_DIRECTORY \
  --control-mode joint_target \
  --close-reset-snapshot runs/tabletop_direct_close_seed_20261023/snapshot.json \
  --replay-episode runs/tabletop_direct_object_train_20261023_v23/episode_0007.json \
  --replay-zero-steps 901 --headless --device cpu
```

Replay executes no PPO and never sets top-level policy qualification. It marks
source exhaustion/partial episodes separately from real termination. Physical
failure produces exit 2. `max_mimic` in this CLI samples control frames, unlike
the diagnostic wrapper's 240 Hz substep peak.

## Training validation

V24 completed 8192 sampled steps / 244 optimization epochs, then 5 deterministic
policy evaluations; all physically safe, exit 0. Same seed42, reset jitter .3 mm,
checkpoint period1024. All evaluations reached lift but timed out: grasp 0/5.
Across logged control frames, peak mimic error .0180013 rad and maximum object
bottom .3000163 m: no meaningful pickup. This is finite-run stability evidence,
not a guarantee for arbitrary exploration.

Nine focused files: 186 passed, 1 skipped. Includes all ten mimic mappings and
both error signs, coupled-limit clipping, replay partial/physical failure
reporting, and direct continuation argument/SHA guards. py_compile and git diff
--check passed. Full repository tests were not rerun.

V25 serial continuation started from v24, up to 8 rounds of 65536 new steps each.
Monitor: runs/tabletop_direct_continuous_20261023_v25/monitor.json. Stops on
physical error and resumes only safe budget exits; runtime sources/reset inputs
are pinned. Actual future outcomes must be read from its monitor/round summaries.

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/continue_dh116_tabletop_ppo.py \
  --resume runs/tabletop_direct_object_train_20261023_v24/ppo_checkpoint.zip \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --close-reset-snapshot runs/tabletop_direct_close_seed_20261023/snapshot.json \
  --output runs/tabletop_direct_continuous_20261023_v25 \
  --timesteps 65536 --max-rounds 8 --checkpoint-every 2048 --eval-episodes 5
```
