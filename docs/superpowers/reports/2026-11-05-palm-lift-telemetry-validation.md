# DH116 actual palm-lift telemetry validation — 2026-11-05

## Measurement and scope

The user requested actual left/right palm upward displacement alongside actual object lift. Each palm is the measured Isaac articulation `base_link` body-link pose, in left/right hand order; world Z is compared with the measured world Z immediately after that environment's physical episode reset. This is neither a commanded wrist target nor the nominal reset seed, object height, rod endpoint height or hand COM. The rod retains the existing net-lift definition, excluding the support's initial 60 mm elevation.

`PalmLiftTelemetry` observes every live physics substep before any terminal reset. A selected physical reset replaces only those reference rows and preserves already collected terminal peaks. The task bookkeeping reset at the end of a control tick does not replace the physical reference. Telemetry retains signed negative values, uses validity masks and clones its outputs; no RNG is consumed.

Each completed 32-control-step rollout aggregates, separately for each grasp style:

- `left_palm_lift_max_m`, `right_palm_lift_max_m`: maxima over valid live physical substeps in that rollout.
- `left_palm_lift_last_mean_m`, `right_palm_lift_last_mean_m`: group means of each environment's last valid sample in the rollout. These are not necessarily simultaneous samples.
- `left_palm_lift_max_env_id`, `right_palm_lift_max_env_id`: global environment IDs attaining each independent peak.
- `left_palm_lift_valid_envs`, `right_palm_lift_valid_envs`: valid environment counts. Missing groups return null values and zero valid count; masked non-finite samples cannot manufacture zero peaks.

Left/right maxima can come from different environments and instants; a table of maxima is not a paired-hand trajectory or a successful-grasp claim. Historical logs preceding this instrumentation have no actual palm displacement and remain unrecorded. The telemetry is optional/default-off for general TensorEnv consumers and enabled by visual training. It does not change actor/critic observations, actions, rewards, success criteria, model contract or checkpoint layout.

## CPU regression

Executed on the server in `/mnt/mydisk/xk/projects/dh116-palm-lift-20261105`, using the installed Isaac USD libraries but no Isaac simulator instance:

```bash
usd_root=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$usd_root:src" LD_LIBRARY_PATH="$usd_root/bin:${LD_LIBRARY_PATH:-}" \
  PYTEST_DEBUG_TEMPROOT=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
  /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
  tests/test_palm_lift*.py tests/test_vision*.py tests/test_physics_cloning.py \
  tests/test_native_height_batch.py tests/test_depth*.py tests/test_front_depth_camera.py \
  tests/test_mixed*.py tests/test_dh116_mixed_reset_launcher.py tests/test_surface_*.py \
  tests/test_dh116_tensor_env.py tests/test_combined_failure_reset.py \
  tests/test_object_drop_reset.py tests/test_continuous_motion*.py -q --tb=short
```

Result: **1429 passed, 2 GPU tests skipped, 83 subtests passed**, 58.80 s. One pre-existing zero-element tensor initialization warning. Targeted red/green tests covered actual-vs-commanded measurement, partial and terminal resets, negative-only and missing samples, rollout aggregation, per-style IDs, mm TensorBoard tags, unchanged default-off transitions/RNG and absence from model contracts. Independent code review found no P0/P1/P2 defect.

## Native continuation

The previous fixed-camera run reached complete update 129. Its checkpoint was copied, loaded on CPU and SHA-validated before sending SIGINT only to its verified training PID. The operator-triggered stop's unfinished next iteration is excluded. The server was not rebooted. Original checkpoints and run data remain intact.

- Source: `/mnt/mydisk/xk/dh116/runs/palm_lift_switch_20261105/source_visual_checkpoint_iter129.pt`.
- Source SHA256: `5c85bc7a9516f9124a5f6dc99d39fe185016c60c0dff6f1c97346b2d429ecd0c`.
- Source complete counters: iteration 129, 4718592 transitions, 2669 episodes, successes 0.
- New run/tmux: `dh116_gpu3_fixed1024_palm_lift_resume129_20261105T122312Z` under `/mnt/mydisk/xk/dh116/runs/`.
- New code: `/mnt/mydisk/xk/projects/dh116-palm-lift-20261105`.
- Same configuration: GPU3, 1024 physical environments, pinch/tripod 512 each, fixed 224x96 RGB-D, `usd_parse`, explicit collision groups, Kit32, rollout32, micro512/infer256, `support_deficit`, absolute update target2048.
- Strict same-contract `--resume` from source129; no downsize flag or legacy privileged actor was used. Policy/Adam/LR/RNG and global counters were restored. As with ordinary resume, physical state and sensor history begin from reset rather than serializing the old physics trajectory.

The new training process completed two full PPO updates:

| Update | Transitions | Seconds | Episodes | Native successes | Camera/object valid |
|---|---:|---:|---:|---:|---:|
| 130 | 4751360 | 39.7723 | 2669 | 0 | 100% / 100% |
| 131 | 4784128 | 44.6096 | 2669 | 0 | 100% / 100% |

Both updates report 512 valid left-palm and 512 valid right-palm environments in each grasp group. Update131 measurements, converted from recorded metres:

| Grasp | Rod net peak mm | Left palm peak mm | Right palm peak mm | Last left mean mm | Last right mean mm | Left/right peak env |
|---|---:|---:|---:|---:|---:|---|
| Pinch | 0.001228 | 0.502139 | 0.542551 | -0.125745 | -0.128602 | 129 / 365 |
| Tripod | 0.170358 | 0.569493 | 0.861406 | -0.042936 | 0.055240 | 533 / 913 |

The visual policy's full rod-history chain is: original4096 run through update5; fixed1024 run through update129; this resumed run thereafter. At the update131 audit, historical rod peaks remained **pinch 0.949237 mm at90**, **tripod 2.533216 mm at57**. Palm history begins at130; all four historical palm peaks at this audit occurred at131. Old 371D privileged actors, demonstrations, deterministic tests and failed capacity tests are excluded.

A later read-only CPU checkpoint audit at update135 confirmed 4915200 transitions, all 80 tensors finite, and all 19 Adam state entries at step5400 (=135 x40). The source129 SHA256 remained unchanged. The 10 new mm scalar tags each contained six completed updates130-135, and the verified native PID150108 remained on GPU3. The inherited resume_migration source_iteration5 describes the earlier 4096-to-1024 conversion; training_start.json records this same-contract continuation from129. The two audit stages are distinct. TensorBoard logs `lift_mm/{grasp}/rod_peak`, `left_palm_peak`, `right_palm_peak`, `left_palm_last_mean` and `right_palm_last_mean`; these tags contain millimetres directly. Existing group metre tags remain available. The monitor is retargeted to the resumed run and retains quiet notifications during healthy progress.

## Limits and next step

This validates measurement, continuation and two complete optimizer updates; it does not establish reliable grasping. Native successes remain0. The goal remains2048 absolute PPO updates. Success still requires native object net lift30mm, hold5s and fixed-anchor slip at most0.5mm. The prior 1-second VRAM peak measurement belongs to the previous fixed-camera capacity validation; this change does not claim a new driver-memory peak from a single snapshot. Current native update rows report about21.263GiB driver usage.

Raw run logs, checkpoints, sensor captures, credentials and caches stay on the server/local artifacts and are not uploaded to GitHub.
