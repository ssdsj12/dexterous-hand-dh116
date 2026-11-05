# 2026-11-05 fixed-camera throughput validation

The user changed the priority to throughput and allowed 2048 or 1024 environments instead of forcing 4096 into memory. Lower-resolution RGB-D was explicitly permitted. The selected run uses 1024 physical environments, equally split between pinch and tripod, with 1024 stationary cameras at 224x96.

## Implemented changes

`FixedVisionCamera` captures all environments in one tiled product. It never moves cameras or changes visibility. After initial warmup, each sample starts with one render and accepts only fresh frame metadata aligned to the current physical time; time, step and palm FK stay frozen during capture. Numeric camera IDs are independent of USD traversal order. At most two first-observation RGB-D/calibration/FK/point-mask captures are saved for diagnosis; these contain no object-truth labels.

The previous bounded provider remains available. CPU float32 pinned rollouts, logical PPO microbatches and exact CAD geometry are retained. Actor3844D, critic371D, 128-point masked inputs, four-frame history, 24D actions and the native success definition remain as documented. The new fixed images preserve horizontal field of view and use less vertical background; actual intrinsics are verified.

An explicit `--vision-resume-downsize` accepts only the specified4096-to2048/1024 camera/count/resolution transition. Model, Adam, LR, RNG, absolute update and transition counts are retained. Counters and both known critic reference arrays select the same subset from each grasp group. Unknown differences and numeric/bool/type substitutions fail. `resume_migration.json` and later checkpoints retain the full source contract and retained/dropped totals. Physical state and sensor history restart, as in regular resume.

## Initialization and capacity observations

On this server, fast PhysX replication stalled in `replicationAttachEndFn.replicate` with 32 Kit workers; a 64-worker comparison aborted with `malloc(): invalid size (unsorted)`. Neither reached camera creation and neither is a camera-capacity result. The installed IsaacLab fallback loads inherited USD physics without replication and calls `filter_collisions` with the actual physics-scene path and all environment roots. In the2048 trial this passed reset/bind in about 27.39 s. Original geometry, materials, actuators and solver settings are loaded; this is not a promise of bitwise-identical trajectories between initialization paths.

| Environments / fixed cameras | Image dimensions | Native result |
| --- | --- | --- |
| 2048 | 320x240 | Vulkan `ERROR_OUT_OF_DEVICE_MEMORY` allocating NRD transient texture; no PPO update. |
| 1024 | 320x240 | Vulkan GBuffer allocation failures and no valid fresh RGB-D; no PPO update, exit2. |
| 1024 | 160x120 | Images valid, but pinch retained only 1 point/environment and tripod 0. Initial perception rejected; no PPO update. Initial camera driver memory19.283GiB. |
| 1024 | 224x96 | All1024 images and objects valid;55–80 initial points/environment, mean69.684. Worst per-environment P95 surface distance0.338320mm, below the existing3mm gate. Continued real PPO. |

The filter, calibration convention and GT audit threshold were not relaxed to admit224x96. Its wider pixel sampling resolves the thin rod while the shorter image limits background pixels. Actual first-observation images show both hands and the rod. Cropped vertical coverage and appearance-specific brown-pixel segmentation remain deployment limitations.

## Training evidence

Source checkpoint: preserved copy of the old4096 visual actor at update5,655360 transitions, SHA256 `6e5b0215f84eef59d16e015156a290b7eb31daf8ce93ff9b0055773071e25942`. Original files remain on the server.

Selected run: `/mnt/mydisk/xk/dh116/runs/dh116_gpu3_fixed1024_224x96_parse_mixed2048_20261105T104905Z`; staged source: `/mnt/mydisk/xk/projects/dh116-vision-fixed-20261105`.

- Updates6/7/8 completed in39.29/42.38/40.22s, approximately800 transitions/s. Every recorded final observation used one render and one camera batch. All three updates recorded100% image/object validity and no failures.
- Checkpoint update8 contains753664 transitions; all 80 tensors are finite, with 19 Adam states all at step320. Migration metadata remains present.
- One-second driver sampling through update7 recorded21776MiB (21.266GiB). Torch peak allocated/reserved memory was1.009/1.381GiB. Sampling is not a hard upper bound or long-run memory guarantee.
- Native successes and completed episodes were0 at the checkpoint check. This validates capacity, sensing, conversion and optimizer execution; it does not establish grasp reliability.

The older4096 run required approximately400s/update and gathered131072 transitions/update. The new1024 run gathers32768, with smaller images. The approximate tenfold reduction in update time includes the smaller rollout and resolution; measured aggregate throughput rose from about309 to about800 transitions/s. It is not a controlled single-factor performance comparison.

## Commands and regression

Native launch in the activated environment uses GPU3 by UUID and a valid NVIDIA Vulkan ICD:

```bash
bash scripts/python.sh scripts/run_dh116_surface_student.py \
  --output /mnt/mydisk/xk/dh116/runs/dh116_gpu3_fixed1024_224x96_parse_mixed2048_20261105T104905Z --num-envs 1024 --updates 2048 --rollout-steps 32 \
  --vision-student --surface-feedback --surface-reward-profile support_deficit \
  --depth-camera-config configs/dh116_training/front_depth_camera_v2.json \
  --pregrasp-candidates configs/dh116_training/effective_pad_center_pregrasp_v1.json \
  --mixed-pregrasp-candidates configs/dh116_training/effective_pad_triad_pregrasp_v1.json \
  --vision-camera-mode fixed --physics-clone-mode usd_parse \
  --vision-width 224 --vision-height 96 --vision-visibility-mode all \
  --vision-ppo-microbatch-size 512 --vision-inference-batch-size 256 \
  --resume /mnt/mydisk/xk/dh116/runs/vision_fixed_switch_20261105/source_visual_checkpoint_iter5.pt --vision-resume-downsize
```

Final related regression:1194 passed, 2 GPU tests skipped, 83 subtests passed, one existing zero-element tensor warning. Red/green checks cover stationary/fresh camera capture, sensor-only evidence, strict resize projection/type rejection and clone/filter routing. The final command was:

```bash
usd_root=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$usd_root:src" \
LD_LIBRARY_PATH="$usd_root/bin:${LD_LIBRARY_PATH:-}" \
PYTEST_DEBUG_TEMPROOT=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
tests/test_vision*.py tests/test_physics_cloning.py tests/test_native_height_batch.py \
tests/test_depth*.py tests/test_front_depth_camera.py tests/test_mixed*.py \
tests/test_dh116_mixed_reset_launcher.py tests/test_surface_*.py -q --tb=short
```

Training continues toward absolute PPO update2048. Monitoring tracks this policy's current run plus its4096 predecessor only through update5; teacher, privileged-actor and unrelated capacity-test results are excluded. Raw captures, logs and checkpoints remain outside Git.
