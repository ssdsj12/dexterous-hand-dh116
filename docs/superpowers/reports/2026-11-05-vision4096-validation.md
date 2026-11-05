# 2026-11-05 visual student capacity validation

## Implemented behavior

The 3844D actor consumes four frames of real RGB-D surface points transformed into both palm frames, encoders, wrist FK, issued actions and sensor validity/timing. Object truth and contact state remain available only to the 371D critic, simulation rewards and independent audits. Teacher-action distillation is disabled because the archived privileged demonstrations do not contain synchronized camera observations.

Two grasp families coexist in each rollout, with half of the physical environments assigned to each. One shared policy learns from both. The absolute stopping condition is 2048 completed PPO updates; episodes are reported separately.

Memory changes retain the physical scene and reward definition: exact float64 CAD-height calculations are chunked; float32 rollout storage is pinned in host RAM; PointNet inference and PPO microbatches bound GPU activations. Weighted gradient accumulation retains 5 epochs, 8 logical minibatches and 40 Adam steps per complete PPO update. RGB-D processing uses exact 64-environment chunks independently of the camera pool.

Camera movement renders every environment at the same frozen physical time. The first render after moving is discarded; same-render-product timestamps and actual extrinsic readback are checked. Native anti-aliasing and frame generation are disabled and read back. The default all-visible mode avoids repeated USD visibility edits; optional batch visibility is a different strict resume contract.

## Offline and GPU component checks

The final related regression command completed with **1132 passed, 2 GPU tests skipped, 83 subtests passed**, and one pre-existing zero-element tensor initialization warning:

```bash
cd /mnt/mydisk/xk/projects/dh116-vision4096-20261105
usd_root=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$usd_root:src" \
LD_LIBRARY_PATH="$usd_root/bin:${LD_LIBRARY_PATH:-}" \
PYTEST_DEBUG_TEMPROOT=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest \
tests/test_vision*.py tests/test_native_height_batch.py tests/test_depth*.py \
tests/test_front_depth_camera.py tests/test_mixed*.py \
tests/test_dh116_mixed_reset_launcher.py tests/test_surface_*.py -q --tb=short
```

The CAD-height CPU/GPU parity checks separately passed 14 tests at 1e-12 tolerance. A GPU replay of actual captured RGB-D verified all ten postprocessing outputs bitwise equal between full-256 and chunk-64 processing, with point-coordinate maximum difference zero. Temporary allocated-memory growth fell from 1537.50 MiB to 385.85 MiB in that component test. This is a component measurement, not total Isaac memory.

## Native observations before the final capacity run

All runs used GPU 3, Isaac Sim 4.5, IsaacLab 0.54.4, RSL-RL 3.1.3, 320x240 RGB-D and 32 control steps per PPO rollout. Tests ran serially, preserving all outputs outside Git.

| Physical environments | Camera pool | Result |
| --- | --- | --- |
| 128 | 64 | Two complete PPO updates, about 42–43 s/update. |
| 512 | 256, batch visibility | Two complete updates, 116.25 / 115.07 s. This preceded the final rendering and postprocessing fixes. |
| 512 | 256, all visible, AA Off, postprocess 64 | Two complete updates, 53.96 / 57.95 s, terminal exit 0. Driver memory reported after updates was about 19.84 GiB. All 512 initial geometry audits passed; worst per-environment P95 was 0.390726 mm. |
| 4096 | 256, batch visibility, AA Off | Initial geometry passed, then the first control observation exhausted GPU memory before a PPO update. |
| 4096 | 256, batch visibility, AA Off, postprocess 64 | Initial geometry passed for all 4096 environments; observations continued without the previous allocation failure. Sampling took about 151.76 s/control step. The capacity experiment was deliberately stopped before a complete update to compare visibility overhead. |
| 4096 | 128, all visible, AA Off, postprocess 64, Kit 32 workers | Two complete 32-step PPO updates, 423.29 / 424.65 s, 262144 transitions. Both groups remained active and all image/object validity fractions were 100%; native successes were 0. Training continued into update 3. |

The selected run is `dh116_gpu3_vision4096_all_kit32_mixed2048_20261105T094038Z`, under `/mnt/mydisk/xk/dh116/runs/`. All 4096 initial geometry audits passed, each with 128 real points; worst per-environment P95 was 0.390714 mm. The 32-worker comparison returned from reset/bind in about 18 s; the otherwise identical 8-worker all-visible experiment had shown no progress for 10 min. This establishes sensitivity to the Kit tasking configuration without claiming a proven C++ deadlock mechanism. Startup diagnostics distinguish reset from binding and cancel their traceback timer once binding returns.

Final checkpoint and sampled memory evidence: checkpoint iteration 2 / 262144 timesteps; all 80 stored tensors finite; 19 Adam parameter states all at step 80. Float32 rollout storage is CPU and pinned. Across 425 driver samples at roughly two-second intervals, peak usage was 19058 MiB (18.611 GiB); Torch tracked peak allocated/reserved memory was 1.720 / 2.410 GiB. The first two updates had no recorded failures. Pinch peak net lift was 0.001196 mm at update 2; tripod peak was 0.003763 mm at update 1 (latest update 2: 0.003540 mm).

Lower-environment results and initial renders alone do not establish 4096 PPO capacity. The final run continues from its own fresh visual initialization to the absolute target of 2048 completed updates; capacity-test policies and the older 64-environment model are not merged into it.

Two-second phase sampling puts collection at approximately 405 s/update (physics, RGB-D rendering and point extraction), and PPO optimization at approximately 19 s/update. Collection accounts for roughly 95% of elapsed update time. One update gathers 4096 x 32 = 131072 RGB-D observations; the bounded camera pool makes two renders per group to reject the frame immediately after camera movement. At the observed short-run rate, 2048 updates would take about 10 days. This is an estimate, not a completion-time guarantee; further throughput work should profile collection components rather than assume the optimizer is the main cost.

## Learning and evidence limits

The fresh visual policy has no native PPO grasp successes in the checks above. Historical teacher demonstrations and the older privileged actor are separate policies and do not count as visual-student success or lift records. Reaching 30 mm alone is insufficient: native success still requires the existing 5 s holding and 0.5 mm slip conditions.

Initial geometry auditing uses truth only to reject bad sensor calibration; it never fills or corrects actor points. Brown-pixel segmentation assumes the current rod appearance. Occlusion, changed materials and transfer to real hardware require separate validation. A short capacity test does not establish long-run memory stability or grasp reliability. Raw runs, checkpoints, camera captures, caches and logs remain on the server and are excluded from this publication.
