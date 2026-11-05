# DH116 sensor-only visual student

The visual actor receives four frames of measured object-surface points, active joint encoders, wrist FK, previously issued actions, sensor timing and validity. It does not receive simulator object pose or velocity, contact forces, penetration, slip, task phase or success flags. The 371D native state belongs only to the privileged critic and simulation reward/safety/evaluation. Teacher-action distillation is disabled: the existing privileged demonstrations have no synchronized RGB-D labels.

## Camera and coordinates

The front camera placement is defined by `configs/dh116_training/front_depth_camera_v2.json`: eye (0.10, 0, 1.05) m and target (0.48, 0, 0.43) m in environment-local world coordinates. CLI width/height choose the actual sensor resolution; the current fixed-camera run uses 224x96 with the same horizontal field of view and less vertical background. Intrinsics are recomputed and checked for the selected dimensions. Depth is optical-Z in metres; optical axes are +X right, +Y down, +Z forward. RTX pixel centers are (u+0.5,v+0.5), with principal point (W/2,H/2).

Visible brown pixels are selected from rendered RGB, then rejected if the full 3x3 depth neighbourhood is invalid or spans more than 20 mm. Up to 128 retained pixels are sampled uniformly in pixel order. Missing entries are zero-padded with a false mask; simulator truth never fills missing points. Each 3D point is transformed camera -> calibrated world -> actual left/right palm body-link frame. RGB-D processing uses at most 64 environments per temporary block, independently of the camera pool size; row order and full-batch float-RGB normalization are preserved.

Vision mode explicitly disables anti-aliasing and DLSS frame generation, then verifies the native rendering settings. Temporal anti-aliasing mixed silhouette RGB/depth in the 4096-environment audit; disabling it fixed those observed floor outliers without adding simulator labels or synthetic points.

The current `--vision-camera-mode fixed` creates one stationary camera for each physical environment, sharing one tiled render product. All environments are captured together; camera poses and visibility stay fixed. After at least eight startup renders, each observation starts with one render and retries only if frame/time metadata is not fresh and aligned. Numeric IDs are checked independently of USD traversal order. Same-render-product ReferenceTime/IsaacReadTimes, initial actual USD camera calibration and unchanged physics time/step/FK remain mandatory. First-observation evidence can retain at most two actual RGB-D frames without consulting object truth.

The older `bounded` mode remains available: a pool of at most256 cameras visits environment groups sequentially at frozen physical time and discards the first render after each camera movement. This lowers rendering memory at the cost of collection throughput. Its configuration is a separate exact resume contract.

The default `--vision-visibility-mode all` leaves all environment roots visible and avoids changing thousands of USD visibility properties at every observation. Cameras still observe their own calibrated environment; startup geometry validation checks every environment. The optional `batch` mode hides non-current environment roots for rendering only, keeping source env0 visible and explicit local overrides on clones. Neither mode deactivates physics or changes collisions. Rendering mode belongs to the exact resume contract; changing it requires a separate run. Native capacity measurements remain necessary for either mode.

## Network and optimizer

Per frame: 2x128x3 hand-local coordinates +128 masks +62 proprioception/action values +3 timing/validity values =961 values. Four frames give3844D. Both hands share a masked PointNet3->32->64->64 encoder. Per-hand max pooling produces two64D features; four frame features feed a772->128->128->24 action MLP. History is cleared independently when an environment resets.

Rollouts stay float32 in CPU RAM (pinned on CUDA). Inference and optimizer microbatch sizes are configurable. The production memory configuration is recorded in each checkpoint contract. Five epochs and eight logical minibatches still produce40 Adam steps per PPO update: microbatch losses are weighted by sample count, gradients are accumulated, then clipped and stepped once per logical minibatch. GAE, global advantage normalization, old raw actions/log-probabilities and adaptive KL semantics are retained. No float16 rollout compression is used.

Exact float64 CAD minimum-height checks are split over environments and vertices. All original vertices remain; the physical geometry and contact parameters are unchanged. This removes the old expanded [4096,34,3,2771] temporary without approximating the safety calculation.

## Launch

Use the installed Isaac Sim4.5 / IsaacLab0.54.4 / RSL-RL3.1.3 environment. This server uses GPU3; the inherited smoke launcher contains matching Kit GPU flags. Set a valid NVIDIA Vulkan ICD and limit Kit task workers, as in the launcher.

The current run uses `--physics-clone-mode usd_parse`: original inherited USD physics is loaded per environment, with explicit collision groups between environments. This bypasses the observed PhysX fast-replication initialization stall/crash. The previous `replicate` mode remains the default for compatibility. Geometry, materials, actuators and solver settings retain their authored definitions.

```bash
python scripts/run_dh116_surface_student.py \
  --output runs/vision_fixed1024 --num-envs 1024 --updates 2048 --rollout-steps 32 \
  --vision-student --surface-feedback --surface-reward-profile support_deficit \
  --depth-camera-config configs/dh116_training/front_depth_camera_v2.json \
  --pregrasp-candidates configs/dh116_training/effective_pad_center_pregrasp_v1.json \
  --mixed-pregrasp-candidates configs/dh116_training/effective_pad_triad_pregrasp_v1.json \
  --vision-camera-mode fixed --physics-clone-mode usd_parse \
  --vision-width 224 --vision-height 96 --vision-ppo-microbatch-size 512 \
  --vision-inference-batch-size 256 --vision-visibility-mode all
```

The absolute stopping target is2048 completed PPO updates, not2048 episodes. A fresh visual actor starts at0. Resume accepts only the same visual version and exact actor/critic/camera/reset/memory contract; the old371D actor cannot silently resume here.

The current policy continues the prior 4096 visual actor at update5. Add `--resume SOURCE_VISUAL_CHECKPOINT --vision-resume-downsize` only for the explicitly supported 4096-to-2048/1024 fixed-camera conversion. It preserves policy, Adam, LR, RNG, absolute update and transition counts. Per-environment counters and critic reference arrays select the same balanced subset from both grasp groups. Only the recorded camera/count/resolution changes are allowed; unknown differences and type changes are rejected. `resume_migration.json` and subsequent checkpoints retain the source contract and retained/dropped counter totals. This starts from a physical reset and empty sensor history, as does normal resume.

`monitor.json` and `iterations.jsonl` describe completed optimizer updates. `collection_status.json` records current rollout progress without counting an unfinished update. `vision_student_ppo.pt` is committed atomically after complete updates. `initial_camera_observation.json`, `initial_camera_geometry_validation.json` and `vision_memory.jsonl` record sensor and capacity evidence. The independent initial GT geometry audit can abort invalid perception, but never changes actor inputs. Real success is counted only by native PPO `successes`, using the unchanged30mm /5s /.5mm criterion.

The README and CHANGELOG contain this publication's actual native capacity and training results. A short memory test is not evidence of reliable grasping or completed2048-update learning.
