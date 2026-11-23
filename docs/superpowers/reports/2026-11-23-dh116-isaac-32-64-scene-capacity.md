# DH116 Isaac Power scene capacity at 32 and 64 lanes

Baseline `37ae4ba641057e6437f6cd834ee4cdf5a9b69ba5`. Tested on GPU 2, a 24,564 MiB RTX 4090, in order 32 then 64. Each lane authors the original left and right DH116 hands, 34 source colliders, a collidable table, and both static end supports. Camera, rendering during physics ticks, model inference, and PPO were outside this probe.

| Lanes | Actual articulations | SDK steps (setup + measured) | Foreground GPU memory after readback (MiB) | Eight ticks with all-lane reads (ms) | Natural exit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 2 + 8 | 2,628 | 239.28 | 0 |
| 64 | 128 | 2 + 8 | 2,642 | 238.32 | 0 |

Every lane used the archived Power seed and its own translated scene prefix. The probe read each hand link and joint through its actual PhysX view. A 1 mm wrist-coordinate write in the last lane was read back, left the other lanes unchanged, and was restored without an SDK tick. The source-envelope table clearance remained above the original wrist minimum during the eight held-state steps. The 32- and 64-lane runs passed full weak-reference checks for 257 and 513 Python owners respectively. The GDB inferior and outer process both exited naturally with code 0. Memory was read at observed boundaries and does not represent a peak.

The first 32-lane attempt exited 1 during SimulationContext creation, before any stage lane was authored, with an empty USD type for `physxScene:timeStepsPerSecond`. The corrected probe runs the established clean-import check and reads source geometry after Isaac starts. The second attempt completed all 32 lanes and physics reads, then the existing lifecycle diagnostic rejected more than 32 weakrefs. The corrected diagnostic applies that same observer in groups of at most 32 with all owners retained and still uses the original STOP, context clear and App.close sequence.

This closes scene capacity and hand-state isolation for Power at 32 and 64 lanes. The test did not have a live deformable rod, a full episode Reset, a per-lane terminal event, or a PPO rollout. The original DeformX solver-only capacity is reported separately. The production Teacher entry still accepts one environment and one update. Full multi-environment testing requires the live DeformX bridge to follow the shared SDK clock and route native forces to the owning hand, a lane-local episode Reset, and terminal-observation packing before PPO.

Reproduction uses the archive-backed Power receipt already used by the solver-only probe. Run each invocation in a separate supervised Isaac process with an empty output directory. After a natural exit and successful 32-lane record, repeat using 64 and a new output directory.

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/python.sh -B -m scripts.audit_dh116_isaac_batch_scene \
  --num-envs 32 --gpu 2 --receipt /absolute/path/to/power-reset/outer.json \
  --receipt-sha256 d1aa0fbe8ba495ede23da425d4c46cc7f8c2dbc5377471dad1ad788cea4af8af \
  --output /absolute/path/to/new-evidence-directory
```

CPU regressions: 70 passed.

```bash
python -m pytest -q tests/test_dh116_isaac_batch_scene.py tests/test_teacher_physical_group.py \
  tests/test_teacher_batch_inputs.py tests/test_native_teacher_worker_contract.py \
  tests/test_dh116_power_solver_capacity.py
```

SHA-256 values for the raw local logs, shutdown receipts and scene observations are in the [readiness record](../../contracts/dh116_teacher_ppo32_readiness_v1.json).
