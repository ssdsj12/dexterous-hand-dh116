# Fixed-camera visual PPO implementation plan

> For agentic workers: root controls native Isaac; independent camera and resume tasks may run in parallel.

Goal: prioritize throughput over retaining 4096 environments, as requested. Try 2048 and 1024 physical environments with a fixed real camera per environment; preserve the absolute 2048 PPO update target. The user permits 160x120 RGB-D if 320x240 does not fit.

Architecture: add a fixed CameraView provider that captures every environment in one render product without moving cameras or changing visibility. Retain sensor-only 3844D actor, 371D critic, 128-point masks, exact geometry, 32-step rollouts and existing PPO semantics. Fixed cameras render at least once per observation after initial warmup; timestamp/frozen-state checks remain mandatory. Preserve the current learned policy and Adam using an explicitly authorized, narrowly validated downsize conversion and balanced style-index selection. Never overwrite the source checkpoint.

- [ ] Camera agent: test and implement `tabletop_vision_fixed_camera.py` and `tests/test_vision_fixed_camera.py`; prove fixed poses, fresh frames, numeric order, one observation group and no GT fallback.
- [ ] Resume agent: test and implement `tabletop_vision_resume.py` and `tests/test_vision_resume.py`; default strict contract, explicit 4096-to-2048/1024 conversion only, same actor/optimizer/RNG/update counts and projected per-environment counters/critic references. Unknown physics/reward/calibration changes fail.
- [ ] Root: test CLI forwarding/limits, fixed contract/provider selection, guarded downsize integration and migration metadata. Keep old bounded mode compatible. Set viewTile limit to fixed camera count at launch.
- [ ] Root: copy and verify the latest complete old checkpoint, pause the existing monitor, stop only the verified own Isaac process, and keep all original outputs.
- [ ] Native capacity ladder, serially on GPU3: 2048x320x240 and 1024x320x240, then authorized smaller images. The measured 1024x160x120 perception failure led to 224x96, which increases horizontal rod sampling while cropping vertical background. Require complete initial sensor/geometry audit and two complete 32-step PPO updates for a viable candidate; record frames/renders, throughput, memory and finite checkpoint/Adam state. Stop failed attempts without touching other users.
- [ ] Continue the selected fixed-camera run to absolute update2048, retarget TensorBoard/current metadata and the existing quiet monitor; preserve predecessor lift records as one actual policy chain.
- [ ] Update README/CHANGELOG with native evidence and limits, review the exact source, commit and fast-forward GitHub11-5 using the existing authorized credentials. Verify remote SHA and a clean local worktree.

Implementation remains in the existing isolated 11-5 worktree and the separate server stage `/mnt/mydisk/xk/projects/dh116-vision-fixed-20261105`. Configuration and measurement choices are authorized by the user's throughput preference; no further approval is needed to execute this plan.

Runtime initialization fallback: fast PhysX replication stalled or aborted before camera creation. Use original inherited USD physics with replication disabled and explicit per-environment collision groups, matching the installed IsaacLab fallback. Keep the replicate mode available for compatibility.
