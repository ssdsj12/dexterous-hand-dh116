# Actual left/right palm lift telemetry

Goal: display actual palm Z displacement separately from rod net lift, per grasp family. The user selected measured left/right palm motion rather than rod endpoints.

Definition: actual palm body-link world Z minus that environment's measured Z immediately after its physical episode reset. Use existing wrist FK measurements, never palm commands or nominal root seeds. Record finite live measurements before any terminal reset, preserving the old episode's values while establishing the new measured baseline.

Implementation in the existing isolated11-5 worktree and server stage `/mnt/mydisk/xk/projects/dh116-palm-lift-20261105`:

1. Add an optional device-only tracker and TensorEnv reset/substep hooks. Default disabled for existing callers. Snapshot last/peak/valid[N,2] diagnostics into extras without changing policy inputs, rewards, actions, guards, success or RNG.
2. Add a pure group summarizer: signed per-hand maxima and last-sample means, valid counts and global environment IDs. Missing values stay null. Separate left/right maxima may come from different environments.
3. Enable the tracker for native visual training before its reset, aggregate across completed PPO rollouts, and write monitor/iterations/TensorBoard scalar tags. Do not add telemetry to the actor/critic contract or alter the model.
4. Prove reset/terminal semantics, actual-versus-command separation, invalid handling, input/RNG nonmutation and unchanged existing transitions with red/green CPU tests. Run related regressions and independent review.
5. Preserve the current atomic checkpoint and original run. To load the new instrumentation, switch only the owned training process at a completed update boundary. Resume with identical1024 fixed cameras,224x96,rollout32 and exact model/optimizer/RNG/counters; keep the absolute2048-update target. No second Isaac or server reboot.
6. Verify at least two new complete native updates, finite checkpoint/Adam continuity and real hand diagnostics; keep training running, retarget TensorBoard/current pointer/monitor and show measured millimetres. Older missing hand history remains unknown.
7. Update README/CHANGELOG and publication evidence, commit and fast-forward GitHub11-5; verify remote SHA. Preserve all raw checkpoints and telemetry outside Git.
