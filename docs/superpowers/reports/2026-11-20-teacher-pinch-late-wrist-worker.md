# Pinch late left-wrist comparison driver

## Evidence and decision

The sealed 74-SDK-step reference measured right-hand opposition in its final 15 solver substeps. The left index remained unloaded with a terminal 0.232 mm gap. The independent asynchronous replay found a possible 0.238 mm right non-target overlap when fingers reach their old target ahead of the wrist. Antigravity independently flagged the ordering risk and checked the runner's fixed window budget. This comparison keeps the right command path and ends at the existing boundary.

`scripts/run_dh116_pinch_late_wrist.py` changes only the left wrist's six requested action components during the last eight SDK steps. It retains 25% of the old terminal measured-to-command wrist residual and leaves the finger and right-hand requests unchanged. Before issuing the changed window it verifies the newly observed SDK66 rod nodes, both measured wrist poses and joint positions against the sealed prefix and reruns the conditional left path certificate. An invalid prefix or failed certificate is recorded as a guarded stop. Right-hand measured reactions and the existing guards still determine the outcome of the comparison.

## Verification

- 161 CPU tests passed. The readiness file contains the exact test command and the log SHA.
- The conditional CPU check over nine historical rod snapshots reported a 0.292 mm table-clearance margin above the guard and at least 0.044 mm non-target endpoint clearance with independent 0.050 mm per-node sensitivity. These are static conditional results. Measured left and right opposition remain for native verification.
- Maximum budget remains 2 setup, 8 quiet and 64 closing SDK steps, 864 solver steps, zero Teacher queries or PPO updates. This driver has no lift phase.

For preparation, use the project interpreter with `python -B -m scripts.run_dh116_pinch_late_wrist --sealed-reset <manifest> --baseline <sealed_run> --output <new_native_directory> --plan <new_plan>`. The actual execution requires the resulting exact frozen SHA and a fresh output directory.

Next, compare observed four-pad reactions, protected terminal events and the real process exit against the sealed baseline.
