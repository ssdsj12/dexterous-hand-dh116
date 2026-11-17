# Implementation plan

1. Inspect actual USD visual meshes and the existing visual_pose_sync conventions.
2. Add failing CPU USD tests: geometry fidelity, no physics components,
   independently posed parents, wxyz rotation, position/rotation errors and
   physics-clock invariance.
3. Implement independent visual-only display owner; no asset files modified.
4. Connect the existing viewer to that owner and replace stale-USD pose auditing
   with actual display-versus-SDK link auditing.
5. Run focused CPU regressions; verify original files and checkpoint hashes.
6. Update report/readiness and prepare exactly one source-frozen native run for
   approval. Do not reuse consumed native budget.
