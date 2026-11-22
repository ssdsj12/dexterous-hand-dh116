# Tripod Refined Contact Work Implementation Plan

> Use executing-plans within the existing Ubuntu worktree; independent readonly review already completed.

**Goal:** omit the unused coarse tube calculation only where the existing full-node contact refinement supplies the authoritative result.
**Architecture:** PolicyOwnedTripodGeometry._evaluate passes None instead of tube for target shapes. Shared guard/default API stays byte-identical; non-target coarse separation, supports, full history, .00015m limit, finite/pass checks remain.
**Tech Stack:** existing Python3.10/NumPy/SciPy CPU evidence; no Isaac or environment replacement.

- [x] Extend existing tests/test_native_teacher_policy_tripod_distance.py primary test with real guard/continuous certificate call counts. Old source must perform redundant coarse calls and fail the new invariant.
- [x] Change only shared guard call in tabletop_native_teacher_policy_tripod_distance.py to check_supported_shape(world, boxes, None if target else tube, ...).
- [x] Run Tripod policy, continuous envelope and existing source/reset regressions. Check all protective failures remain.
- [x] Replay authentic v9 geometry with unchanged inputs; compare reference/new complete packets and failure reasons, alternate timings in actual control_source_scope.
- [ ] Verify 528 sources/3 old checkpoints, source-bound manifest, fixed RGB. Update Registry/readiness/report/README/CHANGELOG; Ubuntu commit and Windows same-commit publication.
No new native attempt or forced cleanup in this CPU package. New bounded recipe preparation is a later separately frozen package.
