# Static Manifest Handoff Implementation Plan
**Goal:** remove duplicate full historical replay inside one static contract while retaining validation at every independent later boundary.
**Architecture:** private preflight helper returns existing JSON report plus the exact loader-verified immutable manifest. Public static_preflight keeps its original dict/schema. verify_static_contract uses that exact result; volatile budget checks remain unchanged. No global or cross-call cache.
**Tech Stack:** existing Ubuntu Python3.10, actual source-bound Tripod manifest, original PPO contract.

- [x] Verify current static_preflight loads full manifest, then verify_static_contract immediately loads it again.
- [x] Run failing public contract handoff test using real file verification and synthetic CPU-only budget I/O.
- [x] Extract common private preflight body/return; preserve public report and independent builder/source/live full verify calls.
- [x] Prove exact object handoff, reject changed test-owned file on the next call; run public preflight/session regressions.
- [x] Measure full static contract before/after in alternating CPU runs; report exact scope, no native claim.
- [x] Freeze current recipe, run regressions, report/Registry/readiness, protection and Windows same-commit publication.
