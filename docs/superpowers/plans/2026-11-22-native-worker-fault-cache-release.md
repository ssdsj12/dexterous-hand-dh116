# Native Worker Fault Cache Release Implementation Plan

**Goal:** cover success and exception paths with the existing source-bound Kit cache release at the existing pre-GC boundary.
**Architecture:** only actual_provider changes; scope-bind the old module's run_owned_call, forward an exact once-only before_collect hook; preserve sealed lifecycle/Reset source. Read weak sim and bounded actual SDK step, compare completed end when present.
**Tech Stack:** current Ubuntu Python3.10, original Python lifecycle and Kit source contract; no environment changes.

- [x] Verify v5 failure has no stat_cache_release receipt and retains constructor frames/PhysX tensor references.
- [x] Add CPU regression using the real lifecycle/run_owned_call and an explicit fake cache I/O boundary.
- [x] Observe failing test before production changes.
- [x] Implement private worker lifecycle boundary; remove success-only release, retain scalar completed end and weak sim reader.
- [x] Run normal/source-fault/cache-rejection regressions, legacy lifecycle and actual worker regressions.
- [ ] Independently review spec/quality, protect all528 files/3 old checkpoints, report/readiness and publish via Windows.
- [x] Freeze new current source recipe; do not consume native budget before runtime readiness.

Boundary requirements: actual SDK step integer>=2 and<=frozen max; completed end must equal actual step. Synchronize and clear traceback frames precede release; original GC/STOP/App flow follows; no second cache call, no new physics/update, no os._exit, no signal, no qualification inference.
