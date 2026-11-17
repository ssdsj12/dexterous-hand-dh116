# Tensor ownership: instrument the missing cached SimulationView chain

## Actual implementation
Integrated four reviewed diagnostic commits from an isolated Ubuntu worktree. Protected source, assets, controls, installed SDK and teardown order are preserved; isolated diagnostic/runtime wiring changes are listed below. No Isaac, solver, policy or PPO run occurred.

New runtime_owner_ids records only construction-boundary IDs for factory locals, force_state, namespace and known resources. create_force_runtime's keyword-only owner_observer defaultsNone; old default callers do not invoke it.

probe_simulation_owner_edges now emits bounded scalar caller chains and two-hop Python referrer descriptions. Worker uses16 descriptions/level and4 caller frames. gc.get_referrers itself enumerates direct referrers before bounded description; the limitation is explicit. No f_locals materialization, frame clearing or added GC.

TensorViewOwners records14 known aliases using scalar IDs/types and weakrefs: manager SimulationView/backend, both assets/Data and their cached SimulationView/backend. Only source-confirmed __dict__ fields are read. Descriptors and native methods are not invoked. Missing dictionaries or unsupported weakrefs remainunknown, without a strong-reference fallback.

The worker observes registered/before_clear/after_stop boundaries. New files: factory_owner_ids.json, before_clear.owner_edges.json, after_stop.owner_edges.json and three phase.tensor_view_owners.json. These files do not exist for old runs and must not be backfilled with CPU fixture values.

## Installed source chain
Eight actual installed SDK/Lab Python files are read-only audited and now pinned by external_input_paths before native startup. Complete paths/SHA are in readiness; snippets remain only in local evidence.

| Source | Actual behavior |
|---|---|
| Core SimulationContext.physics_sim_view | delegates to SimulationManager |
| Articulation._initialize_impl, line1508 | stores manager SimulationView in _physics_sim_view |
| ArticulationData.__init__, line57 | stores the same manager SimulationView |
| Articulation._invalidate_initialize_callback, line1702 | clears _root_physx_view |
| AssetBase invalidation, lines324–329 | invalidates asset initialization/visual state |
| SimulationManager STOP, lines118–120 | invalidates manager view and sets manager fieldNone |
| Tensor SimulationView.invalidate, api.py147–160 | delegates backend invalidation; it is not a Python _backend release marker |

Therefore managerNone plus root ArticulationView weakrefs gone does not demonstrate that asset/Data cached SimulationView or its backend wrapper is gone. The installed-source chain is confirmed; the actual remaining native object's identities and responsibility for SIGSEGV are not yet measured.

## What CPU models prove and do not prove
Project lifetime tracing stores weakrefs and scalar metadata; it does not persist arbitrary frames or materialize f_locals. Factory locals, force_state, closures and a namespace copy alone can be reclaimed after dropping ownership. An explicitly retained constructor frame can retain its factory caller and resources through f_back; releasing the actual holder makes the CPU objects collectible.

This demonstrates a mechanism, not the real upstream holder. Archived frame liveness is not sufficient crash attribution, and no arbitrary SDK frame is cleared. No successful natural-exit claim is made.

Shared source-shaped view tests preserve a SimulationView/backend through assets/Data even after a fake manager becomesNone. Deleting the real CPU owners releases all test objects while the observer/returned JSON remain alive. Unsupported weakrefs, descriptor-backed fields and missing dictionaries are explicitlyunknown. None of this is native-release proof.

## Diagnostics must not break cleanup
Review found the first factory ID writer could raise before runtime ownership handoff. A failing real adapter/owned-call CPU test reproduced it. The worker now records only a textual error, still hands runtime to owners.keep and follows the original operation-finally/synchronize/checkpoint sequence. The final existing error gate still fails the run. KeyboardInterrupt andSystemExit are not swallowed. View/probe evidence failures follow the same rule and do not retain the caught exception traceback.

## Verification
Combined relevant CPU regression467 passed,0 failed,0 skipped with CUDA_VISIBLE_DEVICES empty. This includes the prior geometry/force/seed/support/worker/storage tests and25 new owner/view/source-pin cases. Targeted agent regressions and independent reviews are retained with exact commands and SHA in readiness.

An additional legacy lifetime run exposed pre-existing fixture failures before its subprocess could execute: it assumed one __main__ guard, while the production scheduler correctly has a module-identity guard and a CLI guard. This is being repaired in a separate CPU-fixture package; it is not hidden or counted as native failure, and this report does not claim the whole repository is green.

All528 protected file hashes remain unchanged. No historical checkpoint was loaded, copied or overwritten. Only isolated diagnostic/worker wiring, tests and compact reports/status are published from Ubuntu commits through Windows. PR2 remains Draft, unmerged.

## Next bounded observation
A fresh native attempt must record factory/force_state IDs, actual cached wrapper/backend weakrefs, both original SDK clocks and the actual external process outcome. The previous once-only budget remains spent. The diagnostic keeps the originalGPU2/N1/two-SDK-step/zero-control/solver/PPO scope, but a new frozen SHA and authorization are still required before execution.

Do not add fast_shutdown, repeated GC or speculative SDK field deletion. A possibleFabric contrast is a separate one-variable experiment; it is not implemented or run by this package. Formal Teacher admission remains blocked by actual source/contact/exit qualifications, with original30mm/5s/0.5mm standards unchanged.
