# P0 contact diagnostics and geometry implementation plan

> **For agentic workers:** Use subagent-driven-development for the independent diagnostic and geometry units; root owns native integration, regression, documentation and publication.

**Goal:** Explain missing real grasp contact and produce auditable static grasp candidates before any further long PPO training.

**Architecture:** A default-off native read-only substep adapter supplies a deterministic contact-event reducer. A separate source-CAD centerline candidate generator screens and ranks proposals. A bounded inference-only contact probe composes these units without changing policy contracts or physical settings.

**Tech stack:** Python, PyTorch, USD/SciPy source-hull geometry, existing TensorWorld/TensorEnv and native fixed RGB-D evaluation.

## 1. Preserve current experiments and prerequisites

- [x] Verify clean `codex/dh116-lift-repair-20261106` at f29eb51 and repository AGENTS.
- [x] Cancel only the CPU controller waiting to launch R0, preserving its prior status and all data; keep actual1439 running.
- [x] Audit raw PhysX contact buffers, CAD atlas, FK and pre-reset observer hooks.

## 2. Contact reducer and raw native adapter

Create `src/go2_pvcnn/control/dh116/tabletop_grasp_contact_diagnostics.py` and focused CPU tests. First verify missing API/test failure, then implement event identity, per-pad establishment/loss/censoring, seven requested reasons, explicit simulated torque and observable-only slip, two style ratios and streamed evidence. Test wrong-facing normals, duplicate/out-of-order steps, nonfinite measurements, terminal/reset boundaries and all reason combinations.

Root creates the native-only adapter and its tests; reuse the already read PhysX buffers where available. Validate object filter/body ownership, source point/normal/separation identity and timestamps. Hook before terminal reset only when explicitly enabled; disabled legacy path performs no extra sensor read or diagnostic file creation.

## 3. Source CAD centerline screening

Create a focused geometry module and CPU CLI plus tests. Enumerate left/right arc-length positions; retain original joint bounds and mimic FK. Cover pinch/tripod, per-pad orientation, upward bearing feasibility, object/self/inter-hand penetration and clearance. Every rejection has a reason, accepted list never includes failures, and full-hull checks follow proxies. Unit fixtures establish arithmetic only; the offline execution report binds the actual source USD and seed.

## 4. Bounded native contact validation

Add an explicit inference-only short probe option, with bounded environment/step budgets and no PPO construction. Tests must reject probe+training/resume, fixtures/native certificates, invalid geometry and reused output. Use the existing first-terminal observer and actual RGB-D actor. After the running pair completes, acquire the same exclusive lock, verify no Isaac process, and run in a new directory. Record raw pads/events/summary and preserve failed/unobservable outcomes. Do not advertise a probe as 100-case validation or grasp success.

Actual scope: the historical-reset observation smoke and independent full raw replay are complete (8env/120 control steps). All source-CAD searches returned zero accepted proposals. The optimized-candidate source/scene/reset admission gate is explicitly not implemented; no optimized candidate is executed and `native_experiment_eligible` remains False. This remaining gate is documented separately from completed diagnostic validation.

## 5. Review, documentation and publication

Run focused CPU suites and an appropriate baseline regression; compare existing failure identities rather than hiding them. Independently review diagnostic semantics and geometry screening. Update README, CHANGELOG and a validation report in the same commit, including actual commands, results, limitations, pending native stages and no long-PPO claim. Commit on the requested branch, update Draft PR only, verify remote SHA. No merge/Ready transition.
