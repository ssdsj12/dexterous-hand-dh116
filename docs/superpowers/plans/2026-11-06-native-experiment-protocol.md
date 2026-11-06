# Native experiment protocol frozen before reward pilots

This supplements the T2 implementation plan. It authorizes no merge, readiness change, physical change, lowered success threshold or modification of historical checkpoints. All launches remain serial on GPU3 after the recorded user-authorized original exit.

## Fixed inputs and splits

- Actual historical models:129 and1439. Neither is the missing269/482 peak policy.
- Fixed camera224×96, original calibration, material/load/cable/reset parameters,24D tanh-Gaussian action for new training,3844D sensor actor and371D privileged critic.
- Authored cold validation: seed11061101; held-out test: seed11061102; each contains100 unique writer states per grasp. Real readback uniqueness, camera validity and first-terminal physical streams must certify every native result. No synthetic trial or repeated episode counts.
- Historical129/1439 comparison uses the same test manifest and deterministic v2 inference, not optimizer updates. Its outcomes diagnose those actual stages and do not choose reward weights or new models.

## Adam comparison and reward pilots

Use the explicit1439→128 balanced-environment derivative,64 new PPO updates, rollout32, and identical source model, saved Torch RNG, launch seed and authored initial reset for each pilot. This is262144 new transitions and absolute target1503; it is a controlled diagnostic budget, not a claim of converged2048-update retraining. Preserve original physics, command scales,0.2Nm efforts and30mm/5s/0.5mm native success.

First compare R0 preserve-Adam with R0 reset-Adam. Immutable complete checkpoint candidates are retained after16,32,48 and64 new updates (absolute1455/1471/1487/1503). Use the same independent100-per-grasp validation manifest for all candidates. The implementation retains by absolute iteration, so these pilots use `--vision-checkpoint-interval 1` to preserve all64 completed updates while admitting only the four frozen candidates. Opt-in native initial-state audits compare every environment before the first reset-Adam PPO update; matching seeds/configuration alone is insufficient. Both modes start from the same1439 source file; reward-fork references with a different source SHA require a separately reviewed interface and are not silently accepted. The externally bound native certificate plus raw evidence replay selects `best_validation`; only positive native successes permit `best_success_rate`. Every candidate remains available. No selection uses training lift peaks or held-out test results.

Run R1,R2,R3 separately with the validation-selected Adam mode and the same64-update/source/initial-state budget. Coefficients are fixed by the versioned reward contract before the first pilot: qualified-height-hold1.0, qualified upward0.5 and descending-segment penalty0.5. R0 retains the old support_deficit reward exactly. Positive quality uses valid native opposition, observable slip≤0.5mm, signed upward support≥runtime weight, finite measured velocity and no failure. Contact loss/low support are diagnostic negative cases, not success.

The primary factor outcomes are qualified hold for R1, the lift distribution for R2 and measured descent/slip for R3. Compare per-grasp paired native cases and report coverage/invalid cases with denominators. A factor cannot be called effective solely because its training reward rose or one maximum increased. Require no deterioration in native success and valid telemetry; report paired uncertainty rather than claiming significance from a point estimate.

R4 combines hold+up only after both factors receive independent native support. R5 combines only independently supported factors. If admission fails, record R4/R5 as not admitted with the measured reason; do not silently launch all weights or report fabricated ablation results. The attached implementation document explicitly requires validating factors before combinations.

## Final evaluation and reporting

Evaluate the selected new model on the independent test manifest once selection is fixed. Report each grasp's100-trial Lift P50/P90/P95/max,30mm rate, qualified hold, observed slip and coverage, native success, contact loss, support diagnostics, action/estimated-torque saturation and invalid/terminal counts. Simulator estimates remain privileged diagnostics.

The final report must distinguish actual results, failed/incomplete native attempts and unexecuted combinations. Keep the original1750 final model,129/1439 originals, all derivatives, validation candidates and immutable manifests. No native result may be replaced by a fixture. Keep PR#1 Draft until the user decides readiness and merge after reviewing the native report.
