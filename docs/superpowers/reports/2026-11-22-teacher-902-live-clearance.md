# Actual open Reset clearance and remaining Tripod contact gap

Base: 08d39aaea9e7e842eaf371b38d0d8fda97625ee5, branch 11-17.

audit_dh116_902_live_clearance audits the completed native run, not a new simulation. It binds the explicit outer SHA and each artifact SHA. SDK2 full body/q/rod Reset capture plus SDK3..240 give 239 frames. Native base_link xyz/xyzw and named measured active q drive the original USD FK. Every saved pad/proximal pose and all SDK2 source-body poses cross-check that FK. Other later link poses remain derived, never independently measured.

temporal_shape_bound uses all recorded material nodes in conservative support-plane rod envelopes. Each convex hull's first-frame gap is reduced by the maximum corresponding vertex displacement over time; support bounds use the same displacement deduction. Table clearance is the minimum vertex Z across all frames. Original >1mm rod/support and >=2.1mm table gates are retained. No body mass, force, material, USD, cameras, joint limits or controller is changed.

The existing parked-reference reader adds explicit keyword task with default power, retaining natural-exit, artifact, identity, SDK/solver clock and every-substep dual-support load checks. Tripod and Pinch require their own prefixes and exact task scope; default Power behavior has regression coverage.

Results from saved native data:
- Left rod/support/table lower bounds: 1.2311466 / 22.9422547 / 57.3823270 mm.
- Right rod/support/table lower bounds: 6.0684550 / 1.7950125 / 46.1515562 mm.
- Each side has 1683 recorded-link FK comparisons. Maximum position residual left 5.98095e-8m, right 7.07138e-8m; largest rotation residual 1.00324e-6rad.
- Model-derived full-shape geometry passes; this does not certify independent all-link measurements, rod steady state, opposed contact, grasp or Teacher. Final rod node speed remains approximately 12.9mm/s as reported in the native package.

Focused test-first checks initially failed for the missing audit module and unsupported task keyword. New tests protect late moving-hull collision, quantitative clearance/motion, nonfinite input, quaternion convention and explicit task identity. Final related suite: 223 passed, 0 failed in 69.15s. There are five newly added cases, eight total in the two affected audit/reference test files. No new Isaac/GPU/solver/PPO steps were run.

Antigravity R1 verified the conservative bound direction and highlighted quaternion, FK provenance and SDK2 coverage. R2 read exact final sources and found no blocking issue. It could not read docs through its source-only MCP whitelist; this limitation is explicit. Codex independently inspected the saved native data and report.

## Endpoint evidence changes the next action

A private bounded 8-point CPU endpoint sweep used actual SDK232..240 rod geometry and the original request/geometry guards. At the original target, thumb/index/middle gaps are -0.0159919 / +0.0248581 / +0.2372645 mm. Middle +0.002rad leaves +0.0421516mm; +0.004rad reaches -0.1402601mm at that pad but the contact source-envelope lower bound becomes -0.3668893mm, below the unchanged -0.150mm floor. Index still remains open. All eight middle-only candidates fail the full contact predicate. No candidate action was executed.

Do not solve this by noise, overclosing, source exemptions or relaxing penetration. Next jointly search a small opt-in wrist/finger alignment against the actual rod, preserve the user 902 final grip semantics, and certify the complete original-rate path. Then require measured opposing reactions and fixed-wrist hold before Lift.

Private evidence/recovery: ref902/live-clearance-proof-239.json, live-clearance-full-regression.log, live-middle-contact-sweep.json. Original 238-only proof remains as a scope-limited intermediate; final 239-frame proof supersedes it. Original three models and 528 protected files are checked before release. RGB calibration files remain unchanged.

Registry regression: 60 passed, 0 failed. First release diff --check rejected a trailing blank line in the updated reference test; formatting corrected before commit. No runtime/test semantics changed.
