# Shared continuous rod envelope guard

Baseline: 09a26fee479b5f31f6a8af40856e87f6a51ad713, Ubuntu branch 11-17.

## Implemented behavior

The old 1,800-direction bound can reject a geometrically safe selected pad. The private offline refinement and native source/cooked checks previously used different calculations. The versioned shared helper now resolves eligible conservative rejections without relaxing the original 0.150 mm selected-contact limit.

- audit_dh116_certified_segment_distance.py::rod_envelope_clearance preserves all original 20 groups, each containing every supplied frame and three adjacent nodes, plus the original 5 mm radius. It accepts 1–9 frames. At most 40 certified segment QPs per shape propose separating directions. Each proposed direction is reevaluated against every hull vertex and all frames in that group. A subset upper bound can reject but never admit a shape.
- verify_dh116_power_shared_rod.py::check_supported_shape/measure_links and verify_dh116_supported_distal_transfer.py::validate_live_continuous_hulls use this same helper only with explicit dh116-continuous-envelope-support-v1. SDF, noncontact 0.250 mm, supports, table, ownership and reaction delivery checks remain.
- SupportedDistalProbe binds complete source and cooked evidence to one post-SDK/pre-rod boundary, with observed SDK, actual preceding rod SDK and node-buffer SHA. observe consumes it once. Missing, stale, duplicate or incomplete evidence fails. Ordinary same-clock pre-step geometry reads do not leave stale pending evidence.
- A failure-ledger test exposed duplicate rod-source keys in the first implementation; the recording callback now retains ownership of that clock field and preserves the original failure evidence.
- screen_dh116_902_flexible_contact.py freezes the certificate file as a dependency and explicitly rejects refinement in the historical v1 request/entry. Its 24-window cap stays unchanged. No new native recipe is enabled by this package.

## Verification and review

231 CPU tests passed (171 related including 23 new cases, 60 Registry), exit 0. Final log SHA: 087cc72192ec3eeccb2454cb9083ce636b958e1bf5b1df46c656fc608b996bf7. Test command and explicit files are in readiness. Tests cover an analytic rotated box whose old lower bound is -1.418554 mm although a certified separating plane gives -0.100000 mm; real penetration; nine-frame coverage; QP failure/budget; version gates; and the actual measure_links → source/cooked guard → observe call chain using mock native buffers. No test result is described as native contact validation.

Fresh-process imports passed assert_clean_import_boundary: no USD loaded before AppLauncher. All 528 protected files, three original checkpoints and both fixed RGB configurations are byte-identical.

Antigravity completed two read-only code/plan rounds against the same HEAD and file SHAs. Codex independently corrected the pre-step/post-step evidence distinction. Final code review request R3 encountered the account quota limit; it did not return an approval. Codex completed source review, negative tests and regression independently. No account settings were changed.

## Existing-data computation, not new simulation

The sealed Tripod SDK400–408 geometry was reused. A 66-control-window / 528-substep candidate uses the real command functions and float32 finger/wrist writes under an explicitly ideal-tracking hypothesis. Temporary right finger22/finger32 relief returns to the same final grip. This is neither measured dynamics nor policy training.

The shared helper accepts all 528 tested states against the full archived envelope: minimum contact lower -0.149957695 mm; noncontact lower 2.622281 mm; 238 certified QPs. The original fast support bound falls to 0.001565 mm at ordinal381, right finger22_link versus support0.

A controlled alternative increasing intermediate index relief from 0.001 to 0.005 rad did NOT improve that bound (minimum 0.000477 mm); it is not selected.

Further investigation identified a loose axis-aligned bounding-box bound. Using the existing unmodified full convex-polyhedron/AABB separating-axis calculation wherever the fast support bound is below 1 mm establishes a minimum conservative support separation of 1.387084 mm over the same 528 states. No support mesh, physics, native guard threshold or final pose was altered. Thus the earlier micrometre number is a weak lower bound, not measured penetration or the actual closest distance. The SAT witness computation is private diagnostic evidence; it still requires source-bound integration into a new full-route proof and tracking-reserve audit before native use.

## Readiness and next work

No Isaac run, solver run, camera capture, Teacher query, PPO update or new successful grasp occurred. Pinch, Tripod and Power remain outside the success expert pool.

Next: seal a separately versioned actual-command trajectory proof, including the full support SAT witness and tracking reserve; bind its explicit live refinement version, unchanged endpoint, source SHA and finite budget. Then run an eligible bounded real-contact test. Pinch needs fixed-wrist force hold; Power needs actual palm/fist wrap. Preserve the two fixed cameras, real flexible rod, physical hand identity and original 30 mm Lift / 5 s Hold / 0.5 mm Slip standards.
