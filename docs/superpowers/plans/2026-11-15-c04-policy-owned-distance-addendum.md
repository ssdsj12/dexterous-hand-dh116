# C04 policy-owned Pinch distance and trusted-guard remediation

Parent-approved SPEC remediation, 2026-11-15. CPU only; no native/GPU execution, installation, training launch, push or sub-agents.

## Observed defects and RED

The first C04 candidate used BilateralMeasuredPinchApproach for policy intervals. Its geometry successor is intentionally frozen to the diagnostic action schedule, policy step=(SDK-3)//8 and SDK<=234. This is not a random PPO action contract. Independently, NativePinchSources.observe treated every passed=false geometry row as SourceFailure, including valid original joint/force guards that must settle a known terminal negative sample. The previous 176-pass suite did not establish these actual source paths.

`spec-fix-red.log` preserves both failures: the new reader API was absent and a source_complete=true original joint guard was incorrectly quarantined. The old frozen validator's legal .123 request rejection remains a separate passing baseline test; its behavior must remain unchanged.

## Separate source contract

Create `tabletop_native_teacher_policy_pinch_distance.py` with independent `dh116-policy-owned-pinch-geometry-v1` and `dh116-policy-owned-pinch-distance-v1` identities and implementation SHA. Reuse only the existing locked asset/SDF loader, same-clock SDK capture, pure SDF distance and conservative geometry mathematics. Do not call, patch, weaken or filter the old diagnostic schedule validator. Do not require zero wrist requests or an absolute SDK origin/cap.

At capture, copy actual SDK link transforms/velocities and complete q/qd/target readback against the current clock and fixed41 raw nodes. Verify actual current parameter source inventory, native radius/raw-derived data, side/pad/SDF bindings and projected/submitted targets. Preserve the original joint/force/clearance, source-envelope/non-target separation, wrist tracking/angular and whole-hand safety guards. Source completeness (finite correctly bound evidence plus required SDF coverage) is independent of constraint pass/fail. The support certificate's constant-radius lower bound is adjusted algebraically to the actual verified radius, without changing task constants.

The original physical row is fsynced and then accepted by the original action-history validator. Only then can the new reader bind the complete row hash, actual request, SDK-target hash, window clock, token and frozen policy identity, and compute distance. Its full evidence is separately fsynced as `policy_pinch_distance`; this avoids a circular hash between the physical row and its derived distance packet. The packet contains new version/source SHA, geometry source/pose/radius provenance and history_binding{collection_token,policy,physical_row_sha256,sdk_targets_sha256,history_owner_uid,physical_samples}. It never grants native qualification.

C01's exact old distance-reader type remains used only for Reset readback, which has no action schedule or absolute SDK restriction. Policy windows always use the new reader. The adapter stops immediately on a source-complete trusted guard and settles -40 once with terminal Phi0. Unknown/corrupt evidence takes precedence and cannot produce a negative training example.

## Required CPU regression

Use NativePinchSources construction and its production exchange-enrichment/observe path, original physical episode/history and real reward. Isolate only unavailable native parameter/health getters into explicit CPU packets; construct owned41-node and SDK-pose fixture buffers without relabelling historical native/backend evidence. The source-interface fixture's model input generation remains CPU synthetic, and it does not qualify a live Reset/session.

Positive cases: legal non-recipe .123 request at SDK3; a real original-control prelude and fresh no-step Reset gives policy0 at SDK11; a 232-step CPU control prelude and fresh Reset gives policy0 at SDK235. Full windows have eight delivered rows. A real third-substep joint guard gives exactly three delivered rows, complete sources, passed=false, one -40 and terminal Phi0. Negative cases: wrong owned pose/clock/source inventory/radius/parameter clock/scope, NaN and SDK-target mismatch, each stopped after its durable third row without update/checkpoint.

Reviewed dependencies are exact C02 `712f7fcf3e3c0a6992951228071ba08ecf66762f` seven files and C03 `fee8de1b34294c294213e6e2eb77e9f8f826f40e` reward/two JSON. Keep these separate from C04 deliverables. Re-run the full combined CPU suite and scheduler528+checkpoint2 protection after remediation; no result grants native authorization.


## Explicit remaining Mesh/Numeric Reset source bound

The unchanged `tabletop_mesh_reset_order._clock` has an SDK240 upper bound. The new policy reader's SDK235..242 interval test is not proof of a Mesh/Numeric Reset above240. A separate `reset_clock_preflight` reads the actual SDK counter before Reset mutation or budget reservation and rejects >240 with `bounded_mesh_reset_clock_v2_source_missing_for_reset_above_sdk240`. It never changes the SDK/solver clock or the old reader. `actual_admission.reset_source_limits` discloses the bound, long_term_reset_supported=false and release condition for an independently reviewed bounded Reset v2 source. Initial Reset2 plus a single episode through SDK258 is not unconditionally blocked. A future Reset>240 and long-term multi-episode native Teacher remain unsupported. CPU baseline ordering RED and boundary/admission GREEN are separate from any native proof.
