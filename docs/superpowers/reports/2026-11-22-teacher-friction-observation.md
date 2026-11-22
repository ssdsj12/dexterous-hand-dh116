# Teacher complete-task source: raw friction observation
## Why this advances the grasp/Lift mainline
The current native contact PPO profile has Close/prelift terms but no complete Lift/Hold
objective. Its successful short update is not a learned grasp. The already approved fixed-S0
V2 design needs actual contact/slip sources and a full phase/reward/context/collector route.
This package supplies a missing real low-level displacement producer instead of fabricating
slip labels or continuing to add closure dose to the rejected Tripod ray.

## Actual source changes
prepare_dh116_deformx_contact_friction.py verifies all72parent sources and the existing
88170101 backend, copies them into a new independent tree, appends raw surfaceReadout dxRel
to SdfPairSample and adds contact_friction_samples_v1. Exactly3generated sources change:
YarnBall.h, contact.cuh and bindings.cpp. It does not alter surfaceContact force/Hessian,
step order, materials, timestep, controller or the original getters.
sdf.h aliases the same PairSample type; its native sizeof-based copy uses the extended
record. All translation units were rebuilt; no mixed record ABI is used.
The new field is float32[K,3] solver_friction_displacement_world_m, before tangent projection.
It preserves normal motion too. It is not velocity, a calibrated slip error or ForceProxy.
slip_uncertainty_m remainsNone; slip_qualified remainsFalse.
Generated DeformX/YarnBall delta remains GPL-3.0-only. Parent source, license, binaries,
vendor, assets and historical checkpoints are unchanged. No generated source tree/binary/raw
capture is uploaded; repository code only prepares the explicit new copy.

## Actual independent build
Existing Python3.10.12, CUDA Toolkit12.8.61, SM89 and GNU12.3.0; no environment upgrade.
Configure8.090s, build36.200s, install0.117s; all exit0, total44.415s.
New binary SHA847c1f49c195b275ac23175182194ad0764a235e5adaed9407d8d6dcce251cf4.
The complete source tree and observation delta are hashed in readiness.
CPU import verifies the new method and unchanged old v1/v2 methods coexist.

## Actual standalone GPU validation
verify_dh116_deformx_contact_friction.py checks the loaded binary path/SHA before constructing
a rod. GPU2, existing exclusive native lock, baseline/candidate processes serial under GDB.
Each backend runs static/translation/rotation plane-mesh cases; each has21free rod nodes,
3settling advances and1observed advance, each480Hz with12substeps.
Total24rod advances/288substeps; zeroIsaac/Teacher/PPO, no DH116 task was executed.
Independent outer and GDB inferior exit0 on both (6.033s/5.884s), no timeout/kill/retry.
Each case captured252real contact pairs. Positions, velocities, contact forces, mesh wrenches,
old pair vectors and v2 pose/gap/normal-force fields have maximum numeric difference0.
This is bounded exact numeric agreement, not a universal GPU bitwise-equivalence claim.

Raw dxRel was independently checked against consecutive contact-point-derived node motion
minus the known mesh linear/angular motion. Maximum residuals:
static1.005e-9m, translation3.697e-9m, rotation9.397e-10m.
Tangential force dot tangential displacement was nonpositive in all observed pairs.
These controlled arithmetic checks do not calibrate uncertainty for the real hand/rod task.
The first private analysis failed to serialize NumPy bool_; failure/partial JSON were kept.
Corrected versions re-read unchanged raw outputs; no solver rerun occurred.
The final version also corrects an overly broad bitwise label to exact numeric equality.

## Review and next integration
Antigravity Teacher-Lift-TrainingR1–R3 actually read the relevant current source.
Codex rejected R1's invented5mm uncertainty, premature prelift predicate, detached-anchor
reset and old FULL NA compatibility assumptions. R2/R3 accepted the raw dxRel extension,
separate compilation and unchanged old API, and required native parity.
Antigravity cannot access the private raw GPU logs: the result above is Codex's independent
execution evidence, not an Antigravity native qualification.
Next: an independent V2 backend/source binding, SnapshotV2/SlipLifecycle, real phase witness,
RewardEngineV2 and Collector using the existing PPO. Keep old profile/model/checkpoint
namespaces exact; do not relabel the new binary as the old BackendProfile.
Keep calibrated slip and full success unavailable until their own sources pass.
The task still requires real opposing grasp and off-support30mm Lift/5sHold/.5mmSlip in
Pinch, Tripod and Power. This package does not claim any of those successes.

## Tests and protection
Five new source tests first failed, then passed. Related CPU regressions35passed,0failed;
the Registry publication check is recorded separately in readiness.
528protected files and3historical checkpoints verified unchanged.
