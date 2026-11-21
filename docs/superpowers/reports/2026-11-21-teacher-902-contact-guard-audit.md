# Native 902 Tripod command contact: tracking and representation failure

Baseline: 98d06fdcddd8587b2899bfa96e99b4914247dd34, Ubuntu branch 11-17.

## Actual native result
The independently frozen v2 route was executed once, with the original 240-step settlement and 21-window approach followed by the new command route. The budget ceiling was SDK936 / 11208 solver substeps / 87 control windows / one environment / zero PPO. It stopped at SDK476, before that SDK's rod advance, on the unchanged source-envelope guard for right.finger33_link. SDK241–475 has 235 complete consecutive command/exchange records; the v2 tail completed 67 substeps (409–475). The preceding rod is SDK475 / tick5676. SDK476 is a failed post-SDK/pre-rod measurement, not a completed exchange.

The process naturally returned 1 in 228.576 s. GDB independently reported inferior exit 1, no signal; app.close returned, owned references were released, and the single-instance lock was available afterward. No SIGTERM, SIGKILL, retry, extra PPO, Teacher query, or model update occurred. This is a correctly stopped failed contact experiment, not native success or exit-zero qualification.

- Frozen fingerprint: fe4fe3f312b46fdc0ff17c1d00c5a43c32dfd6d01c45155a7c75675140de1041.
- Outer receipt SHA: ffc57e0423eaffbd22afa942d9f7a571253bb7859c49c83b137b96e06ae97dfa.
- Failed geometry SHA: 29e8170899108d969862f699a1e4bcf7eb613e08caac50fb0772c41906590221.
- Native log SHA: fd83fd483f9fac4073e9b8f41edf05d0c3ce8e5b2f5c8b3826250281703a6d31.

## What the failure does and does not prove
Using the same actual SDK476 pad pose and the preceding sealed rod nodes (SDK475), the continuous finite-segment certificate gives:

| Representation | Continuous rod gap | Minimum discrete-node gap |
| --- | ---: | ---: |
| Source USD conservative convex envelope | -0.1528123 mm | +0.2772477 mm |
| Frozen original F4 cooked pad | +0.0205048 mm | +0.5806743 mm |
| Original SDF, discrete nodes | Not measured here | +0.5932100 mm |

The source envelope exceeds its existing 0.150 mm allowance; that guard remains unchanged. It does **not** prove actual cooked collision or a solver defect. The cooked geometry remains clear. The continuous segment and material-node checks are distinct; neither may replace the other. All recorded preceding hand forces and torques are zero; only support_0/support_1 appear as partners.

At the same complete SDK475, independently substituting actual roots and active joint measurements into the existing FK gives:

| FK input | Pad origin discrepancy from actual SDK475 |
| --- | ---: |
| Commanded root + commanded active joints | 1.141831 mm |
| Actual root + commanded active joints | 0.025770 mm |
| Actual root + actual active joints | 0.0000548 mm |

Measured passive Mimic residuals are about 1e-8–5e-8 rad. The primary pose discrepancy at this sample is wrist tracking, not broken Mimic. This is a geometric decomposition of recorded data, not a new physics experiment or a universal causal model. SDK475 q/qd must not be relabelled as SDK476.

A separate CPU counterfactual using ideal SDK476 hand pose gives +0.253938 mm against archived rod and +0.211669 mm against actual rod. Actual hand pose gives -0.103626 mm against archived rod and -0.152812 mm against actual rod. Rod movement and tracking both matter; these differences are not additive physical causation.

## Implemented audit and tests
scripts/audit_dh116_902_contact_guard.py now reads the sealed outer receipt, guard, preceding exchange and frozen plan. It checks artifact SHA, side-specific link ownership, post-SDK/pre-rod identity and exact rod buffer. It computes and labels source and frozen cooked results separately; it cannot grant contact or Teacher admission.

joint_tracking separates measured commanded joints from passive Mimic relations. Passive backend target zeros are never interpreted as real follower commands. AABB lower bounds only prune candidates proven unable to beat an already certified upper bound, avoiding unnecessary numerically ill-conditioned distant-point QPs; no geometric tolerance was widened.

Six new CPU cases cover source-only rejection vs clear cooked shape, segment vs node separation, rigid transforms, absent failures, forced/unsealed exits and artifact tampering. Initial missing-module and two representation-contract failures are retained; a far-point QP failure during implementation is also retained. Final related regression: 197 passed, 0 failed (41.52 s), log SHA 88b94e2b40b7a16913d688c163adf546fbafc667214748e1f0c73dc3149fc577. Registry result is recorded in readiness.

Reproduce the read-only native audit with the existing project CPU environment:
```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B scripts/audit_dh116_902_contact_guard.py \
  --receipt /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ref902/command-v2-native-tripod/outer.json \
  --guard tripod_geometry_guard_476.json --output NEW_AUDIT.json
```
Output must be new. Raw receipts, geometry, logs and models remain outside Git.

## Review, protection and next work
Antigravity's configured SSH reviewer read the live v2 source, tests, caller and dependency files after repo_info verified the workspace. Its response was delayed by repeated model-unavailable errors; the actual native failure and corrected source/cooked interpretation were sent as follow-up. No final approval is claimed unless recorded separately in readiness.

528 protected files, three historical checkpoints and both fixed RGB camera configs are checked unchanged. The original USD, supports, thresholds, joint safety, action/state contracts and controller are unchanged. No new stereo capture occurred.

Next: use actual wrist tracking and velocity to define a bounded precontact settle/progression gate; verify its entire measured clearance before closing fingers. Independently establish whether the locked cooked shape can serve a narrower, source-bound guard at the intended pad; never bypass the existing source guard based only on this audit. Re-screen continuous geometry and node/SDF contact together before another distinct frozen native attempt. Preserve the 902 final Pinch/Tripod grip; Power still needs full palm/finger enclosure. Opposed forces, fixed-wrist loaded hold and support-free Lift/Hold/Slip remain unqualified for all three styles.

### R1 received and independently checked
Antigravity read seven key tracked sources with matching SHA and accepted the source/cooked distinction and actual wrist tracking diagnosis. It proposed original-F4 binding and complete nine-frame digest checks in independent entry paths; these are the next separate software patch. Codex corrected two overstatements: the formal launcher already freshly recomputes the entire proof before Kit; and its required precontact branch already sets the 240 s soft wall. The native run did not use an unbound fake hull and no budget increase is needed. R2 was sent with these corrections. The new audit was untracked during its initial read attempt, so external review of this audit itself is not claimed.

Registry: 60 passed; combined related regression plus Registry: 257 passed, 0 failed.
