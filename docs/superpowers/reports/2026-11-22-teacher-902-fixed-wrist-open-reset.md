# 902 fixed-wrist open Reset execution path

Base: 5d0f99acc27ddc7e02c97d9c7c1d1b753fb4cb5b, branch 11-17.

The independent entry reuses acquire_supported_reset, run_shared and the existing explicit resource owner. Only right-hand initial wrist and opening change, using the source-pinned original 902 endpoint and exact float32 joint targets. The original scene, left hand, supported free DeformX rod, fixed stereo configs and safety thresholds are retained.

reset_config validates the original target and remaining finger targets. checked_reset_proof requires both full 22-shape, 239-frame initial audits and retains the independently failed closure/contact flags. run_screen finishes standalone USD in a child and explicitly accepts its contact-failure exit 1 only with complete initial evidence. main freezes the exact sources, checks fresh proof source hashes, binds measured palm/proximal SDFs and executes the unchanged no-action path.

Validation: first red test failed importing the absent entry; final related CPU regression: 79 passed, 0 failed, including 17 new cases; Registry regression: 60 passed. Fresh CPU plan freeze exited 0. Private logs: ref902/open-reset-red.log, open-reset-green.log, open-reset-regression.log and fixed-open-reset-native-tripod/freeze.log. Run using the private argv.json; without --approved-frozen-sha256 this only creates a plan.

Native not yet run at this commit. Proposed single attempt: GPU2, 1 environment, 2 SDK initialization + 238 measured SDK steps, 2856 solver substeps, 0 control/Teacher/PPO, 180 s inner soft bound. No retries or automatic kill. A later run must check natural outer exit, all SDK3..240 rows, no internal failure file, actual q/qd/target, live rod clearance and support forces. Original endpoint middle gap remains +0.237273 mm; this package does not certify contact or Teacher grasp.

Antigravity R1 independently read all four current sources and proposed this minimal reuse. Its identified exit-1 and internal-failure risks are retained. R2 read the actual entry and found proof/frozen SHA TOCTOU and unconditional exit 0 after an internally guarded failure. Both are fixed and covered by red-to-green tests. R3 final reread is pending. Main freeze was exercised against real CPU evidence; not every main branch is unit-tested.

The entry now checks complete per-step files and exact SDK/solver/pair counts after native cleanup. A natural exit and a successful measured Reset are distinct. The first CPU-only plan is superseded and was never executed. One Registry command initially named a nonexistent test file (exit 4); the corrected tests/test_dh116_work_packages.py run passed all 60 cases.

## R3 supported-rod correction
Antigravity found that pair_counts includes support_0/support_1: rejecting every nonzero pair would reject valid support settling. A fixture with actual support names and positive support counts reproduced the bug (1 failed, 16 passed). completed_interval now requires both supports present with positive integer counts and a nonempty hand set with zero integer counts. Missing support, zero support and missing hand evidence have explicit tests. Latest related regression: 82 passed / 0 failed (20 new cases); prior Registry run: 60 passed / 0 failed. Two prior CPU-only frozen plans are superseded, never run; native attempts remain zero. Before execution use a fresh plan and current frozen SHA.

## Final R4 and actual native interval
Antigravity reread final entry d5929c77322c732c8deba7779d37fddea0a9c0cd5360d29863634fb5c64a4c31 and test 41792d7b4cab9e90448cbd1cd33150703f4e95bc0ed35854af3db484a32c6b58 and found no new CPU wiring blocker. No native qualification was inferred from review.

One actual run used final fingerprint 7b01c0999e5cfc2a3911d1551d4508e3f30710b220405a47fc9fc3dd1e473aaa, code HEAD 6a644fece0cd0a4735ba0fccc3b15615fe7ee20c. It completed SDK 2+238=240 and 2856 solver substeps with 238 exchange records, zero action/Teacher/PPO. Outer natural exit 0; forced_exit=false, exceeded_budget=false; wall 91.826s. No internal failure file. Each end support has 2856 captured pairs, hand pairs zero; support peak loads {'table': 0.0, 'support_0': 0.1564640571747694, 'support_1': 0.15651294064896087}; pair force balance residual 0 N. Receipt SHA: 5681a8cddffb664104b0d8ef7699cc8354693407a1c9e3b2427f2339de2d1022.

Tail SDK232..240 active joint error is 0 rad at float32 readback on both hands; maximum passive affine mimic residual is 4.0046871e-8 rad. Backend targets are constant. The passive backend zero drive target is not a valid tracking target; passive motion is evaluated against the original affine mimic instead. Minimum recorded table clearance is 0.046151561m. Final maximum rod node speed 0.012912691m/s is reported without a settling-qualification claim.

All-22-shape live rod/support clearance recomputation remains pending, as does endpoint contact and all grasp/hold/lift qualification. This interval is complete, not a qualified Teacher. First private metric reducer used scalar conversion on an array-valued clearance field and failed; fixed to np.min, retaining tool/error evidence. Raw native data/maps/logs remain private at ref902/fixed-open-reset-native-tripod-r4.

Recovery: do not rerun this consumed fingerprint. Next read real rod frames and measured q/qd, finish clearance audit, then plan force-guided closure. Fixed RGB configs unchanged; this diagnostic collected no RGB.
