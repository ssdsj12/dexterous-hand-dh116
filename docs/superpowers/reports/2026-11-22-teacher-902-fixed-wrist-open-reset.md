# 902 fixed-wrist open Reset execution path

Base: 5d0f99acc27ddc7e02c97d9c7c1d1b753fb4cb5b, branch 11-17.

The independent entry reuses acquire_supported_reset, run_shared and the existing explicit resource owner. Only right-hand initial wrist and opening change, using the source-pinned original 902 endpoint and exact float32 joint targets. The original scene, left hand, supported free DeformX rod, fixed stereo configs and safety thresholds are retained.

reset_config validates the original target and remaining finger targets. checked_reset_proof requires both full 22-shape, 239-frame initial audits and retains the independently failed closure/contact flags. run_screen finishes standalone USD in a child and explicitly accepts its contact-failure exit 1 only with complete initial evidence. main freezes the exact sources, checks fresh proof source hashes, binds measured palm/proximal SDFs and executes the unchanged no-action path.

Validation: first red test failed importing the absent entry; final related CPU regression: 79 passed, 0 failed, including 17 new cases; Registry regression: 60 passed. Fresh CPU plan freeze exited 0. Private logs: ref902/open-reset-red.log, open-reset-green.log, open-reset-regression.log and fixed-open-reset-native-tripod/freeze.log. Run using the private argv.json; without --approved-frozen-sha256 this only creates a plan.

Native not yet run at this commit. Proposed single attempt: GPU2, 1 environment, 2 SDK initialization + 238 measured SDK steps, 2856 solver substeps, 0 control/Teacher/PPO, 180 s inner soft bound. No retries or automatic kill. A later run must check natural outer exit, all SDK3..240 rows, no internal failure file, actual q/qd/target, live rod clearance and support forces. Original endpoint middle gap remains +0.237273 mm; this package does not certify contact or Teacher grasp.

Antigravity R1 independently read all four current sources and proposed this minimal reuse. Its identified exit-1 and internal-failure risks are retained. R2 read the actual entry and found proof/frozen SHA TOCTOU and unconditional exit 0 after an internally guarded failure. Both are fixed and covered by red-to-green tests. R3 final reread is pending. Main freeze was exercised against real CPU evidence; not every main branch is unit-tested.

The entry now checks complete per-step files and exact SDK/solver/pair counts after native cleanup. A natural exit and a successful measured Reset are distinct. The first CPU-only plan is superseded and was never executed. One Registry command initially named a nonexistent test file (exit 4); the corrected tests/test_dh116_work_packages.py run passed all 60 cases.
