# Coordinated left Tripod target increment candidate

The previous minimum-pad feedback regressed on the matched 224-SDK interval and was rejected. This package reuses an earlier unpublished four-joint candidate instead of increasing the isolated index dose.

## Runtime change

The original supported-rod Reset and right/left seating replay remain exact through SDK1704. A new opt-in `--coordinated-seating-reference` appends 12 dose windows and 32 zero-command hold windows. The existing runtime validates the archived hand/rod pose, contact pair buffers and issued reaction wrench at the boundary, then sends the four-joint ray through its real 24D exchange. Loaded left-hand evidence stops the whole ray; it does not independently remove one component. Right fingers and both wrists retain their existing targets.

The joint target increments are finger11/finger12/finger21/finger31 = [0.00123842094, 0.00071899769, 0.00058776207, 0.00044589402] rad. These are increments on the existing backend targets. They are not absolute measured-q goals.

## Critical distinction

At the real SDK1704 boundary, the corresponding target-minus-measured-q values are [0.00032261, 0.00139993, 0.00131255, 0.00072825] rad. The previous eight SDK samples have constant targets but a 0.000317 rad measured index oscillation. The old SDF screen used measured-q kinematics; its 37–79 micrometre target-pad gaps do not prove the actual target-command trajectory. The new stage states this limitation explicitly. Existing live geometry, force routing and original joint constraints remain active.

Antigravity reviewed the same source chain and the distinction between requested targets and measured state. Codex corrected its SDK/control-window denominator and unproven causal claims. No physical success is inferred from that review.

## Verification and next run

119 relevant CPU tests passed (52.59 s), including 31 owner/hook cases. The real runtime function is exercised with a CPU fake physical driver to confirm the coordinated four-slot command reaches control. The module first failed to import, then wrong-side contact rejection was reproduced and corrected. The original private overlay is retained unchanged.

Real-source CPU freezing succeeds for 2056 total SDK steps, 227 control windows and 24648 solver substeps, one environment, zero Teacher/model/PPO operations, 600 s soft wall. No native run has yet occurred for this source version.

The one-shot candidate is falsifiable: after the dose, the 256-SDK hold must show sustained real left and bilateral opposing reaction. If it remains intermittent or triggers the original guards, reject this direction and do not repeat the same fingerprint. Only actual stable load permits a later off-support Lift test. The 30 mm / 5 s / 0.5 mm standards and fixed stereo remain unchanged. This is a diagnostic, not a learned PPO Teacher or successful BC data.
