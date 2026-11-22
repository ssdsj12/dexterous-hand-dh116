# Fixed-Reset lift measurement for the flexible Teacher

`observe_supported_lift()` reads one completed 17-mesh solver packet. It owns the 41 world-space material node positions, both hands' grasp-specific contact coverage and opposition, and peak loads for the table and both end supports. Pinch and Tripod reuse the existing distal-contact reader. Power uses all five distal pads and the palm per hand, checks their actual load and opposition over all 12 solver substeps, and rejects required contacts on the wrong rod half. The snapshot also lists observed additional hand links, including the registered proximal Power link. Whole-hand contact remains a native validation item.

`FixedLiftReference.from_rod_reset()` fixes both height origins at the measured Reset. The lift signal uses the smaller change in median and minimum node height. It requires bilateral loaded contact, all three registered support loads below the existing tolerance, and the rod surface above the table. The signal records the current endpoint. Sustained lift, attachment, applied force, stability and slip require their own evidence.

`scripts/audit_dh116_supported_lift.py` reads the archived Reset and both geometry records, checks the recorded artifact hashes, and recomputes each physical endpoint without launching Isaac. It uses the geometry configuration radius and table top alongside the actual Reset node positions.

## Verification

- Eight new CPU cases initially failed on the missing feature and two review regressions reproduced before the follow-up fix. The final three-file CPU regression passed 35 tests in 50.79 seconds. The separate Registry suite passed 60 cases. Commands and log hashes are in readiness.
- Recomputed 256 historical Tripod physical rows from SDK 1801–2056. The fixed Reset was SDK 2. The archive has eight bilateral rows, zero rows with all three support loads clear, and zero off-support Lift rows. The highest minimum node height in the cohort remained 23.56 mm below the Reset minimum height. No new native simulation or PPO was executed.
- The archived run exceeded its soft wall-time budget. Its independent outer exit code is unknown. This replay only checks the saved physical rows against their recorded hashes.

## Review and next integration

Antigravity reviewed the existing physical source and reward route. Codex verified that the original initial support load is ordinary setup evidence and that single-endpoint contact cannot establish stable holding. The new output names the registered supports explicitly. A future reward consumer must use the existing phase history, the same measured Reset, and the PPO discount for the potential difference. A positive height change alone does not imply a positive discounted reward.

Antigravity read all three initial code files and matched their SHA-256 values. Its second pass flagged Boolean radius inputs and loss of additional hand-link evidence. Both concerns were reproduced and corrected with regression cases. That pass proposed routing the lift signal through the existing 480 Hz event callback. Codex verified that the legacy Collector and factories use exact type and mode checks, so a new training profile still needs an explicit versioned construction path. The fixed Reset origin takes precedence over the proposed sag offset. Stable prelift promotion requires actual motion evidence.

The native collector still needs this observation wired into the actual phase and reward. The next diagnostic should first show sustained opposing contact and a real unload-and-lift transition, then apply the existing 30 mm / 5 s / 0.5 mm completion criteria.
