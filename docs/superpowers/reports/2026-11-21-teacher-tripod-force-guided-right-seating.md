# Tripod force-guided right seating

The measured SDK 1296 boundary is a complete slice of the naturally closed bilaterally contacting Tripod run. At that boundary the left tripod passed all 12 load and opposition substeps. The minimum right middle load was 0.033 N, and right total load was below qualification. The worker freezes and verifies its actual hand, rod, contact pairs, and applied reactions before changing actions.

At fixed wrist targets, the frozen draft can issue small positive increments on the original right index and middle channels only. It computes minimum force from all 12 solver substeps in the last completed SDK sample. Completed qualification suppresses the increment. When total force is adequate, a comfortably loaded pad is held. Scheduled increments still cap each joint at 0.003 rad of requested additional travel in ten windows. Eighteen zero-action windows then check whether contact persists. Input actions and the force used for each decision are written against the actual SDK clock.

A 33-sample CPU replay of simultaneous maximum travel from the sealed boundary gave a minimum non-pad rod clearance of 1.032 mm. Estimated index and middle SDF distances changed by about −233 and −124 µm. Those estimates authorize only a bounded candidate. The original 0.25 mm rod guard, per-pad penetration guard, hand force qualification, and support and table collision checks still apply.

The independent preflight requires the earlier retarget inputs plus `--seating-reference <Tripod retarget outer.json>`. It emitted up to 1520 SDK steps, 160 controls, 18216 solver substeps and 600 seconds. Combined CPU regression passed 216 tests. This package includes no new native result, grasp pass or Teacher admission.
