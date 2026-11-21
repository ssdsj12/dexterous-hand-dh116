# 902 flexible Pinch and Tripod hands-clear approach

Each candidate approaches the settled 41-node DeformX rod from its own verified native Reset. Candidate thumb and partner pad normals oppose, with the original left and right articulation identities and rod half enforced throughout the path. The screen uses nine sealed consecutive native rod states and 17 sampled poses per hand, 17 source collision envelopes and five locked pad hulls at each pose. A first Tripod fit passed point-SDF checks while failing the full right middle-finger source envelope by 1.451 mm. Requiring the whole envelope and increasing right wrist height inside the existing limits produced a valid hands-clear approach.

| Mode | Minimum rod clearance L/R | Minimum support clearance L/R | Final intended pad gaps |
|---|---:|---:|---:|
| Pinch | 0.491 / 0.524 mm | 1.348 / 4.708 mm | 0.708–1.255 mm |
| Tripod | 0.435 / 0.300 mm | 22.942 / 1.310 mm | 0.857–1.476 mm |

The original native runtime retains the per-physics-step guard and consumes a SHA-verified same-grasp settled receipt and a passing pose-specific screen. It issues only the existing 24D motion contract. Each approach has a maximum of 432 SDK ticks including the 240-step settling prefix, 24 control windows, 5,160 solver substeps, 0 model queries and 0 PPO updates. It requires the hands-clear support condition and grants no contact or lift admission. All existing stereo calibration bytes are locked through the same path. The physical run for this stage is headless.

CPU regression: `python -m pytest -q tests/test_dh116_902_flexible_precontact.py tests/test_dh116_902_flexible_reset.py tests/test_dh116_power_palm_contact_evidence.py tests/test_dh116_supported_distal_transfer.py tests/test_dh116_pinch_supported_reset.py tests/test_dh116_tripod_supported_reset.py tests/test_teacher_force_closure.py tests/test_dh116_power_supported_task.py tests/test_native_phase_contract_v2.py`. Result 163 passed. Next evaluate actual reach and use the measured pad gap and reactions to close with held wrists.
