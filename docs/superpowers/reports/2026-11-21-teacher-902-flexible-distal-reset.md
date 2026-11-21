# 902 distal grasp references on the flexible task

The videos set the target grasp topology. Pinch uses the thumb and index. Tripod uses the thumb opposite the index and middle. Either mode may begin with its own hand positions and clearance. The old replay supplied geometry references only. The two-eye full-resolution Student rigs retain their existing files and calibration digests, including the optional hand-order rig.

The Pinch reset comes from the 902 archive initialization (SHA `500e9f25f6b05f8a07d68358f8fd38c0d8d5a7172213d31b769bd38846484a68`), with the historical 60 mm support-height translation applied once. Tripod uses its independently screened flexible-task seed. The Tripod rigid initialization (SHA `c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b`) was referenced by the old contract but not recovered. Both archives retained 371D observation vectors and 24D actions and did not provide the absolute wrist translations needed to reconstruct the historical Tripod seed directly.

Both new candidates use the original supported DeformX scene, with one free 500 mm, 5 mm radius rod and two physical supports at the existing solver-compatible locations. The historical rigid rod radius was 8 mm. Both support tops remain at 360 mm. The historical supports were at (430, ±243) mm in X/Y. The retained flexible supports are at (436, ±238.5) mm and maintain the original 24 × 10 × 60 mm dimensions. With the old supports, the closest 41-node material station along Y would be 0.5 mm beyond each box footprint. This round keeps the existing registered flexible support geometry.

The native distal evidence reducer checks both fixed material index and current world Y to stop crossing hands or accepting contacts on the opposite half. It reports failures with the physical link, native solver substep and node ID. It keeps the force vector and contact source checks. The bounded native entry reuses the original 44-envelope Reset verification, 17-mesh contact capture, and outer exit accounting. The CPU convex-hull preflight runs in a separate process to keep the Isaac process free of standalone USD imports.

| Initial clearance | Left rod | Right rod | Smallest support |
|---|---:|---:|---:|
| Pinch | 2.269 mm | 2.742 mm | 24.768 mm |
| Tripod | 1.247 mm | 2.970 mm | 1.920 mm |

CPU screening command for each grasp: `python -B -m scripts.verify_dh116_902_flexible_reset --grasp pinch --output /tmp/dh116-902-pinch.json`. Substitute `tripod` and a separate output file for Tripod. The CPU regression command was `python -m pytest -q tests/test_dh116_902_flexible_reset.py tests/test_dh116_power_palm_contact_evidence.py tests/test_dh116_supported_distal_transfer.py tests/test_dh116_pinch_supported_reset.py tests/test_dh116_tripod_supported_reset.py tests/test_teacher_force_closure.py tests/test_dh116_power_supported_task.py tests/test_native_phase_contract_v2.py`, with 160 tests passed. No original policy, checkpoint or camera settings are used as authority for the new native result.

Next measure the same-clock 240-step native Reset and settle for each grasp. Following native clearance, establish opposing pad forces, hold a fixed wrist until contact is stable, and evaluate unsupported lift, drift and hold. The original 30 mm, 5 s and 0.5 mm standards apply.
