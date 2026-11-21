# Power guarded high wrist bypass

The two original hands retain the measured SDK 880 opened finger targets. A two-phase world translation and reset-relative intrinsic XYZ rotation takes each wrist around the wire with its own waypoints. The 24D action compiler uses the original translation and rotation rates. The route does not request any finger movement, change the contact geometry, or attempt a lift. It terminates at a higher pre-grasp wrist pose for later approach.

| Certified lower distance across the sampled CPU path (mm) | Left | Right |
| --- | ---: | ---: |
| Rod | 2.142 | 2.358 |
| Support | 1.493 | 27.540 |

Each stage used 65 samples with a per-shape motion bound between samples, from the actual original stage pose, active joints and observed supported rod. Requests use the same intrinsic XYZ interpolation as the controller. A shorter direct left route had only 0.280 mm rod clearance in that screen, so the two-phase route retained that extra clearance. Left and right paths had separate world Y envelopes throughout. These are CPU geometry results.

`verify_dh116_power_retreat_then_open` adds `--high-bypass-reference <opening outer.json>`. It replays the exact certified opening, checks the live hand and rod states against the sealed SDK 880 snapshot, and audits native tracking after each new stage. All original runtime clearance, force, rod, and support guards remain active. Preflight permits 3584 SDK steps, 418 controls, 42984 rod substeps and up to 1080 seconds in one environment. Combined CPU regression passed 219 cases. This package contains no new native results, grasp pass or Teacher admission.
