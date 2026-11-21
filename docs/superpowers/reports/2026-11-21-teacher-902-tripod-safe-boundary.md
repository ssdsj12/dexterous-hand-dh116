# 902 Tripod safe boundary before distal contact

The original 24-window approach stopped at SDK 411 when the right middle-finger source envelope reached 0.247996 mm, below the unchanged 0.250 mm guard. This remains a failed native run. Its successful prefix is not relabeled as a naturally completed run.

bounded_prefix() now selects the first 21 original action windows, without retiming or recomputing their target increments. The optional --precontact-windows 21 ends a new run at SDK 408 (2 initialization + 238 settlement + 168 controlled SDK steps; 4872 solver substeps, 21 control windows). Default 24-window behavior remains unchanged. The independent boundary must finish and close naturally before it can support a contact continuation.

Antigravity desktop review R1/R2 on HEAD 227fe18 read the actual script SHAs and identified the hard-coded old Tripod waypoint dependency as a later contact-entry issue. Codex independently verified that live q/root-driven FK matches measured link positions to 0.000104 mm; there is no evidence for a millimetre-scale frame error. The controller must continue integrating from backend targets rather than measured q.

## Validation

- First red test failed because prefix selection did not exist; test verifies identical commands, no speed-up, invalid window rejection.
- CPU regression: 164 passed, zero failed. Native boundary pending.
- Native guard, table/support contacts, original USD/joint limits/24D contract, 480Hz physics and both fixed stereo camera configuration bytes remain unchanged.
- No Teacher queries, PPO, new RGB collection, or grasp qualification.

Command extension: append --precontact-windows 21 to the same original Tripod 902 native recipe with its settled receipt and precontact proof, then freeze a fresh plan. Full raw logs stay on Ubuntu and are SHA-bound by readiness.

Next: verify this single shorter native run and exact boundary identity; screen the measured distal closure separately. Pinch wrist alignment and Power full wrap remain independent tasks.

## Actual native result

Executed one new frozen recipe on GPU 2 at source commit 3f2a23ea. Completed SDK 408 / 4872 solver steps / 21 control windows in 124.50 seconds. App.close returned; GDB inferior and outer process naturally exited 0, no timeout or forced signal.

All 168 controlled rows have matching SDK/rod/link clocks and no guard failure or hand contacts. The final hand geometry, nodes and action state match the earlier failed run's SDK 408 prefix exactly through verify_waypoint_boundary(..., exact_geometry=True). The earlier SDK 411 failure remains unchanged and separate.

Readiness binds outer SHA ba230eed88c2b003f58bb0a7104d3811312d3bafabde999c0ca9ba5435f1e0a9 and final row SHA 3c066ff00d19e1b2fd640fdfc5b68ffa050854f1025090ee1cfe64b7cacbdf19. Both cameras retain their previous bytes. This qualifies an independent safe precontact boundary, not distal contact, grip, Lift/Hold or Teacher training.
