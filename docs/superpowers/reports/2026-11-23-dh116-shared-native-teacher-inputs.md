# Actual shared Teacher input readback

The opt-in --teacher-source-audit path now reads the original SDK-vs-IsaacLab
mounted-hand sources directly from real left/right articulations. Both actual
view paths, 17 DOF readbacks, authored seeds, 66 measured common fields, all 41
material nodes, native full-N parameter getter and per-lane context/Reset history
are bound to the shared parent source. Diagnostic CUDA means are computed by
the independently versioned shared Teacher; actual issued commands still come
from the authenticated historical action sequence. This is not on-policy PPO.

## Actual GPU 3 result

Two environments, four DH116 articulations, one shared real DeformX solver.
226 SDK steps: two initialization, eight setup/isolation, 216 coupled exchanges.
29 Teacher input captures, each with 132 valid common fields and the original
305 NaN/false fields per lane. Complete native parameter layout stays N=2.
Nonzero reaction writes: 14/0. SDK 218 selected lane 0 Reset, followed by eight
SDK steps; final resets 2/1, rod ticks 96/2592. References and previous request
clear only for the selected lane; no borrowed previous action becomes measured
joint state. The raw shared contact/Reset proof remains separate from policy
source and grasp qualification.

Outer natural exit 0, no signal or forced end, app.close returns; 58.533 seconds.
Capture median 35.229 ms, mean 42.723 ms, cold maximum 238.589 ms. These are
two-lane source/diagnostic-forward times, not 128-lane control-window timings.
Zero optimizer calls, zero policy-issued actions and zero native PPO updates.

Outer SHA: 58d9b86e378b07e4157e7706e85e1e58df41e45ae3d200470f178574f6064a29.
Input capture SHA: 4c25d6a6d07582aca9aab4b620d572faae67e77170f5361022f005fa42f66dfe.
Raw clock input SHA: 4fd34be763bdf1abe88e002f83d9b00bc7e52f4b89f44bee31a18f05fa43d8d9.
Raw data remains local in shared-teacher-inputs-v3-gpu3-r_agumkc under the existing
teacher-collision-ordered-fk-20261119 evidence directory. Full argv/frozen code
is in attempt.json; no source hash changed during execution.

## Two preserved failures and clock correction

First attempt stopped at SDK 10, naturally exiting 1. Actual time was
0.020833334419876337 seconds while nominal 10/480 is 0.020833333333333332.
The original clock contract's 1 ns tolerance rejected this 1.087 ns difference.
Second attempt also stopped at SDK 10 because get_physics_dt reports nominal
1/480, but the actual core current_time callback accumulates float32 step_size.
Both failure logs/receipts remain local; neither is a lifecycle pass or training.

Installed Isaac 4.5 SimulationContext._physics_timer_callback_fn adds the native
callback step_size and increments the integer step count. The revised source
separately retains configured SDK dt, float32 callback timer dt, measured SDK
time and the integer-tick policy time. It verifies raw time against the explicit
callback dt and integer tick; no free offset is fitted and no clock value is
overwritten. Context/action schemas use their declared nominal tick time.
A real one-tick lag still fails. At final SDK 226 the retained raw-vs-nominal
difference was 24.556 ns, not a millisecond lag. This correction grants no RGB
synchronization or camera reprojection qualification.

The exact measured failure was reproduced before each clock change. CPU group:
36 passed, two CUDA skips. Combined legacy and shared source/model/PPO regression
on GPU 3: 60 passed, no failures, in 45.25 s. The GPU tests use explicit synthetic
source data and remain distinct from this actual Isaac replay.

Command: scripts/python.sh -B -m scripts.smoke_dh116_shared_rod_native with the
existing explicit receipt/smoke/backend paths and their SHA arguments, plus
--contact-receipt, --contact-receipt-sha256, --reset-last-window and
--teacher-source-audit. Complete argv stays in the frozen attempt, not inferred.

## Remaining training integration

Next consume original source-qualified R0 endpoints exactly once per physical
tick, settle shared on-policy windows, bootstrap terminal observations before
local Reset, and run actual original PPO updates. Then scale all three grasps
to 128/512 with measured memory/window/update metrics. Original 24D, guards,
success criteria, R2 closure and expert-pool gates stay intact. Current input
readback success is not grasp or Teacher training success.

All 528 protected files match their inventory. Historical /tmp checkpoints remain
absent and were not recreated. No raw assets, model binaries or ledgers upload.
