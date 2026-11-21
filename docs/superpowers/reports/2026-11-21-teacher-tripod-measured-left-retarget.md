# Tripod left approach retarget

The previous left approach stopped at SDK 1178. This package uses the measured right-hand contact, rod position and active left-hand state from the complete SDK 1136 row. The failed receipt is accepted only for this diagnostic input. Its outer exit code was 1 without a signal and its native close marker was present. The original failure remains recorded.

The new left target retains the original three distal pads and material stations. The screen uses 11 measured rod states spanning SDK 1128–1136, 1170 and 1177. Across 17 samples of the candidate path the minimum non-target clearance was 0.899 mm. Terminal distal SDF gaps were approximately −0.070 mm each at nodes 35, 35 and 33, with estimated normal dots of −0.816 and −0.556. These are offline geometry results.

The worker runs the original commands through SDK 1136, checks measured poses, rod arrays and the right contact and composer records against the sealed row, then switches only the left commands for 24 drive and 16 settling windows. The original 0.25 mm live guard, support collisions, and selected pad SDF limits remain active. Use `python -m scripts.screen_dh116_tripod_bilateral_retarget --receipt <outer.json> --output <screen.json>` for the independent screen, and `scripts.verify_dh116_tripod_supported_reset` with the paired retarget reference and screen for the launcher.

The combined CPU suite passed 208 tests and the launcher preflight passed the no-USD-before-AppLauncher check. The bounded diagnostic permits up to 1456 SDK steps, 152 controls and 17448 solver substeps in one environment. No native run, grasp qualification or Teacher admission is recorded for this package.
