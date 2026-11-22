# Actual Tripod lateral-wrist experiment: stable grasp and Lift remain open

The corrected launcher ran once on GPU2, with one environment, 2192 SDK steps, 244 controls and 26280 rod substeps. No Teacher queries, models or PPO updates. The original replay checked the SDK2056 rod, hand joint/target/pose, solver pair and wrench boundary. Outer and GDB inferior exits were naturally 0 after 625.601 s. The independent replay used artifact hashes from the closed run.

An earlier wrapper incorrectly routed the launcher's Python prefix probe through GDB. It returned 1 before an Isaac instance was created, with zero SDK steps. That attempt and the verified replacement wrapper are retained separately. No physics run was repeated.

| Actual evidence | 128 SDK steps before XY command | 128 SDK steps after XY command |
|---|---:|---:|
| Left loaded | 0 | 2 |
| Right loaded | 128 | 119 |
| Bilateral loaded | 0 | 2 |
| Longest bilateral run | 0 | 1 |
| Left index minimum normal force | 0 N | 0 N |
| Left thumb minimum normal force | 0.0330 N | 0.0118 N |

These intervals are adjacent before/after observations in a single run. They do not isolate the dose from the ongoing motion or generalize to other initial conditions. The command advanced the left target (+25, −125) µm in World XY. At the end of the eight-step command, the measured palm moved (+13.1, −64.1) µm. The geometric screen that used a static wrist displacement cannot predict loaded finger force or rod vibration.

The 2190 measured physical rows reached Close 2121 times and PreliftHold 69 times. There were zero Lift rows, zero rows with all registered supports unloaded and zero complete successes. For the final hold, the longest loaded interval was one SDK step (2.083 ms). This candidate is blocked. Another fixed open-loop dose is not the next design. Use current palm/rod relative pose and individual pad reactions to decide both posture and clamp increments, then test stable grasp before enabling the synchronized World Z command. Pinch, full-fist Power and held-out cable specifications require separate native acceptance.
