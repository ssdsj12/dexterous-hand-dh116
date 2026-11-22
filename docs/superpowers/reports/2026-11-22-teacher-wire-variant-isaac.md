# Resized supported Tripod Reset and settlement

The first Isaac pilot uses a 9 mm diameter, 450 mm training wire. The two DH116 articulations, the table and the end supports use the existing setup. A separate builder retains the original joint seed while reanchoring the initial hands and support positions to the resized free rod. No camera, controller action, model or PPO runs in this pilot.

The frozen recipe allows one Reset, two SDK initialization ticks, eight measured SDK ticks and 96 solver substeps on a single GPU2 instance. The original rod allocation is expanded from 11 to 17 mesh slots at SDK2/rod tick 0. Actual native parameters are checked at Reset and at SDK10. The terminal sample compares the live CUDA positions with the saved positions, on the same clock as the palm poses and the physical reactions. The left and right support loads must be present at SDK10, the table force must remain below 1 µN, and no hand pairs may occur during settlement. Rejected attempts retain the raw exchange evidence.

Antigravity inspected the original Reset and shared registration, then reviewed the opt-in reader. Codex verified its recommendations against the locked source. The final change explicitly freezes its dependencies, uses an independent CUDA endpoint read, and checks the signed mesh reaction for each support.

## Checks before the native pilot

- CPU geometry measured 44 hulls: 46.038 mm minimum table clearance, 2.513 mm support clearance, and 1.747 mm rod clearance. These are independent preflight figures.
- Five focused CPU tests passed. The broader run recorded 89 passes and one existing failure. The historical fresh17 receipt binds the distal transfer file to the earlier Git blob `cefb595f…` and the baseline HEAD already uses `6cfe3e71…`. The strict audit remains in effect.
- No variant native result is asserted yet. Contact adaptation, Lift, Hold, Slip and the held-out specifications still require episode evidence.

Preflight commands and log SHA are recorded in the readiness contract. The native result will be recorded separately from both this preflight and the earlier standalone CUDA getter checks.
