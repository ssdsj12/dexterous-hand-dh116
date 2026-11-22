# Resized supported Tripod Reset and settlement

The first Isaac pilot uses a 9 mm diameter, 450 mm training wire. The two DH116 articulations, the table and the end supports use the existing setup. A separate builder retains the original joint seed while reanchoring the initial hands and support positions to the resized free rod. No camera, controller action, model or PPO runs in this pilot.

The frozen recipe allows one Reset, two SDK initialization ticks, eight measured SDK ticks and 96 solver substeps on a single GPU2 instance. The original rod allocation is expanded from 11 to 17 mesh slots at SDK2/rod tick 0. Actual native parameters are checked at Reset and at SDK10. The terminal sample compares the live CUDA positions with the saved positions, on the same clock as the palm poses and the physical reactions. The left and right support loads must be present at SDK10, the table force must remain below 1 µN, and no hand pairs may occur during settlement. Rejected attempts retain the raw exchange evidence.

Antigravity inspected the original Reset and shared registration, then reviewed the opt-in reader. Codex verified its recommendations against the locked source. The final change explicitly freezes its dependencies, uses an independent CUDA endpoint read, and checks the signed mesh reaction for each support.

## Checks before the native pilot

- CPU geometry measured 44 hulls: 46.038 mm minimum table clearance, 2.513 mm support clearance, and 1.747 mm rod clearance. These are independent preflight figures.
- Five focused CPU tests passed. The broader run recorded 89 passes and one existing failure. The historical fresh17 receipt binds the distal transfer file to the earlier Git blob `cefb595f…` and the baseline HEAD already uses `6cfe3e71…`. The strict audit remains in effect.
- Contact adaptation, Lift, Hold, Slip and the held-out specifications still require episode evidence.

Preflight commands and log SHA are recorded in the readiness contract. The native result below is separate from both the CPU preflight and the standalone CUDA getter checks.

## Native result

Ubuntu ran the published source `6a587c6b30d962023f529439ce6f726a5367ee46` through `scripts/python.sh`. GPU2 completed one Reset, two initialization ticks, and eight measured Isaac–DeformX exchanges. The 41-node solver advanced 96 substeps. Both the GDB inferior and outer process exited naturally with code 0 in 63.024 s. The independent audit checked all 37 artifacts, source hashes, SDK/solver identities and the cleanup sequence. No controller, Teacher query, or PPO ran.

- Native diameter 9 mm, rest length 450 mm and total mass 23.6925 g at SDK2 and SDK10. The 17-mesh runtime recorded 96 real substep pairs per end support and none for table or hands.
- At SDK10, support force magnitudes were 34.774 and 34.738 mN. The recorded force is on the support mesh, with downward Z. The solver reports the equal and opposite reaction on the rod. The table force was zero and the minimum native hand–support clearance was 2.513 mm.
- SDK10/rod tick 96 matched the independently read CUDA nodes. Material stations 35 (left) and 5 (right) gave palm-local positions of (−24.825, −61.839, 121.129) mm and (7.229, −57.410, 107.398) mm. The raw hand reaction was zero, as expected before closure.
- The wire still moved at the endpoint. Peak material-node speed was 56.37 mm/s and the combined support load was below the 232.42 mN wire weight. The evidence qualifies the loaded Reset and short exchange path only. It establishes neither static equilibrium nor a grasp.

Frozen SHA256 `45a2711372c25a2f9cedf3e5e811521d5541bf65835aa7eeb726cb34384c4156`. Outer receipt SHA256 `de97adae1f824eb814f91a489c6df8d7d1130e3635c1760174eeb51338107cce`.

The next experiment needs both stable opposing contacts and a clean unload of the stands before it can count Lift. Reuse the measured wire position and tangent in each palm to place the fingers, and then measure the per-pad force response to small target corrections. Report each of Pinch, Tripod and Power separately and reserve the held-out specimens for evaluation.
