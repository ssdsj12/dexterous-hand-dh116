# Power native supported rod pilot

On GPU 2, the isolated free 41-node rod and both original DH116 articulations completed 2 setup steps, 8 measured SDK steps and 96 DeformX substeps. GDB inferior and outer exit were naturally zero in 53.121 seconds, with no signal or forced exit. The pre-set 480 Hz clock, 12 solver substeps per exchange, gravity and original joint contract were retained.

| Contact body | Solver pairs | Loaded substeps | Peak measured aggregate force |
| --- | ---: | ---: | ---: |
| Negative-Y support | 96 | 96/96 | 0.063631 N |
| Positive-Y support | 9 | 9/96 | 0.017072 N |
| Table | 0 | 0/96 | 0 N |

Both supports were loaded simultaneously in 9/96 substeps. The lowest material-node center fell from 365.000 mm at Reset to 364.144 mm after the eighth exchange. This is the start of a dynamic settling interval, with asymmetric support and no static equilibrium claim. Left thumb/middle/ring recorded 2/12/8 contact pairs. Both palms and all right pads recorded zero contact. The 96-step bounded pilot used zero action and does not establish a Power grip, lift or Teacher qualification.

Post-exit verification checked the bytes of all sealed components, each original 15-slot buffer capacity (7,380), monotonically matched/reset-anchored native SDK and solver ticks, per-hand joint/target and actual link identity, unilateral support isolation, exact previous-step hand reaction routing and per-substep recomputation. The largest pair force reconciliation residual was 2.235e-8 N. The left active-joint tracking error peaked at 0.000699 rad and the right remained zero across these first exchanges. The raw named DOF state contains wrist metres as well as finger radians. Target-error statistics here use only the six named active finger channels.

Recipe SHA-256 `0b35491eb24a557cb1b372b0891bda75279fa97a0fdde66fbf9eceef00741870`. Outer receipt SHA-256 `65a04975f6a9aea501b7e85e2f7d2c2bff923dac32c530c524b9178d3743228d`. The acquisition command is `python -B -m scripts.verify_dh116_power_supported_task` with the retained locked palm fixtures and independent output. Next measure support settling with hands clear and test fixed-wrist full wrap before issuing any lift.
