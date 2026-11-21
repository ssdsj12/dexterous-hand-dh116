# Native Power near-open support settling

One GPU 2 Isaac run completed 2 setup steps and 238 measured SDK steps. The first exchange bound the measured stationary Reset poses. Later exchanges preserved continuous collider motion. The original one-environment budget issued zero actions, Teacher queries and PPO updates. Isaac closed naturally, with the outer process and GDB inferior both returning zero after 64.882 seconds.

Both end supports carried the rod on all 2,856 solver substeps, for 5,712 total contact pairs. There were zero hand or table pairs. An independent readback checked each artifact SHA, the exact rod/Reset/SDK identity, all 17 mounted joint states per hand, and same-rod wrench routing. The active joint positions matched the recorded targets throughout this interval. The solver's measured contact force balance residual was zero.

The final minimum rod centerline was 347.067 mm, 17.933 mm below Reset. Average vertical forces during the final 50 ms were 0.137665 N and 0.137657 N. The largest rod node speed in that interval was 0.015997 m/s. All 238 measured rod position, velocity and rest-length arrays matched the far-parked control exactly. The near-open posture did not alter the rod's supported settling.

This result establishes a native starting boundary for a subsequent measured approach. The rod still moves, and no grasp, hold, lift or Teacher PPO admission was evaluated here. It also leaves the separate Pinch and Tripod qualifications open.

Frozen SHA-256 `ca5033b088e111be61f8e92ac973ea106cce04f0fd26bcb4c1ecfdb54c6b0716`. Outer receipt SHA-256 `2da34c93c7dc967736840f709a04e7d57cc8998e9239e02e51a2a21b78d1f814`. The related Reset repair had 153 CPU regressions pass. The independent native check used the saved full-contact readback from the isolated support-settling receipt.
