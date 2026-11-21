# Tripod seating phase hand ownership

The first left-index stabilization run stopped at SDK 1296 with an explicit hand-ownership exception. Adding slot 2 to the shared slot table allowed the right-phase compiler to put that left slot into its frozen map. Runtime correctly rejected the mixed map before issuing either seating dose. The worker completed normal App close and exited 1. There was no native signal or geometry rejection. The original failed receipt remains available.

The corrected compiler freezes an independent slot map for each hand and validates the exact slot-to-pad mapping at selection. The new regression reproduces the wrong frozen right map before the fix and runs that compiled map through the selector afterward. Offline preflight additionally exercised both real archived boundaries with the actual contact reducer: right index and middle at SDK 1296, and left index at SDK 1480.

The updated implementation passed 221 CPU regression tests. The corrected recipe retains the one-instance 1704 SDK-step ceiling and original collision, force, action, and grasp criteria. The failed run provides no result on the left dose. The corrected recipe requires its own native receipt.
