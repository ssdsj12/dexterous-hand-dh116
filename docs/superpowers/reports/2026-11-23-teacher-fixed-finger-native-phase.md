# Enable held fingers in native Pinch and Tripod phases

The `--fixed-nonparticipating-fingers` option keeps the full 24D action and the original mounted wrist channels. It is available in the two 902 diagnostic launchers.

- Pinch keeps both middle, ring and little finger targets fixed. The hold starts immediately after the verified SDK 432 anchor and covers the 64 new withdrawal rows.
- Tripod keeps both ring and little finger targets fixed. The hold starts immediately after the separately verified SDK 568 anchor and covers the new right-hand continuation.
- Power uses its existing full-hand command path.

The launcher rejects any nonzero held-finger command in the opted-in portion before launch. The original controller and PhysX guards execute normally. The runtime verifies binary32 submitted and backend targets against the phase's first lawful pre-action readback and verifies clock continuity, episode ownership and side-specific joint identities. A discrepancy stops further commands. Actual joint angles remain physical measurements and can move under compliance.

The archived prefix is replayed exactly. Pinch's left middle and Tripod's ring and little channels have small nonzero residuals in parts of that prefix. A fixed-finger policy from Reset and a masked PPO policy distribution will need their own recertified trajectories. These diagnostics do not change an existing checkpoint or its action probabilities.

The integration suite exercised both launchers, archival Pinch in phase and the actual SDK 569–576 native Tripod readback. Nineteen CPU regressions passed. The fixed-enabled Pinch launcher completed an independent CPU-only GDB check, with 2,251 pinned files and exit 0. The new option had not executed in Isaac before this check. Its integration preflight SHA is `cc8607dc60f3d801d82df7915329c01e136142f1a3be67336cd690a558ff4a61`; the CPU test log SHA is `82678a71af1cc2ca6565ee490c3bb6ce59580cb62fbc4e5793a6eab074e3a2ca`.
