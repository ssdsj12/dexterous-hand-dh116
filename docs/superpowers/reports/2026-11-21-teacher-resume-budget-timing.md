# Pinch B budget failure and launch clock repair

The first B process restored its own copy of the authorized A weights and RNG after a distinct measured Reset. It executed eight controls, 64 measured SDK steps and 768 solver substeps. Independent CPU readback of the original raw ledger and checkpoint confirmed update 2, global control step 16, two cumulative accepted minibatches and a parameter change L2 norm of 0.003452503.

The post-update wall check failed. The original process logged only one budget marker, with 37.392 seconds remaining before the update. The prelaunch CPU preparation had consumed 23.510 seconds of the inner timer before the launch envelope was issued. Both the outer process and GDB inferior exited with code 1 after 196.386 seconds. The lifecycle recorded fault close and preserved the original log, raw trajectory and checkpoint. The failed B receipt is 64294b7cbeacc54100da691b7b4ed1f472577ca6b589907d546de501c5b1f36e. It carries no diagnostic or Teacher qualification.

The scheduler now starts the native clock after the read-only A origin preflight and before issuing the envelope and spawning the worker. The live grant still independently re-verifies A, launch and exit remain bounded, and the 180-second inner and 240-second outer limits are unchanged. The two new CPU cases verify that a 23.49-second preflight consumes neither effective native limit. The selected Ubuntu regression passed 220 tests.

The source closure has changed. The retry registers a new bounded A recipe and can create B only from its genuine new receipt. The original failed B checkpoint stays in its own attempt directory. Grasp qualification still requires the original force, lift, hold and slip criteria.

[Readiness](../../contracts/dh116_teacher_actual_native_clock_v2.json).
