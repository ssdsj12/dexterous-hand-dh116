# Tripod coupled wrist re-seat and index closure

The 9 mm flexible-wire run recovered some left thumb load with wrist-only corrections. The corrections consumed five of eleven close windows. The index advanced 0.01662 rad and remained about 2.82 mm from contact. An earlier fixed-wrist run reached index contact after losing thumb load.

The private archive audit checked 88 close-step snapshots against the original outer receipt. It reconstructed required link frames from measured joints. The reconstructed positions agreed with recorded common transforms to about 0.074 µm. All 47 underloaded snapshots admitted a combined wrist and index candidate in the geometric screen, with a minimum non-selected rod margin of about 0.759 µm. The replay evaluates predicted motion from saved states and has no new solver response.

The independent coupled-close option reuses the native thumb normal force, the measured local signed distance and the guarded nominal index step. It passes combined motion through the original 44 hull, support and table guard, and permits wrist alone if the combined proposal fails. A recovered load pauses the current correction until the next control window. The budget reserves the original two hold windows and caps emitted index increments at 88 SDK steps. The existing 3338 SDK, 239 control-window and 40032 solver-substep bounds apply.

The combined CPU regression passed 147 tests. The exact command, hashes, and native budget are in the readiness contract. Native validation must compare measured index travel, delivered thumb and index reactions, same-tick opposing load, support reactions, and natural process exit. This diagnostic covers one contact station of a single 9 mm specimen. Off-support Lift, Teacher PPO and held-out runs remain pending.
