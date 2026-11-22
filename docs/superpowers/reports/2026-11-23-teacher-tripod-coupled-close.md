# Tripod coupled wrist re-seat and index closure

The 9 mm flexible-wire run recovered some left thumb load with wrist-only corrections. The corrections consumed five of eleven close windows. The index advanced 0.01662 rad and remained about 2.82 mm from contact. An earlier fixed-wrist run reached index contact after losing thumb load.

The private archive audit checked 88 close-step snapshots against the original outer receipt. It reconstructed required link frames from measured joints. The reconstructed positions agreed with recorded common transforms to about 0.074 µm. All 47 underloaded snapshots admitted a combined wrist and index candidate in the geometric screen, with a minimum non-selected rod margin of about 0.759 µm. The replay evaluates predicted motion from saved states and has no new solver response.

The independent coupled-close option reuses the native thumb normal force, the measured local signed distance and the guarded nominal index step. It passes combined motion through the original 44 hull, support and table guard, and permits wrist alone if the combined proposal fails. A recovered load pauses the current correction until the next control window. The budget reserves the original two hold windows and caps emitted index increments at 88 SDK steps. The existing 3338 SDK, 239 control-window and 40032 solver-substep bounds apply.

The combined CPU regression passed 147 tests. The exact command, hashes, and native budget are in the readiness contract. The native run compared measured index travel, delivered reactions, simultaneous opposing load and natural process exit. This diagnostic covers one contact station of a single 9 mm specimen. Off-support Lift, Teacher PPO and held-out runs remain pending.

## Bounded Isaac result

The one-environment GPU 2 run reached all 3338 SDK steps and 40032 solver substeps. GDB and the outer supervisor observed natural exit 0 after 763.7 s. The sealed audit checked 6452 artifacts. There were zero Teacher queries, zero PPO updates and zero retries.

Thumb contact first occurred at SDK 2167 and its wrench was delivered at 2168. The index received 75 issued increments, 35 of them with a wrist correction. Its measured travel was 0.04279 rad and its final SDF gap was +0.531 mm. The prior wrist-only recovery reached 0.01662 rad and +2.817 mm. The new index load remained zero. The thumb had positive load in all twelve solver substeps for 52 of 128 close SDK rows and zero of 16 final hold SDK rows. There was no simultaneous opposition and no Lift.

During close, 40 commands were paused awaiting full measured thumb load and 13 while waiting for the next window after load recovery. This is a measurable source of lost close travel. The next candidate should preserve explicit force and geometry guards while testing a bounded substep transition between wrist compensation and index motion. It needs an independent trace of continued load, actual joint travel and off-support lift before Teacher promotion.
