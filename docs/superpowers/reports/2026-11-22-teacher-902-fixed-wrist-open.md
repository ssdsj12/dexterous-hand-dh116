# Fixed-wrist 902 open Reset candidate — 2026-11-22

This independent CPU candidate uses the original right Tripod final wrist pose, opens the thumb/index/
middle fingers, and closes with the wrist target fixed. It preserves the original 902 terminal active
joint target, side-specific USD, original left Reset, supported flexible scene, cameras and physical guards.
It is not a native entry point or a Teacher policy.

## Executed evidence

A bounded original-geometry search checked 135 legal active-joint candidates against the initial rod
plus every archived SDK3..240 frame (239 frames), all 17 original source convex envelopes and five
locked cooked pads per hand. 24 candidates satisfied the original Reset clearance thresholds.

The selected float32 open configuration has:
- Right rod lower bound 6.068451 mm; support lower bound 1.795037 mm; table clearance 46.151581 mm.
- Left rod lower bound 1.231136 mm; support lower bound 22.942230 mm; table clearance 57.382346 mm.

The compiled closure uses 64 windows / 512 ideal substeps. Left actions, right wrist actions and unchanged
finger axes stay exactly zero; original 0.3 rad/s joint rate and mimic limits are checked at each float32
target write. Endpoint joint error is 9.70e-8 rad. Every geometry sample passes original contact-floor,
noncontact, support and table checks. This is archived-rod CPU geometry, not simulated dynamic contact.

**Endpoint contact candidate FAILS.** Terminal pad gaps are approximately -0.0160, +0.0249 and +0.2373 mm
(thumb/index/middle); node IDs 3/2/4 and opposing normal dots -0.971/-0.902 are on the intended side.
The existing endpoint predicate requires all three pad gaps below zero. Therefore closure_geometry_passed
is true, endpoint_contact_candidate_passed and closure_passed are false; the audited CLI exits 1.
There is no measured force, delivered reaction, Lift, Hold or successful Teacher evidence.

## Review, fixes and tests

Antigravity R1–R4 read the actual Ubuntu source. Codex rejected the assumption that fixed wrist targets
eliminate physical tracking error. Its final source-binding and endpoint-semantics findings were reproduced
with seven failing tests, then fixed: the source request is pinned independently of the candidate and
clearance success is separated from endpoint contact. These final fixes were independently checked by
Codex; Antigravity did not reread the post-R4 bytes.

A previous evidence run failed at relative-path serialization after CPU calculation. Absolute-path
normalization fixed that error; its log is retained. Another test caught spurious float32 corrections to
unchanged joints; they now remain exactly zero.

- Focused final tests: 16 passed.
- Related CPU regression: 140 passed, 0 failed.
- No Isaac, CUDA solver, Teacher query or PPO update was run.
- 528 protected files, three historical checkpoints and both fixed camera files unchanged.

## Reproduction

Use the existing CPU environment and run python -m scripts.screen_dh116_902_fixed_wrist_open with:
--candidate configs/dh116_tasks/flexible_902_tripod_fixed_wrist_open_cpu_v1.json
--settling-receipt /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ag2287-power-settling-native/outer.json
--output a fresh private JSON path.

Evidence and full pytest command are in the private ref902/fixed-wrist-open-reviewed-proof.json and
fixed-wrist-open-regression.json. Readiness records their SHA. Raw geometry, logs and archived data stay
outside Git.

Next: independently qualify the new open Reset/settling in native physics, then use actual rod position,
q/qd, contact pairs and delivered opposing reactions to control closure. The existing target is an anchor
for the user's final grip pattern, not proof that the real flexible rod is already enclosed. Neither this
candidate nor the old 480-step replay enters the success pool. Self/inter-hand collision and native
clearance/exit checks remain required.
