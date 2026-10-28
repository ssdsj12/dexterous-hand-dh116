# Subagent-driven progress

- [x] Task 1 — six-channel DH116 middle-hook contract (manual fallback after three implementer capacity failures); focused contract/touch/speed tests: 39 passed. Legacy four-channel integration fixtures remain for Task 3 migration.
- [x] Task 2 — pinned DH116 middle-finger geometry bundle; calibration/build/selector and geometry contract tests: 3 passed.
- [x] Task 3 — six moving meshes, middle Jacobian/reaction routing, active mask, and evidence fields; runtime/Jacobian tests: 38 passed.
- [x] Task 1 (new objective) — 12D DH116 hand-only state and batched parallel MPC; grasp/lift entry contract, previous-action continuity, no Panda fields. Focused tests: 9 passed.
- [x] Task 2 (new objective) — teacher/student hand-only JSONL replay collector and manifest verifier; short smoke generated 12 transitions.
- [x] Task 3 (new objective) — explicit Torch-CUDA Cosserat approximation with fail-fast device selection; backend tests: 2 passed. PyElastica CPU path remains unchanged.
- [ ] Task 4 (new objective) — Isaac hand-only physical grasp/lift validation and full GPU Cosserat parity/benchmark; not yet run.
- Isaac hand-only wire probe is now present, but its first 12-action headless run failed numerical-stability gates (large contact impulses); no grasp/lift claim.
- Student checkpoint is directly executable from Isaac joint feedback. A full 24-action/2400-step trajectory at a non-penetrating wire height remained finite but produced zero contact and zero lift. Moving the wire into the current finger envelope caused >500 N at the first physics step and the guard stopped. Next: calibrate detached-hand root pose and fingertip-to-wire position before retraining with physical contact/lift feedback.
- A 0.36 m wire-height probe did contact, but produced a 363.99 N peak and 33.65 mm uncontrolled wire rise within 8 steps; early-stop guard correctly failed it. Physical grasp/lift remains unqualified.
- Note: teacher/gpu implementer subagents hit model-capacity errors; Task 2/3 were implemented inline after the failure. No approach run was resumed.

- DeformX import Task 1: complete (uncommitted new files; 10 tests passed; task review clean). Source bundle and per-instance skeleton driver ready.
- DeformX import Task 2: in progress; actual coupled Isaac import validation pending.

- DeformX import Task 2: complete (uncommitted new/runtime files; 34 focused/regression and 29 solver tests passed; review clean; actualIsaac2stepdefault+12stepcontact validated, normalexit0). V65 saved2037760transitions/301epochs/10episodes-env, checkpoint integritychecked; intentionallynotresumedoldobject. Flexible512 PPO notimplemented.
- Task 1 (2026-10-27 PPO synchronized force plan): complete (commits fb1f8a7..31b0be9, review clean)
- Task 2 (2026-10-27 PPO synchronized force plan): complete (commits 73495c2..0220b3e, review clean)
- Task 3 (2026-10-27 PPO synchronized force plan): complete (commits c033789..52d6851, review clean; parity fix included)
- Task 3 force-floor follow-up: complete (commit bc70cee, review clean; instant-force CPU/CUDA parity restored)
- Task 4 (2026-10-27 PPO synchronized force plan): implementation and focused validation complete (commits 04664ba, 80e86cd, 25aa38e, 78a84e1; 114 focused tests passed; compileall and diff check passed). Isaac teacher probe was not rerun after the final code fixes.

2026-10-28 teacher/student: recorder, BC loader, mandatory student validation implemented/tested; physical search has zero qualifying success. Command-consistency audit and contact-effort candidate under physical validation. See docs/superpowers/plans/2026-10-28-command-execution-consistency.md. No commits.

2026-10-28 bilateral follow-up: preserved per-hand accepted lift heights with independent clearance correction; existing bilateral close admission regression verified. Contact-effort20ms dropout grace implemented, both scoped reviews clean; combined186tests passed. Right functional-pose candidate retained actualasset/limits and unchangedleft, and realrightthumbcontactsnowexist. Actual rightfit8 and rightfit_grace8 each zero successful firstepisodes; max bilateral>=1N continuousduration .05417s<.2s. Candidate is separate fromdefaults, noPPO. Full evidence in docs/superpowers/plans/2026-10-28-left-right-grasp-symmetry.md. No commits; noIsaacinstanceleft running.

2026-10-28 instant per-hand close hold: implemented individual capture/feedback and bilateral current-sample lift admission with zero close dwell, independent partial-hold heights, explicit contract and probe fields. Scoped review clean after fixing delayed +3 reward regression; root 176 tests passed. Isaac `success_teacher_20261028_181648_perhand_instant8`: 7/8 first episodes entered lift exactly on their first jointly ready physics substep, but zero physical lift successes; existing protection stopped env1 (still close) at right finger22 speed -3.10246 rad/s. Velocity root cause unresolved; no student PPO, no threshold relaxation, no commits. See instant-per-hand-close-hold plan and run instant_close_audit.json.

2026-10-28 velocity penalty: user approved replacing finite >3rad/s global termination with diagnostics plus time-integrated shaping penalty; PhysX caps and nonfinite/other guards retained.162tests passed, scoped review clear after two P2 fixes.8case actual comparison passed previous3.10246rad/s spike and continued; interrupted at490controlsteps to fulfill GUI request, peak3.55222rad/s, longest4.167ms, zero physical lift success. Sole GUI launched teacher_demo_20261028_183709_instant_velocity_penalty (1env,z-1.5mm,4N/hand). No PPO training/commit.
