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
