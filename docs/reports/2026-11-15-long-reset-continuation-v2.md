# W5 v2 CPU continuation and long-clock consumer gate

Scope: Task3/Task4 of the approved 2026-11-15 long-reset-v2 plan. Base
`6d90a7eb49b4dce0275fe4d81b82a9dd68fcb6f5`. New files only; no old
control, reader, physics, gain, joint, USD, registry or readiness edits.
No App, Isaac, GPU, native execution, download, installation or push.

The original CPU `test_dh116_dual_hand_episode.setup` supplies the actual
hand/rod/SDK doubles and setup=2 endpoint. The independent mesh fixture is
attached to those same objects under explicit `cpu_fixture` scope. The SDK
step function performs repeated double-DT addition; no endpoint repair or
hard assignment advances to a boundary. This fixture mode is declared and
is not the short native receipt's observed fp32-term mode.

| Boundary | Actual inherited control progression | Zero-step Reset | Next episode | Total SDK / solver / internal |
|---|---|---|---|---|
| 240 | 29 controls to234, then original right joint guard on actual substep6 | PASS | 248 | 248 /246 /2952 |
| 258 | 32 full controls after setup2 | PASS | 266 | 266 /264 /3168 |
| 2402 | 300 full controls after setup2 | PASS | 2410 | 2410 /2408 /28896 |

Each boundary performs a real independent `MeasuredMeshResetV2.run` on the
same scene/rod, advances neither SDK nor solver, rebuilds all11 mesh slots,
increments reset epoch, invalidates the modeled previous mesh epoch, and
then completes another inherited8-step control. The original guard restores
the failed240 joint to the seed during Reset. The CPU mesh epoch/readback
model does not certify real native borrowed-view behavior.

`FixedS0PhysicalEpisodeV2` overrides only Reset and state-clock; original
`control`/`pre_action_state` implementations are inherited. Its exact
source-locked factory returns exact `ClearanceGuardedExchange`; original
guard five-file inventory and2.1mm floor are rechecked. Wrappers, bare
exchange, replaced Reset factories and changed source locks reject before
advancement. The real post-SDK guard stops on step3 with no fourth step.

Budget evidence uses original `SessionBudget` with explicit3000 limits:
setup charged once, each observed actual advanced raw row charged once,
Reset refunds nothing, partial6 consumes6 solver calls/72 internal
substeps, and a persisted-then-quarantined row remains charged. These are
CPU fixture accounting checks, not a new production session owner.
W3's real CPU history/context51/phase classes are instantiated afresh at
Reset: new owner UIDs and episode/reset identity, Close phase, zero elapsed,
prelift/hold and previous action, no anchors. The native W3 source adapter
and owner/session/collector/checkpoint integration remain missing.

The actual retained S0 manifest
`a39de62de8ea97b3b154af2f910ba9dc097ccf64f9d650dad627ee2d8516cdce`
is loaded by the original loader, including its52MB pinned numeric source;
no621MB stream or identity cache is used. `ResetPreflightV2` checks that
original identity separately; `validate_live_reset_v2` uses independent
v2 numeric/readback/current mounted/rod/backend checks. CPU packets are
rejected as live native evidence. No positive live-native preflight was run.

| Original consumer | Timestamp source | float64 at240/258/2402/9600 | fp32 at240/258 | fp32 at2402/9600 |
|---|---|---|---|---|
| bimanual_step._clock | rod nominal counter projection | PASS | PASS | PASS |
| LiveGeometryReader._clock | actual SDK accumulation | PASS | PASS | REJECT |
| dual_evidence._sdk_boundary | actual SDK accumulation | PASS | PASS | REJECT |
| DualHandPhysicalEpisode._read_state_clock | actual SDK accumulation | PASS | PASS | REJECT |

The table retains both actual SDK timestamp and nominal rod timestamp.
A separate bimanual SDK-stamp diagnostic does not mislabel its production
input. Each rejection has `long_clock_consumer_incompatible:<function>`.
Table results, SHA, current consumer source bytes and the reviewed source
map are verified; re-sealing changed results cannot admit them. Existing
1e-9 and2e-7 readers remain unchanged. Current native history is short234;
actual long-mode precision remains unverified. Both policy gates therefore
have `clock_consumers_ready=false` and `native_ready=false`. Caller True
cannot upgrade the episode's readonly readiness property. New v2 reader
success never changes this conclusion.

Validation:15 new continuation checks +22 new consumer checks, and selected
mesh_reset_v2/clock/native_teacher_session/original_dual_episode regressions:
**231 PASS**. Targeted structured boundary evidence:3 PASS. Original
`Repository.verify_protection()`:528 protected files +2 checkpoints PASS.
No full-suite mechanical run. Commands, RED/GREEN logs, source identities,
compatibility tables and log SHA256 are in
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/long-reset-continuation-v2-20261115`.

Remaining: independent SPEC then QUALITY review of the exact implementation
commit; actual long-clock/source adapter resolution; V2-I01 explicit
worker/session/collector/checkpoint integration; separately approved native
budget that accounts for Close/Lift/full5s Hold and subsequent episodes.
W5 CPU continuation is not native readiness, source compatibility, grasp,
lifting, stability or performance proof.480/60/12/8,24D,30mm/5s/0.5mm remain
unchanged.
