# Complete Expert Collection and Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, Inline Execution as explicitly selected by the user. Steps use checkbox syntax for tracking.

**Goal:** Obtain successful physical close/hold/lift expert data, then train and validate hand-only teacher/student controllers.

**Architecture:** Keep the mounted scene and GPU rod. Add a versioned complete decision record at the existing control/evidence boundary; keep physical execution outcomes separate from reduced MPC predictions. Upgrade observations only after the data interface exists; final training requires successful complete episodes.

**Tech Stack:** Python, NumPy, PyTorch/CUDA, existing Isaac scene and pytest.

## Global Constraints

- Existing working branch `10-21`; preserve unrelated dirty changes and `1332`.
- Inline execution, one Isaac process, single-thread project configuration.
- No approach replay; restore a measured close state without success credit.
- Panda owns transport, hand actions are 12D. Original USD, physical rod and margin requirements unchanged.
- No commit/push or generated-evidence upload in this task.

## Task 1 — Remove the reproduced sampling false stop, resume the expert

Files: `pad_region_guard.py`, `tests/test_dh116_pad_region_refinement.py`.

- [x] Reconstruct V10's failure and compare 0.5/0.1/0.025 mm sampling at the same pose and patch. Fine bounds exceed 1.19 mm.
- [x] Add a recorded regression asserting `passed`, identical partition SHA and a smaller `patch_sampling_allowance_m`; observe failure.
- [x] Refine only ambiguous local-patch checks, preserving the original 6 mm / 30 degree patch, 1 mm margin and penetration checks. Record refinement count/allowance.
- [x] Run patch/front/region plus affected suites: 68 passed; final combined CUDA/control/data suites: 150 passed in12.29s.
- [x] Save V10 last valid step1030; SHA `3572d87e8e862601a0c84b6fbd2686bc8f5ef43fda482a25c13028090d617a9b`. Launch one V11 continuation, no approach replay.
- [x] Verify actual close: same V20 passed the original100ms preload gate at3104/time.3105s.
- [x] Verify actual full1s hold: same V20 transitioned to lift at13104/time1.3105s without phase credit or duration changes.
- [ ] Verify actual lift and whole-wire clearance. Entering lift does not prove a raised object.

## Task 2 — Complete time-aligned hand decision records

Files: new `contact_collection.py`, new `tests/test_dh116_contact_collection.py`, mounted runner.

Interface: `make_decision_record(observation, *, phase, previous_action, action, owner, prediction, control_dt_s)` produces schema `dh116-contact-decision-v2`. Required observation contains q/qd `(2,6)`, pad/goal points `(2,3,3)`, point Jacobians `(2,3,3,6)`, six measured loads/gaps/valid flags and load goals, rod position/velocity `(3,N)`, directors `(3,3,N-1)`, omega `(3,N-1)`, wire radius, clearance rows and hand-only derivatives. Record 12D actions and actual control owner. No Panda columns enter.

- [x] Write tests for required fields, dimensions, lossless privileged dynamics, no Panda actions, independent copies and actual follow-up state; verify RED.
- [x] Implement the record constructor and completion function binding pre-decision step/token to its measured post state; do not invent missing observations or successful outcomes.
- [x] Wire qd, previous contact validity and current rod arrays; record hold/lift explicitly, preserve legacy diagnostic samples. One CPU snapshot per decision; reuse measured post state instead of additional GPU scalar reads.
- [x] Portable/routing/replay tests pass; actual V20 complete replay now reads32close/100hold/4lift intervals, including the true.5ms close→hold boundary, with continuous state/time/token chain. Formal training still rejects the unfinished run; successful terminal lift remains pending.

## Task 3 — Feedback-rich observations and qualified distillation

Files: `contact_replay.py`, `contact_imitation.py`, `train_dh116_contact_imitation.py`, their tests and mounted inference adapter.

- [x] Test new replay retains qd, contact feedback, phase/goal and wire dynamics; rejects missing outcomes, broken interval chains, learned labels and missing hold/lift for final training.
- [x] Version v2 features/checkpoints: student267 with three observed frames and known-history mask; teacher486+19*N (1265 at41 rod nodes). Keep explicit v1 models, units, bounded residuals and parked joints. Future outcomes/privileged rod state do not enter student features. CLI defaults v2 and uses actual submitted targets, not unexecuted prediction suffixes.
- [ ] Train only after successful physical expert data exist. Report sample count by phase, source qualification and independent held-out errors.
- [ ] Run teacher and student separately with Panda transport in the complete scene. Require measured grasp, hold and lift; retain failures without relabeling them successful.

## Completion evidence

- [ ] CUDA rod tests plus actual full-scene rod/contact/reaction execution.
- [ ] Successful physical expert episode with complete close/hold/lift decision records.
- [ ] Saved teacher/student trained from those episodes, not reference-only or failed close data.
- [ ] Actual teacher/student Isaac acceptance with separated Panda/hand ownership.
- [ ] README, CHANGELOG and CODEX_RESUME reflect actual results and outstanding requirements; do not claim global tests pass while known failures remain.

## Measured continuation — tracking and joint-equilibrium correction

- [x] V11 terminal2815/lastvalid2814, coupled lower limit on right thumb. Its preceding completed action intervals show persistent .08–.15 mm radial point prediction error; causal offline joint RMSE .001077→.000152 rad with residual-bias calibration.
- [x] Add bounded causal tracking-bias observer and loaded command-equilibrium inset1mrad, retaining per-update slew and true joint bounds. CPU/CUDA plant fixtures verify learned bias, inactive joints, bounded correction and the loaded-limit case.
- [x] Preserve radial loading reference and optional bias in close initialization; derive calibration only from completed prior physical intervals. Changing Kp invalidates old bias, not loading progress; initial proportional torque is preserved.
- [x] V12 resumes V11 step2400 to avoid a boundary-start transient. Real loads reach .761/.926/1.066/1.153 N, then tracking exceeds1degree and stops at200; no success claimed.
- [x] Start V13 from V12 valid199 with simulation Kp10, original .15Nm cap and torque-preserving rebase. Latest confirmed53 valid close, no approach replay.
- [x] Formal complete replay requires whole_wire_clear, not only raised grasp stations; diagnostic observer behavior unchanged. Related159 tests pass in12.56s.
- [ ] Obtain actual full expert grasp/hold/lift; V13 step259 valid close, not hold/lift. Full regression840 passed/71 failed in58.54s, same failure set as previous811/71; no global passing claim.

## Contact-centre continuation

- [x] V13 terminal468: all four loads .950–1.213N, but unchanged small-pad clearance .9985mm rejected. No grasp success or successful-source training.
- [x] TDD offline align_contact_centres and close-initialization reframe. Preserve actual surface gaps, wire dynamics/qd/initial active drive deflection; no stage credit or live teleport.
- [x] Production candidate from V13 valid467: palm translations<1mm, rotations1.407°/.835°, gap change<.06µm; full50-body, six-region and nominal-initial checks pass. Original refs/caps/USD unchanged. Output runs/dh116_contact_alignment_20261021_v1.
- [x] TDD load-reference back-calculation once actual goal force is reached, preserving existing tests and physical limits. Do not keep winding up an unused depth target.
- [x] Start one V14 from the candidate. Latest1430 valid close, four loads .974/1.082/.948/1.042N, still zero simultaneous strict-valid frames; no hold/lift.
- [x] Focused171 passed in6.39s; full846 passed/71 failed in58.58s, identical failure set; compilation/diff pass.
- [x] Implement preload/target-hold handoff without deleting separation or collision criteria; actual physical stable-grip qualification remains pending below. V14 is terminal, not live.

## Preload handoff continuation

- [x] V14 reached valid close step2891 without simultaneous strict-valid pinch contact; four loads near target but changing MPC commands kept the hands moving. Saved already valid step1200 before cooperative stop; no approach replay or success credit.
- [x] TDD `PreloadTargetHold`: independently capture each hand at95% measured active-channel target with actual contact/known feedback, retain down to90%, release on contact/load loss. This is a command hold, not a grasp qualifier.
- [x] Constrain held coordinates in both optimization and its published command trajectory, without erasing physical Jacobians, load bias or state prediction. Teacher/student adapter applies the same measured handoff; complete raw records retain the held-hand mask in goal/diagnostics.
- [x] While held, use measured contact depth instead of integrating an unused force error. Original force/clearance/separation/slip/dwell gates are unchanged.
- [x] Related179 passed in13.49s; full854 passed/71 failed in59.33s, exactly the same failure set. Compilation/diff checks pass. No formal training, commit or push.
- [x] Launch single V15, session7591/PID142429, from V14 step1200 (`preload_initialization_step1200.json`, SHA f5fac382dd4ceb9fd08d42fcbf9f067eefe9b221fd8883a8a183c41473233ed6). Latest378 valid close, both hands latched, submitted targets unchanged; contact velocities still settling, no stable grip/hold/lift yet.
- [ ] Continue the same V15 physical observer through stable close, hold and lift. Do not restart merely for polling or treat target hold as measured grasp success.

## Correct the contact motion boundary and restored-wire screen

- [x] V15 stopped cooperatively at1412 after saving valid1000. Holding commands alone did not remove the separation artifact. In actual500–900 data, all seven right Panda coordinates were bitwise unchanged; thumb-origin displacement gives about.005mm/s, but reported qd through the verified Jacobian gives3.5mm/s. Gap rate was near0 while reported separation remained2.3mm/s.
- [x] TDD opt-in `mesh_motions(...,velocity_source='pose_delta',prediction_order=1)`: world-frame secant mesh-COM velocity and SO(3) angular velocity from actual consecutive poses. First sample retains reported twist until measured history exists. Label this first-order boundary as a simulation approximation; leave rod physics/contact parameters unchanged. Record both actual-pose and reported twists for diagnosis.
- [x] V16 terminated before physics: direct-close screen incorrectly tested default straight wire against saved loaded hand q. TDD `initial_wire_geometry` and use saved rod nodes/volume-corrected radius at this screen, without skipping the guard. Same actual saved1000: nominal fails.9435mm, restored passes1.2428mm. SourceUSD, regions and1mm margin unchanged.
- [x] Related182 passed in13.31s, including new boundary/initialization tests. Earlier full after boundary change856 passed/71 failed in37.53s, same failure set; final full after initialization change is running in `/tmp/dh116-regression-20261021-restored-wire.log` (session27012); recheck before reporting.
- [x] Start single V17 (session99650/PID143660) from saved V15 valid1000 SHA965fa18ce5e2b92ab139d8458dd4cc8d23ad22790dc15d7703dd57e3adb82ee6, with first-order pose_delta motion. At114 valid close,89 simultaneous strict-valid pinch frames; thumb separation.013/.036mm/s, four loads.930/1.066/.955/1.061N. No stable-grip, hold or lift qualification yet.
- [ ] Continue V17; preserve full v2 records and original dwell/slip/load/whole-wire-clear requirements, then train and separately validate teacher/student only after actual full success.

## Pose-derivative precision continuation

- [x] V17 had many strict-valid frames but isolated100us velocity spikes repeatedly reset dwell (687:644 valid frames, longest17.5ms). Saved valid1000 and stopped cooperatively; not a successful grip source.
- [x] TDD causal `PoseDeltaMotionWindow`, opt-in `--mesh-motion-window-steps 10`: current measured positions, derivative from up to1ms of contiguous actual pose history. No physical timestep, source guard or contact threshold changes. Initial no-history twist remains reported; reset on frozen/dynamic transition. Record actual estimator interval. A quantized.2mm/s fixture loses false2mm/s impulses, while genuine4mm/s remains4mm/s and above the unchanged1mm/s tolerance.
- [x] Related183 passed in15.12s; final full858 passed/71 failed in58.93s, identical failure set. Compilation/diff pass. No commit/push or official training.
- [x] Launch one V18(session1204/PID144225), from V17 valid1000 initialization SHA9f0d43a8392eff43cb91a190becc31a69600156b3f08f68b5be808a1639307aa. Same mounted scene, GPU rod100us/10us/16samples and original caps. At93 valid close,72 strict-valid frames continuously7.2ms, loads.939/1.058/.938/1.045N. No stable hold/lift yet.
- [ ] Follow the same V18 through actual close/hold/lift; do not add more speculative changes or restart for observation timeouts. Formal distillation still requires full actual expert success and whole-wire clearance.

## Existing preload requirement, without an extra capture gate

- [x] V18 at503 had actual.934/1.036/.932/1.041N but both command holds remained off because the added95% capture threshold exceeded the existing physical90% requirement. Save valid500, stop cooperatively and retain evidence.
- [x] TDD regression at.93N; remove the added threshold and use the original90% requirement for command pause/resume. This does not weaken GraspEvidenceGate's physical qualification.
- [x] Final affected184 passed in13.54s; compilation/diff pass. Last full858/71 predates this isolated condition change and is not presented as a fresh global pass.
- [x] Start V19(session70616/PID144617) from V18 valid500 initialization SHAdd3062016e8c597a1343593e714d3bd1331e44c1339f04879671f84e26b44178. At270 passed close, both hands held, continuous strict-valid23.6ms, loads.911/1.010/.902/1.014N.
- [ ] Continue the same V19 physical run. No stable hold/lift or successful expert dataset yet; do not start formal distillation or relabel preloaded continuations as a general from-open grasp controller.

Latest actual V19 check:607 passed close,57.3ms continuous strict-valid pinch contact, both TouchFirst hands reached oppose, loads.912/1.017/.911/1.027N, both command holds active,6 complete physical control intervals. PID144617/session70616 still active; recheck before acting. This is stage progress, not final grip/hold/lift success.

## Approved MPC target synchronization — current V20

- [x] User approved synchronizing TouchFirst to actual MPC load goals instead of waiting on its unused nominal .05→1N ramp. Preserve measured touch/opposition, physical100ms preload, hold1s, slip and lift criteria.
- [x] TDD optional controller targets in TouchFirst and TabletopGraspSequence; mounted MPC runner passes configured goals. Targets alone never establish contact/adequate load; old nonMPC ramp remains. Related195 passed in14.26s; compile/diff pass. No global passing claim.
- [x] Save V19 valid1752 physical state to `runs/dh116_contact_mpc_close_gpu_20261021_v19/mpc_preload_sync_initialization.json`, SHA `cd24c2201270572c21f39c650f589084f075069cf077a151d450a635636ec303`. V19 cooperatively stopped, terminal summary operator-stop, no success.
- [x] Start single V20(session15643/PID145546) from that state, same assets/GPU rod/physical settings, no approach replay or phase credit. Actual first step passed close, four strict-valid contacts, loads.905/1.017/.906/1.024N, `preload_target_source=controller` and targets1N. This supersedes V19 LIVE notes above; always recheck runtime.
- [ ] Follow the same V20 through measured close/hold/lift, preserve full v2 intervals; formal expert training and actual teacher/student validation remain pending successful whole-wire-clear episode.

Latest V20 physical check:1982 passed close, both TouchFirst phases preload/ready with controller targets, four strict-valid pinch contacts. Loads.889/1.009/.923/1.028N; left command hold released on the original.9N requirement and MPC regulation resumed, right held. Earlier left-thumb force dips reset the physical100ms preload dwell; no hold/lift yet,19 complete v2 intervals. PID145546/session15643 remains live. Do not restart or weaken criteria; continue observing actual regulation. The same related suite was freshly rerun after launch:195 passed in11.32s, compile/diff pass.

## Actual close qualification and conditional local-alignment comparison

- [x] Same V20 recovered preload and passed original100ms physical close gate at3104/time.3105s, enteringhold without code/threshold changes. Latest4117 validhold, elapsed.1013/1s, materialslip.0219mm. PID145546/session15643 live; always recheck. No complete hold/lift success yet.
- [x] Verify actual fullv2 collection contains32close plus7hold measured intervals; hold ownerhand_mpc_hold and12D hand commands, not fictitious newMPC actions. The close→hold boundary interval is0.5ms, not10ms; preserve its actual duration.
- [x] Save validclose3103 for necessary later comparison: `pre_hold_initialization_step3103.json`, SHA `7ec4c4fd47c128b99af65b0b2568df0fafb00238dedb2dc1d3cd1d4181d26b9c`. Does not import stage qualification or stopV20.
- [x] User approved conditional local alignment≤3mm per target, same pinch cross-section preferred, same physical gates. Record design in`docs/superpowers/specs/2026-10-21-dh116-local-pinch-alignment-design.md`; no commit/push.
- [x] Read-only reducedMPC/staticUSD comparisons using realdecision3100: unaligned16posespass; full/half/quarter/tenth aligned goals fail left-index region at later predicted poses. Twentieth goals(.111/.072mm)16posespass but actualmesh axial alignment improvesleft/worsensright, so NOT a proven improvedcandidate and NOT deployed. Logs inV20/local_pinch_probe_*.jsonl.
- [ ] Continue sameV20 through fullhold, Pandatransport, actuallift/wholewireclear. If it actually fails and local alignment is needed, first review/implement the geometric joint-pose solve; do not simply turn onopposed_pinch or move reference labels.
- [ ] Obtain qualified complete expert data, then formal teacher/student training and actual separate Isaac validation. No new runtime source changes this turn; do not repeat195unchanged tests merely for observation.

## Same-instance physical hold and actual replay boundary verification

- [x] Same PID145546/session15643 confirmed live at7257; validhold elapsed.4153/1s, slip.0614mm, all four contacts valid. No restart, no additional pose/force/gate changes, no formal training.
- [x] Run actual `load_complete_contact_run(V20,require_success=False)` on the live completed prefix:73 intervals(32close/41hold),72 actual10ms plus1 actual.5ms phase-boundary interval. Time/token/state chain accepted. Default formal loader correctly rejects this unfinished run without grasp/lift/whole-wire success. No synthetic data or new models generated.
- [ ] Continue this same run to fullhold and actuallift. Do not interpret waiting on a live process as a blocker or rerun tests when source is unchanged.

## Actual full hold passed, lift started — same V20

- [x] Follow the same live PID/session through full1s measured hold. First lift row13104/time1.3105s, originalhold since3104/time.3105s; no reset, gate changes or imported credit. Originalfour pinch contacts remain valid, materialslip.1077mm at entry.
- [x] Verify actual routing once lift starts: arm_owner=panda, hand_owner=hand_mpc_hold,12D hand targets unchanged. Panda-only transport increments accumulate into its seven-coordinate targets; not a hand planner controlling Panda.
- [x] Actual complete replay reads136 intervals(32close/100hold/4lift), all intervalpassedTrue and state/token/time chain accepted. Source remainspartial_action_diagnostic, not a successful final expert.
- [x] Continue same V20 until its actual outcome, not an observation timeout. Terminal15805/lift.2701s: rightthumb separation1.0943mm/s invalid under original1mm/s gate, load.9295N/gap-.2973mm retained. Slip.2437mm<2mm, stationrise~.0022/.0016mm. close/holdPASS,liftFAIL. PID145546/session15643 finished; no newIsaac.
- [ ] After actual full expert success, qualify all phases/whole-wireclear, trainteacher/student, and separately validate each actual Isaac controller. No new runtime source changes or repeated tests in this waiting turn.

## Terminal V20 diagnosis — do not replay unchanged configuration

- [x] Verify actual summary and missingPID/session completion. No exception: sequence abort is a real failed criterion; script exited0 withsummary.passedFalse, not a successful trial.
- [x] Load actual failed complete replay:160 intervals(32close/100hold/28lift), terminalfailed interval.0001s preserved; default successful-expert loader rejects it. No training or successful relabeling.
- [x] Check Panda transport implementation accumulates7D targets and actual FK targets rise, so it is not an absent/lost command. Atlift.2322s targetpalmrise~.660mm, measuredpalmrise−.297/−.207mm. Entry targetFK lies.979/1.194mm below actualpalms; measured reportedvelocity discrepancy persists, extra PhysXcontact forces>1mN not reported.
- [x] Independently check actualq FK against measuredpalm atentry/terminal: errors<.0004mm and.00003deg. No coordinate-display explanation found. WorldPhysxCfg extra stabilization defaultsFalse and runner doesnot override; do not change solver settings under an unverified stabilization hypothesis.
- [ ] User chooses next design: recommended real-palm pose feedback for independent Panda transport, versus local≤3mm contact/pose alignment. Keeporiginal caps/timestep/gates; do not restart before an implemented/verified change or actual directed experiment.
- [ ] After design approval, write/execute focused TDD implementation and verify realgrasp/hold/lift withcomplete expertcollection. Overall goal remainsactive, formalteacher/student stillpending.

## 2026-10-22 — Local alignment continued; old V21 passed close

- [x] Execute the user's selected local≤3mm alignment without a new Panda controller. Static hand-only candidate and initialization details remain in the separate local-pinch plan; no physics or qualification thresholds changed.
- [x] Original V21 actually entershold at14700/time1.4701s on the same old loaded runtime. Latest15728 validhold elapsed.1028/1s, materialslip.0031mm. Do not label1s hold/lift complete yet.
- [x] Actual complete replay153 intervals (148close/5hold) preserves its.1ms phase boundary and validates state/token/time chain. It remains diagnostic-only until terminal grasp/lift/whole-wire success.
- [x] Save last valid close14699 with full physical state, without phase credit or stopping V21. Preserve all older backups.
- [x] Repair the existing preload contract's missed-physics-sample implementation separately: same original90% load/contact loss predicate now consumes measured interval extrema, with raw source metadata. No new grasp goal, force cap or Panda architecture. Baseline8passed; RED8new/19old, focused27GREEN, affected138passed in7.52s, compile/diffPASS. See `2026-10-22-dh116-preload-window-feedback.md`.
- [x] Continue the original V21 through the full1s hold: actual first lift24700/time2.4701s, exactly1s after hold14700/time1.4701s, no reset or phase credit. At24792 liftelapsed.0092s, all4 contacts valid, slip.0365mm. Actual ownershipPanda/hand_mpc_hold and12D hand targets. Complete replay249intervals(148close/100hold/1lift) validates time/token/state chain; not a terminal successful source.
- [ ] Continue the same original V21 through actual Pandatransport/lift and whole-wire clearance; it has NOT loaded that offline patch (old manifest has no feedback tag and old data no window field). Do not interrupt current qualified progress to load it. If the original run actually fails and a tested reload is needed, use a saved realclose state, one instance, original gates and complete collection.
- [ ] Qualify a complete successful physical expert, then train teacher/student and separately validate each in Isaac. Offline tests and partial phase success do not complete the goal.

## 2026-10-22 — V21 terminal and isolated TGS integration reproduction

- [x] Confirm V21 terminal27063/time2.7064s, liftelapsed.2363s; rightthumb separation1.215433mm/s exceeds original1mm/s despite.954386N load, gap−.307029mm and slip.185191mm. close/holdPASS,liftFAIL. PID missing/session21797exit0. No successful training source.
- [x] Actual terminal replay272intervals(148close/100hold/24lift), final6.3ms interval retains failure. Time/token/state chain validates; default formal loader rejects it.
- [x] Trace actual24700→26826: target commands change, four actual Panda joints per side remain bitwise unchanged; measured palms descend.265390/.095665mm. Arithmetic at100µs/16 float32 substeps predicts exactly those frozen axes; this alone is not a physical root-cause proof.
- [x] Run isolated CUDA Isaac joint, no gravity/rod/payload, same100µs/velocity.005rad/s/1000steps.16position iterations: actualdelta0 with reportedqd~.005;1position iteration: actualdelta.000476837rad vs expected.0005. Both sessions terminal (92294exit0 after completed-data shutdown,67335exit0), no concurrent Isaac. Evidence/source inV21/tgs_joint_integration_{16,1}.json andprobe_tgs_joint_integration.py;16original source separately pinned. This proves an integration freeze mechanism, not whole-robot convergence or grasp success.
- [x] Run V22 bounded500step wiring baseline from actual prehold14699 (`94426ef00b6a5cfb01ce1f4c0ed1cdfca29fda2976e52b1ff43b39220305dea7`) with current window feedback and original16position iterations. PID167123/session95002 natural terminalexit0. All500rowchecks pass;5completeclose intervals/4past100frame windows validate, no phase qualification. SummaryFalse/closeFAIL means intentionally incomplete50ms, not a new contact abort or successful source.
- [x] After V22 terminal, start one full-budget V23, PID167430/session76710. Actual manifests differ only in steps500→200000 and position iterations16→1; unchanged originalUSD/material/effort/speeds/dt/16contact subsamples and all physical gates. Source/initialization identical; no new Panda feedback/cache/rod/gains.
- [x] Equal actual first400frames both pass original rowchecks; V22 all14Panda positions unchanged, V23 all14have nonzero differences. Evidence/recipe: V23/first400_integration_comparison.json andcompare_initial400.py. This is real mounted integration response, not full lift acceptance. Actual V23decision300 current thumb loads.901272/.902873N, past minima.893021/.892667N and bothunheld;400leftcurrent.910818/past.899197/unheld. The repaired window is executed causally and prevents false capture despite recovered snapshots; do not call these previously unheld samples a latched→released transition.
- [x] Same V23 actually passes original100ms stable close at2025/time.2026s and entershold. Latest7073validhold, elapsed.5048/1s, slip.2256mm, all4contacts valid. New actual71completeintervals(21close/50hold) validate through outcome7026/time.7026s; original2.6ms boundary and hand_mpc_hold owner12D retained. Still diagnostic/source not terminal, not full expert success. Same PID167430/session76710; no restart or repeated unchanged tests.
- [x] Save actual lastclose2024 full physical state, SHAfb520d675454f8ea4239b669e04a48d3edb5878de55f7d25bc662e9d7516c365, helper20874exit0/restore accepted. No phase credit, no main stop/newIsaac; preserve14699 and older backups.
- [x] Same V23 passes full original1s hold: firstlift12025/time1.2026s, exactly1s after firsthold2025/.2026s. No restart, phasecredit or gate changes. Actual122intervals(21close/100hold/1lift) time/token/state chain validates. Panda owns transport, hand_mpc_hold keeps12D; entering lift is not a raised object.
- [x] Attempted optional hold12024 initialization export is rejected by existingclose-only helper (session54946exit1); no file, phase rewrite or helper change. Keep actual2024close backup and full rawtrajectory. This optional unsupported export does not block current simulation or justify a restart.
- [x] Read-only actual lift motion/reaction comparison, V23/lift_tracking_probe_v1.{py,json}, exit0. At.2126s palms rise.0994/.1321mm (profileprogress.5645mm is not a targetFK certificate); wire still ontable. Actual hand→rod verticalforce−.52685N. Sign traced fromGPU body_force through impulse bridge to robot application. This is local evidence, not final unique cause or grasp qualification; no control/force/physics change.
- [x] Follow V23 to its actual terminal16922: original source-convex guard rejects right non-pad finger13 at .999886mm<1mm; all6pad-region checks pass. Reaction was published/applied and PhysX advanced before postphysics rejection; missing postguard/material fields denote an incomplete failed-step record, not an unexecuted physics step. Last fully validated16921 retains all4contacts/slip1.413684mm<2mm, but stationrise−.003094/+.003461mm, no lift. close/holdPASS,liftFAIL; process/session ended, no identical relaunch.
- [x] Actual terminal complete replay validates170intervals(21close/100hold/49lift), lastterminal outcome16922failed. Default successful-source loader rejects it; no formal training. Wrapper numeric1/core0 evidence predates SimulationApp's env overwrite and is not a live BLAS-pool introspection.
- [x] Independently reconstruct actual full18D failed-step q from original USD. Original face-envelope bound1.000032mm differs from actual recorded .999886mm at submicron FK precision; a support-projection witness gives1.147995mm with the same78.700µm motion envelope and8.000052mm radius. This is offline source-convex diagnostic evidence of a conservative edge/corner bound, not bitwise measured-pose acceptance or runtime success. Report/source: V23/source_clearance_witness_probe_v1.{py,json}. Original runtime algorithm and1mm requirement remain unchanged.
- [x] Complete the bounded Panda-fixed screening using V23 validclose2024; details in the local-pinch plan/V3 reports. Cumulative3mm,10µm gap, all source/pad checks and parked fingers preserved. Static viable aligned candidates still lack upward normal support; no new physical initialization or successful source. This completes the finite screen, not grasp/lift or a proof of global impossibility.
- [ ] Obtain successful whole-wire-clear expert data, formal teacher/student fits, then separate real Isaac policy acceptance. Goal remains active.

## 2026-10-22 — Recorded-state support and CPU/CUDA contact-force parity

- [x] With no Isaac running, restore the exact pinned CUDA physical model/mesh identity and use actual valid rod snapshots12025/14151/16921 plus their recorded GPU boundary end-mesh poses. Evaluate instantaneous forces only; no time advance, changed robot command, physical parameter or expert label.
- [x] Explicitly distinguish recorded boundary twist from two counterfactuals: zero relative velocities, and stationary rod/hands-up5mm/s. At entry, weight .9862088N, measured-twist normalz−.5369901N/frictionz+.0004269N; instantaneous total differs from original integrated frame by.0000191N. In the5mm/s counterfactual, friction+.0979976N but total handz−.4370229N. Current geometry remains pressing downward; no claim this alone explains every terminal guard failure.
- [x] Same GPU closest-distance/normal inputs into the original CPU `_calculate_contact_forces_rod_mesh`:54channel cases agree on nodal forces, rod torques, mesh reaction force/material-frame torque. Max force2.22045e−16N/torque3.46945e−18Nm. This verifies the shared viscous-regularized Coulomb force/distribution, not independent distance algorithms or an integrated successful trajectory. No GPU friction omission found at these inputs.
- [x] Diagnostic scripts runexit0; py_compile/diffPASS. Reports inV23 `contact_support_audit_v1.json` and `contact_support_audit_cpu_parity_v1.json`; preserve originals. Production backend remains unchanged and all successful-expert gates remain unmet.
- [ ] User choice remains required before relaxing the Panda-fixed local-pose design to allow small palm/arm orientation adjustment. Do not take automatic goal continuation as approval. No identical failed configuration relaunch, changed friction, middle hook, fabricated success or formal fit while awaiting that choice.

Blocked audit: the same missing palm/arm-posture design choice has persisted through the local-screen turn, force-parity continuation and current revalidation (three consecutive goal turns). The preceding turn made diagnostic progress; it was not a live-process wait. Current authoritative summary remainsclosePASS/holdPASS/liftFAIL, no verifier/audit process is running and no user approval has arrived. Safe fixed-Panda screening and force-law comparisons are complete; successful-source training remains invalid. Mark the goalblocked pending that decision, notcomplete; do not keep producing identical trials or audits solely to keep it active.
