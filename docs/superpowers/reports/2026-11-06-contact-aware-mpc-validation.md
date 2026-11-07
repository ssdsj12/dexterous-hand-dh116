# Contact-aware Teacher MPC and separate pinch/tripod validation

## Scope and evidence rules

This work starts from `b63501e8619f777afedc995a7cf88b6524ef0adc` on `codex/dh116-lift-repair-20261106`. The PR remains Draft. All experiments use independent outputs under `/mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z` and a frozen source snapshot. One Isaac process runs at a time under `/tmp/dh116-native-single-instance-xk.lock`.

The 3844D sensor Actor, 371D privileged Critic, 24D command contract, Tanh Gaussian PPO, default v2, original physical properties/actuator limits and native net-30-mm / continuous-5-s / 0.5-mm drift criterion are unchanged. Contact force, depth and Jacobians are simulated expert-only information. No new real-hardware feedback is claimed and no long PPO is launched.

## Measured failure mechanisms

The previous local-pad MPC improved its own aggregate same-state geometric objective by 15.795%, yet pinch native success fell from 4/4 to 0/4. Its four environments accumulated absolute command offsets of 0.013664 / 0.012370 / 0.012474 / 0.016595 rad. At loss of opposition, replaying zero Teacher finger velocity retained the shifted command instead of restoring the successful reference. Integration of the issued action differences explains the measured offsets to less than 7.1e-15 rad. The joint prediction error gate did not detect this contact failure.

Tripod failed earlier, before the old 3-mm activation threshold. Teacher actions and pre-terminal commands agree with the historical AK source, while actual contact responses diverge after contact. At control tick 400, current left thumb/middle loads were approximately 9.99/9.03 N, compared with historical 0.97/0.39 N. Two terminal thumb loads reached 33.697/34.572 N on `finger14_link`; the other two cases included approximately 18 N of rejected contact. The unchanged failure guard is a **single rigid-link object-filtered normal load above 30 N**, not the legacy load-profile target of 12 N. Missing raw per-link terminal data prevents uniquely identifying the terminating link in those two cases.

Initial pose, action order, drive, mass, inertia and material readbacks do not explain a gross configuration mismatch. The exact cause of the first contact-response divergence is unresolved. Contact numerical sensitivity/layout is a hypothesis, not a proven root cause. The actionable problem is that time-indexed action replay keeps compressing despite the current trajectory's excessive load.

External read-only audits, retained outside Git:

- `absolute_reference_command_audit.json`: reconstruction matches all eight baseline environments' non-reset native commands exactly; canonical integration uses eight 480-Hz substeps per control period.
- `tripod_failure_diagnosis_v2.json`, SHA256 `d6a493f8f36b9ec5fa3f57be74d5f3818e08ac7e4ff95675bf536d175c5c549a`.
- `tripod_pressure_offline_design_review_v1.json`, SHA256 `f6854fff1eff30481816d6cc69c61ac1a481e02cc4649d06cc2b79f07e7f4f79`.

## Separated controller experiments

`local_pad` remains the default existing Teacher MPC. Explicit `absolute_reference` reconstructs the cold-reset Teacher command sequence, uses future time-indexed actions in the four-step horizon, limits total geometric deviation to 0.001 rad and recovers the absolute target when opposition is lost. The canonical environment still applies its measured-state lead and physical guards; the offline reference does not pretend to reconstruct that state-dependent lead guard.

The pressure-target helper converts actual force-on-hand vectors into inward directions with the correct negative sign. It requests bounded geometric normal motion from measured force error (0.0004 m/(N s), at most 0.001 m/s), with the historical expert's 0.6-mm depth retreat. These are controller requests, not changes to simulation contact parameters or fabricated measured forces. Missing/invalid contacts have zero weight; nonfinite measurements remain explicitly invalid. It predicts local kinematics, not force, friction or object dynamics.

The historical AK pressure goal was constant at `[1.5, 0.75, 0.75, 0, 0] N` per hand. An additional absolute-reference box of ±0.001 rad can oppose unloading: the actual tick-400 snapshot predicts the right middle finger moving inward under that box despite an outward goal. Therefore pressure acquisition must be evaluated separately from the small absolute-reference residual experiment, retaining the original physical limits and actual-command rate limits.

`pressure_reference` leaves pinch on the absolute-reference route and enables tripod planning from the first actually bearing pad. It uses the current measured contact point and pressure instead of accumulating the historical compressive displacement. Bearing digits hold their nominal target while the local optimizer requests pressure correction; missing digits retain the source acquisition velocity. It removes the experimental absolute-reference box for tripod, while admitted plans retain original joint bounds, measured `q±0.1 rad` and `0.005 rad/control-step` slew. Rejected-contact hands request retreat from known bearing normals. Unknown normals or invalid/rejected plans hold the affected finger commands rather than restoring the old overcompressed absolute target. The actual environment's original guards remain authoritative for fallbacks.

The admission check rejects predicted inward motion at overloaded pads relative to commanded common wrist transport. All twelve wrist channels still equal Teacher actions. This local sign check does **not** predict object transport, physical pressure reduction or grasp success. Unknown non-pad normals cannot be guaranteed to unload by this model.

Independent review identified a roundoff bug before the first absolute-reference pilot executed: eight native 480-Hz additions could differ from a single 60-Hz addition by about 8.88e-16 rad and cause a false empty trust intersection. The queued job was cancelled before Isaac launch; its snapshot is retained. The corrected solver collapses only roundoff-sized trust gaps inside the unchanged physical/rate interval. Real infeasibility still rejects the plan. Regression cases at initial 1.0 and 1.1 rad now activate correctly. A second review fixed an unknown-hand hold box that could override the modeled lead interval: it now intersects the lead bounds and rejects an impossible hold.

Saved-native-snapshot CPU replay covers ticks 32/100/200/350/400 for all four tripod environments: 16/20 rows activate; the four tick-32 rows have no bearing contact. At tick400/env4, predicted first-step relative normal motion is outward by 7.121/12.533 µm at left thumb/middle and 3.103 µm at right middle; all recorded wrist channels and admitted lead/rate checks pass. This is offline replay of real states, not 20 new Isaac trials or measured force reduction. The external `pressure_adapter_saved_native_snapshots_v1.json` binds the five implementation hashes.

### Executed absolute-reference MPC result

`absolute_reference_pilot_v2` completed 1175 controls in 1258.017 wall seconds, exit0, with 1175 real RGB-D frames and 3359 active MPC environment-steps (3355 changed labels). All eight first episodes terminated. The actual cold reset equals the previous reference run exactly. Independent frame/action/snapshot checks and replay of the unchanged physical success observer passed.

| Grasp | Replayed environments | Lift P50 / P90 / P95 / max (mm) | 30-mm rate | 5-s hold rate | Native successes |
|---|---:|---|---:|---:|---:|
| Pinch | 4 | 31.5645 / 31.6168 / 31.6223 / 31.6277 | 100% | 100% | 4/4 |
| Tripod | 4 | 0.11637 / 0.13876 / 0.13935 / 0.13994 | 0% | 0% | 0/4 |

Pinch ends at controls1174/1175/1175/1174 with success, failure0 and no recovery; its complete synchronized labels qualify for the success pool. This restores the four baseline replay successes lost by the old local MPC (0/4, maximum12.489 mm). It is a measured improvement over that failed MPC in this fixed initial-state experiment, not a 100-case generalization result or a Student success. The original reference alone also achieved4/4.

Tripod remains unchanged because it fails before the preserved 3-mm activation gate: controls419/418/420/419, `excess_force`, all unqualified. The pressure-profile experiment ran separately after this run released the Isaac lock.

Audit: `absolute_reference_pilot_v2_independent_audit.json`, SHA256 `824e9d9b57ae5cb5627306da4ed4f8e4ea452f6f980fdcb742d9369a70e99e21`. It also records contact acquisition/loss, conditional rod-marker drift, actual target/reference bounds and missing torque data. Whole-episode pre-grasp depth/load peaks are not confused with the final successful hold window. Commands:

```bash
/mnt/mydisk/xk/envs/dh116/bin/python /tmp/audit_teacher_mpc_contact_v2.py \
  --root /mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z \
  --mode absolute_reference_pilot_v2 \
  --output /mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z/absolute_reference_pilot_v2_independent_audit.json
```

### Executed pressure-reference MPC result

`pressure_reference_pilot` completed1175 controls in1462.712 wall seconds, exit0, with all eight first episodes complete. Its independent frame, state, issued-action, pressure-target, model-direction and original-bound audit passed. The native success observer replay agrees with all terminal records.

| Grasp | Environments | Lift P50 / P90 / P95 / max (mm) | 30-mm rate | 5-s hold rate | Native successes |
|---|---:|---|---:|---:|---:|
| Pinch | 4 | 31.5645 / 31.6168 / 31.6223 / 31.6277 | 100% | 100% | 4/4 |
| Tripod | 4 | 0.41835 / 0.97267 / 1.02400 / 1.07533 | 0% | 0% | 0/4 |

All four tripod episodes now pass the previous approximately419-control overload termination point, but eventually drop the object: controls684/938/783/864, failure7=`object_dropped`, no recovery. Thus the change did **not** produce successful tripod grasping. Opposed-contact time fractions are only5.71/12.91/7.29/5.21%; contact acquisition/retention remains inadequate. The preserved wrist reference is still time-indexed rather than a validated physical-readiness phase controller.

Tripod safe-prefix maximum pad/total-hand load decreases from19.779/33.894 N in the old reference to15.892/25.645 N, and maximum effective depth from2.067 to1.327 mm. These safe-prefix summaries exclude the final failed control period to avoid mixing reset data; the audit separately retains terminal extrema. They must not be compared as if they included the old33–35 N terminating thumb samples or represented measured hardware forces. Peak rejected-hand load remains7.171 N. Lower load and higher1.075-mm peak are insufficient evidence of a stable grip.

No tripod label enters the successful supervision pool. Both new MPC profiles retain four qualified pinch episodes; the next Student candidate uses the independently verified absolute-reference pool, not failed pressure trajectories.

Audit: `pressure_reference_pilot_independent_audit.json`, SHA256 `987caeeb4f8291cb8aa76d34e37861dd976c7c8f79cbd9585678468602d9953a`. Run the same external audit command with `--mode pressure_reference_pilot`. These are four fixed-source replay environments per style, not100 independent trials.

## Full Student evaluation protocol

The fixed Student is `visual_actor_pointnet_reset_bc100_v2_warm_start_v3.pt`, SHA256 `949a014921ffea83ea3fe57fb53550042a748f8b9decffe72eff69c11c6602d3`. It was fitted only on one successful pinch source, with 4698 frames from four replays and no heldout trajectory. There are no qualified tripod training demonstrations in that fit. Its 100 offline BC Adam steps are not 100 PPO updates.

The existing production evaluator loads only this actor and projects the inference batch from 8 to 200. It uses the same frozen `cold_test_100_per_style.json` as the prior actual-129/1439 paired tests, 100 unique initial identities per grasp, deterministic inference, original 224×96 cameras, micro512/infer256 and a 2400-control-step cap. Teacher/MPC is disabled during Student inference. This is a paired regression set already used in earlier evaluations, not a newly unseen selection set.

The first attempt also enabled detailed raw contact-manifold tracing for all 200 environments. Actual measured overhead was approximately ten wall seconds per control period. It was interrupted with SIGINT, its data and interruption record preserved, and none of its incomplete trials counted. A fresh identical evaluation retains the normal pre-terminal conditional telemetry without the expensive optional manifold log. Detailed contact snapshots remain available in the small Teacher/MPC experiments.

An independent audit replays conditional raw samples, verifies unique first terminals and physical identities, and reports each grasp's Lift P50/P90/P95/max, 30-mm attainment, hold, contact loss, opposed-contact time fraction, simulated torque saturation and native success. Slip without an established anchor is **not applicable**, never silently zero. Native execution, telemetry completeness and successful grasping are separate outcomes.

### Executed BC100 Student result

The pure Student run completed 1800 control steps in 2400.819 wall seconds, exit0. All 200 first episodes ended, with zero invalid measurement samples. The independent audit replayed **1,503,649 native physical samples**, verified every case identity, report/evidence SHA and the native execution certificate, and reproduced the statistics below. These are the actual BC100 model's results, not historical peak policies.

| Grasp | Episodes | Lift P50 / P90 / P95 / max (mm) | 30-mm rate | Maximum qualified hold | Opposed contact time | Native success |
|---|---:|---|---:|---:|---:|---:|
| Pinch | 100 | 0.00004545 / 0.00005899 / 0.00005929 / 0.00005941 | 0% | 0 s | 0% | 0/100 |
| Tripod | 100 | 0.00003384 / 0.00005863 / 0.00005950 / 0.00008371 | 0% | 0 s | 0% | 0/100 |

The very small height values are numerical settling, not meaningful lifting. Neither group established bilateral opposed contact; contact-loss counts are zero because contact never established. Slip is not applicable (`no_anchor`) throughout, not measured zero slip. Simulated torque saturation was 0% for pinch and 0.000006418% for tripod; this is an implicit-drive estimate, not measured motor torque.

The old unconditional report flags remain `legacy_telemetry_complete=false` / `native_validation_complete=false` because no slip anchor exists. The explicit conditional telemetry is complete and the certificate attests native execution, not grasp success. The independent audit verifies this distinction rather than rewriting the legacy flags.

Audit: `student_bc100_complete_independent_audit.json`, SHA256 `6cd7f5863fc9ee2b18ac3527e0c34415ef7dab5e6d323f7deff9b18b2c803473`. Its command is:

```bash
/mnt/mydisk/xk/envs/dh116/bin/python /tmp/dh116_native_student_audit.py \
  --run /mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z/student_bc100_complete_100_per_style \
  --output /mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z/student_bc100_complete_independent_audit.json
```

This result demonstrates that the short fit and visually active PointNet have not produced a working grasp policy. The evaluation set is retained as a paired regression set; no model has been selected by its historical maximum height.

### Verified Student input and fitting failures

The actual issued initial actions agree with CPU inference to 1.49e-7 or better, so the observed failure is not explained by accidentally running another actor. All 100 pinch initial left-wrist quaternions are the negative of the Teacher's initial quaternion (`dot=-1`), representing the same physical orientation. The old Actor consumes those raw components. Changing only that quaternion's sign in recorded inputs changes action by as much as 0.22483 and reduces the native-input versus source-input prediction discrepancy from approximately 0.225 to 0.000250. Replacing the point cloud/mask alone changes action by only 1.12e-6. This is a verified representation problem, independent of the model's fitting error.

The old BC100 fit is also insufficient: its source MSE 0.00197447 improves only 6.11% over a per-channel constant-mean baseline of 0.00210292. Startup frames have MSE 0.0175984, about 8.9 times the whole-source average. Of the 16 startup rows at ticks0..3, five were never drawn in the actual 6400 seeded sampling draws. Even the Teacher's own initial observation produces markedly incorrect predicted finger and rotation actions. The actual Student executes persistent incorrect nonzero actions; it is not merely outputting zero.

The explicit opt-in `wrist_quaternion_hemisphere_v1` Actor adapter chooses a unique sign by making the largest-magnitude quaternion component positive (fixed first-index tie break), separately for both hands and all four frames. It changes no position, encoder, point, mask, previous-action or timing feature and adds no parameter or input dimension. Raw observations remain preserved. Historical contracts omit the field and retain `raw`; new models carry `actor_wrist_quaternion_preprocessing` and a distinct BC v3 artifact/derivation version. Inference, PPO construction and static/exact resume check the mode. Old models are not silently normalized.

The independent source/input/action audits are retained as `bc100_inputs_and_source_fit_readonly_v1.json`, `bc100_native_action_trend_v1.json` and `student_bc100_phase_observability_readonly_v1.json`. A source phase-boundary label ambiguity was observed near lift stopping, but no initial idle/start contradiction was established. The new representation and a longer controlled BC fit require their own evaluation; they are not evidence that grasping has been repaired.

### Controlled offline fitting and startup coverage

Two 1000-step fits use identical source weights, encoder initialization, 4698 successful Teacher pinch rows, batch64, LR1e-4 and seed42. Raw and canonical fits produce byte-identical Actor/BC-Adam/sampling-RNG state because this training source already uses the canonical sign. The canonical adapter instead protects inference when a physically equivalent quaternion is returned with the other sign.

Increasing uniform training to1000 still underfits startup. An explicit canonical-BC-only option now samples a mixture of uniform training rows and the first four control frames: `p_i=(1-f)/N+f/K` for startup rows and `(1-f)/N` otherwise. With `f=.25`, all rows retain positive probability and no heldout row enters training. The default `f=0` preserves the exact historical `torch.randint` path. Weighted sampling uses a private CPU Generator and stores its actual distribution, SHA, algorithm, counts and RNG; derivation rechecks the original immutable dataset and split.

| Offline artifact | All-source MSE | Startup MSE | Startup MAE | Startup maximum action error |
|---|---:|---:|---:|---:|
| raw1000 | 0.000231797 | 0.00449366 | 0.0470969 | 0.229928 |
| canonical1000 | 0.000231797 | 0.00449366 | 0.0470969 | 0.229928 |
| canonical1000_startup25 | 0.000240999 | 0.0000122437 | 0.00215487 | 0.0164650 |

Startup25 is not an unconditional improvement of the whole loss: whole-source MSE increases3.97% and phase-transition errors remain. Its source-t0 maximum error is0.00259670; on the actual100 pinch initial observations the maximum error against the source-t0 target is0.00273628. Both canonical models are exactly invariant across eight independent hand/frame sign-flip interventions on all200 recorded native initial inputs; raw1000 changes by up to0.162462.

Independent audits verify132 protected-state/sampling checks. Startup probability is25.25542784%, with16162 of64000 reproduced draws at startup (25.253125%); every startup row and all4698 source rows were sampled. Critic, latent noise, PPO Adam/LR/counters/RNG and source files remain byte-identical. Only the explicitly trained Actor and, for canonical warm starts, the known versioned adapter-contract field change. These three fits still contain one pinch source, no tripod labels and no independent holdout; they are offline diagnostic candidates, not a demonstrated new grasp policy.

Evidence: `quaternion_bc1000_actual_artifacts_independent_audit_v1.json` and `quaternion_bc1000_sampling_and_protected_state_audit_v1.json`. Model artifacts remain outside Git under `quaternion_bc_diagnostics/`.

### Actual MPC-supervised Student candidate

The next candidate is fitted to `absolute_reference_pilot_v2/successful_visual_supervision.pt`, SHA256 `4b337fdbb80e64980cb2730faa47b681dcb04227cc729a9d6bcbf22adb318215`. All4698 rows match actual issued actions;3355 include MPC corrections. Its four successful pinch episodes exactly match the independently replayed native success pool. Failed tripod/pressure trajectories are excluded.

`student_mpc_canonical1000_startup25_warm_start_v3.pt`, SHA256 `8dd9928366c231f8bd67894e67b72f0f37803628b658e4b156a016f8e0009b53`, uses1000 offline BC updates, batch64, LR1e-4, seed42, explicit PointNet reset, canonical quaternion preprocessing and25% startup mixture. Training MSE is0.000241638; startup MSE is1.21555e-5, maximum0.0164945. There is still only one independent pinch source, no tripod label and no independent heldout source. It is not selected as a best-success model.

Independent candidate audits verify101 checks and73 protected/sampling checks, bind the source/label/native-proof SHA values, and preserve all source PPO state except trained Actor weights and the explicit known adapter-contract field. The gate is `student_mpc_candidate_verified.json`; input/model checks are `student_mpc_candidate_actual_artifact_audit_v1.json` and `student_mpc_candidate_protected_sampling_audit_v1.json`.

The new pure-Student evaluation uses the same200-case regression manifest, original physics/cameras, no Teacher/MPC controller and at most2400 controls, in `student_mpc_canonical1000_100_per_style`. Initial actual actions from all200 environments match CPU canonical inference within1.19209e-7. The same weights with a raw-input counterfactual differ by up to0.133752, while all independent quaternion sign flips leave canonical outputs exactly equal. This CPU replay of recorded native inputs verifies adapter execution and is not an additional set of native trials. Audit SHA256: `8248a7607084e72fb4b27c1c87e47e45adf5925ed5096cebf57af406ead989d1`.

### Executed MPC-supervised Student result

The pure-Student candidate completed 1800 controls in 2423.595 wall seconds, exit0. All 200 unique first episodes ended, with no incomplete episodes or invalid samples. Independent replay of **1,534,579 native physical samples** verifies the frozen case manifest, model, physical identity, terminal records and execution certificate. Both styles still have zero native successes.

| Grasp | Episodes | Lift P50 / P90 / P95 / max (mm) | 30-mm rate | Maximum qualified hold | Opposed contact time | Native success |
|---|---:|---|---:|---:|---:|---:|
| Pinch | 100 | 0.00388034 / 0.01119644 / 0.10205120 / 24.53798565 | 0% | 0 s | 0.74622% | 0/100 |
| Tripod | 100 | 0.00005582 / 0.00008432 / 0.00010043 / 0.00080066 | 0% | 0 s | 0% | 0/100 |

Every pinch episode establishes bilateral native opposed contact at least once, compared with none for BC100. However, every episode also loses it; there are 5765 valid true-to-false transitions in total, and opposed contact occupies only 0.74622% of recorded physical time. First contact-loss P50/P90/P95 is 2.11875/2.22208/2.46156 s. A 24.538-mm maximum does not establish reliable lifting: the median is 0.00388 mm and P95 is 0.10205 mm. Neither model reaches 30 mm or holds the success criterion for 5 seconds.

Tripod never establishes bilateral opposed contact. Its zero contact-loss count means no contact was established, not stable retention. No qualified tripod demonstration was available for this candidate, so this evaluation tests a known training-data gap rather than a trained tripod skill. The supplied 200-case set remains a reused regression set; it is not an independent unseen model-selection set.

The contact measure uses native bilateral all-five-pad opposition eligibility. The 200-case run does not include the expensive optional style-masked manifold trace; it does not certify that every contact uses only the requested pinch/tripod fingers. Simulated torque-saturation rates are 0.016261% for pinch and 0% for tripod, explicitly implicit-drive estimates rather than measured hardware torque. Pinch has only 1.43542 s of applicable anchored-slip evidence, with reported slip 0.911748 mm and 100% coverage within that applicable duration. All remaining no-anchor/no-contact intervals and all tripod slip are **not applicable**, not zero. Legacy unconditional completeness flags therefore remain false; conditional telemetry completeness and the native execution certificate are verified true. None of these execution checks certifies grasp success.

Audit: `student_mpc_candidate_independent_audit.json`, SHA256 `119108491d5afe0e289a42e4240a9e93ed0f7ca9d06bded730ac3a528f1cb2a5`. Reproduce with:

```bash
/mnt/mydisk/xk/envs/dh116/bin/python /tmp/dh116_native_student_audit.py \
  --run /mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z/student_mpc_canonical1000_100_per_style \
  --output /mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z/student_mpc_candidate_independent_audit.json
```

## CPU verification and review

Targeted tests cover source indexing, absolute fallback, total residual bounds, real future commands, numerical trust intersections, actual-issued prediction, mixed-style isolation, terminal state, force-vector sign, pressure retreat and unknown-hand lead intersections. Independent code review reproduced the numerical boundary issue and verified its correction; the final pressure review found no remaining blocking issue.

The first full run reported one additional shell-launcher failure because the Git archive retained CRLF, whereas the preceding test environment used LF. Twelve staged `.sh` files were normalized to exactly the previous environment's bytes, with before/after SHA recorded in `test_environment_shell_eol_normalization.json`; repository shell content and native physics were not changed. The affected test then passed. The full suite was rerun after this environment correction:

```bash
cd /mnt/mydisk/xk/projects/dh116-mpc-contact-v2-20261106
USD=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$USD:src:.:vendor/DeformX" \
LD_LIBRARY_PATH="$USD/bin" PYTEST_DEBUG_TEMPROOT=/tmp \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider tests \
  --continue-on-collection-errors -q --tb=short \
  --junitxml=/tmp/dh116-contact-quaternion-startup-final.xml
```

Final result, including the quaternion and startup changes: **5497 passed, 96 existing failed, 28 existing errors, 49 skipped, 91 passing subtests**, 271.08 seconds, exit1 because of the preserved existing failures/errors. All 5529 prior nodes retain their status with none missing; 141 new nodes all pass. JUnit SHA256: `6d3a27c95a226627715aaf46496e0db01307b2d2631b550710dfbbf5fd6aa09a`. Comparison: `/tmp/dh116-contact-quaternion-startup-final-comparison.json`. This is not a claim that the entire legacy suite is green.

## Preservation and unresolved work

The final historical-model audit confirms unchanged SHA256 values for the actual 129, 1439 and 1750 Checkpoints: `5c85bc7a9516f9124a5f6dc99d39fe185016c60c0dff6f1c97346b2d429ecd0c`, `d8d94faca22638584e10eb67ff55f9cfde904984c7232433ffd68f867ab5bda9` and `837791335e5f1c2153db763ed419381c708ec8b1f8ff2d4abfc6fa025aab249d`. Proof remains outside Git in `historical_checkpoint_final_hashes.json`. Every native experiment has finished and GPU3 has been released; no long PPO was started.

- Absolute-reference MPC restores four fixed pinch replay successes. It matches the Teacher reference and does not prove broader superiority or Student success.
- Pressure-aware MPC reduces the measured tripod safe-prefix overload but still drops all four objects. Contact acquisition/retention and a physical-readiness-based phase controller remain unresolved. The present model predicts local kinematics, not force or object dynamics.
- The new Student improves pinch contact acquisition relative to BC100 but remains 0/100 successful for both styles. One independent pinch source and no qualified tripod source are insufficient supervision; low startup fitting error is not closed-loop success.
- An independent heldout set, best-Checkpoint selection, paired Adam-retain/reset experiments, Grasp/Hold/Lift curriculum, R0-R5 single-factor reward studies and broader cable generalization remain deferred. No successful outcome is inferred for those unexecuted experiments.
- Native execution still uses the original rigid cylindrical object proxy and existing uncalibrated simulation materials. It provides no deformable-cable or real-hardware validation, and simulated contacts remain outside the Student input.

Keep PR #1 Draft. Do not claim Lift degradation has been solved, relax 30-mm/5-s/0.5-mm success, or promote a rare height maximum to reliable grasping.
