# New aligned-wrist independent Reset path

Base bd80a5b8727b24453ca699f1dbc0f333a5bb6105, branch 11-17.

The existing independent open Reset entry now accepts an explicit all-or-none opt-in trio: --joint-candidate, --joint-search and --joint-path-proof. No opt-in keeps the original run_screen/version/mode/default behavior. The new branch reconstructs the full CPU proof in a separate USD child process, compares the complete fresh proof and supplied artifact, checks candidate/search/proof SHA again at final freeze, and changes only the right wrist Reset pose. Measured float32 opening q, left hand, rod/support/table/gravity, both fixed RGB configs and original safety/action contracts are retained.

adapt_joint_reset preserves the no-admission flags and maps receipt/source field names for the existing checked_reset_proof/check_frozen_proof interfaces. joint_reset_inputs binds the three artifacts; check_joint_frozen_proof prevents a changed artifact from being frozen after its CPU check. Execution continues through the original owner, shared.run_shared(action_after_settle=None) and explicit natural-cleanup lifecycle. This branch cannot issue closing actions.

certify.path_row_passed now requires right-side material-node identities at every substep. The original REFERENCE file SHA is compared with the actual parent frozen plan. The derived geometry auditor accepts the independent new Reset version only with candidate/search/proof byte hashes matching its frozen sources and screen, and exact right pose/opening equality.

Test first: 3 initial AttributeError red cases plus later missing-audit/missing-hash-check red cases were observed and retained. Final related CPU suite: 235 passed, 0 failed, 74.87s; seven new tests cover only-right-pose change, failed CPU evidence, wrong opening, material-node side, new-recipe source identity, changed search hash, incomplete CLI opt-in and false force qualification (some cases share one test). Antigravity R1-R2 read the exact three production files and found no blocking CPU interface/freeze defect. Its test-source reread predates the final two cases; their final execution is independently verified by Codex.

A fresh full path proof was regenerated after the guards changed. The actual main entry recomputed it in a child and froze successfully with clean native import boundary. Recipe fingerprint:
386c8ffedcabc305e7674dc4738d14851d03716797b22e80a397832f84f1a53e
Budget: GPU2, one instance/environment, 240 SDK steps (2 setup + 238 measured), 2856 solver substeps, zero control/Teacher/PPO, 180s inner soft wall, no retries or automatic kill. All old fingerprints remain consumed and are not reused.

At this source publication, native_runs=0. After one authorized attempt, independently record outer/debugger exit, internal failure/result, all 238 exchange files and source SHA; audit actual initial SDK2 + exchanges as 239 measured frames. Do not call this qualified contact, grasp, Lift or Teacher.

Closure must preserve two distinct identities: the original search-source receipt and the newly measured execution-boundary receipt. Do not rewrite the search source SHA to disguise the new Reset. Native closing needs a newly bound proof and real opposed force / next-step reaction evidence.

Private evidence: ref902/joint-reset-*.log/json and joint-candidate-reset-native-v1/{argv.json,plan.json,freeze.log}. No raw records/models are uploaded. 528 protected files and three original checkpoints are verified before commit.

Registry regression: 60 passed, 0 failed.

## Actual sealed native result

The first prelaunch check rejected fingerprint 386c8ffe because required source publication changed frozen docs/contracts/dh116_release_manifest.json. No GDB/Isaac attempt or SDK step had started. That CPU-only plan is superseded; no retry of a consumed native fingerprint occurred.

After final publication, the recipe was freshly sealed and executed once: 8f5d710152a3d77bf09c59fc23261faa7513a2a014bb542ab42bd73de82f2df6. Source HEAD c5dc15203356cf657076d0fe2cdd3ce9a6d5b6e2. The outer/GDB inferior both naturally exited 0, signal=None, forced_exit=false, exceeded_budget=false; wall 94.931s. 240 SDK / 2856 solver steps, all 238 exchange records, no internal failure; zero control/Teacher/PPO. Both supports have 2856 contact-pair records; hand/table pair counts are zero; max pair force residual 0 N. No process was force-terminated.

Fresh 239-frame measured-base/q-derived audit exited 0. Right rod/support/table lower bounds are 6.1970886 / 2.3684033 / 46.5378628 mm; left bounds are 1.2311466 / 22.9422547 / 57.3823270 mm. Both hands have 0-rad active target error in the last nine saved frames, constant backend targets, passive mimic residual <=4.0046871e-8 rad. Final rod maximum node speed 0.012912691m/s remains reported without a steady-rod qualification claim.

Outer receipt SHA 655a24bb551c655685a0e168712c13e282226ea6f93de167ee3bb49e3286add2; derived-clearance audit SHA 631a917b8dd434d97c1b4746f9c8f8ae0c358e3eae5b27a8d75e66d5f60352e3. Raw files are private under ref902/joint-candidate-reset-native-sealed. The native budget is consumed and must not be rerun.

This passes the bounded operational Reset interval and its derived geometric audit. It does not certify independently measured full-link motion, contact force, grip, Lift/Hold, or Teacher admission. All qualification flags remain scoped accordingly.

Next work must distinguish the original search-source receipt from this new execution-boundary receipt. The current certify API binds those as one receipt, so it needs an explicit execution-source path with candidate/Reset recipe identity checks before a fresh closing proof can be generated. Do not replace the original search SHA. Then keep the existing continuous guards and SupportedDistalProbe and require actual opposing force plus next-step reaction delivery.
