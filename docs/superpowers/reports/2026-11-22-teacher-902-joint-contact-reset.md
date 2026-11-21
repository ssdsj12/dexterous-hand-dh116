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
