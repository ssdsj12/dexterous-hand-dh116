# 902 continuous contact guard

A node-only SDF screen admitted a right Tripod target with collision between rod nodes. The new independent contact entry rejects it before Kit starts; live source and cooked hull checks also cover the full polyline. Original 0.15 mm selected-pad penetration, 0.25 mm noncontact clearance, support/table limits and both fixed RGB camera files remain unchanged.

## Independent review

Antigravity R1–R4 read the actual Ubuntu worktree through the DH116 read-only SSH role. Codex checked file SHAs, callers and archived data. Adopted: local measured-pose correction, independent receipt validation, continuous rod protection, source-bound stations/commands and reference-derived table height. Rejected: treating node SDF as proof of continuous clearance or executing a failing candidate.

Measured FK matches SDK links to at most 0.000104 mm. Use FK_new @ inverse(FK_old) @ SDK_old; keep controller increments based on issued backend targets. No evidence supports a millimetre-scale coordinate error.

## Actual geometric result

| Right link | Node SDF gap (mm) | Exact cooked continuous gap (mm) |
| --- | ---: | ---: |
| finger14_link | -0.050 | -0.826960 |
| finger23_link | -0.050 | -0.209635 |
| finger33_link | -0.050 | -1.957356 |

Middle-finger source gap is -1.984922 mm. Its critical segment spans 12.500904 mm; endpoint cooked gaps are -0.049994/+0.890807 mm. This is real between-node penetration. No tolerance relaxation, geometry exemption or hull edit is used. Runtime uses an existing conservative full-polyline lower bound and may reject safe geometry; exact distances here establish the cause, not a weaker runtime gate.

## Changed behavior

- screen_dh116_902_flexible_contact: measured_fk_pose, contact_route_passed, stage_actions, independent_entry, checked_request/stations, prepare/load_stage. Preserve encoded JSON archives, side-specific material stations, source SHA and original 24D command identity.
- screen_dh116_tripod_measured_precontact: independently verified boundary receipts and contact_rod_clearance_m across source/cooked hulls.
- verify_dh116_902_flexible_reset: optional contact receipt/request/proof, freshly recomputed in a CPU child before Kit imports USD; default entry unchanged.
- verify_dh116_power_shared_rod / verify_dh116_supported_distal_transfer: continuous selected-pad protection alongside node SDF; only explicit independent entries avoid the old hard-coded Tripod boundary.
- solve_dh116_tripod_sdf_constrained_direction: remove per-call HiGHS threads=2. After another default linprog initialized the process-global pool, this conflicting setting produced status 4 / HiGHS Status 0: Not Set. A real subprocess regression reproduces and verifies the fix. Optimization bounds and budgets remain unchanged.

## Verification

287 related CPU tests passed in 61.46 seconds, including 15 contact tests and the subprocess LP-order test. Initial failing tests and logs are retained. The real old candidate exits 1 with all_passed=false; attempted native preparation rejects during CPU preflight, creating no native run. Source and cooked continuous probes independently reject it.

An older receipt suite separately fails its pre-existing /home versus /mnt path-alias validation. Original code was restored byte-for-byte; that evidence remains. This is not a claim that every repository test passes.

No new Isaac run, Teacher query, PPO update, RGB capture or grasp qualification in this package. The previously published SDK 408 boundary naturally exited 0 but recorded no hand contact. Raw logs/hulls/search traces remain outside Git; readiness binds their hashes. Check all 528 protected files and three original checkpoints before publication.

## Static alternative and remaining route

Reusing the bounded minimax solver found a static alternative after 362 evaluations / 20.93 s: node gaps -0.0210/-0.0216/-0.0332 mm, continuous bound -0.1393 mm, noncontact clearance 2.6221 mm, support clearance 0.3723 mm. It is terminal geometry only.

The direct approach requires at least 66 original-rate control windows and still reaches -0.2074 mm at the index source hull. Its proof remains false. The committed 12+6-window request is the deliberately rejected historical candidate, not an executable success recipe.

Next: screen the actual controller Euler/target sequence with staged or delayed index closure. Only freeze a native recipe when its complete route passes. Preserve physical hand identity, user-requested 902 final posture, flexible wire, fixed cameras and Power full fist/palm wrap. Contact load, fixed-wrist hold and net Lift remain unverified.

## Reproduction

Use the Ubuntu registry CPU environment and the 17 test files in readiness. Negative candidate:

    python -B -m scripts.screen_dh116_902_flexible_contact --receipt "$TRIPOD_SDK408_OUTER" --request configs/dh116_tasks/flexible_902_tripod_right_distal_contact_v1.json --output "$NEW_EVIDENCE/continuous-contact-rejected.json"

Expected exit 1, all_passed=false. Never launch Isaac with that proof. Keep 24D/371D contracts and Lift 30 mm / Hold 5 s / Slip 0.5 mm.

Registry regression: 60 passed in 13.78 seconds, exit 0.
