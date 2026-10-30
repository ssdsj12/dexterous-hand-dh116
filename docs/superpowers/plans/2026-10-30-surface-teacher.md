# Surface Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for bounded implementation and independent spec/quality reviews.

**Goal:** Generate multiple DH116-native grasps using multi-point surface geometry and publish teacher cases only after measured physical lift qualification.

**Architecture:** Pure differentiable geometry objectives; USD-derived Torch FK/collision sampling; bounded candidate optimization; existing six-axis PhysX drives for serial real validation; strict raw-trajectory teacher acceptance. Existing PPO actions and physics remain unchanged.

**Tech Stack:** Existing hand Python/Torch/NumPy/SciPy/USD/Isaac environment; pytest. No new installed dependencies.

## Task 1: Geometry objectives (bounded worker)

Create `src/go2_pvcnn/control/dh116/tabletop_surface_objectives.py` and `tests/test_surface_objectives.py` only.

- [x] Write failing tests for oriented finite-cylinder SDF (outside/inside/caps), differentiable surface attraction, distinct penetration, table cost, optional normals and supplemental convex/self penetration. Reject invalid shape/nonfinite/zero quaternion.
- [x] Implement `finite_cylinder_sdf(points,center,quaternion,radius_m=.008,length_m=.5)` and `surface_terms(patches,collision_points,center,quaternion,*,patch_normals=None,table_z=.30,radius_m=.008,length_m=.5,contact_points_per_patch=3,self_penetration_m=None,object_hand_penetration_m=None)`. Inputs patches[N,2,2,P,3], collision_points[N,K,3], center[N,3], wxyz[N,4]. Return per-row `reach`, `penetration`, `table`, `normal_alignment`, `self`, `object_in_hand` and `patch_distance_m`[N,2,2]. Scale reach by8mm and penetrations by1mm; table floor0.3mm.
- [x] Reach uses mean absolute SDF over nearest3 points per patch; zero crossing must remain distinct from inside penalty. Geometry never creates contact evidence. Return rates/energies, no dt or actuator outputs.
- [x] Run red/green CPU tests using server wrapper, no Isaac, no commits or other file edits.

## Task 2: Native differentiable geometry (main)

Create `tabletop_surface_kinematics.py` and `tests/test_surface_kinematics.py`.

- [x] Test six-active to eleven joint mapping, transform chain, gradients and collision sample provenance before implementation.
- [x] Use current source USD joint frames and hulls, topological FK and named MIMIC coefficients; sample surface triangles deterministically and select multiple finger14/finger23 inner-patch points near existing pad reference. Include normals and complete-hand collision samples.
- [x] Calculate object-in-convex and nonadjacent self-penetration checks; exclude same/adjacent collision bodies explicitly. Verify against existing HandPointFK and actual SDK poses before accepting candidates.

## Task 3: Candidate optimization and teacher acceptance (main)

Create `tabletop_surface_teacher.py`, tests and `scripts/search_dh116_surface_teacher.py`.

- [x] Test multistart bounds/diversity and strict acceptance of raw bottom/force/speed/dwell; reject missing digit contact, excessive force, imbalanced pair, fault/reset or insufficient actual height.
- [x] Optimize 12 variables/hand with Adam in normalized bounded coordinates. Recompute geometry from FK each iteration. Keep multiple ranked candidates, never label geometry-only output qualified.
- [x] Physical entry reuses current CLI scene constructor with a teacher diagnostic callback. Integrate joint/wrist velocities through the existing controller; record actual observations and actions. Reset only to selected candidate initial pose, retain original rod and physics.
- [x] Publish qualified teacher initial cases/trajectories only after the strict acceptance function; otherwise write failure reports only. No long PPO, no old checkpoint mixing.

## Task 4: Validation and delivery

- [x] Preserve source before deployment and run combined targeted CPU regression and static checks.
- [x] Run offline/native FK comparison, bounded candidate search and serial actual lift trials with unchanged physical limits.
- [x] Spec review then quality review; repair findings and repeat only affected checks.
- [x] Save raw evidence outside Git, update README/CHANGELOG/report with actual outcomes, commit locally on10-29 and verify deployed hashes. Height/contact records are explicitly rejected for teacher use where closure penetration is excessive.

## Task 5: User-requested real support wrench

- [x] Create failing tests for gravity-based upward target, normal/friction separation, tangent direction, measured-contact-only torque, Jacobian virtual-work sign and shared actuator cap.
- [x] Add `tabletop_surface_support.py`: measured normalZ/frictionZ plus object weight/velocity produce a bounded tangential contact wrench; map through actual folded six-active Jacobian to torque. Existing `write_contact_effort` enforces total active motor budget.
- [x] Enable only in the teacher diagnostic wrapper, retain production PPO freedom and current physical parameters. Record actual signed force components and support/error/torque telemetry.
- [x] Run focused review and real lift attempt, then publish only after final raw-substep revalidation with separate producer and qualifier source identities.

## Task 6: User steering — pinch and keep holding

- [x] Add explicit pinch/hold diagnostic: latch the chosen target, keep wrist commands, let coupled fingers close on their natural arcs, and continue real-contact normal feedback without scheduled ascent.
- [x] Reproduce and repair finite-mimic master retreat counteracting thumb/index force control only in this mode; keep original passive tracking, hard actuator limits, invalid-state recovery and ordinary PPO behavior.
- [x] Run red/green force-latch, wrist-lock, natural-arc and master-retreat regressions; previous combined suite863 passed.
- [x] Complete physical F/G and later H/I/J diagnostics, retain raw force/pose evidence and report actual sustained contact separately from force accuracy and lift qualification. None qualified.

## Task 7: Latest steering — force-unrestricted physical lift

- [x] Remove target-force/force-error and digit-distribution gates from the new diagnostic; hold bounded closure targets and drive the wrists upward through real actuators.
- [x] Red/green tests for held closure, finite state recovery and actual cylinder-bottom dwell; combined41-file regression940 passed.
- [x] Static independent review; save full480Hz guard evidence and replay it before reporting qualification.
- [x] Run40-case A, isolated B, matched40-case C,36-case D and40-case E; record actual height/hold and full raw sources. A/C meet height/contact dwell but deep closure penetration disqualifies teacher use; B/D/E fail. No PPO started.

Completed implementation/diagnostic checklist items do not imply a qualified grasp or teacher exists. Actual outcomes must be recorded in the report.
