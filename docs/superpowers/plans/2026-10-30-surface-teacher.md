# Surface Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for bounded implementation and independent spec/quality reviews.

**Goal:** Generate multiple DH116-native grasps using multi-point surface geometry and publish teacher cases only after measured physical lift qualification.

**Architecture:** Pure differentiable geometry objectives; USD-derived Torch FK/collision sampling; bounded candidate optimization; existing six-axis PhysX drives for serial real validation; strict raw-trajectory teacher acceptance. Existing PPO actions and physics remain unchanged.

**Tech Stack:** Existing hand Python/Torch/NumPy/SciPy/USD/Isaac environment; pytest. No new installed dependencies.

## Task 1: Geometry objectives (bounded worker)

Create `src/go2_pvcnn/control/dh116/tabletop_surface_objectives.py` and `tests/test_surface_objectives.py` only.

- [ ] Write failing tests for oriented finite-cylinder SDF (outside/inside/caps), differentiable surface attraction, distinct penetration, table cost, optional normals and supplemental convex/self penetration. Reject invalid shape/nonfinite/zero quaternion.
- [ ] Implement `finite_cylinder_sdf(points,center,quaternion,radius_m=.008,length_m=.5)` and `surface_terms(patches,collision_points,center,quaternion,*,patch_normals=None,table_z=.30,radius_m=.008,length_m=.5,contact_points_per_patch=3,self_penetration_m=None,object_hand_penetration_m=None)`. Inputs patches[N,2,2,P,3], collision_points[N,K,3], center[N,3], wxyz[N,4]. Return per-row `reach`, `penetration`, `table`, `normal_alignment`, `self`, `object_in_hand` and `patch_distance_m`[N,2,2]. Scale reach by8mm and penetrations by1mm; table floor0.3mm.
- [ ] Reach uses mean absolute SDF over nearest3 points per patch; zero crossing must remain distinct from inside penalty. Geometry never creates contact evidence. Return rates/energies, no dt or actuator outputs.
- [ ] Run red/green CPU tests using server wrapper, no Isaac, no commits or other file edits.

## Task 2: Native differentiable geometry (main)

Create `tabletop_surface_kinematics.py` and `tests/test_surface_kinematics.py`.

- [ ] Test six-active to eleven joint mapping, transform chain, gradients and collision sample provenance before implementation.
- [ ] Use current source USD joint frames and hulls, topological FK and named MIMIC coefficients; sample surface triangles deterministically and select multiple finger14/finger23 inner-patch points near existing pad reference. Include normals and complete-hand collision samples.
- [ ] Calculate object-in-convex and nonadjacent self-penetration checks; exclude same/adjacent collision bodies explicitly. Verify against existing HandPointFK and actual SDK poses before accepting candidates.

## Task 3: Candidate optimization and teacher acceptance (main)

Create `tabletop_surface_teacher.py`, tests and `scripts/search_dh116_surface_teacher.py`.

- [ ] Test multistart bounds/diversity and strict acceptance of raw bottom/force/speed/dwell; reject missing digit contact, excessive force, imbalanced pair, fault/reset or insufficient actual height.
- [ ] Optimize 12 variables/hand with Adam in normalized bounded coordinates. Recompute geometry from FK each iteration. Keep multiple ranked candidates, never label geometry-only output qualified.
- [ ] Physical entry reuses current CLI scene constructor with a teacher diagnostic callback. Integrate joint/wrist velocities through the existing controller; record actual observations and actions. Reset only to selected candidate initial pose, retain original rod and physics.
- [ ] Publish qualified teacher initial cases/trajectories only after the strict acceptance function; otherwise write failure reports only. No long PPO, no old checkpoint mixing.

## Task 4: Validation and delivery

- [ ] Preserve source before deployment and run combined targeted CPU regression and static checks.
- [ ] Run offline/native FK comparison, bounded candidate search and serial actual lift trials with unchanged physical limits.
- [ ] Spec review then quality review; repair findings and repeat only affected checks.
- [ ] Save raw evidence and any qualified teacher cases outside Git. Update README/CHANGELOG/report with actual outcomes, commit locally on10-29 and verify deployed hashes.
