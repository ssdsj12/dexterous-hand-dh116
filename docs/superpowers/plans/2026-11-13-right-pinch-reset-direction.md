# Right Pinch measured Reset direction

Base 6709e72e170456ae05d30275a33d4a6dfd9d9833. Ubuntu-only CPU implementation; standing publication approval.

- [x] Preserve SDK234 actual right source, original USD/cooked SDF, source17+cooked5 shapes and original limits.
- [x] RED→GREEN tests for decode/limits, exact fixed Reset transforms, bounded LP, nonlinear rejection and no native qualification.
- [x] V1 4D: thumb spread, two closure distances, rod-axis fixed Reset angle. Two-scale Jacobian rejected SDF normal discontinuity after17/42 measurements.
- [x] V2 adds fixed X/Z Reset translation <=1mm, smaller finite differences .002/.001,58/58 measurements. Actual nonlinear candidate still fails.
- [x] Independent CPU line-sampling diagnostic exposes discrete-node/SDF gap difference; does not modify41 solver nodes or create visual points.
- [x] V3 adds fixed Y hand Reset <=6.25mm (half original rod-node spacing),99/132 measurements; exact nonlinear FK rejects later LP direction.
- [x] V4 reduces local trust radius from .5 to .05 without enlarging physical domain; pinned V3 seed,132/132 measurements, wall120s, 0 native/0 PPO. No fifth batch.
- [x] Every candidate checks reset+endpoint, both actual and task-reset rod at initial state, independent right source, original table2.1mm plus optimization reserve0.1mm, other-shape clearance, SDF and updated normals.
- [x] Full161-sample path only if endpoints qualify. None qualifies, so no path/native recipe generated. Native contact, dynamics and Teacher readiness remain false.
- [x] 90 related CPU tests pass. Preserve failed diagnostics, update report/Registry/readiness; protection528+2; Ubuntucommit→Windows publication→remoteSHA/DraftPR verification.

Next: use failed constraints and actual native geometry to design a different right Reset/contact arrangement. Do not interpret finite local-search failure as global infeasibility. Continue independent R0/state work.
