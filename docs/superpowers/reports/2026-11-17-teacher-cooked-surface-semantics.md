# Cooked surface semantics: distinguish closed topology from approximate faces

## Actual change and scope
Expanded scripts.audit_dh116_teacher_cooked_readback.diagnose_surface and analyze_hull. The CPU-only auditor now reports index coverage, used vertices, opposite directed edges, connected components, vertex one-ring cycles, Euler characteristic, centered signed fan volume, winding and diagnostic SVD face planarity separately. Diagnostic fits are never applied to the input or original validator.

No Isaac, GPU solver, control, Teacher query or PPO run occurred. Base e100fb6e0ef510d56631a9355c57b7df31f80f92. Existing34-owner capture only; input SHA c122c87408d681166cf9a437479d0010e1a8dd335dfdf55cb7e7ec0e2610982e.

## What the actual data supports
| Check | Result |
|---|---:|
| All34 hulls: complete index slots and used vertices |34/34|
| Opposite directed edges, connected component and Euler2 |34/34|
| Single vertex-link cycles |34/34|
| Positive signed volume and outward winding |34/34|
| Existing protected full validator |0/34|
| Original plane checks rejected |189/1890 polygons|
| Diagnostic best-fit nonplanar beyond original numeric scale |188 polygons|

These are partial surface checks, not proof of geometric convexity, absence of self-intersection or native contact qualification. That distinction is explicit in every output.

The remaining189th face is rightHand/finger12_link polygon2: source-plane residual3.747132651e-7m versus original numeric tolerance3.687562504e-7m. It is a near-boundary case rather than one of the188 best-fit nonplanar faces. This link observation does not establish a joint-limit or contact-force cause.

All10 current distal hulls match the old F4 CPU data exactly in vertices, indices and polygon records. Historical CPU/GPU parity and current strict-plane rejection can coexist: parity checks equality, not exact coplanarity. New asset corruption is not established by these residuals.

## PhysX source and version limits
Installed omni.physx extension version is106.5.7. Official family reference: PhysX tag106.5-physx-5.5.1, commit32ae713f6cb35f1df2544ffa499fc98a39639a62. Exact installed binary-to-source correspondence remains unproven.

[QuickHull merge and plane construction](https://github.com/NVIDIA-Omniverse/PhysX/blob/32ae713f6cb35f1df2544ffa499fc98a39639a62/physx/source/geomutils/src/cooking/GuCookingQuickHullConvexHullLib.cpp#L277-L315) uses its specific fixed-edge normal calculation and centroid offset; adjacent-face merging is not an absolute coplanarity test. [The reference hull validation](https://github.com/NVIDIA-Omniverse/PhysX/blob/32ae713f6cb35f1df2544ffa499fc98a39639a62/physx/source/geomutils/src/cooking/GuCookingConvexHullBuilder.cpp#L242-L301) excludes a face's own vertices from its enclosure loop. These criteria differ from the project's protected exact-face/enclosure predicate.

Replaying the exact reference formula on saved1890 faces gives maximum normal-angle difference0.000021611 degrees and offset difference1.22726e-8m; for the189 rejected faces,0.000007859 degrees and3.40006e-9m. Ordinary Newell comparison differs by up to2.554642 degrees. This strongly supports the reference cooking-representation explanation; it is not binary provenance proof or permission to widen the project validator.

No native plane was replaced, refitted or re-cooked, no contactOffset/restOffset changed, and no original mesh or physical parameter was changed. Original full qualification remains0/34. Any new representation-specific admission contract must be independently reviewed and versioned; this package does not introduce one.

## Tests and failed evidence
Eight new cases first exposed missing surface diagnostics or nonempty JSON serialization, then passed. They distinguish exact cube/nonplanar closed surface/open edge/reversed face/overlapping slots/pinched vertex/origin translation and real JSON output. Pinched-vertex rejection prevents opposite-edge counts alone being treated as manifold proof.

Full relevant CPU regression429 passed,0 failed,0 skipped; targeted15 passed. A NumPy bool serialization failure was caught using the actual packet and reproduced in a new test; the incomplete first output is preserved, not overwritten. surface-audit-v2 is the successful complete diagnostic output. Commands, environment, result SHA, raw input SHA and official source SHA are in readiness. Independent review found no blocking issue in the two changed files and reran15 tests.

All528 protected hashes are unchanged. Original models/raw data are read-only; no model was loaded or copied. SDK source/probe details and large packet remain on the data disk, outside Git. Public commits contain only auditor, tests and compact report/status. Ubuntu development/commit and Windows publication; PR2 remains Draft.

## Remaining work
The exact installed cooking build and a spatial/force qualification for the approximate representation remain unresolved. The separate owner-probe work package investigates runtime/factory/native-view roots; Python object liveness alone is not sufficient to prove the SIGSEGV cause. Formal Teacher PPO remains blocked. Original30mm/5s/0.5mm success standards, physics, assets and old contracts are unchanged.
