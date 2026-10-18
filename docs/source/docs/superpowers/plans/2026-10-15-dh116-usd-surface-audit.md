# DH116 USD Surface Audit Implementation Plan

> **For agentic workers:** Use executing-plans, Inline Execution as selected by user; no new strategy approval needed.

**Goal:** Extract actual instance-proxy collision surfaces in body-local metres and produce inspectable geometry evidence before pad selection.

**Architecture:** Read-only pxr extraction is separate from pure coordinate math and from pad semantic qualification. Mesh ancestry, exact geometry digest and source-layer hashes accompany the extraction. A diagnostic report never authorizes a grasp.

**Tech Stack:** pxr, NumPy, pytest, matplotlib; existing go2 environment.

**Checkpoint:** Task1 extraction and Task2 diagnostic report executed; 14 new tests/105 combined tests passed. Task3 evidence/notes updated. Real pad selection is not completed: independent thumb frames and convexHull/source-surface correspondence remain required downstream gates. No physics run or source edits.

## Global constraints

Use the existing worktree and core0/single numeric thread. Do not edit source USD, invent O6 offsets, or mark pad calibration/PhysX passed. Follow the approved parent design. All meshes must be explicitly from the requested body and branch; reject missing or ambiguous selection.

### Task 1: Strict read-only mesh extraction

Create `Go2Pvcnn/go2_pvcnn/control/dh116/usd_surface.py` and `Go2Pvcnn/tests/test_dh116_usd_surface.py`.

Interface `read_surface(stage, body_path, branch='collisions') -> dict` returns body/mesh paths, mesh-local points, body-local points, triangles, mesh-to-body column transform, collision-owner path, orientation, geometry SHA256. `layer_pins(stage) -> dict[path, SHA256]` pins loaded file layers; reject anonymous/missing layers for provenance rather than silently excluding them.

- [ ] Write tests using real in-memory USD with rigid-body and translated collision owner; assert correct body-local vertices and inherited collision owner. Missing body, invalid branch, multiple meshes, bad indices, nontriangles, no collision owner and wrong metresPerUnit must reject.
- [ ] Run pytest and observe missing-module RED.
- [ ] Implement using `Usd.PrimRange(body, Usd.TraverseInstanceProxies())`, filter exact branch prefix, validate one mesh and triangular finite geometry. Transform `np.array(mesh_world * body_world.GetInverse()).T`; reject nonrigid transforms. Digest explicit little-endian vertices/indices plus orientation; preserve source indices, never silently weld for output. Walk ancestors up to the selected body for collision API; require collision enabled. Pin layers via hashlib.sha256 of file bytes.
- [ ] Test actual mounted left/right five distal bodies and palm: collision/visual geometry equality, nonempty hashes and normals, and both sides' independent topology. Real asset tests skip explicitly if the asset is absent; local run must not skip.

### Task 2: Inspectable evidence

Create `Go2Pvcnn/scripts/audit_dh116_surfaces.py` with required `--asset` and new `--output` (refuse existing directory).

- [ ] Test output-overwrite rejection before implementing the report function.
- [ ] Use `read_surface` for both branches of each palm/distal body. Save SHA-pinned JSON with branch equality, bounds and counts; create side-by-side local-coordinate surface views for thumb/index and an assembled-hand view. Label axes in metres. Save candidate-free views first; avoid calling triangle winding proof of outwardness.
- [ ] Run on actual mounted asset, inspect images. Record whether visible geometry uniquely identifies outward pad regions; if uncertain, preserve ambiguity rather than select by bounds alone.

### Task 3: Checkpoint

- [ ] Run all DH116 tests, update evidence log/dashboard/branch/CODEX_RESUME.
- [ ] Record exact mesh paths and physics collision approximation: source triangle surface is not necessarily PhysX cooked collision surface. This is a required downstream contact qualification, not a reason to delete collision geometry.
- [ ] Next use the inspected surfaces to define pad references and palm axes, then actual USD FK/PhysX Jacobian; no table or grasp run in this audit.
