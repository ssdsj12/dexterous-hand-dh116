# Stereo J0/J1 diagnostics implementation plan

> **For agentic workers:** Use focused parallel read-only investigation and execute the diagnostic implementation with test-first steps. Do not change production matching, capture new native data, train, push, or merge in J0/J1.

**Goal:** Reproduce current failures from the existing ten immutable native captures and attribute errors with explicit measured/unknown boundaries.

**Architecture:** Extend the current stereo system with an evaluation-only diagnostic module and CLI. Read actual probe/reference/calibration/source bindings; preserve masks and identity, produce per-frame/per-env region metrics, sampled cost curves and real image/3D plots. Missing reference views or source masks remain unavailable. Existing depth, controller, reward, and promotion behavior stay unchanged.

**Tech Stack:** Existing Python 3.10/3.12, Torch, NumPy and server OpenCV; no new dependencies, no Isaac imports.

## Inputs and authority

- Task: `D:/download/DH116_Current_Branch_Detailed_Codex_Tasks.md`, 786 lines, SHA256 `14a1373d7c430286942ae309b8f70453083c434885323945c807a0b9359c8e5c`.
- GitHub target/current local HEAD verified `7c808781ebd2cf969ca08776a83a65e3a2834582`; remote branch fetched before analysis.
- Preserve original 528 file hashes and all native artifacts. All generated evidence goes to a new directory.

## J0: latest-source gap matrix

- [x] Record status/branch/HEAD/log and remote HEAD; do not reset/stash.
- [x] Read readiness, latest phase/integration/native reports and contract audits.
- [x] Recompute protected source hashes and map real software/qualification gaps by J0–J10.

## J1: tested diagnostic primitives

Files: create `src/go2_pvcnn/control/dh116/tabletop_stereo_error_diagnostics.py`, `scripts/analyze_dh116_stereo_errors.py`, `tests/test_dh116_stereo_error_diagnostics.py`; update readiness and this work package's independent report.

- [x] RED: known depth bias with holes must retain reference denominator, joint-valid error and correct 3D point completeness as different quantities; all-empty regions return null quantiles.
- [x] RED: 3D error uses original K/pixel centres and Euclidean rays, not Z-only threshold.
- [x] RED: spatial bands/edges have explicit pixel definitions; unknown/right-view occlusion without actual reference is NA, not a guessed ground truth.
- [x] RED: rejection reasons preserve overlapping evidence and first-failure counts separately; saved invalid disparities are not reconstructed from NaNs as if measured.
- [x] RED: pixel cost probes report competing minima, curvature and patch/background mixing without modifying the production matcher.
- [x] Implement minimal pure CPU tensor primitives; run these tests with existing local Python, retain RED/GREEN logs privately.
- [x] CLI validates artifact SHA and per-frame identity, uses exclusive outputs, generates numeric JSON and actual RGB/mask/disparity/error/3D visuals. CLI never calls native bootstrap or qualification promotion.
- [x] Exercise synthetic tensor/identity/CLI output checks first, then a saved-raw CLI smoke and all available actual saved frames; any missing artifact path/SHA is recorded, not silently skipped.
- [x] Explain Block9 and SGBM tails with sampled pixel/env/frame evidence and same-support comparisons. Rank measured causes separately from hypotheses requiring new reference/capture.
- [x] Run relevant regressions, report CPU software versus saved-native reanalysis scopes, update readiness without changing existing accuracy/physical/deployment gates.

## J2 handoff, after J0/J1

Plan exact tests and independent candidate changes to the existing matcher interface, native-instance-labelled segmentation, graph/arc tracking and auxiliary depth decoder. Set explicit CPU/GPU/time budgets from J1 measurements. No licensed model downloads or long native training are authorized merely by this plan; J3–J10 retain the task's original promotion and experiment gates.
