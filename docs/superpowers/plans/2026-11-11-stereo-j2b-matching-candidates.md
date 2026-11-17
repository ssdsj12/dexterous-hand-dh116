# J2-B Matching Candidates Implementation Plan

**Goal:** Continue the approved J2 task with explicit Census, ZNCC, gradient and quarter/half/full pyramid candidates under the existing StereoDepth contract.

**Architecture:** Preserve default SAD, reliable_sad and SGBM arithmetic. An independent RGB-only streaming cost engine returns the same directional tuple and trace; the pyramid orchestrator reuses that engine with scaled calibration, bounded local search and full-range retry. Existing depth/LR/validity diagnostics consume them without GT arguments. Each candidate has an independent opt-in configuration and complete source/config SHA identity. CPU numerical evidence cannot promote native accuracy, training or deployment.

**Tech stack:** Existing PyTorch/Python/pytest, no dependency installation, no Isaac or training. Baseline local a5a15fb; fetched origin remains692e09d. The user explicitly requested continuation after J2-A; the original task specifies these algorithms and no push/merge.

## Work packages

- [x] T1 cost engine and test-first implementation: Census 5x5/Hamming, ZNCC declared block variance, central-difference gradient L1. Stream individual cost planes and bounded winner neighbors, reject invalid support/texture/ambiguity. Refine using the same metric, not a hidden radiometrically different fallback. Preserve any failed qualification evidence.
- [x] T2 pyramid and tests: .25/.5/1 levels, exact realized K and disparity scaling, conservative validity, original-resolution final refinement; rejected local results retry global or remain invalid. Record skipped unsupported small scales explicitly.
- [x] T3 wrapper integration and tests: explicit method dispatch, diagnostic reasons and capability semantics, independent configs and source hashes. Legacy12 config fields and old default tensors stay unchanged.
- [x] T4 direct continuous RGB numerical controls: fractional shifts, fixed reference coverage, brightness changes and degenerate textures. Compare numerical failure separately from API/software success, no relaxed thresholds. No algorithm declared qualified by all-invalid outputs.
- [x] T5 run affected and complete stereo/privileged/surface CPU suites, preserve executed-source snapshot and protection hashes, review code and reports. Update README, CHANGELOG, readiness, file inventory and independent J2-B report. Local commit only.

Production files: new `tabletop_stereo_matching_costs.py` and `tabletop_stereo_pyramid.py`; modify `tabletop_stereo_depth.py`, `tabletop_stereo_preprocessing.py`, `tabletop_stereo_error_cli.py`. New tests `test_dh116_stereo_j2b_{costs,pyramid,integration,analytic}.py`; four independent `stereo_matcher_*_j2b_v1.json` configs. No old camera/calibration, controller, reward or Checkpoint edits.

Acceptance commands use existing Python with PYTHONPATH=src and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, CUDA_VISIBLE_DEVICES empty: `python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_j2b_*.py`, followed by the previously established full stereo/privileged/surface group. All logs and RED evidence stay in exclusive private `artifacts/stereo-j2b-20261111/` directories. Reports list software, analytic, saved-data, GPU/native and physical qualification separately. J2-C onward remain sequential work packages, not implied by this implementation.
