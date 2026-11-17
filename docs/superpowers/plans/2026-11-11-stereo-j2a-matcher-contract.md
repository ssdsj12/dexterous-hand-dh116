# J2-A matcher contract and reliable refinement plan

**Goal:** Extend the existing matcher with explicit capability/rejection metadata and an opt-in reliability-gated SAD candidate while keeping all five old default output tensors bitwise unchanged.

**Authority:** Fully read455-line task document SHA256 `d4497f4a56e705b37f5f4bad6b75449378f1fa41396f3887be22f83712d01d68`; latest GitHub/local baseline `692e09d2473502574d0870dfaffff38b6c0a544c`. The immediate execution section scopes this invocation to J2-A. J2-B follows a passing A gate; no native experiment, training, push or merge here.

**Architecture:** Keep legacy SAD arithmetic/order and compatibility aliases. A small optional trace records decision predicates without changing those arithmetic results. The reliable candidate scans disparities in bounded planes and evaluates the winning five-cost neighbourhood, independent nonlocal ambiguity and RGB-only boundary evidence. It never consumes GT or semantic reference masks. Low-information/multimodal pixels are invalid; unreliable local refinement uses a declared integer fallback or invalid policy.

## Ownership and test-first execution

- [x] Capture the old matcher as a fixed test-only oracle, compare SAD outputs at both integer/subpixel modes and multiple batches with `torch.equal` on masks/finite patterns and exact finite values.
- [x] Add semantic contract tests for aliases, float disparity/NaN validity, SGBM range-flag confidence and unavailable true occlusion/internal uniqueness.
- [x] Add a pure five-cost refiner and J1 pixel-A regression. Preserve legacy offset=-.25 as a measured failure; use five-point support/model evidence to decline that fit without reading GT. Distinguish refinement rejection from final pixel rejection.
- [x] Add independently sampled continuous RGB fixtures at fractional .1/.25/.5/.75, with observable coverage requirements and degenerate-texture rejection.
- [x] Integrate the candidate into `StereoDepth(method='reliable_sad',subpixel=True,...)`; old method/config defaults unchanged. Support image chunks without full `[B,D,H,W]` cost allocation.
- [x] Merge nested diagnostic masks across chunks, preserving every env identity. Rejection keys: low_texture_left/right, search_boundary, left_right_inconsistent, uniqueness, occlusion_composite, true_occlusion=NA, depth_range, subpixel_rejected, patch_border.
- [x] Reuse the new diagnostic metadata in existing offline replay rather than introducing another matcher. Bind new candidate implementation/policy to processing source/config SHA; old model packages remain read-only and are not silently migrated.
- [x] Run new RED/GREEN tests then old matcher/perception/OpenCV and affected raw-RGB/model-contract regressions on existing CPU environments. No dependency installations. Retain every failure and output in exclusive private folders.
- [x] Update README/CHANGELOG, an independent report, current file inventory and readiness. Mark software/analytic/offline/raw-replay evidence separately; accuracy/physics/deployment qualification remain false.

## Result contract

Existing tensors `depth_m`, `disparity_px`, `valid`, `occlusion`, `confidence` remain. New aliases `valid_mask` and `occlusion_mask` share their tensors. `diagnostics.version='dh116-stereo-matcher-diagnostics-v1'`; confidence kind is `relative_cost_margin` for SAD or `compatibility_validity_indicator` for SGBM, always `confidence_is_probability=false`. SGBM internal uniqueness and true physical occlusion are unavailable, not false zero measurements.

No existing success threshold or camera K/T is changed. A passing synthetic test is not native evidence, and the narrow42.784–43.218px J1 scene cannot establish general fractional-disparity accuracy. A J2-A report must retain any failures before considering the J2-B gate passed.

## Execution evidence and boundaries

Completed software gate: 156 new J2-A cases; complete relevant suites Windows751/1skip and server752/1skip. The final candidate additionally validates actual fractional image samples and narrow single-tone recurrence evidence; it does not claim generic occlusion/periodicity robustness. The original five-cost-only configuration and all RED evidence remain. Accepted mixed-occluder edge errors=27 are an open accuracy limitation. See the independent J2-A report and machine summary. No native/GPU experiment, training, push or merge; J2-B is pending.
