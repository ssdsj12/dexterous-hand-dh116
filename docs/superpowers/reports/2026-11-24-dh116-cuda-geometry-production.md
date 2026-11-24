# DH116 CUDA geometry production measurement integration

Production Pinch collection now selects a sealed CUDA measurement subclass with
`DH116_TENSOR_GEOMETRY_BACKEND=cuda_cabi`. Its base backend remains `eager`;
`actual_measurement_backend` records `cuda_cabi` separately. `TensorGeometry.capture`,
source/body bindings, all guards, every final predicate and synchronous CPU
fallback are inherited. Power has no SDF geometry in this collector and reports
no measurement backend; Tripod retains its previous unavailable-guard gate.
The base class and factory reject `compiled` before device allocation. Runtime
graph-compilation calls, specialization helpers and temporary compiled caches
have been removed. The CUDA measurement module accepts source owners and tensor
inputs independently of Teacher or Student model classes.

The four math files under `cuda_measurements/` are byte-identical copies of the
private v6 candidate. The sealed profile verifies the copied sources, inherited
geometry source, production wrapper/build script, private contract, all private
sources/evidence, version, SM89, and binary before loading. GPU input checks
require SM89. Strided inputs accepted by the legacy capture are copied to
contiguous GPU buffers only at the C ABI boundary. Radius is allocated once in
the collector, while Reset still verifies the actual solver radius.

- Binary: `57baa26f73b2973bee46d3e27faa021cb061e37ea1fb4ac366db3a0b7280942a`.
- Production manifest: `0b50ff3be1b6d0d05084173e8505d68e581765d4db3f0035b53bee387a69fc31`.
- Production wrapper: `348316f345495084fe6c54ae4f95181ec7c33dcab3576c35202719b92360dd7b`.
- Eager geometry source: `8a97b1f56b700d59129e6ca73dfcc491685717ba1e023e5a7f4751166020397f`.
- No integration rebuild, dependency installation, Isaac launch, commit or push.
- Geometry uses no `torch.compile`; ordinary PPO Adam can independently import Dynamo.

## Verification

CPU command (CUDA hidden): `PYTHONPATH=src:. python -B -m pytest -q tests/test_dh116_cuda_geometry.py tests/test_dh116_cuda_geometry_gpu.py tests/test_dh116_tensor_geometry.py tests/test_dh116_tensor_geometry_fusion.py tests/test_dh116_tensor_collector.py tests/test_dh116_tensor_collector_replay.py tests/test_dh116_tensor_contact_replay.py tests/test_dh116_regex_cached_readback.py`.
Actual result after compilation cleanup: **81 passed, 14 skipped**. The skips
are explicit GPU opt-ins. The dedicated compilation/receipt tests pass 37 cases.
TDD recorded five compilation-ban failures, six missing/failed cache audit
failures, and a tampered guard-preservation proof failure before implementation.

Reserved GPU2 command: `CUDA_VISIBLE_DEVICES=2 DH116_CUDA_GEOMETRY_TEST=1 PYTHONPATH=src:.:tests python -B -m pytest -q tests/test_dh116_cuda_geometry_gpu.py`.
Actual result after compilation cleanup: **13 passed**. The entire GPU fixture
replaces the runtime compilation entry with a function that raises immediately,
so both eager and C ABI captures prove they do not call it. This exercises the production factory with the
44-hull/28,893-point real source corpus at N2/N128, every result field at absolute
tolerance `1e-12`, owned repeat captures, strided inputs, original finite/unit
guards and epoch invalidation, metadata/shape/SM checks, original ±1/2/3 μm table
boundaries and extreme finite FP64/FP32-overflow pad routing.

Before compilation cleanup, an independent runner also used the production loader with the private test
imports redirected, without changing historical files: **19 unit tests,
196 stress cases, 16 full capture cases passed**. The maximum capture error was
`1.7763568394002505e-15`; stress error was `1.3322676295501878e-15`.
N128 full-capture replay median was `0.004186174692586064 s`; this is a geometry
replay measurement and says nothing about end-to-end native throughput. These
earlier loader artifacts retain their original manifest identity and are
historical; the post-cleanup 13 GPU cases above use the new manifest.

Private artifacts remain outside Git under
`/mnt/mydisk/xk/candidates/dh116_production_cuda_geometry_verify_20261124`:

- `production_loader_result.json`: `60efb35e00b5d18b5d90b1381cd6e85fc08836787c8b3d4507fbda4162549bc9`.
- `production_loader.log`: `2fa4eb587c0d1501f2caa305778480470ceed82e43c9b65219545d7713df1499`.
- `validation_benchmark.json`: `8834e500e4a4388e52fde53daa1fa2a5c6c394b551e252c15c1d76d82460678b`.
- `production_gpu.log`: `0f6a2ff658364d3e68f368246a8f6be31c09abaf17f6fa1233d149ef22cf3227`.

The rejected compiled experiment was preserved before editing under
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/cuda-no-compile-cleanup-20261124-ye8_5ag0`.
It contains the full prior source, test and manifest snapshots, Git diff,
9ddb eager baseline, red/green logs and fresh regression logs. No historical
failure evidence or binary was deleted or overwritten. Its historical failure
was AppClose returning followed by a Py_Finalize/Timeline SIGSEGV; it granted no
qualification.

`guard-preservation.json` verifies byte-for-byte identity of ten original
definitions, including `capture`, eager hull/pad measurements, CPU fallback,
source metadata and readiness guards. The profile authenticates this proof and
the compilation archive, tying the old private oracle to the current sealed
source rather than pretending their full-file hashes are still equal.

- Archive manifest: `d972ff4f3ca291e6f819be258c59708813e8a0cfdd6bccd005e71a187606bf42`.
- Guard preservation: `97a964efd578acd3434b5e6317a594c2a66f1bc6a185023167e01c473a67a07a`.
- Fresh `no-compile-cpu-regression.log`: `29a2395156514dc9f2a03c9aead46cb466b53a382ec1a0ad65f53b52e876f293`.
- Fresh `no-compile-production-gpu.log`: `72be2d971074b0aacd2fbf73e9b052f64f6707ffadc0e6760c543b40d7dc3f83`.

## Native evidence API

`CudaGeometryProfile.load().identity()` reports the binary, manifest and source
hashes. `verify_measurement_native_receipts([(n2_outer, n2_sha), (n128_outer, n128_sha)], style="pinch")`
validates two SHA-pinned real outer receipts. Entry CLI parameters are repeatable
`--measurement-native-receipt PATH --measurement-native-receipt-sha256 SHA`.
The parent and native worker each verify them. Without evidence, CABI `--updates`
fails before creating output or launching a worker; two-window probes can run.

Validation requires actual debugger/inferior exit zero, JSON null signal,
`forced_exit=false`, fixed sibling native result/AppClose/log SHA matches,
natural AppClose and ownership release, matching style and current full geometry
identity, `bounded_diagnostic` scope, N2 real nonzero hand contact/next-SDK
reaction with isolated Reset and continuation, and an independent N128 18-SDK
two-window throughput probe. Both receipts must match the selected native
readback mode and each other's source-cache metadata. Raw receipts cannot admit
`official_cache`. Signal 11, string `"None"`, duplicate N2, wrong style/profile,
missing contact/Reset/continuation and tampered hashes are rejected by tests.
An `official_cache` N2 receipt additionally requires the real batch
`readback_audit()` summary: `enabled=true`, `diagnostic_only=true`, positive
integer `side_snapshots`, zero mismatches (the API's empty list), and exactly
two extra raw reads per side snapshot. This represents raw-before/cache/raw-after
byte comparison; a self-declared enabled flag without raw reads is rejected.
N128 may remain a timing-only probe without those extra raw reads.

The evidence result and geometry profile retain `native_ready=false` and
`teacher_admission=false`; this gate grants no grasp or Teacher qualification.
No new native receipt was created by this integration work.

`scripts/build_dh116_cuda_geometry_candidate.py NEW_PRIVATE_OUTPUT [--nvcc PATH]`
requires a new output directory outside the repository, refuses existing
directories, uses existing NVCC12.8.61/SM89 with `--fmad=false` and the original
precision flags, and writes a separate rebuild manifest requiring fresh
validation. It never replaces a historical binary or updates the production pin.

## Final receipt source-freeze and Timeline gate

The final receipt gate additionally requires SHA-pinned native
`source_freeze_audit.json` and `timeline_release.json`, plus the fixed parent
`attempt.json` hash in `outer.artifacts["attempt.json"]`. The audit must report
the agreed version, startup checked, admissible, no drift, and no process-exit
or qualification grant. For every fixed `SOURCE_NAMES` entry, the audit hash,
attempt inventory and current repository file must agree. The list contains
22 shared runtime, control, mesh/pair/rod/source, cache, batch/setup, lifecycle
and actual entry scripts; documentation is outside this comparison. The list's
module is itself sealed in the candidate profile.

Timeline evidence must report STOP committed, weakref supported, wrapper dead,
local release verified, and native-release/process-exit/qualification flags all
false. Missing artifact hashes, attempt substitution, false admissibility,
nonempty drift, mismatched or missing source entries and any unsupported/alive
Timeline wrapper are rejected. The checked result retains all three new
artifact hashes for downstream review.

TDD first observed **23 failing source/Timeline negative cases**. The final CPU
profile command `CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. python -B -m pytest -q -p no:cacheprovider tests/test_dh116_cuda_geometry.py`
then passed **54 tests**. These are explicitly synthetic unit fixtures; no real
receipt or source-freeze audit was forged. This correction changed no runtime,
geometry math, native binary or physical parameter.

In the cleanup evidence directory above:

- `final-receipt-source-freeze-red.log`: `b2b24b434e2ae1ccf8aa3c7091f6a5f1393d1b2abc57a41ebcad2a8a5c64c708`.
- `final-receipt-source-freeze-green.log`: `26358b6bbedaa3c31cdff2c576e3141cf2f8ad58e72b278b3a12ce751cc19997`.
- `final-receipt-profile-green.log`: `336e7af304a7a6f7c137851c0d6748159fe925d40bcec355ef018cd7853e64c0`.
