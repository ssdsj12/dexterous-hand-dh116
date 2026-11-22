# Bounded owned JSON decode reuse design

Goal: reduce repeated CPU deserialization during actual Teacher initialization without changing input,physical qualification or native budget.

Alternatives: keep stdlib decoding on every independent verification (correct but measured2.924s per large document); swap to existing orjson (faster but changes big integer/NA/surrogate semantics); reuse the exact existing decoder result for identical verified raw bytes (selected). Real198718823-byte Reset document occupies42447124 decoded bytes;deepcopy0.041-0.083s with exact numeric digest.

Implementation: extend tabletop_fast_owned_codec.py with decode_owned_json(raw). Every caller still receives an owned tree. Cache only large successful pureCPU trees under actual SHA256(raw),bounded to64MiB total and2 entries,never retaining raw bytes. Require independent deepcopy on both miss and hit. Use one private lock for parse/insert/evict/copy bookkeeping. Small or oversized trees follow existing stdlib+decode_owned directly;errors remain errors and are never cached. Account Python containers/scalars and owned NumPy allocations,including dtype. No CUDA,Torch,owner,profile,receipt qualification or derived manifest objects enter the cache.

Integration: only the five owned JSON reads in TripodResetManifest._derive use the helper. They consume checked_file bytes;each load/verify still verifies all26 inventory items and current physical sources and executes all geometry/Reset/common66 validators. Derived manifests are never cached.

Validation: failing owner tests for first-return mutation isolation,hit isolation,large ints/negative zero/nonfinite/dtype preservation,bounded eviction,oversize bypass and errors. Extend existing real-manifest test to prove one large parse across independent full revalidation. Benchmark complete baseline/current derivation,cold/hot decode and RSS;report native gains only after a new frozen attempt. Original24D/371D,fixed stereo,Reset,assets,success standard and old models remain unchanged.
