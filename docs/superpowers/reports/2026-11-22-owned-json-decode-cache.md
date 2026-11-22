# Teacher initialization: bounded owned JSON decode reuse

## Problem and actual change
The real198718823-byte Reset JSON takes2.924s to parse/decode but its pureCPU tree occupies42447124 bytes and deepcopy takes0.041-0.083s. Repeating this during separate required provenance/geometry checks consumes the finite actual Teacher wall budget.

decode_owned_json in the existing tabletop_fast_owned_codec now retains successful decoding of large identical raw bytes. The stdlib JSON parser and existing owned decoder stay authoritative;orjson is not used or installed. SHA256(raw),weak parser/decoder identities and the integer policy form the key. Raw bytes,exceptions,tracebacks,owners,profiles,derived manifests and qualification outcomes are not retained. Every return is an independent deep copy,including the first miss. A private lock serializes parsing,LRU and size bookkeeping.

Eligibility starts at32MiB raw. The cache has at most2 entries and64MiB estimated retained decoded objects and entry metadata;it is NOT a raw/temporary/deepcopy/process RSS cap. Whole Python containers,scalars,owned NumPy allocation,dtype and key/size metadata are included. Non-pureCPU/oversized/small inputs keep ordinary decoding;errors remain uncached errors.

Only five owned-JSON reads in Tripod _derive use this helper. They still consume the exact checked_file bytes. Every load/verify independently checks all26 inventory files,current physical sources,Reset,numeric state,common66,geometry and support clearance. No physical check or source boundary is skipped;no derived manifest is cached.

## TDD and review
9 new owner cases cover first/hit mutation isolation,typed arrays,large integers,negative zero,NA,LRU,total size eviction,oversize/small bypass,errors,concurrent ownership and parser instrumentation owner collection. The original eight cases failed before helper implementation;the real manifest integration failed with two large parses before hookup. A corrected weak-owner test failed with a strong function key and passed after weak references;its monkeypatch context is closed before the GC assertion so the fixture itself cannot retain the owner.
The existing real-manifest test still requires every inventory item to be checked separately by load and verify,while only one large JSON parse occurs. No duplicate scene fixture or new physical interface is introduced.
Final10-file CPU regression:213 passed,0 failed in104.92s. Logs and hashes remain outside Git and are recorded in readiness. Independent read-only review found no remaining blocker. Antigravity remains quota-limited;no new external review is claimed.

Command:
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_native_teacher_owned_json_cache.py tests/test_fast_owned_codec.py tests/test_native_teacher_owned_codec_parity.py tests/test_owned_codec_comparison.py tests/test_tripod_reset_manifest.py tests/test_native_teacher_tripod_admission.py tests/test_native_teacher_static_manifest_handoff.py tests/test_native_teacher_tripod_frozen_worker.py tests/test_native_teacher_actual_launch.py tests/test_dh116_work_packages.py

## Whole-path CPU measurements
Independent baseline/current CPU processes,three complete actual-source derivations per mode:
- Baseline cold9.518s;repeated median8.956s.
- Cached cold9.405s;repeated median5.683s.
- Full returned digest matches exactly:f68160b0375443a31e169c3d9854913e397f0a6f73dc975503e9c02f81d37687.
- Retained cache42448181 bytes,1 entry;limit67108864 bytes.
- Observed process peak RSS1986736KiB baseline /1946152KiB cached. Single isolated runs do not establish a general RSS saving.

The actual scripts/python.sh CPU launcher at DH116_NUM_THREADS=4 reports torch_threads4. Complete static-contract times14.206s,8.978s,9.124s,identical source/config/manifest signatures. Only CPU budget I/O is synthetic in this benchmark;no native lease,Isaac app or Solver is started.

This demonstrates repeated CPU initialization savings,not guaranteed native wall success. First parsing still costs time. The next run must record actual initialization and full update timing.

## Preservation and next execution
0 native runs/0 PPO updates on this implementation. Spent v10 remains failed and its recipe/fingerprint is unchanged. Pinch,Tripod and Power remain separately unqualified for full grasp success. R2/Force Proxy and formal BC/DAgger stay gated.
528 protected files,3 original checkpoints,two fixed RGB cameras,24D/371D,original Reset/assets/material/physics and Lift30mm/Hold5s/Slip0.5mm remain unchanged.
Next is a separate4-thread Tripod smoke8 candidate at the same original66 SDK/8 control/768 internal/1-update/180-240s bounds,after source publication,current resource checks and exact fingerprint binding. No same-version retry or budget expansion.
