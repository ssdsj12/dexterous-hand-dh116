# Bounded owned JSON decode reuse implementation plan

> Execute with executing-plans in the current isolated Ubuntu worktree;independent read-only review is already active.
**Goal:** Reduce actual Teacher startup deserialization while retaining every qualification check.
**Architecture:** Private size-bounded CPU decoding cache keyed by exact raw bytes;fresh independent copies and complete source/geometry checks every time.
**Tech stack:** Existing Python3.10.12,stdlib json/copy/hashlib,NumPy1.26.4;no dependency changes.

- [x] Measure real decoded memory,parse and deep-copy cost;review alternatives and private ownership.
- [x] Add failing tests in tests/test_native_teacher_owned_json_cache.py. Verify first and hit return mutations cannot poison cache;preserve large ints,negative zero,NA and typed arrays;exercise LRU,size bypass and parse errors.
- [x] Implement decode_owned_json in tabletop_fast_owned_codec.py with64MiB/2-entry private cache and fresh copies.
- [x] Replace only Tripod owned JSON deserialization with decode_owned_json(documents.pop(name));preserve checked bytes and all validators.
- [x] Extend existing real-manifest test:all26 files verify per load and verify,but large raw parsing occurs once.
- [x] Run CPU cache,codec,Tripod manifest,source admission,current launch and Registry regression. Preserve RED/failure logs.
- [x] Benchmark complete baseline/current derivation plus cold/hot CPU cost and RSS;reject unsupported performance claims.
- [x] Update readiness/report,verify528 protected files and3 historical checkpoints,commit in Ubuntu and publish same commit through Windows.
- [x] After demonstrated benefit,freeze a separate finite native recipe;do not refresh or rerun spent v10.

Commands: /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_native_teacher_owned_json_cache.py tests/test_native_teacher_owned_codec_parity.py tests/test_tripod_reset_manifest.py
Broader admission/launch/Registry regressions follow only after focused checks pass. All logs/raw data stay outside Git.
