# DH116 per-environment DeformX pair capture

The separate CUDA 12.8.61 build for SM89 adds a bounded raw-contact bucket per solver environment. CUDA chooses the bucket from its own environment index. The legacy single-rod API remains single-rod. The new API returns a distinct row count, overflow flag and raw witness per bucket plus per-lane aggregate forces and wrenches. The live Teacher still uses the reviewer-pinned original binary.

| Grasp seed | Solver lanes | Raw pairs per tested step | Slower of two step and readback times | Capture on/off difference | Max mixed pair-vector difference vs independent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinch | 128 | 62,484 | 7.89 ms | 5.96e-08 | 7.24e-05 |
| Pinch | 512 | 251,412 | 28.09 ms | 5.96e-08 | 7.24e-05 |
| Tripod | 128 | 3,048 | 3.87 ms | 0 | 0.000242 |
| Tripod | 512 | 12,264 | 10.34 ms | 0 | 0.000242 |
| Power | 128 | 3,048 | 3.96 ms | 0 | 0.000242 |
| Power | 512 | 12,264 | 10.39 ms | 0 | 0.000242 |

The nine GPU 3 probes use the respective Pinch, Tripod and Power Reset evidence, with 2, 128 and 512 solver lanes. Each probe used two measured steps, validated every lane against its own aggregate forces and wrenches, then verified a zero-step local Reset, invalidation of the stale ledger, and another step after Reset. A separate overflow witness verified that an overflow on lane 1 did not affect lane 0. All cases are solver-only. The 39 DeformX mesh and batch regression tests passed on GPU 2.

Independent-shared parity at 1e-5 is still open. Exact IDs, contact counts, sampled mesh poses and sampled normal-force values matched in the two reference lanes. The largest aggregated force- and wrench-component differences were about 4.68e-5 for Tripod/Power, and Pinch node-force max difference was about 4.21e-5 N. Pair vectors include quantities with distinct units. The installed solver tests describe cross-thread Gauss–Seidel neighbor reads; the different layouts have not been demonstrated equivalent at the original current guard and reward boundaries.

## Native Isaac

- GPU 3 Pinch 64 using compact collection committed window one in 97.659 seconds versus 267.383 seconds in the older acquisition. Phase times included 60.986 seconds of native completion, 16.190 seconds of observation/guard, 8.456 seconds of admission, and 0.271 seconds to close the successful evidence window. Peak sampled VRAM was 3326 MiB.
- Window two stopped at SDK 15, after 832 total lane solver calls, when lanes 6, 39 and 53 hit the original `right:non_target_not_proven_clear:source:finger13_link` guard. At SDK 15 the three conservative lower bounds were −11.54, −7.47 and −5.39 µm. They remained positive in each preceding measured step of that window. The rejected window preserved 701 raw evidence records, and natural exit 1 recorded zero PPO updates.
- Tripod and Power independently captured original 17-slot data at SDK 2 in current recipes, both exited naturally with code 0, and both sealed and reloaded their actual Reset manifests. Tripod's separate full-tree reread later rejected a changed readiness JSON hash after the report was edited. Its training manifest's source subset reverified against the original native run.
- The three-grasp collector is still pending. The timing measurements and shared solver proof do not advance the grasp/Teacher admission flags.

## Reproduction

Apply `scripts/patches/deformx_shared_pair_buckets.patch` with `git apply --unidiff-zero` to a copy of the previously prepared mesh-observation source tree. Keep the original binary and source tree available for parity. Build the copy using the verified 12.8.61 toolchain, Python 3.10.12 and `CMAKE_CUDA_ARCHITECTURES=89`, then execute `scripts.audit_dh116_deformx_shared_pairs` for each grasp and size with the archived Reset receipt and compiled candidate SHA. The Pinch CLI also accepts the separately hashed archived two-lane smoke ledger. Raw evidence and the new binary stay outside Git.
