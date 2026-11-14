# Mesh reset, health and pair-v2 owner preparation

This isolated package wires the compiled observation backend into the existing projected-Pinch owner. It runs CPU tests only. There is no Isaac/native launch, build, optimizer update, publication, new recipe, or claim that native reset, contacts, Slip, grasp or Teacher passed.

Base: 8e2dbd5884b8e735ac385a505a0cde7d550fdb11. Worktree: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-reset-health-owner-20261114. Branch: codex/mesh-reset-health-owner-20261114. The parent agent owns the separate entry, worker, recipe and registry integration.

## Actual wiring

The new operation is prepare_dh116_mesh_reset_health_projected_pinch. scenario_scope is the outer owner scope: it applies the explicit mesh-observation source profile, the original health/projected chain, the actual-backend radius checker aliases, and the complete v1 pair reader. It adds PairV2EvidenceMixin to the same director exchange exactly once. The source profile verifies the actual 8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340 binary, 72 source files with five observation-only deltas, patch and build/import evidence. Old defaults restore after nested scopes and exceptions. No frozen module is edited.

The explicit read_complete_pair_capture binding belongs to this production scope. A review found that the projected entry does not pass through the two older origin/complete entry wrappers which already perform this binding. The first CPU fixture had added that binding itself, hiding this production integration gap. A new scope test first failed, the binding moved into production scope, and the test-only binding was removed. The disk audit continues to require retained original v1 raw arrays and exact projection agreement with v2; it never fills missing original pairs.

The reset factory calls the original full hand/rod reset and the separately verified zero-step eleven-mesh rebuild through MeasuredMeshReset. It saves mesh_reset_state.json with owned source plan, actual before/after mesh getter packet, and the completed original reset receipt. Getter failure after a completed reset preserves that real receipt, poisons the rod and retains a pure error string. Callback/native aliases are cleared in finally. Evidence files use exclusive creation, flush and fsync.

After every original complete SDK/rod exchange, the observer calls the actual same owner's contact_pair_samples_v2 exactly once. It captures before/after actual SDK time, runtime identity and native solver time without stepping. Online work checks loaded module/class/instance, exact schema/dtypes/shapes, finite values, capacity/count and unchanged clock; it does not recompute source inventories, pair geometry or the heavy parser per row. The original v1 solver/coupling path remains active.

Any new getter or boundary failure adds error/unknown evidence to the completed row. The original guard has priority. The existing owner durably records that physical row and control end before stopping. No completed row is discarded to conceal an observer failure. Health remains the original small live getter; R0 evaluation remains post-exit.

inspect_observations(output, complete=True, parent_lifecycle_qualified=False) independently rereads the journal, reset JSON, direct eleven-mesh snapshot, every v2 packet, retained complete v1 source, same-owner material-director anchor and original health packet. Sequence validation requires one reset at rod tick zero, then SDK +1 and rod +12 for all 232 rows. Complete source qualification additionally requires the caller's separately validated parent lifecycle. Partial inspection retains real source errors and never clears the durable terminal to obtain full qualification. The default parent gate is false.

Returned narrow source flags are reset_mesh_qualified, substep_contact_frame_qualified and same_clock_health_getter_qualified. They are not standalone native qualification. native_qualification, contact_success_qualified, negative_contact_qualified, slip_qualified and teacher_admission remain false. The owner action schedule is unchanged: two setup plus 232 measured SDK steps, twelve solver substeps per measured row, one environment and zero PPO updates. No budget extension is introduced.

## CPU verification

Environment for all commands below:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh

The SDK/solver transport in the tests is explicit CPU doubles. Original builder, reset, controller, episode, journal and shutdown Python paths execute, and actual locked SDF files/table sampler are read. CPU pair generation is synthetic and cannot qualify native contact. The v1 and v2 getter doubles share the same generated raw source packet, rather than reconstructing v1 pairs inside production code.

The full normal owner test passed once: 1 passed in 444.11 seconds, producing all 234 SDK steps, 232 measured physical rows, 29 complete control windows and application-close evidence. This run used the identical complete-pair reader through a test-level binding before the final production-scope correction. It is retained as full-owner functional evidence, not misreported as a second final-scope run. The final scope itself is tested by the new nested/exception guard and all three actual failure-prefix owner cases. Per parent instruction the expensive full 234 CPU run was not repeated for this binding-location correction.

Commands:

    -m pytest -x -q "tests/test_mesh_reset_health_owner.py::test_real_full_owner_or_failed_actual_prefix[None-False]" --basetemp=E/full-final-tests
    -m pytest -x -q tests/test_mesh_reset_health_owner.py -k "not real_full"
    -m pytest -x -q "tests/test_mesh_reset_health_owner.py::test_real_full_owner_or_failed_actual_prefix[pair-False]" "tests/test_mesh_reset_health_owner.py::test_real_full_owner_or_failed_actual_prefix[pair-True]" "tests/test_mesh_reset_health_owner.py::test_real_full_owner_or_failed_actual_prefix[reset-False]" --basetemp=E/prefix-exact-tests
    -m pytest -x -q tests/test_complete_pair_capture.py tests/test_deformx_pair_observations_v2.py tests/test_mesh_observation_profile.py tests/test_flexible_health_postexit.py

E is /home/xk/.local/share/dh116-autonomous-3c80166/evidence/mesh-reset-health-owner-cpu-y5kv2kvi. Final-scope results: 15 lightweight tests passed in 4.48 s, three exact owner failure prefixes passed in 132.59 s, and 119 related regressions passed in 97.01 s. Thus 137 CPU checks passed on the final production-scope wiring, plus the separately retained earlier one-test full 234-step owner pass. The final-scope prefix runs exercise SDK11 getter failure with and without an old guard (nine real completed rows, two ended control windows), and SDK2 reset getter failure (zero measured rows, completed original reset retained).

The initial combined CPU run retained 3 failures and 1 pass: the three failures were missing retained complete v1 raw data at post-exit, while the producer had already completed its requested physical prefix. Those logs remain; no production gate was relaxed. A later broad -k expression accidentally also selected the normal owner through its module name; that CPU run was deliberately interrupted, and prefix-final.log is an interrupted diagnostic, not a passed test. The replacement uses three exact node IDs.

## Durable CPU artifacts and limitations

The retained passing full-output directory is E/full-final-tests/test_real_full_owner_or_failed0/closure.

- physical.jsonl: 488608245 bytes; SHA256 4ec235ea3f68e9273a761de294641663106595fa8efe727dea0e2061529287d2.
- mesh_reset_state.json: 52704370 bytes; SHA256 e6f286eadc7d10491336868ee356c90a19562961f5559a4716fe0eee1ae1a6cc.
- sdk_clock.jsonl: SHA256 09e8c4a6b193305d3668aa7abc779c411483c91d283966955138015f331c2a3c.
- result.json: SHA256 3553fe764b65d859ae5884ca7603e6a002a97c5987478ea2a1daeec91de9b323; failure is null.

The reset JSON retains the full source-bound grid plan, about 53 MB in this CPU fixture. This adds bounded I/O to the future native owner; there is no native timing evidence for the new instrumentation. The 444-second CPU test includes repeated independent large-journal validation and is not a native runtime measurement. The separate root entry/worker must freeze all new and inherited source dependencies, including tabletop_complete_pair_capture.py, and enforce the original 234-step/180-second budget. No native execution is authorized or performed by this subtask.

Camera-agent read-only review found no remaining identified blocker in evidence SHA 431d265e9a1ac94df5ee70c14ecc7157f5c187f90cb4d05fa1fa2b6f19925969 and operation SHA ffbba0d78a802da1e53e1922815dfd4b8a1b8363045da29552e70de0a5d9fe2c, including the final production-scope binding and unchanged old backend defaults. That review did not execute these CPU tests or native. Repository.verify_protection() completed normally: 528 protected source/script/config files and two historical Checkpoints remain unchanged.


## Final source and evidence hashes

| File | SHA256 |
| --- | --- |
| scripts/prepare_dh116_mesh_reset_health_projected_pinch.py | ffbba0d78a802da1e53e1922815dfd4b8a1b8363045da29552e70de0a5d9fe2c |
| src/go2_pvcnn/control/dh116/tabletop_mesh_observation_evidence.py | 431d265e9a1ac94df5ee70c14ecc7157f5c187f90cb4d05fa1fa2b6f19925969 |
| tests/test_mesh_reset_health_owner.py | ab5ea80d496a062a117c1baedff2a3df7aecf3b9c4feab08942b4e6599eba134 |

| Log in E | Actual result | SHA256 |
| --- | --- | --- |
| owner-red.log | 14 expected initial failures | d17427c23aa1ae92afed7668105639fdf99fc0c028d85b4e17d9834a5faba16c |
| owner-reset-first.log | 3 failures / 1 pass; missing complete raw source identified | 3eb821a55cc548e70deb1dd583bdc6b59b0a677b4d50a5df95cca64b8d9f219a |
| full-final.log | earlier full owner: 1 passed / 444.11 s | a3ab70b4e57e095eed9368fcd2fb4358f3bc69cf72a9eb5011819f1880ae7786 |
| scope-binding-red.log | production binding negative: 1 failed / 3.42 s | d78be99159b26ea11c4f97a97d741268d1a801f2168e56cd43a19b7d1c234ff9 |
| scope-binding-green.log | final light scope/observer: 15 passed / 4.48 s | ef2f357d3a86fe69a28a9024a230a7689eeaf7ea7d2ff178dabaeddbd3875aab |
| prefix-exact.log | final actual Python owner prefixes: 3 passed / 132.59 s | 996aa7e5b0f107ad34106d53d4849cbfe2ce7e77775c5f3917699b9372727fe5 |
| final-regression.log | final related CPU regressions: 119 passed / 97.01 s | 481f08d1ace7ca90aa8b020e3defbad00e01a185d2a463a65241ddf187e46c8a |
| protection-details.json | 528 protected files + 2 Checkpoints passed | 70d7df7b1e8d1d6d704648ba2facdb9a8643af2d43357aa866c547e30a7b7073 |

The complete_capture projection source, new operation and observer, existing reset/source profile, v2 parser, original health/director/owner chain and tests must all be included in the parent's independent entry fingerprint. This commit supplies no scheduler SHA or recipe SHA: those remain the parent agent's separate package.
