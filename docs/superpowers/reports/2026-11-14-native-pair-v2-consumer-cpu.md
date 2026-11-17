# Native pair v2 CPU consumer and validation

Date: 2026-11-14. Base: `c9e8a703fa3651d064fc906a521323936bd6e4ad`.
Worktree: `mesh-pair-observations-v2-20261114`; branch `codex/mesh-pair-observations-v2-20261114`.

This package adds only a CPU parser/validator, synthetic CPU tests, and this report. It changes no original API, runtime, controller, geometry, safety threshold, source profile, recipe or default backend. No GPU/Isaac/solver, extension import, native API call or additional build ran for this package.

## Concrete interface

New module: `src/go2_pvcnn/control/dh116/tabletop_deformx_pair_observations_v2.py`.

`parse_pair_observations(raw, *, dt_s, identity, source_instance_id, mesh_names, source_profile, evidence_kind)` consumes an already returned packet. `identity` is the exact existing ten-field `tabletop_bimanual_table_contacts.IDENTITY`; `source_instance_id` is an explicitly supplied owner ID. Evidence kind must be `synthetic_cpu` or `native_api_return_unqualified`.

It returns owned raw arrays/scalars and their digest, source-profile and producer-source provenance, the original `legacy_capture`, and separately sorted `observations`. Mesh names are owned once, including generator input. The old `complete.project` is directly reused, so capacity/attempt/overflow, integer substep/node/mesh ranges, duplicate keys, outward normal, compressive force, step-averaged node/mesh force/torque conservation and original impulse projection remain required.

New raw fields must match the exact separately compiled `contact_pair_samples_v2` schema, including float32 arrays and int32 indices. There is no fallback to endpoint pose, source-derived interpolation, missing depth, unknown normal force, or a reconstructed capacity. Actual `P.pos/P.q` must be finite, unit XYZW and consistent across all nodes of a mesh/substep. The original atomic array order is not assumed sorted: one shared lexicographic ordering aligns all additional columns with the legacy pair order.

`validate_pair_observations(capture, *, source_profile)` reopens the retained raw packet, reruns the parser and original gates, and compares the entire rebuilt result. Changing a projected row, original raw field, identity, source, or qualification flag is rejected.

`PairObservationSequence(source_profile=..., reset_identity=..., source_instance_id=..., mesh_names=..., expected_substeps=..., evidence_kind=..., dt_s=1/480)` stores only owned CPU metadata. `consume(raw, identity=..., source_instance_id=...)` requires:

- the same run/environment/episode/reset/rod/owner;
- exactly one new SDK step and exactly the original solver substep increment;
- actual solver time progression accepted by the existing `solver_clock_matches`;
- raw owner times before/after exactly equal that frame's actual `solver_time_s`.

Duplicate consumption, stale packet time, skipped frames, owner/episode changes, and invalid captures poison the sequence. A new reset requires a new consumer. No SDK object, native reference, step or reset is retained/performed by this class. Identity assertions cannot themselves authenticate a caller-supplied owner ID; the later native owner must obtain it from its actual instance and bind the surrounding clock/readback boundary.

## Numerical meanings kept separate

Each observation retains `native_surface_gap_m`, the signed native `phi-R` result; it is not the original pad atlas's `surface_tolerance_m=0.001`. A measured negative gap outside that atlas tolerance remains a valid raw overlap measurement, not effective pad membership. The parser does not apply or expand the atlas. `original_region_membership` and `effective_support` stay null.

`native_penalty_normal_force_n` is the native unaveraged `N=-k*gap` value before friction. `dot_total_force_normal_n` is the original validator's projection of the recorded total force onto its recorded normal. Both are preserved, with their difference; neither overwrites the other. Float32 evaluation and normal normalization can make their numerical representations differ. The parser checks native N is finite/nonnegative; it does not independently reconstruct the penalty coefficient or claim to prove N from an endpoint. Neither a negative gap nor native N alone establishes bilateral effective support.

The torque cross-check uses the recorded native mesh origin and original force:
`cross(contact_point_world_m - mesh_position_world_m, -force_on_rod_world_n)`.
The actual compiled `contact.cuh:200–207` defines `r=(pos-P.pos)+dx-R*n`, `tq=cross(r,-fm)`, and stores `point=P.pos+r`. Thus adding R again would be wrong. The check reuses the existing aggregate numerical comparison tolerances (`atol=3e-4, rtol=5e-5`) to allow float32 world-position addition/subtraction; it changes no physical threshold. A dedicated nonzero tangential-force fixture with R=0.005 m locks this source relationship.

Owner times retain the actual native host metadata at readback, not SDK nominal time. `substep_timestamp_s` must be null; none is manufactured. The existing legacy `substep_dt_s` is the configured duration used by the unchanged impulse formula, not a measured substep timestamp.

A complete empty packet is accepted as a complete zero-row capture, while **negative-contact qualification remains false**. Positive contact, native qualification and Teacher admission also remain false for every result, even if a supplied profile dictionary claims success.

## Source identity and integration boundary

The consumer binds the new backend path/SHA `8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340`, full 72-file tree `34741b437191fd66bf151331f20476a86f8708414dcc2782a8c69dd15b0efb64`, and five-file patch `1918ebebfde3a00a9c0c3bf979313674d8015f9f5fa42220767b562d169bde41`. It is compatible with the separate physics-agent `tabletop_mesh_observation_source.source_basis()` fields; that module is not copied or modified here.

A dictionary/digest is not proof of native execution. The result explicitly states `source_profile_verification=required_from_outer_source_basis_and_native_receipt`. The future owner/audit must run the complete source profile verification and bind the actual module/class/instance and clock. It must also preserve a returned raw packet and failure diagnostic before stopping if parsing fails. This pure consumer does not own journal writes or grant execution permission.

The earlier compiled/import-only binary is background provenance, not new measured data. No historical v1 pair ledger can be upgraded by filling pose/gap/N from later endpoints. No native recipe or authorization is added here.

## CPU verification

Evidence:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pair-observations-v2-cpu-h5hc7m0z`.

TDD evidence includes a missing-module collection error, one real generator-consumption failure (the second pass previously lost mesh names), and three constructor failures for string/unordered mesh-name inputs. Fixes own the names once and reject unordered/string sequences.

Final related regression: **124 passed in 3.52 seconds: 58 new + 66 existing**.

```text
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short tests/test_deformx_pair_observations_v2.py tests/test_complete_pair_capture.py tests/test_dh116_deformx_pair_capture.py tests/test_dh116_flexible_pair_contact_diagnostics.py tests/test_deformable_pad_region.py tests/test_deformx_mesh_observations_preparation.py
```

Tests cover source-shaped unsorted data, shared new-array ordering, complete original projection, aggregate/normal/force checks, actual pose/torque consistency, strict unknown/error/overflow rejection, all episode identity fields, repeated/incorrect frame consumption, raw ownership, JSON codec round-trip, independent disk revalidation, direct compiled-schema matching without importing the extension, and qualification remaining false. Every new packet fixture is explicitly synthetic CPU evidence.

`Repository.verify_protection()` exited 0. Relevant logs:

| Artifact | SHA256 |
| --- | --- |
| `related-final-02.txt` | `61f0252f4e645111b983080887ad59054a64369c3aad2e868322387dea546e0a` |
| `red-missing-module.txt` | `6a620f78a57fd4923a0a2e26598bc018234a6f6c27cb3d19b3cb37cb5cefe34c` |
| `red-generator-binding.txt` | `2ca9bfb74d2c596fb9e1758bf454ecf24da1c6c941c1f7686cc4b0e3dc2aa421` |
| `red-sequence-names.txt` | `36d8f72fba72e1ea45df8ff2d94915c3d178d454d6f8036e09c74124c6886e75` |
| `protection.txt` | `af5d8a21858f42808c9bcc17012b77bcc2fd6ab3b071dbd4a05f13f410470541` |

Physics performed a source-only review, with no remaining identified blocker. Its initial torque concern was withdrawn after checking the exact compiled source definition; the numerical test now covers that distinction. That review did not execute native code. Root owns integration, freezing and any future bounded native experiment.
