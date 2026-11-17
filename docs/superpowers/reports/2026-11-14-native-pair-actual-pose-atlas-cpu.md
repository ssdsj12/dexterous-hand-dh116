# Native pair v2 actual-pose atlas classifier — CPU preparation

Date: 2026-11-14. Base: `8e2dbd5884b8e735ac385a505a0cde7d550fdb11`.
Worktree: `native-pair-atlas-20261114`; branch: `codex/native-pair-atlas-20261114`.

This package adds an offline consumer only. No native attempt, solver/API invocation, GPU work, Isaac application, training, recipe, registration or live guard change was performed. Existing source files, original atlas, thresholds and consumed recipe identities are unchanged.

## API and source boundary

- `load_original_atlas(project)` reuses the existing `audit_dh116_pinch_initial_pose.load_geometry` and `tabletop_effective_pad._patch`. It loads both original USD geometries and the existing locked F4 source identity checks, then retains only owned CPU atlas dictionaries and source hashes for the ten original distal pads.
- Loading the atlas imports USD for this **offline CPU** operation. This loader must not be moved into a native worker's pre-App path.
- `classify_observations(capture, *, source_profile, atlas)` first runs the existing complete v2 raw revalidator. It requires the original eleven mesh-slot names and exact same ordering/identity of legacy pairs and new observations. Every pair uses its own recorded `mesh_position_world_m` and XYZW quaternion from the CUDA `surfaceReadout` event. It never reads endpoint hand poses or reconstructs/interpolates a substep pose.
- Classification calls the unchanged `tabletop_deformable_pad_region.classify_pair`, including the original normal sign conversion and original `_on_patch` predicate. The 4 × 3.5 mm half-axis ellipse, 1 mm source-surface tolerance and 0.8 normal-cosine threshold remain exact.
- Each pair retains env/run/episode/reset/rod/SDK identity, solver substep index, node and mesh identity, source instance, actual pose, point/normal, old projected normal load and impulse vector, native penalty N, and native gap. Table pairs remain present with `region=None` and `not_a_hand_pad`; they are never counted as hand contacts.
- Native gap (`phi - rod radius`) is separate from the atlas supporting-plane signed distance. Native N is separate from the legacy total-force normal projection. No force/depth threshold is inferred from the atlas's 1 mm geometry tolerance.
- Owner times are copied from the validated getter metadata. `substep_timestamp_s` stays `None`; no nominal or actual substep timestamp is fabricated.
- Full profile and producer-source identity are retained. A profile dictionary and a CPU parse do not authenticate a native run. Actual original mesh-to-body binding still requires the outer eleven-mesh owner audit and receipt.

The result is an original-atlas classification diagnostic. Positive and negative contact, native and Teacher qualification remain false. Effective support and fixed-material slip remain unknown. An accepted original predicate includes its positive-load condition; it is not a bilateral support, holding, friction, slip or success determination. An empty complete packet does not qualify negative contact.

## CPU validation

Environment: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1`.

Final command:

```text
scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_native_pair_atlas.py \
  tests/test_deformx_pair_observations_v2.py \
  tests/test_complete_pair_capture.py \
  tests/test_deformable_pad_region.py
```

Actual result: **105 passed in 7.73 s**, exit 0: **19 new + 86 related existing**.

The new tests load the locked original two-hand atlas but generate explicitly synthetic CPU packets. They exercise unsorted raw atomic rows; per-pair rotated/translated P poses; both sides; table separation; original ellipse/surface/normal/no-positive-load rejection and overlapping reasons; a deliberately wrong endpoint-pose comparison; pose/source/profile/order tampering; owned output and codec replay; and empty capture with no negative qualification. They do not validate an actual GPU pair packet.

TDD evidence is retained:
- Missing-module assertion: 1 failed / 16 deselected before implementation.
- First implementation exposed the exact old impulse field mismatch: 7 failed / 10 passed; fixed by copying the existing `impulse_on_rod_ns` vector, not inventing a normal-impulse field.
- Provenance addition: 1 failed / 18 deselected before retaining the full profile and predicate-source fingerprint.
- Final full related command above is green.

`Repository(Path.cwd()).verify_protection()`: exit 0. No protected-source changes.
A peer review was requested but the peer was occupied; **no completed independent review is claimed**.

## Files and durable evidence

New files only:
1. `src/go2_pvcnn/control/dh116/tabletop_native_pair_atlas.py`
2. `tests/test_native_pair_atlas.py`
3. This report.

Module SHA256: `cde2b8d1d3de7f2cf5e25d47e046be69de394fd5e37224748be484570c007407`.
Test SHA256: `764db2f1903ebe2a0fc107b3f275565346bf9f5e1826b508296d4dbb09d8f4d5`.

Evidence directory:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-pair-atlas-cpu-4067eodv`.

| Evidence | SHA256 |
|---|---|
| `red-missing.txt` | `640bb577d8df405c25864cd75e1853787f6eaf79710fb127f736cd6232707617` |
| `focused-01.txt` | `d671267633177c520601a3b17fcb93a6c6327c10d0a922a0eda5e1a538428812` |
| `red-provenance.txt` | `3318a97aa13e1413510438b27843bdf9d81b440884208c8b5f8b6314ed8f7b8f` |
| `related-final.txt` | `54449e4e5370b641e0c9f338165b7847046cfe1b4c37450329b8059ba30ac93c` |
| `protection.txt` | `af5d8a21858f42808c9bcc17012b77bcc2fd6ab3b071dbd4a05f13f410470541` |

After a separately authorized native receipt exists, the outer audit can parse each retained v2 raw packet under its exact source profile and call this offline classifier. This package neither issues that authorization nor grants native qualification.
