# Original-pad Pinch targets and bounded approach correction

Date 2026-11-14; base `326c1acdb73aff03b0c8b3f60c4212623909b7dc`.
**CPU geometry only. New physics 0; PPO 0; no native recipe or Teacher admission.**

## Concrete change

`scripts/solve_dh116_pinch_effective_targets.py` reuses the existing finite minimax search, own-side USD FK/coupling, locked cooked SDF and original effective-pad atlas. It adds an endpoint requirement that the predicted solver contact point belongs to the original effective region. The solver unit normal weight is a geometry predicate input, never a measured or applied force.

`predict_region` retains node index, world/local point, original ellipse, source-plane distance and ON_HAND cosine. `decode` proposes an independent reset/endpoint within finite bounds; original active/passive limits are enforced. `isolated_sides` lets a failed side leave the other side executable. No old controller, asset, material, actuator or checkpoint changes.

`scripts/audit_dh116_effective_pinch_target_path.py` reuses original FK displacement bounds and trilinear SDF Lipschitz bounds. It checks 1,025 samples and their intervening geometry intervals. A 100 um World-X bow is represented by exactly twenty piecewise-linear 60 Hz velocity segments, compatible with eight 480 Hz physical substeps per policy step. Original finger velocities remain unchanged; the wrist trajectory changes inside original per-axis limits. This is a proposed approach, not a right-hand policy reward/amplitude adjustment.

## Source failure was already present in commanded geometry

Under the unchanged straight Reset rod, the old commanded endpoint ellipse values exceed1:

| Side | Thumb ellipse | Index ellipse |
|---|---:|---:|
| left | 1.81998760 | 1.73581648 |
| right | 1.24318324 | 1.13697404 |

Thus tracking error alone cannot explain the original missed region. This geometric diagnosis does not identify the sole cause of historical PPO bias.

## Search and retained failed trials

- run01: extra required-hull distance constraints were mistakenly added beyond the actual runtime guard. Evidence is retained, not used for qualification.
- run02: corrected to the actual guard split: all22 source/cooked table bounds,18 non-target rod bounds, required depth from locked SDF. Both candidates still failed with an additional planner ellipse cap0.64.
- run03: planner-only cap0.95, still stricter than the original acceptance1.0; no physical threshold changed. Left428 evaluations stopped without local progress and remains failed. Right303 evaluations found an endpoint satisfying all static margins.

Right reset/endpoint delta from the archived commanded endpoint:
translation **[+0.302384,+2.000000,+0.750668] mm**;
world rotation vector **[-0.030000,+0.00203393,+0.00846376] rad**;
active-joint delta **[-0.00359877,-0.00041753,+0.00145040,0,0,0] rad**.
Right endpoint predicted SDF gaps: thumb **-0.100117 mm**, index **-0.289671 mm**.
Ellipse values: **0.949921 / 0.949744**. These are predictions, not real contact measurements.

Straight path run01 failed at sample703/1024: source `finger13_link` non-target bound -0.069383 mm. Independent continuous convex-solid/capsule calculation gives a geometric surface gap **-0.057877 mm** (not measured penetration depth). Translation sensitivity is approximately [0.989221,0,-0.146427] m/m, motivating the bounded positive-X detour rather than dropping the collision check.

Bow path run02 used average root arc length as a Lipschitz bound; this is invalid for varying segment rates and that proof is withdrawn. A failing regression now requires the maximum segment rate. Final path run03 uses that corrected bound.

## Final right path (static reference only)

| Check | Final continuous lower bound |
|---|---:|
| Full source/cooked table clearance | 2.106441 mm |
| Non-target rod separation | 0.001528 mm |
| Required-pad SDF gap | -0.290936 mm |
| Initial all-shape rod separation | 0.146830 mm |

The original gates remain table>=2.1 mm, non-target>=0, required SDF>=-0.3 mm. Original effective atlas remains unchanged.
The very small non-target reserve is not a tracking guarantee. Actual mounted articulation dynamics, evolving rod state, inter-hand scope and native owner/validator freezing remain required before native execution. No recipe is emitted by these CPU tools.
Left has no passing endpoint or path; one side's result cannot qualify the pair. Tripod/Power remain independent unresolved tasks.

## Reproducibility and failure retention

Target run03:
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-effective-target-20261114-run03/result.json`
SHA `cc6fcfcbf001d597fcc46d0a27768974f8e0f4b1714a872074ac8a6a65d8ddbe`.

Final path run03:
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-effective-path-20261114-run03/result.json`
SHA `03b70ba97423665eac710ddd0fe3e868282885428efe1e0775bf304ccf9aa280`.

Failed straight path SHA `81340b52b641abdb23dbb46238f9543329fcf3bad1f1e4b7c22c8c1a7048e25c`.
Target run01 SHA `7e914bee82e622bb4ac7aae16f15552700400baa0390c62dd6af5bc0a7edfd77`; run02 `f34ba3df4892dc883efdf4c012e4f74916482bb929214faa76303b85d6ddd8c3`.
Each result pins the exact executed source/config. Prior solver source snapshots and red/green logs are retained outside Git under `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-effective-target-20261114`. Version1/2 configuration files are retained failure evidence; current executables select version3. No raw data/models are uploaded.

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.solve_dh116_pinch_effective_targets --output /new-independent-target-output
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_effective_pinch_target_path --output /new-independent-path-output
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_pinch_effective_targets.py tests/test_pinch_effective_targets_actual.py tests/test_dh116_owned_approach.py tests/test_tripod_contact_closure.py tests/test_deformable_pad_region.py tests/test_surface_contract.py
```

CPU regression **47 passed,0 failed**. These include real pinned CPU-result tests, not native-validation counts. Protection checks cover528 files and two old checkpoints. No new WireSeg training steps. Next: bind the right candidate to measured tracking/inter-hand margins and an isolated native owner; continue left geometry separately. Formal Teacher still requires effective native contact, Reset/state/action/R0 source qualification; no training success claim.
