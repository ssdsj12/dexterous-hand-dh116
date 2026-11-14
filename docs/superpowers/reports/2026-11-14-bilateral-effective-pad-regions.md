# Bilateral native contact: original effective-pad region audit

Date: 2026-11-14. Baseline: `91af148408bdbe908987aa0c5403ab5d72ecdc79`.
This is a read-only replay of consumed native evidence, not a new Isaac run.

## Finding

All **2,111** recorded hand contact pairs fail the existing inner-pad ellipse.
All pass the existing source-surface distance and ON_HAND normal checks.
Positive raw contact load and the previous 493 same-substep raw paired windows therefore do not establish effective opposing fingertip contact.

| Own hand / actual pad | Raw pairs | Accepted by original region | Unique first rejection |
|---|---:|---:|---|
| left finger14 | 526 | 0 | outside_inner_ellipse |
| left finger23 | 495 | 0 | outside_inner_ellipse |
| right finger14 | 530 | 0 | outside_inner_ellipse |
| right finger23 | 560 | 0 | outside_inner_ellipse |

This identifies a contact-placement failure in this particular native Pinch candidate. It is not proof of the sole cause of historical Student policy failures.

## Actual implementation

- `tabletop_deformable_pad_region.classify_pair`: measured world contact point to its own actual PhysX link frame. Solver outward normal is on rod; negate once for the original ON_HAND predicate. Reuse `tabletop_effective_pad_runtime._on_patch` and the original source-USD atlas. Audit inner ellipse, supporting plane and normal independently; preserve unique first and overlapping rejection reasons.
- `bind_geometry`: exact episode/SDK/rod/policy clock, native batched link paths, own-side body index/path, XYZ+XYZW measured pose and proper rigid transform. No mirrored-hand substitution.
- `audit_dh116_bilateral_effective_pad_regions.run`: verify immutable native receipt/artifacts, source USD, source geometry, F4 capture and SDF bytes; preserve all 232 physical rows, 2,784 solver windows and all hand pairs. Output full per-pair identity, local point, UV, ellipse, signed plane distance, normal cosine and rejection. Preserve raw loads separately.
- Positive region membership alone never supplies contact depth, fixed-material slip, stability or Teacher admission. Original controller, atlas dimensions, collision geometry, actuator limits and reward are unchanged.

The actual runtime performs SDK step, reads link poses, holds those obstacle poses for twelve rod solver substeps, then captures closure geometry before the next SDK step. The replay checks endpoint clock equality and this source order. It does not invent per-substep PhysX pose or clock readbacks.

## Reproducible source and outputs

Actual receipt SHA: `8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8`.
Native ledger SHA: `d53b446425640fbbbf59ba952c52ec84ff952f0eca8b539131f9197bc899158d`.
Local result: `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/effective-pad-region-20261114-run01/result.json`.
Result SHA: `275c7f6359f668288efd8b120ba921e524c691d40bccf7321d682b71e792934d`.
The result pins source modules, own-side SDF/capture identities, atlas and full JSONL output hashes. Raw evidence remains outside Git.

First right index pair: env0, episode `dual-pinch-76ef2a55c15642c8a40a405096485601:one`, SDK188, policy23/substep1, rod solver substep4. Ellipse 1.19425224779 (>1), source-plane distance -0.10458184 mm, ON_HAND cosine -0.99876669.
First left thumb ellipse 1.72780757; left index 1.68828819; right thumb 1.72378659. These are different samples, not a reconstructed simultaneous hand trajectory.

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_bilateral_effective_pad_regions --output /path/to/new-independent-output
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_deformable_pad_region.py tests/test_deformable_pad_region_actual.py tests/test_effective_pad.py tests/test_effective_pad_batch.py tests/test_pair_substep_load_audit.py tests/test_dh116_bilateral_native_evidence.py
```

## Validation and failed evidence

CPU: **100 passed, 1 skipped** (optional CUDA profiling; CPU run intentionally hides CUDA). Initial missing-module red test and the real batched link-path red test are retained under `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/effective-pad-pair-audit-20261114`. The latter caught a fixture-versus-native layout mismatch before replay.
New physics steps **0**, PPO updates **0**. No new natural-exit claim, contact qualification or successful grasp. Original 528 protected files and two historical checkpoints are checked by integration publisher.

## Next work and remaining gates

Correct the independently versioned approach geometry against original actual pad-region constraints, not just the minimum SDF over full cooked hull. Retain full source/cooked table clearance and original guards. CPU proposals remain proposals; no native recipe before complete geometry/path verification and a new frozen budget.
Pinch still lacks qualified effective contact, material slip and full R0 sources. Tripod's existing closure search remains infeasible within its tested local domain, not globally impossible. Power lacks validated five-pad reachability. Formal Teacher PPO is blocked; R2 remains disabled. The protected Teacher/MPC/Student and old success paths are unchanged.
