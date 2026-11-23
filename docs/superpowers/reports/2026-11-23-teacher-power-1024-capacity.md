# DH116 Power solver capacity at 1024 lanes

Baseline `41eef7fe564192dac4d8d337b3959f3b4d9e8b80`. GPU 2, RTX 4090 (24,564 MiB as reported by `nvidia-smi`). The immutable DeformX binary and Power Reset receipt are the same as in the [32-environment preflight](2026-11-23-teacher-ppo32-preflight.md). Each rod has 41 nodes, 17 actual task SDF meshes, and the original 480 Hz step and 12 internal substeps. The archived poses drive fixed mesh targets for this capacity probe.

## Separate owners with raw contact pairs

| Requested rods | Allocated | Final GPU used (MiB) | Two ticks plus pair readback (ms) | Result |
| ---: | ---: | ---: | ---: | --- |
| 32 | 32 | 1,400 | 75.64 | Raw pairs and selected Reset verified |
| 64 | 64 | 2,398 | 153.75 | Raw pairs and selected Reset verified |
| 128 | 128 | 4,394 | 308.51 | Raw pairs and selected Reset verified |
| 256 | 256 | 8,384 | 622.56 | Raw pairs and selected Reset verified |
| 512 | 512 | 16,364 | 1243.99 | Raw pairs and selected Reset verified |
| 640 | 640 | 20,352 | 1566.96 | Raw pairs and selected Reset verified |
| 1024 | 699 | 22,042 | NA | Stopped at GPU reserve before first tick |

The 640-rod run captured both ticks without overflow. The isolated rod had zero pairs and each other rod had 24 pairs per tick. Reset restored the isolated rod and left all other positions bitwise unchanged. The 1024 request stopped after allocating 699 owners. CUDA reported 2,039 MiB free at the 2,048 MiB reserve check. This was a deliberate memory-reserve stop with status 2. The run performed zero solver ticks and did not attempt an OOM.

## Shared SDF with aggregate forces and wrenches

| Rods | Final GPU used (MiB) | Two ticks plus aggregate readback (ms) | Rods with force and reaction per tick |
| ---: | ---: | ---: | ---: |
| 2 | 436 | 3.21 | 1 |
| 32 | 438 | 3.85 | 31 |
| 64 | 440 | 3.66 | 63 |
| 128 | 446 | 3.80 | 127 |
| 256 | 454 | 4.08 | 255 |
| 512 | 474 | 4.60 | 511 |
| 1024 | 508 | 6.70 | 1,023 |

The shared batch uses one set of 17 SDF grids and per-environment mesh poses. One negative-control rod stayed outside its meshes. Both force and reaction were zero there and nonzero for the other 1,023 rods at both ticks. All returned tensors were finite. Reset restored the selected rod and preserved every other position bitwise. GPU 2 returned to 6 MiB after each recorded run.

The locked backend explicitly rejected multi-rod pair capture with `pair capture requires one rope, mesh forces, and a bounded one-time allocation`. Aggregate force and wrench tensors are actual DeformX outputs. They provide no per-pair identity or applied PhysX wrench evidence.

The numbers are snapshots after allocation and readback, and short wall-clock timings. The modes read different evidence, so they do not establish a PPO speedup. No Isaac environment or PPO update ran in this work package. The 1024-lane aggregate result is Power solver evidence only.

## Reproduction and launch status

On Ubuntu use the installed DH116 Python, the locked backend first in `PYTHONPATH`, and the archived Power outer Reset receipt whose SHA-256 is `d1aa0fbe8ba495ede23da425d4c46cc7f8c2dbc5377471dad1ad788cea4af8af`.

```bash
CUDA_VISIBLE_DEVICES=2 python -B scripts/audit_dh116_power_solver_capacity.py \
  --receipt /absolute/path/to/power-reset/outer.json \
  --receipt-sha256 d1aa0fbe8ba495ede23da425d4c46cc7f8c2dbc5377471dad1ad788cea4af8af \
  --gpu-index 2 --num-envs 1024 --shared-sdf
```

Omit `--shared-sdf` to reproduce the separate-owner reserve stop. The first shared-SDF probes failed before a solver tick on an extra archived environment dimension and a batched quaternion variable typo. Both failure logs were retained locally. The repaired batch passed all seven widths.

CPU test result is 66 passed.

```bash
python -m pytest -q tests/test_dh116_power_solver_capacity.py \
  tests/test_teacher_batch_inputs.py tests/test_teacher_physical_group.py \
  tests/test_native_teacher_worker_contract.py
```

The native Teacher still uses one scene/rod lease. Its worker expects one environment, and the existing group barrier stops all lanes on any terminal event. The next full-system test is two original Isaac rigs sharing one SDK clock, with an independently terminated and reset lane, full pair/wrench evidence and GAE terminal-observation checks. Further scale requires a qualified shared geometry path with per-environment pair provenance.

Per-run SHA-256 values are in the [Teacher readiness record](../../contracts/dh116_teacher_ppo32_readiness_v1.json).
