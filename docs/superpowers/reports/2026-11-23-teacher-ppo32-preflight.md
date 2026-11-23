# DH116 Teacher PPO 32-environment preflight

Training targets are 32 simultaneous lanes of one grasp per run, in order Pinch, Tripod, Power. The single Isaac instance remains exclusive. The original 24D controls, 480 Hz physics and 60 Hz policies are retained. Training data must come from actual, isolated PhysX and DeformX sources and use separate grasp checkpoints.

## Exercised paths

- `control_group` reuses the per-lane command transfer and captures the same clearance reports. All hands submit before one global PhysX tick. Each owned solver advances once, then the journal receives every current lane row. The diagnostic group closes after a terminal barrier until per-lane episode recovery is integrated.
- `pack_teacher_group` independently checks SDK, context and rod metadata per environment before stacking the original six tensors, retaining raw 371D validity and NaNs. Live sources require distinct solver instances. The original PPO implementation made a CPU fixture update for each grasp with 32 source-specific lanes and 64 transitions.
- The existing 32-environment legacy script disables PhysX table collision, omits the new force evidence identity, and ends with a hard process exit. That runner cannot produce this work package's native evidence.

## Verification

100 CPU tests passed in the physical barrier, bimanual action, flexible exchange, seven-route Reset, clearance and routing suites. 46 CPU tests passed in batched inputs, native model and actual-input reconstruction. 60 CPU scheduler and registry tests also passed. No native Isaac instance started in this work package.

The CUDA-only capacity probe read the checksum-bound Power native Reset's 17 mesh slots and real SDF bytes. Each owner contains one free 41-node rod and capacity for 8,364 pair rows. In each of 2 steps with 12 substeps, one isolated rod had zero pairs and every in-chamber rod had 24. Reset of the isolated lane bitwise preserved every other lane's positions.

| Independent DeformX owners | GPU used before/after (MiB) | Solver and pair-readback wall time for 2 ticks |
| ---: | ---: | ---: |
| 2 | 6 / 466 | 4.34 ms |
| 32 | 6 / 1,400 | 77.37 ms |

These GPU observations are from one sizing probe per width. Full Isaac memory, training throughput and performance under real contact remain unmeasured. The first sizing setup generated an unexpected palm contact when the control rod was raised. The revised control rod was moved outside the chamber and the isolation probe passed.

Reproduction uses the locked backend in `PYTHONPATH` and a matching `CUDA_VISIBLE_DEVICES`:

```bash
python -B scripts/audit_dh116_power_solver_capacity.py \
  --receipt /absolute/path/to/power-reset/outer.json \
  --receipt-sha256 d1aa0fbe8ba495ede23da425d4c46cc7f8c2dbc5377471dad1ad788cea4af8af \
  --gpu-index 2 --num-envs 32
```

## Launch blocker

C06 still creates one stage/rod lease and validates one environment per worker. Native multi-environment work must author each original rig in one stage before its global initialization, establish independent native Reset and material state, and bind the physical barrier and PPO inputs to the durable lane journals. The first integrated run is two lanes and must verify a targeted Reset and preserved reaction routes before increasing the first learning run to 32 environments. Source changes to the exchange require new live recipe fingerprints. The recorded single-lane results remain attributed to their tested commits.

No multi-environment Isaac PPO has started. Existing Pinch, Tripod and Power evidence is one verified single-environment pose diagnostic update each. The 30 mm, 5 s and 0.5 mm success requirements remain in force.
