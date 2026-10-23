# Hand-only stability investigation

## Contact-Guarded Open Approach Update

The tabletop entry point now defaults to PGS (0), 16 position iterations,
240 Hz physics and 60 Hz control. TGS remains selectable for comparison.
Hands start at 5 percent of the distance from coupled lower limits to the
reference joint vector. A smoothstep reference blends toward the grasp as
the initial 40 mm root offset decreases. This is a joint-space opening, not
yet a visually qualified pad pose.

Descent stops and latches when any wire-filtered pad load reaches 0.1 N or
measured mesh clearance reaches 2 mm. Translation is clipped to the remaining
clearance at each physics step. Reference progression freezes on the next
control step after the latch. This supervisor is not a full swept-geometry
collision predictor, and does not certify closing or exploratory PPO actions.

Actual runs:
- `tabletop_guarded_contact_20261023a`: old initial joint posture, guarded TGS,
  15 second timeout without numerical safety failures.
- `tabletop_open_guarded_20261023a`: open blend, TGS, mimic_error.
- `tabletop_open_guarded_pgs_20261023a`: open blend, PGS, 15 second timeout.
- `tabletop_open_guarded_default_20261023b`: repeated default PGS run, identical
  outcome. Contact latch at 1.5 s; sampled maximum mimic error 0.006125 rad,
  maximum pad load 0.77552 N, minimum mesh clearance 0.008856 m.

The two PGS runs qualify only this deterministic approach/contact-hold
trajectory. They do not complete approach's old root-goal gate, close, or lift.
Reports correctly retain `passed=false`, `failure=timeout`, and PPO updates 0.
The successful-grasp training gate has not yet been replaced by a separately
validated exploration stability gate. Reward coefficients remain unchanged.

Current focused tests: 9 passed, 1 schema-plugin test skipped.

## Earlier Solver-Only Investigation

Scope: two DH116 hands, table and dynamic rigid wire proxy. No arms, reward
changes, collision removal, joint-state overwrites after reset, or relaxed
safety thresholds.

The runtime now advances root references at each physics substep instead of
applying the entire control-period displacement in the first substep. Control
frequency remains 60 Hz. This correction alone does not resolve instability.

All experiments used `scripts/train_dh116_tabletop_ppo.py`, CPU, headless,
`--evaluate-only`, and pose summary
`runs/reference_aligned_native_20261023_fix1/summary.json`.

| Run under runs/ | Solver | Position iterations | Physics Hz | Outcome |
| --- | --- | --- | --- | --- |
| tabletop_stability_pgs_20261023a | PGS | 16 | 240 | close, mimic_error |
| tabletop_stability_pgs64_20261023a | PGS | 64 | 240 | close, mimic_error |
| tabletop_stability_pgs64_960_20261023a | PGS | 64 | 960 | approach, excess_force |
| tabletop_stability_smooth_20261023a | PGS | 64 | 960 | substep roots; approach, excess_force |
| tabletop_stability_tgs_smooth_20261023a | TGS | 16 | 240 | substep roots; approach, mimic_error |

Solver selection, position iterations and physics frequency are explicit CLI
comparison options and recorded in summary.json. Defaults remain unchanged:
TGS, 16 position iterations, 240 Hz. None of the alternatives qualifies as a
stable grasp configuration.

The PGS64 close failure includes left finger13 mimic error 0.02035 rad and
mesh/table clearance -0.000172 m. The fixed-root descent and closure therefore
need contact-feasibility investigation, not just more solver iterations. A
causal PhysX defect has not been established. Proposed next investigation:
contact-constrained palm motion, actual pad targets, and loaded joint response
without forcing an infeasible root trajectory. Keep all current safety gates.

Verification: `bash scripts/python.sh -m pytest
tests/test_dh116_tabletop_task.py tests/test_dh116_runtime_mimic.py -q`
returned 7 passed, 1 skipped. The skipped schema test requires Isaac plugins.
No successful grasp/lift, PPO training, or hardware qualification is claimed.
