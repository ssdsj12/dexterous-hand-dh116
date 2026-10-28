# Instant per-hand close hold and bilateral lift

User-approved behavior: each side independently enters hold when its current gripping conditions and its own pure-normal Z force >=0 are satisfied. No 0.2 s close dwell. Once BOTH current sides satisfy the conditions, and the current aggregate pure-normal Z force is >=0, enter lift in that same physics substep. Earlier readiness is never substituted for a weak or downward-force current sample.

## Implementation

- `tabletop_close_grasp_hold.py`: per-hand accepted command, force target, applied palm offset, current readiness and short contact-loss state. Capture once per acquisition, not on each policy step. Reset only selected environments. Sustained loss below 0.05 N for 20 ms permits reacquisition; brief losses retain the anchor but cannot count as current readiness.
- `tabletop_tensor_task.py`: coordinated rows use the instantaneous bilateral barrier. Existing load bounds, motion, coupling, contact validity and four-second timeout remain; legacy uncoordinated compatibility paths retain their old dwell.
- `tabletop_coordinated_grasp.py`: the accepted side holds its references with bounded internal force feedback while the other side continues exploring. Shared Z search ends during partial hold. No copying of left numeric joint values into the right model.
- Reposition controllers exclude accepted sides from opening, moving and shared reclose barriers. Capture cancels only that side's completed reposition state, preserving the partner and attempt accounting.
- `tabletop_tensor_world.py`: independent per-hand clearance protection during partial close hold and lift. Existing common lift increment and physical guards remain; accepted heights need not match.
- Training contract explicitly records zero dwell and current per-hand normal-force predicates; incompatible older checkpoints are not silently resumed. Probe and teacher evidence expose held/ready sides and accepted references.

## Verification

Regression coverage includes either side ready first, independent partner exploration, feedback while held, pressure/mimic guards, current weak/negative side blocking stale readiness, immediate first-substep transition, no friction admission credit, selected reset, reposition exclusion, and per-hand clearance.

Physical validation uses the same eight-case teacher search and separately stored right functional-fit candidate. Existing real lift-height/stability success criteria remain unchanged. Offline tests or entering lift are not evidence of a completed lift, and no student training starts before qualifying teacher replay evidence.

## Results (2026-10-28)

Implemented and reviewed; no outstanding P1/P2 finding in scoped review. Review caught and fixed a delayed grasp-ready reward regression: a zero-normal-force lift entry must still receive the existing one-time +3 bonus on a later qualified positive-force sample. Reward coefficients remain unchanged.

Root verification: `DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_close_grasp_hold.py tests/test_dh116_lift_grasp_hold.py tests/test_dh116_tensor_training.py tests/test_dh116_tensor_world.py tests/test_dh116_contact_effort.py --tb=short`: **176 passed**. Reviewer controller/reposition/world/training suite: **175 passed**, probe/teacher suite: **32 passed**. `git diff --check` passed.

Actual Isaac run: `runs/success_teacher_20261028_181648_perhand_instant8`. Seven of eight first episodes entered lift; all seven transitions occurred on the FIRST simultaneously ready physics substep, with each hand's normal Z nonnegative. Example env4: both hands first qualified at 1.233333 s and entered lift that same substep; loads 1.866/4.344 N, normal Z +0.0400/+0.9257 N. Detailed transition audit: `instant_close_audit.json` in that run.

The run stopped on existing physical velocity protection: env1 was still in close at 3.785417 s, right `finger22` velocity -3.10246 rad/s exceeded the 3 rad/s bound. The root cause of this velocity excursion has not been established. Maximum object bottom was 0.30000957 m (table 0.30 m), with **zero completed lift successes**. Phase admission is validated; physical grasp/lift remains unqualified. No student PPO was started and no protection threshold was relaxed.
