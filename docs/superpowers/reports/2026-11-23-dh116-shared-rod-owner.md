# Shared solver ownership and first real Isaac startup

SharedRodOwner owns one actual shared solver advance per SDK step. It requires
the previous reaction to be consumed before the next advance, and transfers
each reaction exactly once for the immediate next SDK write. It binds run,
solver instance, ordered episode IDs, separate reset IDs and rod ticks.
Failures invalidate pending data. The native rod-only Reset verifies unchanged
owner time and neighbor nodes, clears only that lane's pending contact data
and increments its episode. This method does not restore an Isaac articulation
or collider poses and grants no complete task Reset qualification.

The original runtime already enforces a 1e-3 N mesh/rod force balance check.
The shared owner preserves it, together with the compact per-field tolerances.
A regression proves that native Reset cannot clear and conceal a prior solver
error. Other tests cover duplicate consumption, wrong clocks, overflow,
independent ticks, neighbor state preservation and stale-reaction rejection.

## Results

- Scoped CPU regression passed 53 tests with 1 CUDA skip.
- The same scoped regression on GPU 2 passed all 54 tests, including CUDA parity.
- GPU 3 actual shared solver ownership probes passed for each grasp with 128
  solver lanes. They verified one-time reaction consumption, a zero-step rod
  Reset, clearing its pending reaction, and another native step.
- The first two-lane Isaac attempt failed in collision-clearance tuple unpacking.
  It naturally exited 1, without any shared rod solver step. Its receipt and
  full log are retained.
- The corrected source ran 18 shared SDK steps in a real Isaac scene: 2
  initialization steps, 8 original hand isolation/setup checks, and 8 shared
  DeformX exchange steps. There were 4 original DH116 articulations.
  Original joint/MIMIC, table-clearance and 30 N reaction checks remained active.
  It naturally exited 0, with app.close returned and no forced exit.
- Both lanes recorded zero nonzero hand-reaction steps. This static initial
  pose probe established no hand grasp/contact. Its result remains explicitly
  native_qualification=false and teacher_admission=false. No PPO ran.

## Evidence

Successful native outer receipt SHA-256
383606fb0ab5b1d6e6d6fb1352c9a7ba2b3db0909fc4e41650f7ce5ec94a62b3.
The frozen source digests in its attempt were rechecked after natural exit.
Readiness records hashes of failed and successful attempts and regression logs.
Raw native logs, binary and model data remain local.

## Next

Apply the archived, independently verified approach actions through the original
24D target and wrist projection adapters to establish actual hand contact and
next-step nonzero reaction. Then complete per-lane hand/collider/rod Reset,
connect actual policy rollouts and validate the original numerical boundaries.
Pinch, Tripod and Power 128/512-lane Isaac PPO remains incomplete.

## Antigravity R3 review

The reviewer examined the supplied current design excerpt, while its readonly
MCP remained on the old tree. It asked to verify consume-before-next-SDK timing,
per-lane pending reaction preservation through Reset, and single-lane collider
teleport on complete Reset. In this implementation consume targets the upcoming
SDK index while the owner remains at the preceding completed boundary. It does
not require the SDK to have advanced. CPU tests cover clearing a reset lane
without clearing neighbors. The native static probe did not perform complete
hand/collider Reset or establish nonzero reaction and does not prove those gates.
