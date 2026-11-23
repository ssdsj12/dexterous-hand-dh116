# Shared DeformX contact with real DH116 articulation

The opt-in contact-receipt path restores the authenticated historical approach
seed after the original eight-step hand isolation test. It selects the historical
trajectory's own pair proposal and replays each original 24D command for exactly
eight SDK steps through BimanualStepConnection and project_policy_wrist_targets.
The neighboring lane receives an exact zero 24D command. Measured link poses
update the shared SDF meshes after each SDK step. Captured mesh reactions are
routed through the original world-COM wrench conversion and actual hand composers
for the next SDK step. Original joint/MIMIC, table-clearance and 30 N checks remain.

## Actual GPU 3 run

- Two environments, four original DH116 articulations, one shared CUDA solver.
- 226 SDK steps in total. These comprise 2 initialization, 8 hand setup/isolation,
  and 216 shared exchange steps with 12 rod substeps each.
- 216 per-step records. Every row contained separate measured action readbacks
  for lanes 0 and 1. Lane 1's actual requested 24D vector was zero throughout.
- Lane 0 had 22 nonzero hand-reaction writes. Lane 1 had zero. Maximum norm of
  force on an individual hand collider was 0.129055594 N and 0 N respectively.
- Natural inferior exit 0, no signal or forced exit, app.close returned.
  Total outer wall time was 57.409 seconds including application lifecycle.
- Zero Teacher queries and zero PPO updates. No grasp success, complete task
  Reset or 128/512-lane Isaac PPO qualification is granted.

Outer receipt SHA-256
e3db3c4016ca73bdf254b9b6b67872122a7aa6f9f1fb83149ea6d43202269d9b.
Journal SHA-256
b7ff1c7f855b931e7b9d5fb475a146d692e214d7c4ed36737e91424604c0519f.
Frozen native script and solver-owner source bytes matched after exit.
Raw logs and data remain outside Git.

## Tests and a source contract correction

The command schedule failed before implementation and then passed with the
ownership/pair suite: 22 passed, one CUDA skip. A broader suite reported
115 passed, one skipped and one failure in the historical wrist-evidence owner.
The failure independently reproduced: NativeParameterEndpoints now emits
source_scope=native_getter, while the offline audit's exact field set omitted
source_scope. This source field was introduced before this work package.

A new focused regression first reproduced the rejection. The audit now validates
the explicit native_getter tag and also accepts the old untagged sealed v1
packets. Other source tags and extra fields remain rejected. Fifteen focused
parameter audit tests passed after this correction. The full wrist-evidence owner, parameter-task and command-schedule
rerun passed all 22 tests in 456.01 seconds; its log SHA is in readiness. No physics or source SHA check is downgraded.

## Remaining integration

Complete lane-local hand/collider/rod Reset and verify continuation under real
reaction. The existing CUDA set_mesh_poses API already supports per-environment
teleport flags. Its pending targets become current at the next solver step,
so a zero-time Reset still needs verified current/target/solver pose alignment.
Then integrate actual policy/source/reward collection and original PPO for all
three grasps, first at the qualified small batch and then at the requested scale.
Shared/independent numerical equivalence at reward, guard and action boundaries
and Pinch's untrained-policy clearance failure remain open.

## Commands and source references

The native attempt invoked the repository's scripts/python.sh with
-m scripts.smoke_dh116_shared_rod_native, explicit --receipt/--smoke/--backend
paths and their SHA-256 arguments, plus --contact-receipt and
--contact-receipt-sha256. Its complete argv, frozen sources and bounded run
limits are retained in the local attempt.json accompanying the outer receipt.

The focused source-scope regression command was
python -m pytest -q -p no:cacheprovider tests/test_dh116_native_parameter_task.py
-k 'not real_owner and not endpoint_failure and not reset_read_failure'.
The native command schedule uses the immutable contact prefix loaded by
archived_contact_prefix from the original naturally exited native receipt.
No model, raw ledger or candidate binary is included in this publication.

The independent archived-command reread verified all 216 lane-0 requested vectors
against the original command sequence, every lane-1 vector against zero, all
consecutive SDK identities and 12-substep rod ticks. The first generated hand
force occurred at SDK 204, and was first written for SDK 205. The final eight
zero-command steps retain the contact in this probe; the original independent
solver smoke instead reset lane 0 at that boundary. Their reaction counts
therefore have different scopes and are not a numerical-parity result.
