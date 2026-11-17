# Quaternion-corrected native verification: matrix fixed, hull topology and exit blocked

## Actual execution
Executed published23820535273de40b5c991914eb5cf2cf2d2019c3 under the user's request to verify.
Frozen SHA2f907b35010971bbea82e9f09c92719a93aa381a10c6d29eae84ce012463025c.
GPU2,1 environment, exactly2 SDK initialization steps;0 control, DeformX solver, Teacher query, model load and PPO steps. No cameras/WebRTC. One attempt, no retry.

Both hands' original supported Pinch seed readback passed at SDK2/time0.004166666883975267s before overwrite. The SDK clock remained unchanged through the explicit STOP update. This is initialization/source inspection, not a force-contact or Lift trial.

## Matrix correction verified against actual native values
All34 raw XYZW body poses were durably saved before conversion. Reapplying the old conversion to this packet rejects right body_id15:
- actual link /World/envs/env_0/rightHand/finger42_link;
- original source /DH116_V2_0_R/finger42_link;
- quaternion norm1.0000001388277882;
- old rotation orthogonality residual1.0496967837170246e-6;
- new residual6.661338147750939e-16.

The corrected conversion passes the unchanged protected matrix check for all34 poses. This is actual native snapshot evidence for the conversion defect. It does not recover the exact missing pose from the earlier spent attempt. finger42 link identity here is not the historical right finger12 joint-tracking failure and must not be used to change that joint's limits.

## Stored cooked geometry: returned34, qualified0
All34 unique bound collision owners returned one stored-cooked hull each:34 count getter calls and34 data getter calls. No cooking request, geometry authoring or new SDF registration was made. The full264096985-byte capture is saved on the data disk; its SHA is in readiness.

The protected geometry validator rejects all34 with COOKED_HULL_TOPOLOGY_INVALID_0 and "polygon plane is inconsistent with convex vertex enclosure". It did not stop at pose preflight this time. Across1890 polygons,189 violate the existing face/plane or convex-enclosure numerical check. Largest face residual0.218929498mm. Example left base_link polygon55:0.121132335mm face residual versus0.000971911mm numerical tolerance.

These measurements do not prove corrupted USD, a particular SDK bug or a unique plane/index convention error. The previous10-pad CPU/GPU parity tested equivalence, not full34-owner plane validity; both facts can coexist. No plane was refitted, no vertex/index changed and no tolerance relaxed. Stored representation return is distinct from topology qualification and live PxShape attachment.

A new CPU-only audit command, scripts.audit_dh116_teacher_cooked_readback, replays the existing protected validator on saved raw packets and lists per-body/per-plane residuals, source identities, duplicates, input SHA and counts. It never starts Isaac, repairs geometry or grants native qualification. CLI exit0 means the audit completed; failed validation remains explicit in JSON.

## Shutdown remains failed
Worker result was saved after68.839669991s. app.close returned, manager view wasNone and the SDK clock stayed2. About71.6s after supervisor launch, SIGSEGV occurred in libomni.physx.tensors.plugin.so -> _physicsTensors -> carb -> CPython Py_FinalizeEx. Full main40-frame and all-thread8-frame stacks plus dynamic mappings are preserved.

Both hand view weakrefs were gone afterSTOP, but SimulationContext and both Articulation Python objects remained alive through generated factory locals/closures and installed initializer-frame references. The exact failing late native destructor is still unresolved. No fast shutdown or forced-exit success was substituted.

The paused debugger remained after the180s soft marker without further physics progress. The user separately authorized cleanup of inferior524288/GDB524158. Only those processes were ended (SIGKILL inferior, SIGTERM debugger); both are gone, GPU2 returned to9MiB and the single-instance lock is free. GDB returned0 but supervisor returned1; native natural exit0 is not qualified.

## CPU tests, protection and publication
The7 new audit tests failed before implementation, then passed. They cover immutable valid input, unaltered numerical tolerance, source-plane mismatch, malformed/nonfinite/index-invalid payload, recomputation despite forged cached flags, duplicate body identities, SHA and output-overwrite refusal. An intermediate output-newline bug was caught and fixed; its failed log is retained.

Full relevant regression421 passed,0 failed,0 skipped. Exact command/environment/log SHA and all34 per-body measurements are recorded in readiness. All528 protected hashes and3 original candidate checkpoints remain unchanged. The protected collector was not edited. Raw meshes,264MB capture, logs and models stay outside Git.

Ubuntu commit and Windows publication carry only the isolated offline auditor, tests, compact status and reports. PR2 stays Draft and unmerged. Teacher/BC/DAgger/PPO admission remains blocked.

## Next work
1. Use the saved34 raw hulls to check installed SDK polygon-plane/index semantics, independent manifold/winding/volume evidence and cooking representation assumptions; preserve all original data and thresholds.
2. Repair explicit owned runtime/factory/native-view release before plugin unload, with CPU ownership tests.
3. Freeze a changed, tested version before any separately authorized native attempt. Do not rerun this consumed version/budget.

Original physical parameters, assets, limits, controller, checkpoint and30mm Lift/5s Hold/0.5mm fixed-anchor Slip standard remain unchanged. No grasp success was measured in this zero-control verification.
