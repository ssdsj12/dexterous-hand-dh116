# Single-finger fixed-fixture loading plan

User authorized the proposed single-finger fixed-fixture loading test. Execute inline; no further approval, real hardware command, commit or production parameter adoption.

Goal: measure simulated load authority at the actual grasp posture, independently of straw slip. Compare four isolated fingertips (left/right thumb/index), using one Isaac CPU process and CPU0/numeric single thread.

Design: retain source hand articulation, affine rigid mimic, current grasp posture, fixed palm, disabled gravity and gains20/damping.2. Only the selected distal pad collider contacts a fixed, frictionless plate. Place the plate beyond that pad's support surface with .5mm initial gap; normal follows the actual grasp's world-X closing direction. Other colliders disabled for this isolated fixture test, explicitly documented. Ramp the selected flexion joint target at .05rad/s to +.15rad, hold2s, then unload; retain coupled joint limits. Test uniform active effort caps .2/.5/1/2 Nm sequentially. Record filtered plate-specific normal force, actual active/follower position, mimic error, command error, implicit-PD estimated torque, plate immobility and force vector. Raw force peaks never count as sustained capacity. A .02rad mimic error, nonfinite state or >30N load ends the trial honestly.

Tasks
- [x] Preserve/stop v54 and confirm no other Isaac instance.
- [x] Add failing offline tests for support-plane placement and sustained-window qualification (transients, incomplete hold and unsafe windows rejected). Implement small pure analysis helpers.
- [x] Implement standalone scripts/test_dh116_fingertip_fixture.py using existing source USD and sensor/drive APIs; save complete setup/readback, traces, progress and results in a new run. Do not modify PPO control.
- [x] Run tests, then actual baseline fixture; verify unloaded zero force, isolated loaded-pad force, fixed plate and bounded joint states. Execute cap sweep with reset between caps.
- [x] Summarize force versus effort/geometry, repeat successful capacity conditions, review and document limits. Do not claim manufacturer calibration from a simulation fixture or resume a newly tuned PPO without its own physical admission.

Interpretation: below14N may show insufficient actuator command authority or transmission/geometry limits. Achieving14N demonstrates this estimated simulator can supply that force at this fixture posture only. It does not identify a real motor constant, prove all postures/fingers, or solve grasp/lift.

Executed:v59 32 trials,v60 sleep comparison8,v61 iteration comparison16;all two repeats,0 sustained14N. Report docs/dh116_fingertip_fixture_20261024.txt.20tests passed;read-only review issues corrected. No production tuning adopted;PPO remains paused after0updates. Further steady-load diagnosis is an unresolved result,not a fixture implementation failure.
