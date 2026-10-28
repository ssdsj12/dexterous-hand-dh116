# Policy-owned grasp and transport

User approved policy-owned corrections with actuator limits and hard physical guards retained. Implement inline; no launch, commit or push is requested. Preserve existing dirty work and old controllers.

Design: introduce explicit `policy` tensor controller. Its 20 actions mean 12 named active joint positions (existing envelope), 6 independent wrist position targets (XY +/-3 mm, Z -4..50 mm relative to reset), and 2 per-hand drive stiffness targets within the existing load profile. This is a deliberate new action contract, not compatible with old force-action policies. Grip force is produced by physical PD drives with unchanged effort/damping limits, not a scripted force closure integrator. The two last actions control stiffness instead of becoming unused force targets.

No retained-grasp action replacement, ordinary mimic/lead pause, automatic slip recovery, release-reposition, midpoint compensation, autonomous ascent, or 0.03 rad/s closure servo in this mode. Finger target slew equals the existing 0.3 rad/s active-drive setting; wrist target slew uses the existing 0.05 m/s carriage setting. These are simulation settings, not verified hardware capability. Mechanical bounds, severe mimic correction, overload relief, signed table-clearance projection, finite/velocity/force/transport guards and actual object success checks remain. Stage transitions score readiness; they do not suppress policy actions.

Implementation sequence:
1. Add failing behavioral tests for close/lift policy authority, opening/closing, independent palm requests, actuator slew, severe mimic/overload protections, inactive/reset isolation and world submission.
2. Implement separate controller plus TensorTask dispatch and TensorWorld bounded transport/gain submission.
3. Wire explicit CLI mode, reject conflicting scripted options and checkpoint semantics; add truthful contract metadata. Add shell selection that omits scripted reposition for policy mode.
4. Run focused regression, review and document results. Keep Isaac closed. Actual contact stability and successful lift remain unverified until a new simulation run.

## Verification and review

Implemented explicit policy mode, controller/world integration, action contract, CLI conflict checks and default shell selection. No simulator or training launched. Existing work and histories retained; no commit/push.

Initial behavioral RED: 9 failures for absent controller/world behavior. Integration RED: missing script configuration rejection, missing shaping diagnostics, missing xyz launch flag, misleading safety metadata. Independent review also reproduced opposing severe-mimic/overload requests; added failing regression and neutralized the conflicting target at measured q. Selected reset and inactive rows covered. Tests stub the process check and final interpreter only when checking shell argument wiring; no simulated physics is claimed.

Final command:
`DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_policy_control.py tests/test_dh116_tensor_training.py tests/test_dh116_tensor_env.py tests/test_dh116_tensor_world.py tests/test_dh116_coordinated_grasp.py tests/test_dh116_lift_grasp_hold.py tests/test_dh116_grasp_shaping.py`
Result: **204 passed in 5.00s**. `bash -n scripts/train_dh116_tensor_ppo.sh` and `git diff --check` passed. Independent reviewer confirmed launch/shaping/guard-conflict fixes and found no additional controller/world blocker. Its test-expectation finding was fixed: current preexisting finite overspeed behavior is warning/penalty, not a 3 rad/s hard stop. The new integration test verifies the actual nonfinite-velocity hard stop; this task does not strengthen or weaken that existing guard.

The attempted bare CLI `--help` invocation failed at AppLauncher argument registration because required pose/output were absent; it did not create an Isaac app. CLI wiring was instead verified by the shell argument test and configuration tests. Actual Isaac startup, contact stability, successful lift and wall-clock speed remain unverified. Stage/readiness/reward and object success criteria remain active; policy actions are not frozen by those stage gates.
