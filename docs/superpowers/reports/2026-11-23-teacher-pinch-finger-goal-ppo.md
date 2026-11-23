# Goal-conditioned Pinch Teacher PPO diagnostic

The worker scores the checked-in Pinch preview as an active-finger reference. The reference uses canonical DH116 joint names and a verified SHA-256. SDK readbacks carry the same episode and physics clock as the rod. The reward adds a bounded pose potential and one arrival bonus per episode after 16 consecutive healthy SDK ticks. The original 24D distribution and physical guards still run.

This recipe exercises one PPO update with 8 control windows, the original measured Reset, one Isaac instance and GPU 3. The copied replay metadata may relocate one Pinch task path after both files' bytes and the complete remaining source basis are verified. It uses no offline trajectory as a scheduler dependency.

277 CPU tests passed. This experiment covers active-finger reference error only. Wrist reference, inactive-finger action masking, multi-episode rollout and full-grasp qualification are separate.


## Native diagnostic result

One real PPO update completed on GPU 3 against the two-hand, 41-node DeformX scene. The process completed 2 initial SDK steps and 64 controlled SDK steps, grouped into 8 control windows, with 768 solver substeps. The outer process and the GDB inferior exited naturally with code 0 in 164.782 s. The independent scheduler readback verified the sealed result.

The accepted minibatch changed weights by L2 0.00431824. Exact KL was 0.00169718 at accepted learning rate 0.00001. The clean checkpoint SHA-256 is `79a07e58a5c59073b8a916f7744a42e90f18619b1fd58204e0e6b5ad3f5aeacc`.

Maximum left/right participating-finger error at Reset was 0.045074/0.334631 rad and ended at 0.047744/0.338446 rad. The right hand remained outside the 0.05 rad tolerance, so the arrival bonus never fired. The full contact ledger contains 31,488 table pairs and zero hand-wire pairs. No grasp, Lift, Hold, or generalization is qualified.

The next change is a multi-episode rollout on the same owned scene with a bounded count of fresh Reset events. The existing single-episode worker does not supply that path. Wrist reference, fixed-digit policy distributions, the Tripod Reset source binding, and Power contact sources still require work.

Test commands under the existing DH116 Python environment:

- `python -m pytest -q tests/test_native_teacher_finger_goal.py tests/test_native_teacher_reward.py` (277 passed).
- `python -m pytest -q tests/test_dh116_work_packages.py tests/test_native_teacher_finger_goal.py` (64 passed).
