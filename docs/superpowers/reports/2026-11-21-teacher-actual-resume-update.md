# Actual Teacher resume update

The CPU regression passed 334 tests. The two-process fixture ran the original PPO optimizer for eight controls in each process. Process A sealed update 1 and global step 8. Process B started a distinct fixture episode, restored the exact policy, Adam and Python/NumPy/Torch RNG state, and reached update 2 and global step 16. Its starting Adam step was 1, and its accepted minibatch total was 2. Its policy parameter change had L2 norm 0.033964400. The A checkpoint SHA-256 remained 8001c9cbcc8ac4b8553a208eef4f9427d099769c30eb546a29725315df07cd8f before and after B.

The shared update code verifies an unused control window, the independent episode, the incoming policy and origin, unchanged A checkpoint bytes and cumulative optimizer counters. B writes a separate exclusive checkpoint and journal seal. The test keeps the physical owner and journal structure while using CPU fixture bodies and a fixture rod.

Run the focused regression with:

~~~bash
bash scripts/python.sh -m pytest -q tests/test_native_teacher_actual_resume_update.py
~~~

This package ran on Ubuntu CPU. Native B still needs the runtime lease, role-aware artifact audit and a fresh native A receipt from the same source closure. There is no new grasp qualification.

[Readiness](../../contracts/dh116_teacher_actual_resume_update_v1.json).
