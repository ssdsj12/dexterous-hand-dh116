# Bind actual Teacher resume to immutable checkpoint bytes

The existing C05 loader now accepts the authorized SHA-256 when called by the resume trainer factory. It compares the exact byte snapshot with that SHA before calling the weights-only deserializer. The A diagnostic audit obtains the restored random-state digest from the same checkpoint bytes already checked for policy, Adam, counters and source. Existing default checkpoint metadata stays identical. The actual A origin carries the additional random-state digest.

The resumed update checks weights and Python, NumPy and Torch random state before any simulation control. In the independent CPU processes, an injected random draw causes rejection with zero B transitions and zero fixture solver steps. After restoring the original random state, B still completes its single additional optimizer update and preserves A.

The two initial tests failed at the missing required bindings. A first implementation that returned the digest to every audit caller failed four compatibility assertions. The opt-in actual audit path restored the original default metadata, and the final regression passed 246 CPU tests across two runs. The [readiness record](../../contracts/dh116_teacher_actual_resume_binding_v1.json) binds the failed and passed logs.

Antigravity R3 independently matched the Ubuntu source and confirmed the next authorization boundaries. The companion can bind a producer package and receipt digest, while the parent grant carries the verified exact checkpoint path and state digests. The scheduler should recheck A before grant and after B exits. The in-loop lease checks continue to validate the frozen files, role and kernel lock. This package ran no Isaac or native PPO and leaves B launch qualification disabled.
