# DH116 Pinch v9 wall truncation and profile verification

## Native attempt

The single consumed Pinch v9 attempt used the freshly admitted right Reset on GPU 3. The ledger contains 234 contiguous records, two SDK initialization steps and 48 delivered and durably recorded measured steps. Measurement covered SDK steps 3–50, 576 internal solver ticks and six complete eight-substep control windows. No guard fired. The observed minima were 2.160193 mm on the left hand and 2.236924 mm on the right hand against the existing 2.1 mm floor.

The collector started with 77.304 s remaining. The five completed nonterminal endpoints had 66.086, 53.000, 39.872, 26.548 and 13.092 s remaining. The sixth had zero. The physical interval reached 0.100 s of the 0.533333 s episode cap. GAE recorded a valid original endpoint bootstrap and masked carry on external truncation. The single-episode worker stopped with `fresh_verified_reset_factories_required`, zero accepted PPO updates and no checkpoint. The natural outer process exit was 1 after 194.952 s, with no GDB inferior signal. The close returned through the fault path, with resource release unverified by the lifecycle gate.

Attempt SHA `1484e90ac79677f02d965750b1cf2b1cc40886a6155ceee3bad2365d9322c7a2`. Receipt SHA `aeb29f649367ba8823f067987d32d6579502e219a09d985ef76069939f0adc1b`. Original 53,583,709-byte ledger SHA `24cf6d9c3bdc574ec09cb80c26b4f7cf6275af49af25a411b94294c257e17593`. Raw logs and the lifecycle observations are stored with the original Ubuntu attempt. The registry records the consumed fingerprint as failed native evidence.

## Independent profile review and CPU verification

The recorded control path enters the profile reader scope up to 18 times per full window and checks the profile again in decision and bootstrap input packs. Antigravity independently audited that the complete regular-file/SHA checks occur inside `BackendProfile.load()` for every inventoried role and that frozen profile equality detects a missing, new or altered member. The patch removes the extra preliminary pass over the older copy. Each verification still reloads and rehashes the actual files and compares their paths, roles and digests. The source-basis loader, native hooks and reader scopes are unchanged.

The regression counts physical pad-file reads and detects corrupted bytes injected between independent calls. It reproduced two reads before the patch, then confirmed one read and rejection of corruption. The CPU suite passed 153 tests. One archived reader fixture was audited separately as a pre-existing failure. Its historical `read_locked` rejects the legacy aliased evidence path before reaching the reader under test. Unfiltered logs retain the failure both before and after the patch.

Seven interleaved CPU comparisons gave a median profile verification of 252.29 → 166.53 ms and a reader-scope median of 294.52 → 217.86 ms. Eight samples replayed from the original ledger produced a 91.03 ms median for independent geometry revalidation and 44.30 ms for encoding, serialization, write, flush and fsync to private scratch storage. These measurements isolate CPU costs. Full native wall-time attribution still needs native phase timing, including solver and stream waits.

## Recovery

The published code removes one repeated verification pass. At the observed throughput, this change alone does not provide the headroom for 32 controls. The next executable package is to audit verification reuse within the same synchronous native collection boundary, while retaining whole-inventory byte validation, ownership and per-step evidence. This package does not change the 32-control, 258-SDK-step or wall-clock budgets. Pinch remains unqualified for task success, and formal training remains gated.
