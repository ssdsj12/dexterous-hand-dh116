# Dual Pinch exact 2+8 post-exit audit: CPU preparation

Date: 2026-11-13
Base: f7e6f15ec969eca65858ea188bb5aa1b65a5f21e
Branch: codex/dual-pinch-postexit-f7e6f15

## Interface and qualification

The independent helper is:
scripts.validate_dh116_dual_pinch_interval.audit_outcome(output, outer_exit, cleanup)

It returns an audit dictionary. The dedicated outer wrapper writes that result
as dual-pinch-interval-post-exit-audit.json; this helper starts no resource and
writes no artifact itself. It does not modify the original prepare entry,
contact module, protected source, native recipe, registry or readiness.

lifecycle_qualified means exactly one completed zero-action diagnostic interval,
two SDK setup steps plus eight measured SDK/rod steps, followed by the required
release/STOP/App close and independent normal inferior/outer exit evidence.
It never means contact quality, full SimulationContext disappearance or training.
same_rod_contact_qualified, grasp_qualified and teacher_admission stay false;
hand_table_contacts is None, hand_table_source_available is false and
no_hand_table_contact_proven is false on both success and failure.

## Independent checks

The helper reopens original artifact bytes and verifies:
- outer exit 0, observed native lock, finite duration <=180 s, actual log SHA,
  verified inferior exit 0 without signal, and matching outer/debugger status;
- no fatal capture or GDB command-error marker; normal finalize has all seven
  nonempty hash-bound products, actual stack frames, RIP/instruction and full
  mapping attribution through the existing capture_evidence helper;
- stopped_thread_count is an integer greater than zero, without requiring an
  implementation-dependent exact thread count;
- result.json and acquisition have matching 64-hex source fingerprints and
  exact 10/8/8/0 execution counters, no producer failure or false process-exit
  claim, and the original preparation VERSION;
- the original inspect_interval replay reconstructs the durable journal,
  exact zero 24D decision, 66/371 state boundary, eight static table NA rows,
  no terminal guard failure and one settled interval;
- standard native_physics_budget and physics_clock agree between result and
  acquisition and with every journal step; mandatory sdk_clock.jsonl must
  contain phase 1..8 mapped exactly to SDK 3..10;
- every physical row retains the 0.0021 m wrist projection floor.

The initial geometry check binds the exact private paired seed and backend
identities, original mounted 17DOF names, no-step reset clock, two source
geometries and actual reset articulation identities. Backend articulation may
be either the hand container or its /wrist_fixed path, but it must agree with
the corresponding recorded reset hand. No alternate unrelated path passes.

It independently recomputes source17 and locked historical pad5 geometry per
hand using the original validator and actual captured link poses. The original
table center/size comes from the source geometry reference layout. The rod
nodes use the exact proposal cast to np.float32, matching the actual builder
constructor input; radius is unchanged and thresholds are exactly 0.0021 m
table gap and 0 rod gap. Computed source/locked minima must agree with the
initial live clearance summary within 1e-7 m. This is initial geometry replay,
not proof of live PxShape attachment or contact absence at all later frames.

## Ownership and SimulationContext observation

Required weakref phases are dual_reset_complete, owned-reference drop,
before_stop, after_stop, before_app_close and after_app_close. Each has a
process map. Original builder owners must initially be observed alive;
table_contact_source is registered just after the first phase.

All recorded non-Simulation resources, including any extra recorded owner,
must be false after STOP and before/after App close. Manager root view release
must be reported by exactly one SDK STOP event. Every available lifecycle
clock stays at SDK 10; unavailable clocks must not contain invented values.

Simulation liveness is reported phase by phase, including unknown. It is not
forced false. The original lifetime_released_before_close boolean and the
pre-close marker must agree with the complete observed resource set, so a
still-live SimulationContext honestly leaves that original boolean false.
Execution qualification can nevertheless pass when every non-Sim owner,
manager STOP, cleanup, close and independent process-exit gate passes.

simulation_clear_receipt.json must be the exact known helper return:
explicitly_unsubscribed, unsubscribe returned, subscription field cleared,
original clear_instance returned, zero extra GC calls and zero physics steps.
It must retain native_release_verified=false and unknown post-return simulation
liveness. The helper does not reinterpret this receipt as proof of object death.
The exact prior v3 clean-import boundary is required.

## CPU tests and evidence

The positive unit fixture reuses the complete original CPU builder/episode
physical journal with real source geometry. Its outer/GDB/liveness/clear fields
are explicitly synthetic test data, never promoted to a native receipt.
The positive test preserves simulation alive=True and the original full-release
boolean false while proving the intended execution-only audit decision.

Negative tests cover failed outer status/duration, marker-only or empty GDB
capture, missing manager release, live/unknown/malformed non-Sim owner, dishonest full
liveness flag, failed ready unsubscribe, wrong or absent scheduler clock,
source identity mismatch, early USD import, paired seed change and lowered
geometry threshold. Semantic seed/threshold tampering also updates the
producer journal digest, so an unchanged checksum is not the only rejection.

New command:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_pinch_postexit.py -x

17 passed in 14.47 seconds. A missing-module red test and the concrete original
FP32-node versus JSON-double mismatch preceded the passing implementation.

Existing immediate regressions:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_physx_lifetime_validation.py tests/test_dh116_dual_hand_evidence_journal.py

25 passed in 4.87 seconds. Final total: 42 passed, 0 failed. Compile and the
528+2 protected-file checks passed. No native/GPU/GDB execution or push occurred.

Evidence directory:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-postexit-cpu-5vgop0b3

Audit module SHA256:
29c3480591c3a822d795867eeb961e134a33cf2795dc1104f924c7893fbc0cb9

Future exact native worker/source approval, launch, outer capture and publishing
remain separate. A successful audit cannot close the known static GPU
hand-table capability gap or authorize Teacher PPO.
