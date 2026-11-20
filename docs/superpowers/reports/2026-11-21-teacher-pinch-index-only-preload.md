# Pinch left index-only preload

The worker preserves the first 45 windows and the original SDK 202 and 282 gates. At SDK 362 it rechecks current opposition, joint bounds and source/cooked swept clearance before adding +0.0001 rad to left finger21, global axis 2. The other 23 channels are held. Antigravity reviewed the exact source and gate order. Independent CPU recomputation gave the same 0.453 µm left minimum margin with the original flexion certificate and the alternate abduction certificate. The worker used the original flexion certificate, with unchanged guards.

One GPU 2 run completed 55 control windows, 442 SDK steps including setup, and 5,280 DeformX substeps. All three boundary gates passed and no guard fired. App.close returned normally, outer exit was 0 and the independently recomputed raw ledger SHA matched the outer receipt.

| Settled interval | Both hands above original opposition and load gates | Left-only failures | Right-only failures | Left signed opposition lost | Left combined median N |
| --- | ---: | ---: | ---: | ---: | ---: |
| SDK 299–362, unchanged prefix | 420/768 | 341 | 7 | 0 | 0.30415 |
| SDK 379–442, following index increment | 293/768 | 444 | 15 | 39 | 0.29225 |

The longest consecutive qualifying interval after the increment was 17 solver substeps. Four pads remained above 0.05 N in all compared substeps. The left-sided load-deficit P95 rose from 50.95 to 80.97 mN on deficient samples. The minimum table clearance was 2.565 mm after the change, and the maximum captured force residual was under 0.833 µN. Across the full 960 post-increment substeps, 30 of 41 lost signs had an outward left thumb net-force projection with the thumb normal still pointing inward. Table contact remained present. No lift, sustained hold or PPO update was attempted.

The candidate is rejected for motion promotion based on its measured post-command outcome. The two equal-length windows come from different times. A no-increment continuation from SDK 362 is needed to estimate how much of the later deterioration occurs without an index change.

368 CPU regressions passed. The exact worker is `python -B -m scripts.run_dh116_pinch_index_only_preload`. [Readiness and raw evidence identities](../../contracts/dh116_teacher_pinch_index_only_preload_v1.json).

The subsequent [same-initial-state control](2026-11-21-teacher-pinch-same-state-force-control.md) measured 293 versus 197 qualifying substeps at matched SDK times. The candidate remains below the requirements for sustained hold.
