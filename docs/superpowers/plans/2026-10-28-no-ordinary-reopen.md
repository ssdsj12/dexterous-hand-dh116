# Retain accepted grasp without ordinary reopening

User requires accepted close pose to remain held while waiting for the other side and during lift; holding or tightening is allowed, ordinary reopening is not. Saved GUI trace proves signed force correction opens an accepted side even below overpressure; 20ms contact loss then clears pose ownership and resumes search. Eight-case trace proves lift progression is gated to1–2.5% samples, world root targets do apply the allowed increments.

First isolated correction: clamp held ordinary force movement to nonnegative closure; retain close pose ownership until explicit reset/failure, with current readiness still required to admit lift. Keep physical pressure/mimic/lead/mechanical governors and current lift gates unchanged for the first comparison. Tests must cover both hands, close and lift, ordinary excess force below pressure band, current contact loss blocking lift without releasing latch, selected reset, and real overpressure still corrected. Fresh8case physical comparison determines remaining lift starvation. No physical success claim without measured height/stability.


## Second evidenced correction

The first isolated fix passed151relatedtests and finished the360-step8case comparison (`success_teacher_20261028_184355_retain_no_reopen8`) without process failure, but max lift offset was only3.979mm and object lift success remained0. Independent reproduction confirmed a requested held command can still be opened by0.000416667rad in one step with mimicerror-0.006rad and measuredlag0.005rad, without severe error/pressure.

Accepted sides now retain their existing command against ordinary soft lead/mimic projection; those projections still bound new tightening. Per-hand severe coupling>=0.012rad, pressure relief, thumb limit/conflict recovery, mechanical bounds and slew remain authoritative. Any such protective correction is separately exposed as `held_safety_correction`. This deliberately changes soft-governor semantics, not the physical fatal guards.

Once both CURRENT close predicates admit lift, raise at the existing0.01m/s unless severe coupling, pressure or tracking blocks it; do not re-run close load/slip/soft mimic readiness on every lift step. Those two no-longer-blocking load/slip gate counters are zero. Actual physical success still requires original object height/stability and includes friction in lift force.

Regression tests initially reproduced7failures for mild-governor opening and repeated admission blocking. Final targeted controller/hold/world/training/probe suite181passed. Physical validation: `teacher_demo_20261028_184832_retained_pose_lift8`,8cases,GUI,600controlsteps. Outcome pending. No claim of physical lift success.
