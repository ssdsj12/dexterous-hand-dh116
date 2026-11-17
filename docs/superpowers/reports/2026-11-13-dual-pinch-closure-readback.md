# Dual Pinch closure: archived measured response readback

Base: 6cab12fb2416ae3a6d6d13be18c8247c5a46b507. This is a CPU summary of the already completed 2+8+8 native run. There are no new physics steps, SDF calculations, geometry search states, R0 evaluations or optimizer updates. Only this report, one read-only CLI and its tests are added; the worker, profile and physical thresholds are unchanged.

## Measured response

The four positive finger commands reached the actual PhysX backend targets. During the 16.67 ms action window, measured thumb/index q moved only about 33–34% of those target increments. The four saved required-pad node-SDF gaps decreased but remained positive; no hand/rod raw pair was recorded. This establishes the direction of this single measured response, not contact establishment, later tracking or a safe repeated closure.

The first action is 24 zeros. The second has indices 1,2,7,8 equal to 0.1,0.1,0.05,0.1; all other components, including wrist/lift axes, are zero. The unchanged joint rate is 0.3 rad/s. Both sides use positive thumb/index commands; right thumb has half velocity. Backend targets are SDK commands, while q/qd are actual SDK measurements. Error is backend target minus q.

The zero window has zero active backend-target increments. Measured active q is unchanged at stored precision except right finger12 (+3.72529029846e-9 rad); nonzero qd remains recorded. The following table describes SDK10→18 and includes all six active joints per hand, including uncommanded drift. Full per-SDK requested targets, backend targets, q/qd and both windows are in the JSON result.

| Side / joint | Target delta (µrad) | Actual q delta (µrad) | Actual q end (rad) | qd end (rad/s) | Target − q end (µrad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| left / finger11 | 0.000000 | -1.072884 | 0.2881713510 | -2.15038144e-05 | 1.072884 |
| left / finger12 | 499.963760 | 167.518854 | 0.4573759437 | 0.0163496546 | 332.444906 |
| left / finger21 | 500.202179 | 169.634819 | 0.6627964973 | 0.0163953546 | 330.567360 |
| left / finger31 | 0.000000 | 0.000000 | 1.1499999762 | -1.21206426e-06 | 0.000000 |
| left / finger41 | 0.000000 | 0.000000 | 1.1499999762 | 2.26055477e-08 | 0.000000 |
| left / finger51 | 0.000000 | 0.000000 | 1.1499999762 | 7.35883532e-09 | 0.000000 |
| right / finger11 | 0.000000 | -0.566244 | 0.4800276160 | -1.02574613e-05 | 0.566244 |
| right / finger12 | 250.011683 | 83.710998 | 0.0376987159 | 0.00817392021 | 166.296959 |
| right / finger21 | 500.202179 | 169.694424 | 0.6823298931 | 0.0163933299 | 330.507755 |
| right / finger31 | 0.000000 | 0.000000 | 0.2752428651 | -2.26110728e-07 | 0.000000 |
| right / finger41 | 0.000000 | 0.000000 | 0.5636945367 | 1.2422224e-07 | 0.000000 |
| right / finger51 | 0.000000 | 0.000000 | 0.2510532439 | 1.62747767e-07 | 0.000000 |

## Required-pad SDF and contact observations

These are already recorded geometry_from_measured_pose values from locked
node-SDF grids and actual same-clock link/rod readbacks. They are phi-minus-radius,
not native contact force or pressure. The minimum node remains left thumb 35,
left index 34, right thumb 4, right index 3 throughout. Lattice coverage and positive
gradients remain present; all saved contact-predicate node lists are empty.
First post SDK3 values equal zero-end SDK10 values. No reset-time SDF is invented.

| Required pad | First/zero-end gap (mm) | Action-end gap (mm) | Change (µm) |
| --- | ---: | ---: | ---: |
| left / finger14_link | 1.613433473 | 1.603886951 | -9.546522 |
| left / finger23_link | 2.054471988 | 2.040942200 | -13.529789 |
| right / finger14_link | 1.867071260 | 1.861801837 | -5.269423 |
| right / finger23_link | 2.454290166 | 2.439360134 | -14.930032 |

The other three pads per hand (finger33/43/53) have no saved SDF in this run;
they remain null/NA. No additional SDF sampling is performed.

The ledger contains 7,872 rod/table raw pair records (41 nodes × 12 substeps ×
16 SDK samples), zero hand/rod raw pairs. These are substep/node/mesh records,
not 7,872 distinct contact objects. Static GPU hand/table contact remains NA:
"static GPU table filter unsupported; empty buffers cannot prove absence".
Hand/table counts stay null and no-hand/table-contact is not proven.

## All recorded source17 and locked5 table clearances

Values below are mm. Start means first measured post-SDK3, zero-end SDK10,
action-end SDK18. Minimum is over 16 existing samples, not a continuous path
bound. Geometry is original source17 plus locked5, not new live cooking of
the twelve non-pad colliders. No FK or collision query is rerun here.

### Left

| Shape | Start SDK3 | Zero-end SDK10 | Action-end SDK18 | Minimum |
| --- | ---: | ---: | ---: | ---: |
| cooked:finger14_link | 3.097819036 | 3.097819036 | 3.107744791 | 3.097819036 |
| cooked:finger23_link | 3.188979961 | 3.188979961 | 3.195354853 | 3.188979961 |
| cooked:finger33_link | 38.988853029 | 38.988853029 | 38.988853029 | 38.988853029 |
| cooked:finger43_link | 56.485162033 | 56.485162033 | 56.485162033 | 56.485162033 |
| cooked:finger53_link | 73.938516303 | 73.938516303 | 73.938516303 | 73.938516303 |
| source:base_link | 39.344026315 | 39.344026315 | 39.344026315 | 39.344026315 |
| source:finger11_link | 56.030199110 | 56.030199110 | 56.030199558 | 56.030199110 |
| source:finger12_link | 18.223186276 | 18.223186276 | 18.224916983 | 18.223186276 |
| source:finger13_link | 2.119988497 | 2.119988497 | 2.129694519 | 2.119988497 |
| source:finger14_link | 3.018896285 | 3.018896285 | 3.029097598 | 3.018896285 |
| source:finger21_link | 16.935628136 | 16.935628136 | 16.935512756 | 16.935511156 |
| source:finger22_link | 2.110836484 | 2.110836484 | 2.117672500 | 2.110836484 |
| source:finger23_link | 2.773530901 | 2.773530901 | 2.780531144 | 2.773530901 |
| source:finger31_link | 36.241914490 | 36.241914490 | 36.241914490 | 36.241914490 |
| source:finger32_link | 36.266878478 | 36.266878478 | 36.266878478 | 36.266878478 |
| source:finger33_link | 38.848978787 | 38.848978787 | 38.848978787 | 38.848978787 |
| source:finger41_link | 54.662663216 | 54.662663216 | 54.662663216 | 54.662663216 |
| source:finger42_link | 54.852466159 | 54.852466159 | 54.852466159 | 54.852466159 |
| source:finger43_link | 56.340022490 | 56.340022490 | 56.340022490 | 56.340022490 |
| source:finger51_link | 72.288094263 | 72.288094263 | 72.288094263 | 72.288094263 |
| source:finger52_link | 72.369439957 | 72.369439957 | 72.369439957 | 72.369439957 |
| source:finger53_link | 73.805089960 | 73.805089960 | 73.805089960 | 73.805089960 |

### Right

| Shape | Start SDK3 | Zero-end SDK10 | Action-end SDK18 | Minimum |
| --- | ---: | ---: | ---: | ---: |
| cooked:finger14_link | 4.144049822 | 4.144049822 | 4.151581534 | 4.144049822 |
| cooked:finger23_link | 4.082778213 | 4.082778213 | 4.083676303 | 4.082778213 |
| cooked:finger33_link | 21.842686004 | 21.842686004 | 21.842686004 | 21.842686004 |
| cooked:finger43_link | 25.878561848 | 25.878561848 | 25.878561848 | 25.878561848 |
| cooked:finger53_link | 55.725389972 | 55.725389972 | 55.725389972 | 55.725389972 |
| source:base_link | 55.758488622 | 55.758488622 | 55.758488622 | 55.758488622 |
| source:finger11_link | 49.117138875 | 49.117138875 | 49.117143681 | 49.117138875 |
| source:finger12_link | 14.585745658 | 14.585745658 | 14.587531374 | 14.585745658 |
| source:finger13_link | 2.116345323 | 2.116345323 | 2.122657356 | 2.116345323 |
| source:finger14_link | 3.576266049 | 3.576266049 | 3.583555769 | 3.576266049 |
| source:finger21_link | 40.199903508 | 40.199903508 | 40.196536920 | 40.196536920 |
| source:finger22_link | 3.738398317 | 3.738398317 | 3.738914877 | 3.738398317 |
| source:finger23_link | 4.082778213 | 4.082778213 | 4.083676303 | 4.082778213 |
| source:finger31_link | 60.233972879 | 60.233972879 | 60.233972879 | 60.233972879 |
| source:finger32_link | 22.146581195 | 22.146581195 | 22.146581195 | 22.146581195 |
| source:finger33_link | 21.805053169 | 21.805053169 | 21.805053169 | 21.805053169 |
| source:finger41_link | 63.677376076 | 63.677376076 | 63.677376076 | 63.677376076 |
| source:finger42_link | 25.802483472 | 25.802483472 | 25.802483472 | 25.802483472 |
| source:finger43_link | 25.878561848 | 25.878561848 | 25.878561848 | 25.878561848 |
| source:finger51_link | 79.181144653 | 79.181144653 | 79.181144653 | 79.181144653 |
| source:finger52_link | 56.116574693 | 56.116574693 | 56.116574693 | 56.116574693 |
| source:finger53_link | 55.492203590 | 55.492203590 | 55.492203590 | 55.492203590 |

Global minima are 2.110836484 mm (left source finger22) and 2.116345323 mm
(right source finger13), only 10.836484 and 16.345323 µm above the unchanged
2.1 mm runtime floor. Those limiting endpoint gaps increased to 2.117672500
and 2.122657356 mm during this window; that observation is not extrapolated.

Minimum saved non-target rod surface-gap lower bounds are left 2.351110368 mm
and right 0.741537236 mm. They retain the original all 41-node convex-support
calculation and its 2 µm guard; no new bound is computed. Maximum wrist tracking
errors are 1.27272418e-8/2.02002694e-8 m, maximum world COM linear speeds
1.89874925e-6/1.40570171e-6 m/s. No wrist command or lift is added.

## Identity and replay boundary

The archived receipt has process/outer exit 0, tested HEAD
981ebc91a50398638b19a05572aa9594b088a955, qualified only for
dual_pinch_same_free_rod_one_guarded_finger_action_interval. This new summary's
own native/grasp/Teacher flags remain false.

The receipt registry recipe SHA 74f8378ee4648f30c8ecb2ba8f0aa5f2caf9ef396a8e49b9c532cbf44ff7fad6
and executed operation SHA 4e6dd4284c433253cddbab4c1ed610501dc13155f6e2e0d2756d50458389b463
are distinct identities. The latter is checked across result/acquisition/audit.
The existing post-exit audit is consumed by exact SHA; the native acceptance
validator is not rerun. The new summary grants no execution qualification.

| Bound source | SHA256 |
| --- | --- |
| receipt | dd64cce38312cb3fe044e210c8560f44fae33b218e5b8e2ba97435a3969c9bc0 |
| physical.jsonl | 140584e4375b96343478ec973fdbdb94f62d8a2a687fc8dd3b1ba1a5db1a2ec9 |
| native result.json | fe5165f7cb0e9b77e2ef52ca1b0e4c5d7813b98d17b25d88f5444bf77078ecb5 |
| sdk_clock.jsonl | 02782ca1c0ea903148677de9b8d6ac9bbe74000e4d534accce3666293f83048a |
| existing post-exit audit | 80abc7aadcb9f2b5ec03084df5c200497360bcee18e6e5d6ffa3a61e88f9d2c9 |
| unchanged profile | 67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498 |
| parent zero receipt | aa1a6877c29b460e8f12589674d4ba09381faf8aea200400a7cd0e71d1a5a1ec |
| paired reset | 996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37 |
| reset seed | 647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d |

Run UUID is dual-pinch-3ba8faa9acb94efeb8766ce1ee0b313c; episode is that
UUID plus :one, env 0, reset 1, rod rod0. Reset is SDK 2 / rod tick 0. Zero window SDK 3–10 /
rod ticks 12–96; action window SDK 11–18 / rod ticks 108–192. Every SDK increment is 1 and rod
increment 12. SDK timestamp spans 0.00625–0.0375 s; final float32 solver time is
0.03333333507180214 s. The existing float32-aware clock checker is reused.

The original journal parser/sequence reducer checks exact event order, episode,
reset, rod and two complete windows. Existing sdk_clock_sample checks actual
SDK/rod/post clocks. Additional checks bind named active indices, finite 17DOF
q/qd/targets, actual submitted/backend targets, post→next-pre joint continuity,
geometry raw SHA and copied q/node identity, source bindings, complete 22-shape
table values and required SDF minima. These are consistency checks on saved
measurements. Numerical geometry trust comes from the exact ledger and its
existing post-exit audit; no new geometric certification is claimed.

## CPU validation and durable evidence

The actual summary completed in 2.90 s external wall, exit 0 under an external
30 s bound, two CPU threads and empty CUDA_VISIBLE_DEVICES. No native/Isaac/USD/
solver module was imported. New physics/optimizer/search states are all zero.

Tests first failed on missing implementation (2 failed, 19 fixture errors).
All 21 new tests then passed; combined with existing journal tests, 30 passed
in 12.16 s. New tests use archived arrays and damaged copies, not SDK owners;
the old journal regression uses explicit CPU doubles. Cases reject incomplete
or swapped windows, action/episode/rod-clock mismatch, named index changes,
nonfinite measurements, geometry SHA/source mismatch, missing table shapes,
inconsistent saved minima, receipt/artifact SHA mismatch and null→zero table
fabrication. CLI subprocess verifies deterministic strict JSON and no native
imports; occupied outputs reject before replay and are created exclusively.

~~~sh
cd /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/dual-pinch-closure-readback-20261113
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
audit_dir=$(mktemp -d /home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-closure-replay-XXXXXX)
/usr/bin/time -f 'elapsed_seconds=%e exit_code=%x' -o "$audit_dir/process.txt" \
  timeout --signal=KILL 30s /mnt/mydisk/xk/envs/dh116/bin/python \
  -m scripts.audit_dh116_dual_pinch_closure_readback --output "$audit_dir/result.json"
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_dual_pinch_closure_readback.py \
  tests/test_dh116_dual_hand_evidence_journal.py
~~~

Private output:
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-closure-readback-20261113

| Output | SHA256 |
| --- | --- |
| result.json | aa0a34ac373da7170e462e87d22e0b0f32635965143db829cfa8ceab6782afe3 |
| actual-process.txt | c5fd3901babf7025b32f3709fbc3d64f2ba301538b5dbc1a5a1e131277259e47 |
| preservation-proof.json | e7659d0d617402fc063687b87f58ce44a4ab265d078a1bbce620a7e08297041d |
| actual.log | 84e196971d45caaa78f7ba385e61bc21f5acbbc24f9044cc2f3e020013ec1cbb |
| tests.log | 6a069ed6bf234a737bbe10e3d19fdcbacdbf2a7e8ad62f4f6830f05798f6b89d |
| tests.xml | 44b6b6f045ce71d99fd9e54fa9b39d485b2fd7cb49556db52e6c505e226bdcfa |
| red.log | 251fe64180c7ed34861475af9f58c7f908019114b8bd782d6ca371fa1da3b315 |

The result includes consumer implementation hashes and original geometry/SDF
source bindings. Original ledger, result, receipt, profile and all model/data
files remain untouched. This package requests no further native run and changes
no frozen recipe. It supports evaluating the actual short response only.
