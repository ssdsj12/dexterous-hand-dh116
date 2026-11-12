# WireSeg fixed16/full128 read-only loss-domain audit

Base: 0475ebf62ef0e96dcf775c81aec979d5338de35b. The original training manifest
contains 128 rows; its checkpoint contract fixes the first 16 for training.
Those fixed 16 training samples and all 128 rows of the separate validation
manifest were audited with the preserved 128-update depth-only model.
No optimizer was constructed, no training update occurred, and no GPU or Isaac
run started. Model, loss, masks, normalization and manifests were not changed.

## Execution and exact reuse

The Ubuntu CPU process used two threads and an external 60s hard limit with a
50s internal alarm. It completed in 25.92s for the whole process, 23.85s internally,
exit code 0. All 288 requested image/view identities completed: 16 train plus
128 validation, each in center192 and full-field resize192. Exactly 16 completed
views from the earlier 4+4 audit were reused unchanged; 272 views received new
inference. Nothing remains missing. Full field means the complete view resized
to the existing 192 input, not native 1024-resolution depth acceptance.

The new audit binds reuse to the expected report SHA, exact checkpoint SHA,
file identity, RGB-depth pair SHA, split, family, view, original size and
implementation hashes. The earlier report can be reused only through its
reviewed audit/source fingerprints and unambiguous head-to-checkpoint bindings.
The post-run cache hardening and retrospective identity proof below address the
original report's missing helper/aggregation binding; they do not alter its
executed source identity or numerical records.
Changed model/data/view identities are rejected. New-format partial reports
retain those bindings, completed records, fixed denominators and explicit missing
image lists, so a later authorized audit can reuse completed views.

The loaded model state matched all original model tensors after inference.
The legacy import, all seven original evidence files and the earlier report
retained their exact hashes. Reused rows were compared field by field with the
earlier report. Constant depth remains the median of all valid full-frame labels
from the original 16 training images: 0.9858868718m. No heldout calibration was fit.

Private root:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-loss-domains-cpu-20261112`.

- `run01/result.json` SHA:
  `c639628246161ba4cbd2ae48b69a083cdb1bfd6a3f9bee8f7eea01c35791cf48`.
- `run01-process.json` SHA:
  `60405375e72b55080047a57e04d3108e2aa57b2dde21122804f81aaab0f807d6`.
- `run01.log` SHA:
  `0819807c6e472587d375b888dbff492cbfa3f9266fe88847f788431bb7abff2b`.
- Executed source SHA:
  `88d9348da8a191779267f55c6f9cbe4648b3a1afdfcf120e926a6f2ad40dbe18`.
- Source import SHA:
  `1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30`.
- Reused report SHA:
  `d999d36f61a15ba9135b795ab6065a353724fcbbc2bb7ac07ae9b91387eb0a70`.

## Post-run cache identity hardening (no new inference)

Review found that the initial four-file view binding omitted
scripts/audit_dh116_wireseg_depth_error.py, which supplies inspect_sample and
the metric/loss calls. The loader also did not validate the new-format
source_sha256. Subsequent code now binds the actual helper SHA, the aggregation
schema version, source text hashes of both aggregation functions, and the entire
executing audit script SHA. Changed helper/source/aggregation identities reject
reuse. Both legacy paths verify the currently installed helper before loading
any checkpoint. Known old full-cohort reports additionally require the reviewed
aggregation function hash; unsupported old sources remain rejected.

Current hardened audit source SHA:
`6f1e29bcbfe23b54be8c7c726e76070e79927d373ab54c68e4235209579458f3`.
The original run01 executed source remains
`88d9348da8a191779267f55c6f9cbe4648b3a1afdfcf120e926a6f2ad40dbe18`.

No 288-view inference was repeated. An independent proof file,
`run01-cache-identity-proof.json`, has SHA
`2583d25ab88677d270af129f39c36a56d5d8aa224712728bb5d377eea1504ded`.
It confirms:

- Original run01 report bytes and all original evidence hashes are unchanged.
- All four group summaries recompute exactly from the saved rows.
- The installed helper SHA is
  `ecb263c2c41ea31a061b09574c866da8f8cd8e6c7bbd7c9d1eaf260d2bd41125`.
- Both aggregation function sources hash to
  `37de45ce092b2ea5a6a20327289e7c52c167b8ff173fd9b768e22e4900cc175c`,
  matching the frozen implementation.
- Training manifest total is 128, fixed training subset 16, validation total 128.
- New model inferences and optimizer updates for this hardening are both 0.

This is a retrospective source-snapshot/current-byte attestation. The original
run01 did not contemporaneously record a helper SHA field, and the proof
explicitly preserves that fact. It enables reviewed legacy compatibility; it
does not rewrite the report as if that field had originally existed.

## Label support by original scene family

The fixed image-pixel denominator is 589824 per training view
(16 × 192 × 192), and 4718592 per validation view (128 × 192 × 192).
Observed and fixed denominators are equal because all images completed.
For partial runs, observed coverage and the fixed-denominator lower bound are
separate fields; unobserved pixels are not treated as invalid labels.

| Split / family / view | Completed / expected | Valid depth pixels | Valid wire pixels | Valid background pixels | Empty depth images | Empty wire-depth images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train / flying_wire / center | 16 / 16 | 25416 | 25416 | 0 | 3 | 3 |
| train / flying_wire / full field | 16 / 16 | 10786 | 10786 | 0 | 0 | 0 |
| heldout / wire_on_plane / center | 128 / 128 | 4718592 | 180608 | 4537984 | 0 | 35 |
| heldout / wire_on_plane / full field | 128 / 128 | 3684971 | 71746 | 3613225 | 0 | 0 |

The 4+4 support difference reproduces on these complete frozen manifests.
Training-center depth covers 4.3091% of pixels, exclusively wire ROI. Heldout-center
depth covers 100%; background accounts for 96.17% of its valid pixels. All 128
heldout images have background supervision, which these 16 training images
provide nowhere. Empty-wire heldout images remain in the all-valid/background
denominator and are not discarded.

## Per-image loss versus pixel aggregation

The following values evaluate the same frozen model. They are not a replay of
historical optimizer steps. Equal-image SmoothL1 averages only images with valid
depth, matching the domain of a fixed-order depth update; pixel-weighted
SmoothL1 weights each image by its valid-label count. Beta remains 0.01.

| Split / view | Supervised images | Valid pixels per supervised image | Equal-image SmoothL1 | Pixel-weighted SmoothL1 | Equal-image MAE | Pixel MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train center | 13 | 407–5474 | 0.135038 | 0.120134 | 0.139989 | 0.125063 |
| train full field | 16 | 411–1261 | 0.173429 | 0.170221 | 0.178379 | 0.175178 |
| heldout center | 128 | 36864–36864 | 0.133652 | 0.133652 | 0.138595 | 0.138595 |
| heldout full field | 128 | 19390–36864 | 0.289506 | 0.271162 | 0.294473 | 0.276126 |

Sparse variable training support causes a real weighting difference. Heldout
center support is identical per image, so its equal-image and pixel means
coincide. The heldout error gap therefore cannot be dismissed as an aggregation
artifact.

## Wire/background accuracy and range

All errors below are metres. Background sums are computed as all-valid error
sums minus valid-wire error sums, using their actual disjoint pixel counts.

| Split / view / domain | Model MAE | Model RMSE | Constant MAE |
| --- | ---: | ---: | ---: |
| train center / all-valid = wire | 0.125063 | 0.153009 | 0.082404 |
| train full field / all-valid = wire | 0.175178 | 0.213971 | 0.123277 |
| heldout center / all-valid | 0.138595 | 0.161518 | 0.078598 |
| heldout center / wire | 0.141680 | 0.167847 | 0.087620 |
| heldout center / background | 0.138472 | 0.161261 | 0.078238 |
| heldout full field / all-valid | 0.276126 | 0.337263 | 0.218553 |
| heldout full field / wire | 0.227882 | 0.281958 | 0.161336 |
| heldout full field / background | 0.277084 | 0.338270 | 0.219689 |

Model all-valid MAE exceeds the constant for all 128 heldout images in both views.
It also exceeds the constant on 10/13 supervised training-center images and
14/16 training full-field images. Since wire ROI itself underperforms the constant,
background-domain mismatch is not a sufficient or unique explanation.

Full-field range compression reproduces: heldout targets span
0.4384–2.1706m and predictions 0.7360–1.3697m. However, full heldout-center
predictions span 0.7286–1.5775m while targets span 0.7664–1.3931m.
The small-cohort observation must therefore remain specific to the full-field
view; a universal center-view compression claim is unsupported.

Exact pooled P95 cannot be reconstructed from cached per-image P95 values.
To avoid repeating completed inference, this audit reports pooled P95 as null
with an explicit reason. Per-image P95 remains in the unchanged/new records.
MAE and RMSE aggregate exactly from counts and error moments; no percentile is
invented or averaged into a claimed population percentile.

## Validation and next decision

New tests first produced four expected missing-feature failures and then a
missing cache-binding failure. Final related CPU regression:
Before identity hardening: 72 passed, 0 failed in 6.33s. Final hardening regression:
77 passed, 0 failed in 5.63s. Two added tests first reproduced missing helper/source
rejection; new coverage also rejects changed aggregation function code and changed
helpers in both legacy paths before any checkpoint read. An independent reviewer
reran all 10 cache/domain tests and read 288 old run01 and 16 old small-audit rows
through the hardened loader without inference. Tests cover exact image/pixel weighting, background
subtraction, invalid ROI support, fixed partial denominators/missing identities,
cache reuse without model inference, and changed checkpoint/report/view rejection.
Existing read-only, strict-resume, chained diagnostic, decoder and relay tests
also passed.

This package completes the requested full-manifest audit and starts no further
cohort expansion or training. Before any actual loss modification, decide whether
the objective is wire depth or full-scene depth and preserve both existing metric
domains during that decision. Full-scene learning needs appropriate training
background labels; the current heldout family must not be silently moved into
training. Wire-only reporting also cannot turn the current model into a success,
because its wire error still exceeds the constant. The observed domain and
weighting differences are reproducible evidence, not an isolated causal proof or
an approved loss change.

## Executed Ubuntu command

```sh
export PYTHONPATH=src:.:/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86/tools
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
/usr/bin/time -f '{"external_wall_seconds":%e,"process_exit_code":%x}' \
  -o /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-loss-domains-cpu-20261112/run01-process.json \
  timeout --signal=KILL 60s /mnt/mydisk/xk/envs/dh116/bin/python \
  -m scripts.audit_dh116_wireseg_loss_domains \
  --checkpoint /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-legacy-audit-20261112/depth-only-legacy-import.pt \
  --checkpoint-sha256 1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30 \
  --dataset-root /mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86 \
  --reuse-report /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-depth-error-cpu-20261112/run02/result.json \
  --reuse-sha256 d999d36f61a15ba9135b795ab6065a353724fcbbc2bb7ac07ae9b91387eb0a70 \
  --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-loss-domains-cpu-20261112/run01 \
  --wall-seconds 50
```

All requested identities are complete. A future separately authorized reuse must
use a new output directory; original reports are immutable.
