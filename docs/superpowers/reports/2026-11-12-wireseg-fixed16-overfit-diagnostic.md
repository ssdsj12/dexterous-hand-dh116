# WireSeg fixed16 diagnostic: masks, crops and depth validity

Source baseline: eb31c793d60cb50eb8d6dd2e06bae2f15c5bd8d5. Ubuntu only; GPU 2.
Scope: small-set learning diagnostic; not native Isaac, not stereo correspondence,
not Teacher/Student qualification. Original checkpoints remain read-only.
Pinned HF revision: 1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b.

## Implementation and tests

scripts/diagnose_dh116_wireseg_overfit.py reuses verified_sources, read_subset,
prepare_depth, training_sample, WireSegPretrainer and checkpoint writer.
It does not replace subset or resume. A fresh independent model/AdamW is created
for each head; this is not an unrecorded resume. Fixed 16 exact RGB/depth identities,
central 192x192 crop, seed116, LR0.0003, at most128 updates/head and300s optimization.
Depth-only does not train segmentation heads; seg-only does not train depth head.
Gradients, loss, visits, empty masks, target/prediction coverage, depth quantiles,
source SHA, pair SHA and independent checkpoints are recorded.
Full-field 192x192 resize is an additional evaluation view, never reported as
improvement over the original central crop. RGB is [0,1]; mask and depth use
nearest-grid sampling, metric depth is not rescaled and is not assumed Optical-Z.
The native rigid/flexible task camera defaults and physics are untouched.

CPU: 18 passed (7 new +11 existing). RED logs first showed the missing diagnostic
functions, then a missing-label update guard. Tests verify empty masks remain real
negative segmentation labels, full-field retention outside crop, unchanged depth
units, invalid exclusion, ROI metrics, independent head gradients, cohort identity
and training-only constant baseline. Logs: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-overfit-eb31c79/red.log,
red-missing-label.log, green.log.

## Actual data and failure retained

Same fixed16 flying_wire training rows, first16 of the existing verified paired
manifest; same128 wire_on_plane validation identities. Source families are disjoint.
No official real-test rows used. All16 have complete exact depth files, but3
central crops have no wire and no positive finite depth labels. All16 remain in
segmentation training and every evaluation. The initial run finished128 seg updates
then correctly failed at the first empty depth crop. Failure log and source SHA
be68eb2fbf4ec5ed96d1a8f9a539881201b8acbff03591d8883832ed8a974c24 remain in run01; no label was fabricated.

The corrected depth-only run recorded30 skipped unlabelled visits and128 actual
updates. No fabricated zero-loss update and no automatic removal from metrics.
The source dataset flying_wire has valid depth only at wire pixels in these crops;
wire_on_plane includes background depth. Therefore all-valid and wire-ROI metrics
are equal on the fixed training crops but different on validation.

## Measured outcomes

| Measurement | Before | After |
|---|---:|---:|
| Fixed16 central segmentation IoU |0.042608|0.536192|
| Fixed16 full-field segmentation IoU |0.018755|0.147281|
| Fixed16 central depth MAE (m) |0.711040|0.125043|
| Fixed16 central depth RMSE (m) |0.730461|0.152990|
| Fixed16 central depth P95 (m) |0.926811|0.285123|

Seg loss first/last 4.131881/1.421799; depth loss
0.735146/0.179168; gradients finite and positive.
These different images' first/last step losses are optimization traces; the
before/after fixed-cohort metrics above are the actual paired comparison.
The seg-only model's untrained depth output is not a depth optimization result.

| Heldout128 | Central crop | Full field resize |
|---|---:|---:|
| Seg-only visible IoU |0.182945|0.062931|
| Depth-only all-valid MAE (m) |0.138590|0.276113|
| Depth-only wire-ROI MAE (m) |0.141673|0.227875|
| Depth-only wire-ROI RMSE (m) |0.167838|0.281949|
| Depth-only wire-ROI P95 (m) |0.292475|0.561846|

Central validation has35 empty-mask crops; full field has0. Those are included,
not discarded. The scalar median baseline fitted on training depth only has
central train MAE0.082404m and heldout all-valid/wire-ROI MAE0.078598/0.087620m:
the trained depth network still underperforms this simple constant.
128 updates demonstrate gradient/data flow, not overfit convergence or
millimetre-level accuracy. Further optimization must diagnose this gap and
separately handle crop/domain bias before increasing data/architecture.

CUDA max tensor allocation: seg 256094208 bytes, depth
245364224 bytes. Depth optimization 1.841s;
seg update trace gives the optimization elapsed separately (initial result's
optimization_wall_s includes evaluation overhead). Tensor allocator peak is not
NVML total process memory. No Isaac was launched by this work package.

## Commands and durable identities

PYTHONPATH=src:.:<dataset>/tools CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
/mnt/mydisk/xk/envs/dh116/bin/python -m scripts.diagnose_dh116_wireseg_overfit
--dataset-root <dataset> --subset-manifest <existing_verified_depth_pair_subset>
--validation-manifest <existing_validation128> --output <new_private_directory>
--samples 16 --steps 128 --size 192 --device cuda:0 --wall-seconds 300
(use separate tokens --samples 16, etc.; second run selected --heads depth-only).

Train manifest SHA38159e2f94a7528f983a1b9dca3a74f9f8bb91e5576046d31c9413b386ae905d.
Validation manifest SHA5d5cc1331115737bfd039fa8223bd0f23cd67ea385595ffc0f03e71e6115f78f.
run01 result SHA ccc074f599a2bb77052172553d8c3279d13538400c07318501b71834d479a8ca.
run02 result SHA 1c34ca7c8f2f848b198f16598510e5c82d56ae35d77ef0db198808bc4aced5d9.
Seg checkpoint SHA77459b47f26ccfdeea024f298fe59bedbb9c119242cddda305aeea0b71cf5f39.
Depth checkpoint SHA56826811b64075446486fc0309a60a616dabcafc738d40700be04ec39ac21adb.
Private root: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-overfit-eb31c79. Raw rows, models, logs and cache not uploaded.

Independent download resumed4 validated16MiB chunks (64MiB) of
data/train-00000-of-00045.parquet, expected full SHA
17ddf17fe837d17027780dda78a916b4874403c3be0e320cb95d06977d207018.
This incomplete blob is not included in training or verified5/148 count.
Download incompleteness does not gate no-camera Teacher.

Next: preserve failed depth run; diagnose constant-baseline gap with fixed data,
keep independent validation frozen, continue bounded download. No generalization,
stereo, force-feedback or grasp success qualification is granted.
