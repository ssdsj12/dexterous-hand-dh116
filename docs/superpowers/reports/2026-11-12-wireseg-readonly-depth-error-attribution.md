# WireSeg read-only CPU depth-error attribution

Base: 4894f5fd874dcf209e5bfcdc52c35c275b15af55. This package adds a read-only
diagnostic and CPU tests. It changes no model, loss, normalization, dataset or
training configuration. It constructs no optimizer and performs zero actual
training updates, GPU work or Isaac runs.

## Scope and preservation

The two original 128-update segmentation/depth checkpoints were read through their
explicit legacy imports. The inference cohort is the first 4 original train images
and first 4 independent heldout images, each evaluated with the original center 192
crop and full-field resize to the same 192-pixel network input. Full-field here
means the complete view resized to 192; it is not native 1024-resolution depth
acceptance. The original frozen train cohort remains 16 and
validation manifest remains 128; this report does not claim inference coverage of
all 16/128. Training-only constant depth is computed from all 16 original training
images' full-frame valid labels, matching the historical 0.9858868718m baseline.

The new script reuses the existing decoder, network, view generation,
diagnostic_loss, depth_metrics and quantile helpers. It accepts legacy-import and
chained diagnostic-continuation formats through their existing strict evidence
validators. No network is rewritten. Every finished inference record is published
to an independent atomic partial report; a 50s internal alarm preserves completed
records, with an external 60s hard process limit. A hard kill leaves the last
published partial report and is not reported as completed.

Initial run01 completed 32 records in 6.06s internally. After adding an explicit
model-state immutability assertion, run02 completed 32 identical records in 5.24s
internally, 7.31s for the entire process, exit code 0. Both runs used 2 CPU threads,
CUDA_VISIBLE_DEVICES empty, and torch.inference_mode. Both model state dictionaries
matched their loaded checkpoint tensors after inference. Both imports and all
seven original evidence bindings per head retained their exact SHA256 values.
The repeat solely verified the new immutability assertion.

Private root:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-depth-error-cpu-20261112`.

- Final result `run02/result.json` SHA:
  `d999d36f61a15ba9135b795ab6065a353724fcbbc2bb7ac07ae9b91387eb0a70`.
- Process receipt `run02-process.json` SHA:
  `2210581704173724353da9fb1318b0594e6ea274ace0f39b71495c305814a334`.
- Log `run02.log` SHA:
  `24fd9492c9f07ca0088139c22ba1d06f2aa6414770089836e186935e949f856c`.
- Executed audit script SHA:
  `ecb263c2c41ea31a061b09574c866da8f8cd8e6c7bbd7c9d1eaf260d2bd41125`.

The source import hashes remain depth
`1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30`
and segmentation
`776603d4b7008734ec25f157fb84a82411ebb78e18588c58281db5376d55fe0e`.
Original checkpoint hashes remain depth
`56826811b64075446486fc0309a60a616dabcafc738d40700be04ec39ac21adb`
and segmentation
`77459b47f26ccfdeea024f298fe59bedbb9c119242cddda305aeea0b71cf5f39`.

## Observations on this4+4 cohort only

All distances below are metres. Metrics aggregate valid pixels, not the mean of
per-image errors. One of the four training center crops has no valid depth; its
empty record remains present and no fabricated loss is assigned.

| Depth-only model / view | Domain | Valid pixels | MAE | RMSE | P95 | Constant MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train center | all-valid = wire ROI | 2031 | 0.143279 | 0.171896 | 0.331757 | 0.041911 |
| Train full field | all-valid = wire ROI | 3295 | 0.175254 | 0.212478 | 0.387825 | 0.108803 |
| Heldout center | all-valid | 147456 | 0.156147 | 0.177374 | 0.284888 | 0.104036 |
| Heldout center | wire ROI | 11108 | 0.146727 | 0.167421 | 0.285772 | 0.116694 |
| Heldout full field | all-valid | 101451 | 0.355890 | 0.410086 | 0.652437 | 0.287742 |
| Heldout full field | wire ROI | 3185 | 0.218242 | 0.252685 | 0.412881 | 0.139104 |

The depth model underperforms the constant even on training wire pixels. Heldout
background supervision differences alone therefore cannot explain the complete
gap. Seg-only's untrained depth output is an additional negative control:
all-valid MAE is 1.705443m on train center and 1.168727m on heldout center.
It is not a depth-training result or a candidate depth model.

Three reproducible observations narrow the investigation:

1. **The label support changes by family.** Training center valid depth covers
   only 1.377% of pixels, all in wire ROI. Heldout center depth covers 100%, with
   wire ROI only 7.533%; 136348 valid heldout pixels lie outside the wire.
   Full-field heldout has 98266 valid background pixels. This is a measurable
   supervision-domain difference, not proof of a unique causal explanation.
2. **Predicted range is compressed in the wider heldout view.** Full-field heldout
   valid targets span 0.4497–2.1656m, with median 0.8146m; predictions span
   0.7370–1.2141m, median 1.0657m. Center heldout targets span 0.7975–1.3577m,
   predictions0.8016–1.2843m. The same fixed crop and full-field view have different
   depth distributions.
3. **A single global bias/scale does not repair heldout performance.** An
   exploratory correction fitted only to 2031 training-center pixels gives
   offset−0.105206m or scale0.904058. Offset improves train MAE 0.143279→0.112724,
   but worsens heldout center 0.156147→0.161568. Scale gives heldout center 0.155297,
   still worse than constant 0.104036. Neither correction is applied to a
   checkpoint or fitted using heldout labels.

These observations are limited to the deterministic first 4+4 samples; no random
population estimate, new convergence claim or full-manifest conclusion follows.

## Normalization, units and loss trace

The existing path decodes uint8 RGB, divides by 255, then either takes the original
center crop or antialiased bilinear full-field resize. Observed RGB lies
approximately in[0,1]; a full-field maximum 1.00000024 is float interpolation
roundoff. The float32 NPY labels are consumed directly in their declared metre
units; depth is neither normalized nor multiplied by a crop/resize scale.
Nearest-grid label sampling preserves depth values. These remain source metric
depth labels, not an independently established optical-axis Z convention.

The depth head uses unbounded positive softplus, not a sigmoid or a fixed[0,1]
clamp. All observed outputs are positive and none is below 0.01m; this audit does
not show near-zero softplus saturation. The median scale discrepancy is near 1,
not a 1000-fold unit mismatch. No normalization or unit conversion change is
justified by this small audit.

Depth-only loss is SmoothL1(beta0.01) averaged over valid pixels in each visited
image. Between 94.5% and 100% of observed labeled pixels per nonempty view lie in
its linear branch. That does not mean the gradient is zero or that beta should
be changed. Training weights visited images equally, while aggregate evaluation
weights valid pixels; varying sparse label counts can make the two summaries
differ. This is a source-level fact to preserve in the next loss-domain audit,
not an experimentally isolated cause of the current error.

## Validation and next decision

Tests were written first: four missing-implementation failures, then one missing
weight-preservation check. Final CPU regression: 67 passed, 0 failed in 5.02s.
Tests cover ROI/invalid-label accounting, training-only calibration, pixel-weighted
aggregation, partial deadline publication, inference without gradients and
detection of a model forward that mutates its own weights.

Before considering more optimizer updates, separate valid wire/background support
and the per-image training loss domain from the aggregate metric domain on the
unchanged full16/128 manifests in a separately bounded read-only audit. Check
whether the observed range compression and per-image bias recur outside the
first 4+4 samples. A unique optimization cause has not been established. This
package makes no 16-update request and does not automatically advance training.
Stereo, policy, generalization and native qualifications remain unchanged.

## Executed Ubuntu command

```sh
export PYTHONPATH=src:.:/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86/tools
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
/usr/bin/time -f '{"external_wall_seconds":%e,"process_exit_code":%x}' \
  -o /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-depth-error-cpu-20261112/run02-process.json \
  timeout --signal=KILL 60s /mnt/mydisk/xk/envs/dh116/bin/python \
  -m scripts.audit_dh116_wireseg_depth_error \
  --checkpoint /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-legacy-audit-20261112/depth-only-legacy-import.pt \
  --checkpoint-sha256 1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30 \
  --checkpoint /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-legacy-audit-20261112/seg-only-legacy-import.pt \
  --checkpoint-sha256 776603d4b7008734ec25f157fb84a82411ebb78e18588c58281db5376d55fe0e \
  --dataset-root /mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86 \
  --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-depth-error-cpu-20261112/run02 \
  --train-samples 4 --validation-samples 4 --wall-seconds 50
```

The output directory is now occupied and cannot be reused.
