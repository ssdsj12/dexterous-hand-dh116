# DH116 WireSeg diagnostic checkpoint continuation

Base `2ef7f90ed9af9039dd32fb197063040ae11996f3`. Strict checkpoint implementation is CPU verified. The original model and data are available for read-only audit. Strict resumption of either historical diagnostic is blocked by missing state. New real training updates, native runs, and evaluation episodes for this work package all equal zero.

## Historical checkpoint audit

| Head | Completed updates | Last recorded visit | AdamW state entries | Checkpoint SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Segmentation | 128 | Unrecorded | 70 | `77459b47f26ccfdeea024f298fe59bedbb9c119242cddda305aeea0b71cf5f39` |
| Metric depth | 128 | 158 | 66 | `56826811b64075446486fc0309a60a616dabcafc738d40700be04ec39ac21adb` |

Both original files have model and AdamW state and Torch CPU and CUDA RNG snapshots in their legacy fields. They lack Python and NumPy RNG snapshots, an explicit sampler cursor and skip history, explicit scheduler/scaler fields, and the complete v2 data/config contract. The segmentation file also lacks a recorded diagnostic source SHA. Visit 158 in the depth update record is useful for audit and does not reconstruct the complete strict-resume state. The files retain their original bytes.

Both records refer to catalogue SHA-256 `bc9afab256cb247b60d45116b9351810c99d6e0818fd26e3690d1cf9dbe86e93`, training subset `38159e2f94a7528f983a1b9dca3a74f9f8bb91e5576046d31c9413b386ae905d` and validation subset `5d5cc1331115737bfd039fa8223bd0f23cd67ea385595ffc0f03e71e6115f78f`. Audit uses `torch.load(..., weights_only=True, map_location="cpu")`. The strict execution path rejects either legacy file before reading the dataset or constructing the optimizer.

## Versioned state and verification

The diagnostic still runs the existing center-crop segmentation or metric-depth loss with AdamW. The v2 checkpoint preserves model parameters, validated AdamW parameter groups and moment tensors, the absolute update count, Python/NumPy/Torch CPU RNG and the selected Torch CUDA RNG when applicable. The scheduler and AMP scaler are explicitly absent in this diagnostic. The cyclic sampler retains every visited sample, including a visit skipped for missing valid crop supervision. Its cursor, next index, updated/skipped counts, successful updates and ordered RGB–depth identities must reconcile before resumption.

The immutable dataset contract hashes the verified catalogue, train and validation manifests, and their ordered sample identities. The training configuration separately hashes the head, model, optimizer, crop, LR, seed, device, runtime settings and relevant implementation files. An approved checkpoint byte digest is required for strict resumption. The next output is independent and checkpoint publication refuses an existing destination. The prior general pretraining format retains its existing behavior.

Ubuntu CPU command:

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_wireseg_checkpoint_resume.py tests/test_dh116_wireseg_overfit_diagnostic.py \
  tests/test_dh116_wireseg_pretraining.py tests/test_dh116_wireseg_blob_relay.py
```

30 passed in 4.77 seconds. The synthetic diagnostic CLI compared eight uninterrupted updates with three plus five resumed updates. Model tensors, AdamW, RNG states, absolute step and full visit history matched exactly, including three depth-label skips. The regression rejects changed checkpoint bytes, dataset manifest, configuration, sampler cursor, missing optimizer/moments, and an existing output. Both original artifacts were audited separately without loading model state into a trainer. All 528 protected files and both historical DH116 checkpoints match. CUDA continuation has not been run.

## Additional verified training data

The pinned dataset now has 8 of 148 complete official blobs. The new files were checked against their official LFS identities and verified again after admission on Ubuntu.

| New file | Bytes | SHA-256 |
| --- | ---: | --- |
| `data/train-00009-of-00045.parquet` | 622337468 | `b51ef63355390dc333d4139b307bcebd1ecbb40db015046a68921657c1e12307` |
| `depth/depth-00020-of-00100.tar.gz` | 25243688 | `612322216330fedb4d969c7e3b5ee31829571df1c8a045dca06d52c5d128d514` |

The RGB Parquet contains 800 `flying_wire` train rows. All 360 depth members have an exact paired RGB identity, with 713 decoded instance masks and zero missing depth or empty full-frame mask rows. All 360 have valid full-frame depth labels. At the existing 192-pixel center crop, 292 have valid depth and wire ROI labels and 68 lack crop supervision. The sampler records those 68 as skips if visited in the depth-only diagnostic. The new admission, subset, and sample ledger SHA-256 values are respectively `eb32001bb97293b72e5339f9dbbf11e2c98641ef42578d066c514c4d9c155995`, `dafdda5a62a35b401872dd11ead4219888c0928cc7032fcb0bdcc87bb2eab524`, and `ea07e6b56a4a348b51f086efe19851d341e272df3cb4eddbff076846da32ad31`. These are monocular metric-depth labels. No new pair is inserted into the original 128-update manifests.

## Next bounded batch

A valid v2 source must be available for strict continuation. No complete historical source has been found. If an authentic complete source becomes available, first run the read-only audit and bind its exact digest, immutable dataset and configuration. A proposed finite diagnostic is one head, its existing 8–16 paired train cohort, its existing disjoint validation cohort, up to 16 new updates and 120 seconds, with zero Isaac steps. Choose the device and hyperparameters matching the source and a new output path. The old artifacts cannot be used with this command as strict sources.

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
bash scripts/python.sh scripts/diagnose_dh116_wireseg_overfit.py \
  --audit-checkpoint "$VERIFIED_V2_CHECKPOINT"
bash scripts/python.sh scripts/diagnose_dh116_wireseg_overfit.py \
  --dataset-root "$WIRESEG_DATASET" --subset-manifest "$FROZEN_TRAIN_MANIFEST" \
  --validation-manifest "$FROZEN_VALIDATION_MANIFEST" --output "$NEW_DIAGNOSTIC_OUTPUT" \
  --heads "$FROZEN_HEAD" --device "$FROZEN_DEVICE" --size "$FROZEN_SIZE" --samples "$FROZEN_SAMPLE_COUNT" \
  --seed "$FROZEN_SEED" --learning-rate "$FROZEN_LR" --steps 16 --wall-seconds 120 \
  --resume "$VERIFIED_V2_CHECKPOINT" --resume-sha256 "$VERIFIED_V2_SHA256"
```

A diagnostic using the newly admitted cohort would have a separate dataset/config identity and requires its own bounded run decision. No legacy fallback, optimizer reset or implicit fresh training occurs on a rejected resume.
