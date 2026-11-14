# WireSeg legacy audit import and bounded diagnostic continuation

Base: b2de6ec4a0e1879b55052b84e0a2a412d7a0cf7c. Development, CPU tests and
artifact creation ran on Ubuntu, with two CPU threads. No actual training update,
native Isaac run, evaluation episode or fresh-model fallback ran in this work
package. The original checkpoints, manifests and source snapshots retain their
bytes.

## Separate lineage and validation

`scripts/import_dh116_wireseg_legacy.py` accepts seven explicit path/SHA256
bindings: checkpoint, update JSONL, historical diagnostic source, frozen cohort,
training manifest, validation manifest and pinned catalogue. All seven are
verified before and after reconstruction. Only the two reviewed historical source
digests are supported; no historical source is executed. Actual original RGB
blobs are independently verified, the frozen training order and pair digests are
reloaded, and center-crop depth supervision is recomputed. The original immutable
validation split remains disjoint.

The segmentation source emitted update records without visits; its fixed cyclic
loop proves one visit per update. The depth source emitted every visit, including
skips. Ordered file names, pair hashes, update numbers, successful records and skip
counts must reconcile with the checkpoint and observed supervision. A missing
skip, reordered row, changed pair hash or rehashed conflicting JSONL is rejected.

The output format is `wireseg-legacy-diagnostic-import-v1`. It embeds the complete
original state without altering tensors, Adam moments, Torch CPU/CUDA RNG or
historical update records, alongside the reconstructed sampler and evidence.
Python RNG, NumPy RNG and complete historical runtime configuration remain
explicitly unknown. Segmentation source attribution remains
`external_snapshot_only`, because its checkpoint never recorded a source SHA.
This is evidence-based reconstruction, not historical bitwise equivalence.

`scripts/continue_dh116_wireseg_legacy.py` is a separate, opt-in diagnostic
runner. It rechecks original evidence and tensor equality before loading state.
It restores weights, Adam, saved Torch RNG and the actual sampler cursor; only
new Python/NumPy RNG states receive an explicit new seed. It refuses a changed
import, incompatible state/device, an existing output, or a budget outside
1–16 new updates / 1–120 optimization seconds. It writes
`wireseg-legacy-diagnostic-continuation-v1`, current RNG and configuration, the
original evidence lineage and persistent unknown-historical-state labels.

Both formats remain ineligible for the existing strict-v2 resume entry point.
No native, stereo, policy or generalization qualification is granted. The runner accepts both the initial audit import and its own continuation
format. Each later checkpoint binds its immediate parent path/SHA and original
import SHA; the parent chain, original evidence, complete sampler prefix and
historical unknown labels are revalidated. Later segments restore all current
Python/NumPy/Torch RNG states and reject a new seed. The parent chain is bounded
to 64 links. No format is silently converted to strict v2.

The wall clock is checked between optimizer updates, starting after evidence and
data validation and model/state initialization. It is not a whole-process
120-second timeout; an in-flight update and final state publication can exceed
the requested optimization interval. On normal wall-clock stop the runner saves
the current state, including when zero new updates fit. A SIGINT/SIGTERM request during the runner finishes the current update, saves at
that safe boundary and exits with status130. Signals during earlier CLI preflight
can stop before any training update. Forced kills or hardware failures do not
promise a final checkpoint; immutable ancestors remain available.

## Actual historical audit, without updates

Private evidence directory:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-legacy-audit-20261112`.

| Head | Original updates | Reconstructed visits | Skips | Next index / identity | Adam slots |
| --- | ---: | ---: | ---: | --- | ---: |
| seg-only | 128 | 128 | 0 | 0 / flying_wire/2_cable/000840.png | 70 |
| depth-only | 128 | 158 | 30 | 14 / flying_wire/2_cable/000854.png | 66 |

Original segmentation SHA:
`77459b47f26ccfdeea024f298fe59bedbb9c119242cddda305aeea0b71cf5f39`.
Original depth SHA:
`56826811b64075446486fc0309a60a616dabcafc738d40700be04ec39ac21adb`.
Import files and independently verified hashes:

- `seg-only-legacy-import.pt`:
  `776603d4b7008734ec25f157fb84a82411ebb78e18588c58281db5376d55fe0e`.
- `depth-only-legacy-import.pt`:
  `1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30`.

Both real imports passed recursive exact tensor/state comparison with their
original files. No model was instantiated or trained for those audits.

## Additional exact official depth data

Ubuntu HTTPS Range failed with a TLS hostname mismatch. The failure log is
retained; certificate verification was not disabled. Windows downloaded just
`depth/depth-00021-of-00100.tar.gz` through a verified Range response in 3.4s,
then copied it over SSH in 0.7s. Ubuntu independently rechecked its size and full
official LFS SHA, copied to a same-filesystem temporary file, and admitted it
using a non-overwriting hard link.

- Pinned revision: `1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b`.
- Bytes: 24,124,467.
- LFS SHA: `eeddfc47beff6abd0f92f7e3bb9da64fe0d9c734dfa9b4e3d39cc760ebcf7c08`.
- Complete official blobs: 9 / 148, independently verified.
- 360 new exact pairs with the existing verified train09 RGB shard, 718
  decoded instance masks, 360 full-frame valid depth samples, zero empty
  full-frame masks.
- At the original 192-pixel center crop: 294 supervised and 66 unsupervised
  depth samples. These are official monocular labels, not new native data.
- No new rows were inserted into either original 128-update manifest.
- New manifest SHA:
  `84ce17fc28c99b46762d414003fe8b7b57a5351760c84d8526344ce24fbf60f8`.
- New sample JSONL SHA:
  `5a108c58d4f9b606bffbfbfc686c180501cf275b727b1b237e1e8543935c7c22`.
- `depth21-admission.json` SHA:
  `0f0f14263c4d50d05eed289e6c9388aa91980c30a7384992eae1ba287164bee0`.

Raw data, imports, source snapshots and JSONL receipts remain private and are not
committed.

## CPU verification

Tests were written first: 18 expected missing-implementation failures for import,
then 5 expected missing-runner failures. The wall-clock/configuration test also
failed before its configuration snapshot was added. Final command:

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_wireseg_legacy_import.py \
  tests/test_dh116_wireseg_checkpoint_resume.py \
  tests/test_dh116_wireseg_overfit_diagnostic.py \
  tests/test_dh116_wireseg_pretraining.py tests/test_dh116_wireseg_blob_relay.py
```

62 passed, 0 failed, 5.52s. New tests cover all seven evidence digests, immutable
outputs, source-specific visit reconstruction, changed ledger/order/skips,
rehashed ledger disagreement, exact preservation, package tampering, unknown
state propagation, strict-v2 rejection and budget rejection before state restore.
A CPU toy model's next stochastic Adam update matches the saved Torch RNG and
original cyclic sampler exactly, including the next skipped visit. This fixture
comparison does not establish bitwise equivalence with the unknown historical
Python/NumPy execution. A normal deadline saves unchanged model/Adam/Torch state
and sampler when zero updates fit. Added tests prove 8 uninterrupted updates
versus 3+5 chained updates produce identical model, Adam, complete current RNG and
sampler after the explicit new-RNG import boundary. They reject a changed parent
SHA, changed sampler and attempted reseeding of a continuation. A simulated
in-flight update deliberately exceeds a 1s soft deadline and completes at2s,
then saves; an interruption request similarly saves at the next safe boundary.
These are synthetic CPU updates, not real-data training.

## Proposed next diagnostic, not executed

Requires a fresh resource-budget approval. Select the depth import above, keep
the original frozen train/validation manifests, original center192/LR0.0003
configuration and next index14. Request up to16 new updates, reaching at most
global step144, and at most120 seconds of optimization-loop time on the approved
CUDA device. Preflight/data loading and checkpoint saving are additional time.
No Isaac steps or automatic extra head are included. The proposed total resource
budget uses an external180s deadline plus30s safe-save grace (at most210s before
forced termination), in addition to the120s optimization soft deadline. A timeout
or forced kill is a failed/interrupted run, never successful completion; if
forced termination prevents final save, retain the previous approved parent.
Run into a new output path:

```sh
export PYTHONPATH=src:.:/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86/tools
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=2
timeout --preserve-status --signal=INT --kill-after=30s 180s \
  /mnt/mydisk/xk/envs/dh116/bin/python -m scripts.continue_dh116_wireseg_legacy \
  --import-checkpoint /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-legacy-audit-20261112/depth-only-legacy-import.pt \
  --import-sha256 1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30 \
  --dataset-root /mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86 \
  --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-depth-legacy-next16-approved \
  --device cuda:0 --new-rng-seed 116 --steps 16 --wall-seconds 120
```

This command is a proposed diagnostic continuation with documented missing
historical state, not strict native continuation. It has not been run on actual
data. For a separately approved later segment, use --checkpoint and
--checkpoint-sha256 for the previous diagnostic.pt, omit --new-rng-seed, and
choose another new output directory. The newly admitted360 pairs do not alter
this fixed-cohort proposal.
