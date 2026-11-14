# WireSeg paired download and separate segmentation/depth resume preflight

Base: 7e98f864fc09aec09e1500a5359d994846c25850. This package admits one more
official depth blob and adds a CPU-only, per-head checkpoint/data preflight.
Actual optimizer updates, model inference views, GPU work and Isaac runs are all
zero. The two real checkpoints remain at 128 historical updates each. No concrete
new optimizer budget was present, so no continuation command was executed.

## One additional exact paired shard

The bounded Ubuntu Range attempt used the existing pinned relay, 4 MiB chunks,
8s request timeout, one attempt and a 180s outer bound. It failed on a TLS hostname
mismatch. TLS verification was not weakened. Windows verified the Range transfer,
expected bytes and full LFS SHA, then copied the complete file to private Ubuntu
staging. Ubuntu independently reverified size/SHA, copied to a temporary file on
the target filesystem and admitted it through a non-overwriting hard link.

- Revision: `1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b`.
- New blob: `depth/depth-00018-of-00100.tar.gz`.
- Bytes: 24914556.
- LFS SHA: `a81cee6e051d011c7130fdd8bc37088c24cc981f05c15bcae0f1249ff8d66e43`.
- Existing RGB source: `data/train-00008-of-00045.parquet`, SHA
  `9ad438f849cf5718a11abac5e40b4d77c17d559b76c24e285458aa0479493af6`.
- Complete verified blobs: **10 / 148**.
- Exact new pairs: **360**, identities
  `flying_wire/2_cable/000480.png` through `000839.png`.
- Decoded instance masks: **711**. All 360 images have valid full-frame depth;
  none has an empty full-frame mask.
- At the existing center 192 view: **295** depth-supervised images and **65**
  without depth labels. Those 65 also have empty center masks and remain valid
  segmentation negatives. They are not dropped from segmentation.
- Valid center background depth pixels: **0**. This shard does not resolve the
  previously observed full-scene/background supervision mismatch.
- These are official monocular RGB-depth pairs, not new native data.

A new manifest and sample ledger were created. The original training manifest,
its fixed 16 checkpoint cohort and the validation manifest were not modified.
The new pairs do not silently enter either resumed head.

## Separate, read-only checkpoint checks

`scripts/preflight_dh116_wireseg_resume.py` requires distinct segmentation and
depth source hashes. It verifies each existing import/continuation lineage with
the established validator, rejects a swapped head before data loading, reloads
the exact frozen cohort, recomputes supervision, and checks the sampler cursor.
It does not instantiate a model or optimizer, restore global RNG, call forward,
backward or optimizer.step, or save a training checkpoint.

Each head has its own output section and a bounded, explicitly hypothetical
visit preview. Preview rows are not completed optimizer updates. Empty masks
remain supervised segmentation negatives; unlabelled depth visits remain skips.
The preview neither grants a resource budget nor upgrades legacy state to strict
v2. Missing historical Python/NumPy RNG and complete runtime configuration remain
explicitly unknown.

Actual CPU preflight used two threads, CUDA_VISIBLE_DEVICES empty and an external
60s limit; the entire process completed in **8.92s**, exit code **0**.

| Head | Historical updates | Completed visits | Next index / image | Supported images | Adam slots |
| --- | ---: | ---: | --- | ---: | ---: |
| seg-only | 128 | 128 | 0 / 000840.png | 16 / 16 | 70 |
| depth-only | 128 | 158 | 14 / 000854.png | 13 / 16 | 66 |

Both saved optimizer slot step ranges remain [128,128]. The saved Torch CUDA RNG
means the existing continuation contract requires a CUDA execution device if a
later run is separately authorized. This CPU preflight did not initialize CUDA.

For a hypothetical 16-update segment, segmentation would visit 129–144 with zero
skips; depth would visit 159–178 with four skips. Both would end at update 144
only **if executed**. In this package both remain at update 128. New admission
does not change those original sampler identities. An actual model/optimizer
restore and training update are not claimed by this preflight.

The previous full fixed-cohort audit still found depth worse than the training
constant on both heldout wire and background domains. This package preserves
that result and makes no quality-convergence or native/stereo/policy claim.

## Durable evidence and preservation

Private root:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-download-resume-20261113`.

| Evidence | SHA256 |
| --- | --- |
| depth18-admission.json | `8fd78fde30879441e7fc2230f072bba0630accd85e515c7530e4f78172eb8430` |
| depth18-train08-subset.json | `6b0cc32edd28a232f9b2e3db784b117c820569841601e1f66eba4904214e99c7` |
| depth18-train08-samples.jsonl | `acb15c284206436d558dabb64f4db596149bce361b7660be6952b143e82c727f` |
| separate-head-resume-preflight.json | `2a37a6d1891201d2c6b4241ee3b77f64944e9a447a8b5af89d5a3d1369b9f9a6` |
| preflight-process.json | `72797ebf49bd4edd2aa54e30471853aa3f03d3b208ff56c58767bce6fdae7623` |

Executed script SHA:
`2f278fd6416eb079f36bdfe1d51e8c3ae655ccf8aea064fa239a994fd95e6d91`.
After the checks, both import hashes and all seven original evidence bindings
per head were independently rechecked unchanged. Raw data, staged download,
models and private reports are not committed.

Seg import SHA:
`776603d4b7008734ec25f157fb84a82411ebb78e18588c58281db5376d55fe0e`.
Depth import SHA:
`1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30`.

## CPU tests and command

Seven new tests first failed on missing preflight implementations. They now
verify distinct-head semantics, unchanged sampler sources, depth skips, changed
cohort/supervision/cursor rejection, finite previews and zero-update read-only
inspection. The full related CPU regression completed **84 passed, 0 failed in
6.23s**. Existing synthetic resume tests exercise toy state transitions; they
do not update these real checkpoints.

```sh
export PYTHONPATH=src:.:/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86/tools
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
timeout --signal=KILL 60s /mnt/mydisk/xk/envs/dh116/bin/python \
  -m scripts.preflight_dh116_wireseg_resume \
  --seg-checkpoint /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-legacy-audit-20261112/seg-only-legacy-import.pt \
  --seg-sha256 776603d4b7008734ec25f157fb84a82411ebb78e18588c58281db5376d55fe0e \
  --depth-checkpoint /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-legacy-audit-20261112/depth-only-legacy-import.pt \
  --depth-sha256 1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30 \
  --dataset-root /mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86 \
  --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-download-resume-20261113/separate-head-resume-preflight.json
```

The actual process was wrapped with /usr/bin/time to record external wall time
and exit status. The output path is now occupied and cannot be overwritten.
