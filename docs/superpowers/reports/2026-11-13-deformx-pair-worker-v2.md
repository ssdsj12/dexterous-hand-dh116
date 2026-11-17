# DeformX pair worker v2: registered evidence root and post-step error gate

Prepared on 2026-11-13 from integration 781695356e85552e8ff49e1b99d921de8bd1aa0f.
This package contains CPU preparation only. No Isaac, CUDA solver construction, native calibration,
training, or extra attempt was launched. The consumed v1 entry, wrapper, recipe, tests, scheduler,
SDK, and earlier lifetime recipes are unchanged.

## Observed failed v1 attempt

The archived attempt is:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-deformx-pair-capture-native/c65fe22d71a91d167694c90fd9a5113fd0c59f86c87439e882079fb09db36200/attempt-nbf2hogr`.

Its tested HEAD was 98a249bfe2932b66dcf105df8ca83828c3f4615d. The outer process exited 1
after 1.914839394041337 s; native_lock_observed was false. The log ends in verify_worker:
"worker output or current recipe does not match the registered approval".
The intended native output directory is still absent at audit time:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-frame-v2-pair-20261113-adsl7i27/native-runs/flex.f7-deformx-pair-capture-native/attempt-nbf2hogr/native`.

For evidence/native-attempts/package/fingerprint/attempt/intent.json, parents[3] is
native-attempts and parents[4] is evidence. The old verifier incorrectly used parents[3] as
Repository.private, then appended native-attempts again for its identity check. This is a concrete
preflight defect, reproduced with an actual temporary Git repository and an authorized-shaped
intent in a CPU subprocess. The gate runs before native_instance_lock, output.mkdir, and the
lazy ropesim import/constructor. This attempt therefore provides no contact-pair GPU measurement.
The generic receipt's native_execution_performed=true denotes its launched child process;
it does not establish that the CUDA solver executed.

Archived SHA256:

| Artifact | SHA256 |
|---|---|
| intent.json | 9f23e7825f85c728c95be20a00958f7d08cd9daf737414ab5152c1959769c1f8 |
| receipt.json | f791d884f78d237600ab34b4c4c5d30ae103a7cac7ebc8a69f6eea116b28c4ed |
| outer_exit.json | 8756890c448fc2005b3cf4c0cfbf638195110d89aae8efb050376ec8189250de |
| native.log | 17422d5df4073654fb95862e90115c05b6e9267dd778c9bc7b42fba8fec0b3c9 |

## Narrow successor

New entry: scripts/verify_dh116_deformx_pair_capture_v2.py.
New scheduler entry: scripts/run_dh116_deformx_pair_work_packages_v2.py.
New recipe: configs/dh116_execution/deformx_mesh_pair_capture_calibration_v2.json.

The v2 verifier names the attempt/fingerprint/package/native-attempts/evidence directories
explicitly. It checks the registered hierarchy, current Git HEAD, recipe entry/argv,
current code/config identity, approved identity, intent identity, and output derived from the
approved output_root/package/attempt/native. The marker is DH116_DEFORMX_PAIR_WORKER_V2,
set only by the scoped wrapper, and CUDA_VISIBLE_DEVICES must be exactly "3".
The recipe environment contains no worker marker. All checks precede lock, mkdir and construction.

The wrapper reuses the v1 scheduler class, rebinding its entry, wrapper, worker marker, artifact
validator and __file__ within a finally-restored scope. Consequently start_native launches the
v2 wrapper again in the real child, preserving its repository/evidence root and validator.
The original entry main and its preflight/lock/output order are reused with scoped bindings.
No runtime AST generation or SDK monkeypatch is used.

A read-only audit of the pinned candidate source confirms RopeBatch.from_positions accepts
[1,3,3] positions with V >= 3, pinned=[False,False,False], fix_root_orientation=False;
step_free(dt,substeps) and the readonly integer last_error property are present.
Source: teacher-pair-build-zQEenV/source/csrc/bindings.cpp lines 890–938, 952–959 and 1246–1247;
source SHA256 89b3928faf49ec26e2d53614c0e30c5d6242892e54c8345c45f4d5702077d8d5.
The executed extension remains frozen by its existing SHA256
92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0.

The short v2 run_calibration body copies v1, adding only an immediate last_error read and nonzero
rejection after step_free, plus solver_last_error in pairs.json. A CPU AST comparison removes
those exact additions and proves the remaining function equals v1, including all geometry,
gravity, material, solver parameters, free-node configuration, contact capture capacity and reset.
The post-exit validator first runs the existing complete raw-packet reconciliation and then
requires type(solver_last_error) is int and its value is zero. Missing/boolean/float/nonzero
records fail. Existing v1 packets cannot become qualified by omitting this field.

## CPU evidence

Final command (Ubuntu, no GPU construction):

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
bash scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_dh116_deformx_pair_worker_v2.py \
  tests/test_dh116_deformx_pair_capture.py \
  tests/test_dh116_work_packages.py \
  tests/test_dh116_teacher_pair_diagnostic.py
```

Result: 97 passed in 43.97 s, exit 0: 24 new tests, 12 old pair tests, 61 scheduler/teacher-pair
regressions. Initial absent-entry tests were red; absent last_error implementation and absent
recipe tests were separately red before their implementation.

Evidence directory:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pair-worker-v2-cpu-20261113-ukrd579o`
contains pytest.log, verification.json and fingerprints.json.

The successful authorized gate test uses real temporary Git/registry/recipe/approval/intent
identities in a real subprocess, replacing only the native lock and calibration constructor.
Wrong head/code/approval/output/approved-output-root/hierarchy/entry/GPU/missing-marker cases
fail before mkdir or constructor. A further real scheduler parent/child test executes the
v2 CLI worker with the real gate and raw validator; the transport's native body and resource
checks are CPU stubs, with explicit cpu_fixture=true and native_execution_performed=false.
Synthetic test packets and temporary result files are fixtures, not native evidence.
A nonzero last_error fixture demonstrates that post-exit validation revokes an otherwise
generic qualifying receipt. Separate fake RopeBatch tests verify all-free 3-node API shape and
step -> last_error -> packet ordering, including failure before packet read.
Scope restoration is tested for both entry and wrapper exceptions.

Read-only peer review by continuation_wireseg_data found no additional blocking issue in the
gate, child reentry, finally restoration, raw validator, frozen dependencies or budget.
verify_protection passed for 528 protected source files and two historical Checkpoints.

## Exact freeze and later approval

| Identity | SHA256 |
|---|---|
| v2 command_spec_sha256 | 77d0290a350ba6b877991b9831da8fb4fb62441481e1866412935fc6d28af594 |
| v2 code_config_sha256 | 94d855ce6be8700816a85a8e84d41a87fe74b52454bf4f8591661e932c6d08c6 |
| v2 entry source | 8aa0bef19fc85099d57bd99528e1fbdaaf34e6f6edf124991dfa9f61f025b67c |
| v2 wrapper source | 1ff80d4b9716445f8aa1a4832dc0e9f5ae4e7c2765b19737e4374c158d454a24 |
| v2 CPU tests | 8a6fd5cff3eb9a993056181d8249869b42b7aae5e7a55ddf8592aca386647757 |
| unchanged v1 command_spec_sha256 | ddd6a7b6adfc744a87a32c8380d473b8c61b848ee7cb8f9b3c90e06a00468dd0 |
| unchanged v1 code_config_sha256 | c65fe22d71a91d167694c90fd9a5113fd0c59f86c87439e882079fb09db36200 |

The two new/old pair tests are explicit frozen source_files; their inclusion as the registry
package tests produces the v2 identity above. The scheduler automatically fingerprints itself,
python.sh and the entry. Both new/old entry/wrapper layers and existing validation contracts
are frozen. The existing extension, Python binding files and libcudart inputs are unchanged.

Suggested independent package: flex.f7-deformx-pair-capture-v2-native, through the v2 wrapper,
requiring new exact approval after integration. No approval or extra attempt is assumed here.
Keep one attempt per code/config, retry_requires_change=true and requires_approval=true.

Unchanged scope/budget: GPU 3, one environment, one all-free three-node rod, one table SDF mesh,
one CUDA solver step of 1/480 s with 12 substeps, zero Isaac physics steps, zero PPO updates,
90 s qualification limit, at least 4096 MiB free GPU memory and 1 GiB free disk.
The existing scheduler records overdue work and refuses qualification after 90 s; it does not
hard-kill the process at that threshold. This package does not alter that policy or authorize retries.

Native qualification still requires actual outer exit 0, observed shared lock, elapsed time
within budget, exact artifacts, positive measured contact, consistent clock, reset invalidation,
raw pair/force/wrench reconciliation and zero solver last_error. No scene/grasp/full-371/Teacher
qualification follows from this CPU package or from a future isolated instrumentation calibration.
