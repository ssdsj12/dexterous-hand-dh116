# Returned acquisition frame: single v2 lifecycle preparation

Base: 3481c7626417f623faa1f7397cf759918d8e8afe, containing stopped GDB capture from source
commit 32e5187. Worktree: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/lifetime-frame-v2-20261113.
This package ran CPU tests only. No Isaac/native probe, camera rerun, solver experiment or PPO
update was launched. Consumed v1 entries, recipes and SDK source remain unchanged.

## Observed mechanism and remaining uncertainty

Both approved v1 cases reached app.close return but retained the nine recorded objects before
close: simulation plus hand, asset/manager SimulationView aliases, hand view, composer, provider,
runtime and rope. Asset and manager aliases referred to the same object within each process.

The registry case's before_app_close.refs.json SHA256 was
2748f6a7eb74e3c81307ac9dd3c9f0e0b0084fed21dfe9f6cdd5796a67fd15df
(attempt-8g06fzyo). The explicit case's corresponding SHA256 was
f75601b7072249ae48adff77aae11c798278e7c9838da729af7679831961e934
(attempt-35e2gyiv). Both are under the respective
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/
flex.f4-physx-lifetime-{registry,explicit}-native/<approved fingerprint>/ attempt directories,
with native output paths recorded in their immutable receipts.

Explicit outer-scope deletion had an observable limited effect. Provider referrers fell from
three to one; its remaining referrer was the completed build_native frame at acquisition.py:146.
Composer fell from four to two (that frame and hand._permanent_wrench_composer); runtime from four
to two (that frame and provider's bound read_rod_snapshot); the root SimulationView aliases from
five to three (that frame and two _physics_sim_view dictionaries). Hand view fell from three
referrers to a single closure cell. Rope remained held by the runtime's rope dictionary.
The observer did not record who retained the completed build_native frame. It also did not prove
the full C++/CUDA ownership chain, a specific invalid native pointer, or a native crash cause.

The registry log SHA256 was
633ed561ef21e7943ffa70ee5fce0441d935a8292530a49be9bff1129a9a95fc.
Its normal Py_FinalizeEx capture had a real first stack and register/maps files, but
thread apply all bt reported Selected thread is running; gdb_finalize.json explicitly marked
all_commands_succeeded=false. A later continue also failed, and inferior exit was unverified.
An app-close return or partial finalize capture therefore cannot qualify either consumed case.

## Narrow implementation

The new human-readable build_native_detached copies only the pinned original build function.
Its try body and signature are checked AST-for-AST against the frozen original by a CPU regression.
No runtime AST generation, frame scan, f_locals access, tracing/profiling hook, SDK mutation or
additional gc.collect is used.

An explicit finally assigns None to the known resource, temporary-buffer and closure locals,
including sim, hand, view, composer, runtime, provider, both root SimulationView aliases,
snapshot/mesh_poses, materials and full GPU joint-state buffer. Initializing those local names
allows the same cleanup when acquisition raises part-way through. A successful returned bundle
retains its intended resources until the unchanged outer explicit owned scope releases them.
The normal return, failed construction and explicit local cleanup preserve the original setup
body. This is a local Python reference fix; provider/composer/view release in the actual installed
SDK remains unproven.

build_frame_detached.json records completion of the finally suite and whether construction raised.
It deliberately says native_release_verified=false. It does not grant qualification or override
weakref, STOP, real app-close or process-exit evidence. An unavailable marker, raised construction,
incomplete clear-list or missing stopped-thread capture rejects v2 qualification.

The small new entry uses process-local, finally-restored module bindings to reuse v1's argument
handling, source preflight, acquisition graph, frozen owned cleanup, STOP, one explicit shutdown
update and actual app.close path. Only the build function, result version, allowed explicit case
and dedicated worker marker differ. The new worker similarly reuses existing receipt transport;
its real child re-enters the new sibling wrapper and selects the already landed stopped debugger.
Old prerequisite receipts keep their original validation because only the new entrypoint is
subject to the v2 scope.

The independent outer audit retains every v1 clock/liveness/close/log/hash/exit gate, then checks
the new builder marker and a positive stopped_thread_count. It cannot promote a failed generic
receipt. The new debugger's source SHA256 is
2ee4da78216db91ac4d0c3f417285eaeb989f0a608dfcd0c916eba3255affc35.

This experiment may still retain root SimulationView aliases through SimulationContext,
SDK callbacks, other completed frames, cached dictionaries or unobserved native owners.
It must not be described as a proven fix for the archived PhysX crash, camera failures,
material/contact source qualification or Teacher readiness.

## Exact future scope and qualification

One independently approved new recipe only:
configs/dh116_execution/physx_lifetime_frame_explicit_v2.json.

Budget is unchanged: GPU 3, one environment, at most one attempt, 2 actual initialization SDK
physics steps, 0 measured physics steps, 0 rod solver steps, 0 PPO updates, 180 seconds
qualification threshold. No native execution is authorized by this CPU preparation or by the
consumed v1 approvals. No fallback retry, registry control rerun or extra measured step is included.

The same headless no-camera experience and reset path are retained. There are zero created
render products; installed SimulationContext.reset still internally calls render twice.
No claim of zero internal SDK rendering is made. The frozen shared scheduler waits for an overdue
child rather than forcibly killing it, so 180 seconds is not represented as a hard process-kill cap.
An overdue attempt must fail qualification and requires operator supervision.

Normal/fatal capture is outside the breakpoint stop callback, with all threads verified stopped
before stacks, registers, instructions, libraries, raw process maps and GDB mappings are collected.
These callbacks continue automatically. Successful CPU pthread/GDB fixtures are debugger validation,
not Isaac validation.

command_spec_sha256:
48c5869e4de32eea0ca30fa35b062f920c8df752176b1733167dba36a8e06936

code_config_sha256:
5c5654de8d08a9a96a989d0a811f4411db8ded8c99b4ac40e5f7ee04c4e599c6

Package tests used for this fingerprint:
tests/test_dh116_physx_lifetime_frame_v2.py and
tests/test_dh116_physx_lifetime_debugger_stopped.py.
Both are also in source_files. Additional registry tests outside source_files change the code hash.

Consumed v1 code fingerprints were recomputed and remain exactly:
registry 354bf1cb8419f156e3b594f10c0760cc127adbc0d46c39c1a36fe044d26b310b;
explicit 7880b81229cf40611d4dd367e05a5c3dad75d21ce40fc883df9e88d14e4bab25.

## CPU evidence

Baseline: 42 relevant existing tests passed. Test-first missing-scope, missing-entry and missing-worker
failures preceded implementation. A fixture namespace initially omitted weakref; that CPU setup
error was corrected without changing production logic.

Retained-frame tests keep the actual-shaped completed builder frame alive deliberately. The
unmodified scope retains fake native resources after return/error; the exact production finally
suite removes those aliases. Separate tests use the existing actual run_lifecycle to verify
successful and exceptional acquisition have released those fake resources before app.close.
AST compilation is used only to construct CPU test fixtures, never to generate production logic.
The CPU fixture's explicit GC measures Python cycles; production adds no collection beyond v1.

Real CPU worker subprocesses verify the child entry, root/private/output identity, v2 worker marker,
stopped capture source and audit-before-receipt on both passing and rejected synthetic evidence.
Independent negative tests reject missing/failed detachment and missing stopped-thread evidence.
The full regression executes the already landed real CPU normal/signal/threaded GDB fixtures too.

Final command:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_physx_lifetime_frame_v2.py tests/test_dh116_physx_lifetime_probe.py tests/test_dh116_physx_lifetime_scheduler.py tests/test_dh116_physx_lifetime_validation.py tests/test_dh116_physx_lifetime_debugger_stopped.py tests/test_dh116_physx_tensor_lifetime.py tests/test_deformable_grasp_runtime.py tests/test_dh116_deformx_bridge_lifecycle.py tests/test_dh116_pinch_scheduler.py tests/test_dh116_deformx_thumb_contact.py

135 passed, 0 failed, 15.60 seconds: 19 new plus 116 prior regressions.
528 protected source files and two historical Checkpoints passed verify_protection.
All previously tracked files are unchanged; this package adds six scoped files.

Private CPU log:
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/lifetime-frame-v2-cpu-20261113/pytest.log
SHA256: 90114d065b780522d2c3daa77dd015fdef69e6b29746056350547f5dd507d7d9.
The adjacent fingerprints.json retains recomputed new and consumed-v1 fingerprints.

An independent read-only peer review checked binding restoration, actual child reentry, v1 receipt
separation, unchanged outer/weakref rejection and finally/marker ordering; it reported no new blocker.
