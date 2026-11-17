# Native cached-view diagnostic: retained ownership confirmed, exit still fails

## Executed scope and outcome
One user-approved attempt of source a0d48b95222606bf4670d97da7d228dd7b2687f9, frozen SHA cadace12e2c347b4f2f012eb2c6f5924da2f6970dfc0c6ec4b2fc77b39a04849. GPU2, one Pinch environment, exactly two SDK initialization steps; zero controls, rod-solver steps, model loads, Teacher queries or PPO updates. No retry.

Both initialization readbacks pass. All34 unique collision owners return one stored cooked hull each. The original validator still rejects34/34; returned data is not geometry/attachment qualification. No tolerance, hull, asset or PhysX parameter was changed. All34 raw cooked hull payloads exactly equal the previous native snapshot; repeated getter geometry is stable, which does not itself grant geometric qualification.

SDK clock stays2 and0.004166666883975267s before capture, after capture and after STOP. Worker result is saved at67.906895s and app.close returns. SIGSEGV then occurs in libomni.physx.tensors.plugin.so during Py_FinalizeEx. All-thread8-frame stacks, main40-frame stack and dynamic-library mappings are saved. Worker completion/app.close do not establish natural process exit.

## New actual ownership evidence
Registration and after-STOP receipts show the same SimulationView ID128799393884928 and backend ID128799389080496 shared by manager, both Articulations and both ArticulationData instances. After STOP the manager field isNone, root ArticulationView weakrefs are gone, but both assets/Data and their cached SimulationView/backend remainalive. Fourteen observed aliases are retained; these are aliases, not fourteen independent native views.

The factory-local dictionary ID128784702946944 and force_state dictionary ID128799481680512 are actual SimulationContext referrers after STOP. The constructor frame's caller chain reaches the factory; its observed upstream referrer is torch.cuda._lazy_init at installed source line346. This is real ownership evidence, not a CPU fixture.

The holder above the _lazy_init frame is not yet identified. Retention is not sufficient causal proof of SIGSEGV. PyTorch _lazy_call stores formatted stack strings by design; merely seeing _lazy_init does not prove that _queued_calls owns a live traceback. No Torch/SDK globals, frames or cached fields were deleted.

## Authorized cleanup and honest exit classification
The debugger stopped all threads at the fault near70s. Waiting for cleanup approval crossed the180s soft deadline without extra SDK/solver steps. The user then separately authorized only inferior656627 and GDB656497 to be ended. Inferior receivedSIGKILL and debuggerSIGTERM; both are gone, GPU2 returned9MiB, and the single-instance lock is free.

GDB's reported returncode0 is not inferior exit0. The raw supervisor was unaware of externally authorized cleanup and retained force_termination_used=false; the separate cleanup receipt and combined readiness explicitly record forced termination=true. Outer supervisor exit is1. Natural exit, Teacher admission and grasp qualification remainfalse.

## Evidence and verification
Full raw264096985-byte cooked capture and all new owner/clock/stack receipts remain on the Ubuntu data disk, with paths and SHA in readiness; no raw data/model/log is uploaded. The authorization and once-only budget are spent and must not be reused.

This evidence package changes reports/status only. Source tests previously passed467 related CPU cases and124 fixture cases; fresh post-native worker/observer/Registry checks pass141/141 (zero failures/skips), with exact command and log SHA in readiness. All528 protected hashes and original model SHA are rechecked before commit. PR2 remainsDraft.

## Next diagnosis
Inspect the real owner above the observed CUDA frame and reproduce its retention mechanism in CPU isolation. A separately versioned eager-CUDA-initialization contrast may test moving initialization outside the SimulationContext constructor, but it is only a candidate: not implemented, approved or executed here. Do not clear arbitrary frames, add repeatedGC, widen contacts, use fast_shutdown, disable protections or retry this frozen version. No further Isaac run or Teacher training is authorized by this report.
