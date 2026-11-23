# DH116 compact native Teacher rollout

The batch runner retains per-lane native contact, action and reward checks. A successful training window discards its full physical rows after admission. The current window remains in memory until that boundary. A caught acquisition, guard or source failure seals the retained records as typed binary. Full binary evidence remains available with `--full-window-evidence`. The original per-row audit is available with `--legacy-row-json`.

Compact history checks the real 24D command, named target readbacks, hand continuity, lane and clock before passing the exact synchronous event to the reward source. The compact history and geometry handoff apply only to the Pinch batch path. The original geometry check runs once per real step, using the original inputs and thresholds. Audit replay recomputes it independently. The same frozen pending batch is sealed and rechecked at the control boundary. The normal compact path logs PPO updates, with window timings reported by the launcher.

## Results

| Run | Two-lane native window | Detail |
| --- | ---: | --- |
| Full binary evidence and per-lane policy hashing | 8.041 s | 0.210 s evidence seal |
| Compact success evidence and one policy identity per batch | 6.506 s | 2.822 s observer and guard |
| Compact history and original geometry evaluated once | 4.135 s | 0.497 s observer and guard, 1.881 s native completion |

Each test ran a single shared eight-tick 480 Hz window, 16 independent rod calls, and zero PPO updates on physical GPU 3. Every inferior naturally exited 0. The timings compare separate native runs and correspond to this two-lane diagnostic. There is no new 64-lane speedup measurement.

The 512-row disk replay took 15.60 s with 512 row flushes (238,939,865 bytes) and 10.33 s when sealing full typed evidence once (205,098,468 bytes, file and directory sync). Every typed record matched after decoding. The default compact path discards the successful raw window entirely.

The original 64-lane attempt reached SDK 18. Its first 64 transitions were committed. Lane 55 failed `right:non_target_not_proven_clear:source:finger13_link` at a measured conservative bound of −0.0200069 mm during its second window. The next 64 transitions were rejected and the optimizer did not run. The process shut down with exit 1. Offline reevaluation using the stored original geometry reproduced the same reason and terminal flag, and a passing lane 0 control stayed passing.

GPU 3 replay of the 512 real physical rows and 64 reward settlements matched the original numeric quantities and exact guard/target-reason flags. Largest absolute errors were 5.96e-8 for rod feature packing, 2.22e-16 for wrist targets, 5.55e-17 for CPU/CUDA clearance comparison, and 2.17e-19 for reward reduction. This tests component parity, with the original CPU checks still used by the live collector.

- CPU regression 160 passed, 1 CUDA-only skip. Log SHA-256 `2911784a13cbcbb9043a099ed54d4d72335ce377f212d4f374c921436858d92c`.
- Focused suite on GPU 3, 10 passed including CUDA parity unit cases.
- Native fast path receipt SHA-256 `74f2e618d5dbc2878f2b44ba0de43980815e3060e452fd3614963f65b0149a42`.
- Native 64-lane failure receipt SHA-256 `7512bdaa8519fa284ec86728801aa841ec2151c25321e32ef4cd86fd478d9087`.
- Real archived CUDA parity SHA-256 `1203d25464d24eaa3d44ef624866192ce612db7fca7ed65330b4be3632a34027`.
- The 528 protected files matched the original hash inventory.

## Three-grasp scale-up

Antigravity read its read-only server tree at `125f2664`. That mount omitted the current batch files, so those files were independently read on Ubuntu. The native pair validator it read is identical to our live copy by SHA-256. It identified the duplicate geometry evaluation and the current shared-SDF contact schema gap. The locked native backend explicitly rejects raw pair capture when the solver contains more than one rod. Real runs at 128 or 512 require a per-lane pair ABI and verified native parallel paths. Individual Tripod and Power Reset capture and a Pinch rollout past the observed guard are separate outstanding tests. Their previous launches rejected a stale frozen source plan before starting the app.

```bash
CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_teacher_native_batch.py
CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID bash scripts/python.sh -B -m scripts.audit_dh116_teacher_tensor_parity /absolute/path/to/native-journals
```
