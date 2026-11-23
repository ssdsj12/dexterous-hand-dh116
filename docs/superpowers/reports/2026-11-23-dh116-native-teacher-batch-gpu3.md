# DH116 native Teacher batch on physical GPU 3

The shared Isaac process now has a real two-lane Pinch dynamic result. The active lane used the first 26 complete 24D command windows from the SHA-verified native force-feedback ledger. The neighboring lane held its targets. The source Reset remained distinct per lane. The original hand, wrist, table-clearance and per-rod pair guards ran throughout.

| Measurement | Result |
| --- | --- |
| SDK ticks | 2 setup + 216 measured, final SDK 218 |
| Native solver calls | 432, with 216 in each lane |
| Nonzero hand reactions routed into the next SDK tick | Active lane 14, neighbor 0 |
| Lane-local Reset and next window | At SDK 210, lane 0 only. Final rod ticks 96 and 2592 |
| Control-window duration | Median 2.015 s, range 1.966–2.190 s |
| GPU 3 memory sampled at control boundaries | Peak observed 2654 MiB |
| Process | GDB exit 0, inferior natural exit 0, no forced exit |

The Reset retained both hardware and nominal timestamps. At SDK 210, the SDK readback was 0.4375000228174 s and the nominal rod boundary was 0.4375 s. The Reset verifies an unchanged SDK/solver time and unchanged numerical state during clear/add, and verifies the CUDA current, target and solver poses afterward. The final lane 0 reaction starts at zero after Reset. Its neighbor's joint and solver numerical fingerprints match across this transaction.

- Accepted outer receipt SHA-256 `faf2f543c38cf1e42f6c3d19220d5de44ac88ce617f454e166e4865236526195`.
- Physical ledger SHA-256 `48b18e6c9b17855938c7762523fd1e51f91073c60d2585c9f213c9e6e19ac9e4`.
- The first active comparison reached SDK 58, then rejected the old Reset timestamp check. It had no routed hand contact and exited 1 after natural cleanup.
- The controlled-contact smoke uses scripted archive actions. Its 27 windows produced zero PPO updates.

## Native PPO

The 64-lane Pinch worker uses the original finger-goal config with learning rate 0.0001, gamma 0.995, target KL 0.02 and three rollback attempts. Each of the 32 windows can provide 64 PPO transitions. Every actual hand/rod pre-action packet, history row, reward and bootstrap retains its lane identity. A physical guard or source failure rejects the window. The first 64-lane launch stopped at the environment-0-only native parameter reader while binding lane 1. The reader now accepts only its configured real lane ID and derives those lanes' constructor nodes from their actual solver owners.

Tripod and Power retain separate source recipes. The Tripod legacy resolved receipt currently rejects a changed recorder hash. Each supported lane needs its own translated support/collider provenance before its 64-lane process can run. The test and Pinch launch used only GPU 3 and the existing single-instance lock. At the launch check fewer than three cards were idle.


### GPU 3 Pinch rollout snapshot

The repaired worker initialized all 64 lanes and committed its first eight-step window at SDK 10. The journal recorded 64 PPO transitions, 512 separate solver calls, and 96 rod substeps per lane. There was one Reset per lane. GPU memory peaked at 3326 MiB in the two-second device sampling. The first completed window took 267.383 s. There were zero optimizer updates at this snapshot. The active run requests 32 control windows and one update. Its final GDB receipt and checkpoint are pending.

- Failed 64-lane attempt outer receipt SHA-256 `90faa5c2c7bddfe980a6ee691f5168bbbd1e4bafe27cd22bdd95aabe12a18cfe`. It closed the app and exited 1 without any committed transitions.
- Repaired launch attempt SHA-256 `5dd3201517ec24fe35ea3c4e0a0911b171e32f86860fe5d6f4fc7cfd87af3317`.
- Repaired run-plan SHA-256 `4f09c02668a2c0525ed83ff5850b402c4462a49ebf33bc32e5dc45662cb3e702`.
- First immutable control journal line SHA-256 `07f938f8a7f6a8e42295867783d36005077bd6c383c8218e54269f6aaaec4f34`.
- The live worker imported the batch episode methods before their extraction into the separate production class. Its run and attempt receipts identify those loaded bytes.
- Tripod rejected the stored Reset recorder SHA for `scripts/record_dh116_tripod_supported_reset.py`. Power rejected the stored exchange SHA for `tabletop_deformable_physical_exchange.py`. Both need style-specific native Reset receipts and matching source adapters. No 64-lane PPO job was started for either style.

### Verification on Ubuntu

The focused CPU suite passed 136 tests. Log SHA-256 `1a2cc5389be9531d848c59b129282e6bd24b87d00032a0861d1cea5298e23fa3`.

```bash
bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_dh116_isaac_batch_scene.py tests/test_native_teacher_supported_reset.py tests/test_teacher_native_batch.py tests/test_action24_history_owner.py tests/test_dh116_dual_hand_episode.py tests/test_teacher_physical_group.py tests/test_teacher_batch_window.py tests/test_teacher_batch_rollout.py tests/test_teacher_batch_inputs.py tests/test_native_current_clock_consumers.py tests/test_dh116_work_packages.py
```

The captured native command uses `CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID bash scripts/python.sh -B -m scripts.train_dh116_native_teacher_pinch_batch`, with individually SHA-bound Reset receipt, two-lane smoke receipt, and training config plus a new output directory. The native CLI rejects a missing or unevidenced smoke, an occupied GPU, or a mismatched Reset. Use `--help` for the required arguments. It preserves 24D actions and the original finger-goal PPO settings. Both the compact CPU logs and full native rows remain on Ubuntu.
