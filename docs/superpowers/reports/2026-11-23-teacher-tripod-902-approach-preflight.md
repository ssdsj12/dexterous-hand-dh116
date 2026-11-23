# Tripod 902 approach lineage and current-source Reset preflight

The fresh Tripod Reset verified on GPU 2 in round 4 uses seed `5b040a43753b43d2a02136e9c44016b8125c02b5b274fa766a563f2f68ba93dd`. It exactly matches the historical zero-action diagnostic at the recorded Reset boundary. The staged 902 trajectory used for the new two-window screen has a different original seed, `9d16393ecdcc02d007c7518d2d25e79b16d14b268e8e228765a86935b0799704`. The rod node coordinates agree at Reset, but the right wrist translations differ by 11.762 mm. The two lineages are kept separate.

On the dedicated Ubuntu execution checkout, the existing scene and Reset functions rebuilt the exact staged seed. The ordinary `scripts/python.sh` entry point froze its two-step recipe with 2,191 inputs. An independent CPU-only GDB verification reproduced the plan and naturally exited with code 0. The plan fingerprint is `903c7654f6a06e5ecece50d111ae79a9d68e3806007272ca16a8dbf9ffc7d94e`. The native output field has no materialized artifact or attempted run.

The checked 902 native archive contains 39 complete 24D control windows from SDK 241 through 552. Each of the 312 control rows is matched to the original independent receipt, has consecutive clock ownership, unchanged issued action within each window and no hand contact pairs. They are followed in the private draft by two new candidate windows on the same archival seed. The first is right-wrist transverse relief. The second reproduces the exact original forward request in window 40. The CPU proof compares it with both the original and hold-then-forward controls.

The draft combines 2 setup SDK steps, 238 parked SDK steps and 41 whole control windows, ending at SDK 568 with at most 6,792 solver substeps and zero PPO. This is a command draft. A native continuation needs the original single-instance owner, a live same-source SDK 240 starting state, an exact SDK 552 anchor check, unchanged physical guards and an independently budgeted run. The retained candidate does not grant Reset authority to the wrong seed or grasp qualification to either one.

| Evidence | SHA-256 |
| --- | --- |
| Resolved 902 task | `8ad3450db5fcb63dfd79cc07f509c1ed7e791879dfa951fa5b2877020c9690b5` |
| SHA-verified 39-window prefix | `70f97bc859c8f641b4355524b9d06d13fae86b605d9e19739870f3d71c0a3060` |
| Reset plan file | `e9827ff8d517658bec7424f29b025d8ec316c356b6ba731809775a6d2aa7dbd0` |
| CPU GDB verification log | `dc746b0f0ee3c5ba51ce7cda6c0b5e41d715b19cc061f3c5904f5b5b691b9850` |
| 41-window command draft | `dbbd67046b5dd3b051c84b9ae24710b7ab123d7e115f57d17b9ccac311db196a` |

The evidence and exact invocations stay in the dedicated private Ubuntu output directory. Nothing in the original repository, older checkpoints or pending Power flight changed.
