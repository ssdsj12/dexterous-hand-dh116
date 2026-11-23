# Full archived Lift replay and Tripod Reset source recovery

This checkpoint replays the complete authenticated physical journal for three selected supported-wire episodes. The existing Lift reader binds every row to the fixed measured Reset and checks the outer natural-exit result and artifact hash. All 9,122 archived exchanges were replayed from SDK tick 3. The phase replay used the existing 0.25 s candidate PreliftHold, or 120 consecutive 480 Hz ticks, with the original motion limits.

| Grasp | Recorded SDK ticks | Bilateral contact rows | Longest bilateral contact | Off-support Lift rows |
| --- | --- | ---: | ---: | ---: |
| Pinch | 3–432 | 0 | 0 | 0 |
| Tripod | 3–2192 | 69 | 1 SDK tick (2.083 ms) | 0 |
| Power | 3–6504 | 0 | 0 | 0 |

Tripod entered PreliftHold on 69 separate exchanges. It returned to Close after every one. Pinch and Power stayed in Close throughout these episodes. Both supports carried positive load at the last tick in all three runs. These runs show where to focus the next physical test: oppose on the same wire, sustain contact, then verify release from both supports before measuring 30 mm net Lift and 5 s Hold. The original 0.5 mm slip criterion still requires a physically qualified source.

## Tripod Reset

The first Reset rejection in the integration worktree is attributable to an absolute-path-sensitive backend inventory. The original execution worktree still reproduces the native profile, then fails the strict recorder check. Both the route-order-v3 and lift-v5 archives freeze recorder SHA `562e7b948205489e5c92ef394dfa470db21a7e92259c7cf43a5422f9b8fdc37c`. The live recorder SHA is `b9598e3cfc122fd538caa304d69b689c560f4e9677ae4f700b83a19532b8fbcf`. Older evidence remains valid for historical analysis within its authenticated scope. It cannot qualify the current recorder.

The unchanged recorder regenerated the two-step Tripod17 recipe on a dedicated Ubuntu checkout of `d415dd7610b6614daaa60801cd275c994ba33886`. The unchanged `--verify-existing-plan` command independently reproduced all 2,191 pins. The candidate keeps the same task configuration and assets, with one environment, one Reset, zero control/solver/Teacher/PPO operations, and a 180 s soft deadline. Its digest is `9bc3bb1f76a6f17c13b0c0935051e2a949aa15cb2e0f245ddfba9913c5232d4f`. This is a CPU candidate freeze. Native Reset and natural exit are pending.

The dedicated checkout passed 132 CPU tests across Reset acquisition, supported geometry, initialization and package scheduling. The frozen output has no native payload, attempt, or exit receipt. The readiness contract records the commands and log digests. The three previously recorded episodes were read only.

## OakInk source families

The archive member layout distinguishes original object grasps from virtual target transfers under the same source grasp. The earlier inventory's object counts referred to source objects only. Pen has 20 original poses, 633 transferred poses and 65 virtual target objects. Screwdriver has 12 original poses, 38 transfers and 7 virtual targets. Toothbrush has 14 original poses, 29 transfers and 5 virtual targets. Cylinder bottle has 38 original poses across four source objects, 8,516 transfers and 531 virtual targets. Grouped candidate selection must keep derivatives of a source grasp together. These are pose references. DH116 force, material dynamics and Lift qualification remain based on native records. [OakInk shape archive documentation](https://github.com/oakink/OakInk/blob/main/docs/datasets.md#data-folder-structure)
