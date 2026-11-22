# Real bilateral feedback candidate: rejected for stable grasp

One new bounded Isaac run used GPU 2, one environment, the original flexible rod/supports and exact SDK1296 pose/contact/wrench replay. It completed 1808 SDK steps, 196 control windows and 21672 solver substeps in 512.09 s. App close, GDB inferior and the outer process all returned naturally with code 0, no signal or forced termination. There were zero Teacher queries, model loads or PPO updates.

## Paired outcome

Both captures are reduced from their original rod pair buffers with the same unchanged contact reducer and original thresholds. The fixed comparison interval is SDK1297–1520 (224 physical steps).

| Metric | Original right seating | New bilateral feedback |
| --- | ---: | ---: |
| Left qualifying steps | 31/224 | 30/224 |
| Right qualifying steps | 167/224 | 34/224 |
| Both hands qualifying steps | 24/224 | 12/224 |
| Longest right qualifying run | 60 SDK | 2 SDK |
| Longest bilateral qualifying run | 1 SDK | 1 SDK |

The new candidate regresses on this same interval. Its later right-hand run reaches 257 consecutive SDK steps (about 0.535 s), but that longer observation interval is not an improvement claim against the shorter baseline. Across all 512 feedback SDK steps, left/right/bilateral counts are 43/312/25; the longest bilateral run remains one SDK step.

All wrist commands were zero. Actual issued requested target travel was left index +0.0006 rad and right middle +0.0018 rad; no other finger received motion. The selector repeatedly nominated the same weakest pad after its travel was exhausted. The original right recipe moved both index and middle. This exposes a limitation of independent minimum-pad selection; it does not prove that any particular coordinated alternative will work.

Both supports continued carrying load. There were zero fully support-free steps, zero wrong-side pairs and no geometry or reaction-delivery violations. No rod Lift or stable grasp succeeded. No data/model enters a successful expert pool.

## Evidence and next implementation

The outer receipt, all raw per-SDK states, contact pairs, command decisions and reduction are retained on the server data disk. SHA values are in the readiness contract. The first offline decoder failed on a field absent from the older probe; its log is preserved. The corrected decoder recomputes both captures from the original pair buffers rather than filling missing fields with zero. This was offline reduction, not a second native run.

Keep the existing default path. Preserve this opt-in candidate as a failed comparison. Next inspect the already prepared coordinated left thumb-spread/thumb-flex/index/middle geometry candidate and its actual archived phase geometry before another finite native experiment. Do not repeat the same single-index dose, lower force/success thresholds or raise wrists before grasping the rod. Historical archive identity repair remains deferred.
