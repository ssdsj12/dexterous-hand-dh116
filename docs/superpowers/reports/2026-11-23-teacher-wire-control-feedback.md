# Whole-control-window wire feedback

The existing wire feedback packer accepted one 480 Hz SDK frame. The new reducer reads the eight consecutive SDK exchanges ending at the decision boundary. Every complete contact interval contains 96 solver samples. The reducer validates the episode, reset, rod, SDK ticks, solver ticks, solver seconds, supported layout, ordered control substeps, and the next policy index. The offline driver derives that index from the sealed endpoint. Native pre-action timing still requires a separate same-owner comparison. It uses the current measured rod station and palm poses. It averages force-on-rod vectors in World axes, reverses the sign for hand reaction, and expresses the result in the final measured palm axes. The scalar measurements are minimum and mean normal force in N.

Only measured palm-local geometry, measured reaction, and validity enter the feedback packet. A measured zero retains a true validity mask, while a missing contact window or rejected pair source masks the reactions and retains current geometry when available. The offline driver requires the exact outer receipt SHA-256, a zero exit code from both the outer process and inferior, and each raw control row SHA-256 from that receipt. The CLI material IDs select probe stations on the existing rod and are absent from the policy-facing packet.

## Sealed original native replays

| Grasp | Terminal SDK step | Valid rows left / right | Positive rows left / right | Sum of per-row mean normal load, N left / right |
| --- | ---: | --- | --- | --- |
| Pinch | 248 | 2 / 2 | 0 / 0 | 0.000000 / 0.000000 |
| Pinch | 416 | 2 / 2 | 0 / 0 | 0.000000 / 0.000000 |
| Pinch | 432 | 2 / 2 | 0 / 0 | 0.000000 / 0.000000 |
| Tripod | 248 | 3 / 3 | 0 / 0 | 0.000000 / 0.000000 |
| Tripod | 1504 | 3 / 3 | 3 / 3 | 0.452084 / 0.396516 |
| Tripod | 2192 | 3 / 3 | 3 / 3 | 0.502844 / 0.437208 |
| Power | 248 | 6 / 6 | 0 / 0 | 0.000000 / 0.000000 |
| Power | 6400 | 6 / 6 | 1 / 1 | 0.002058 / 0.002717 |
| Power | 6504 | 6 / 6 | 1 / 1 | 0.001109 / 0.001178 |

The Pinch windows show measured zero normal load. The loaded Tripod windows provide valid loads on all three measured channels per hand. The Power windows show only one positive channel on each hand. No new native run or PPO update was executed. These results establish feedback readback and timing checks. They do not establish support-free Lift, a stable grasp, or generalization. The 8 held-out specifications remain reserved for a later frozen-Teacher evaluation.

Antigravity reviewed the original action and observation call chain. The existing collector must receive feedback from its own prior physical exchange window to use it live. Before that is enabled, it needs a separate observation version with first-decision validity, same-clock comparison with the owner's pre-action rod and palm readback, and episode-local window clearing. No existing Teacher input, joint limit, controller, or task success rule changed.

The final coupled CPU suite passed 198 tests. Two accepted-path collector routing fixtures had supplied a missing reward contract even though the existing runtime requires a file-backed contract. They now use the same fixture reward config already loaded from disk. The first combined run also exposed Python import ordering in the new test patch scope. Moving the mock to the reducer's bound measurement function preserved the original lift tests in the combined run.
