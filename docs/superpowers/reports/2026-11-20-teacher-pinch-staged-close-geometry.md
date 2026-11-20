# Pinch open Reset and bounded opposed close

## Change and review

The preceding same-boundary audit rejected the old right-hand target because the source thumb proximal hull reached the rod. The independent Antigravity review at HEAD `628447e464badab887e714a67a85065d7c180e20` identified two limitations of the old lower bound: XZ-only support and the use of one support for the whole rod. The new optional check bounds each adjacent segment separately in 3D, reuses the existing refined support directions and 2 micrometre numeric guard, and accepts explicit per-node displacement bounds. Continuous clearance uses the existing adaptive subdivision and global shape travel bounds. A bound below zero means the check did not prove separation. The original audit mode reproduced its exact archived result SHA.

The candidate fixes one absolute open Reset per hand and one 64-SDK-step close. The simulation's original 480 Hz step, eight-substep control period, wrist translation and rotation limits, finger rates and joint contracts determine a canonical 24D command. A CPU fixture sends the command through the original finger and wrist kernels. Both thumbs and index fingers advance; the other fingers hold. Each wrist uses its exact intrinsic XYZ command in the new mount frame. The new Reset is evaluated as an initial condition. The relocation from the old Reset is never represented as a commanded motion.

The old rod seed (41 free nodes, original ground, radius and gravity) is unchanged. The candidate is separate from the existing Reset profiles and is not enabled in the runtime.

## Results

The exact same Reset and close passed geometry screening on **9 of 9** retained SDK boundaries (8, 16, 24, 32, 48, 64, 80, 96, 112). Only boundary 16 was used for fitting. The other boundaries remain from the same archived episode and are not independent native initial states. An additional authored-FK check passed at the original grounded rod Reset, without using transferred measured link corrections.

| Scope across the nine retained boundaries | Left | Right |
| --- | ---: | ---: |
| Minimum initial rod gap lower bound, all 22 envelopes | 0.4849 mm | 0.7818 mm |
| Continuous table check including an additional 0.050 mm reserve | 9/9 | 9/9 |
| Non-target rod clearance proved along the whole close path | 9/9 | 9/9 |
| Minimum non-target rod lower bound at the close endpoint | 0.2130 mm | 0.1890 mm |
| Worst same-frame nominal target pad gap at endpoint | -0.3099 mm | -0.2860 mm |

The swept-Y interhand gap lower bound was 291.315 mm. The additional 0.050 mm rod node motion and table reserve are sensitivity assumptions, not new tracking measurements. The target pad signed-distance and direction scores use nominal rod nodes. They establish geometry for an experiment, with no force, PhysX self-contact, settling, friction, or sensor certificate. The right thumb open target is close to the mechanical lower end. Native Reset needs actual q, qd, joint margins and cooked hull readback before movement.

The recorded old candidate was run with both the fixed-rod segment checker and with +0.050 mm per-node allowance. Neither audit promoted it. The earlier single-pose fixed-wrist searches did not jointly satisfy pad margin, proximal clearance and table checks. The staged candidate succeeds only in the declared offline scope.

## Reproduce

```sh
python -m scripts.audit_dh116_pinch_staged_closure \\
  --project "$PROJECT" --journal "$V13_JOURNAL" \\
  --journal-sha256 3b46bc46f0b615eca40e26c8bafec9c5df9b873cb89ddaf45b171889f5cf2b7b \\
  --candidate "$PROJECT/configs/dh116_execution/pinch-staged-open-and-close-archived-v1.json" \\
  --candidate-sha256 afc7e79c75a5b0436d4b9224c3b9d556cdcb4b9a933cbd1dea615153f4fd186e \\
  --output "$UNUSED_OUTPUT"
```

Final retained replay SHA-256 `601b32e01119fde2974ad279903383082a3b18b265b7b72dd856aa4b6a427614`. Authored grounded check SHA-256 `c52bd2087f11ac1c0c7a6b272143a429851baf4cdbb000091ef7368106ea2f78`. **118** relevant CPU tests passed. No Isaac, solver or PPO steps were added.

## Next measurements

Bind the independent candidate to the original, source-locked Reset owner. Measure settling and all 44 collision envelopes before issuing the new 24D command. Once its bounded native Reset receipts pass, execute a 64-step close and measure simultaneous loaded opposition and rod movement before considering Lift. Pinch Teacher admission remains pending. Tripod requires measuring a contact approach beyond the verified free-space tracking and Power still requires palm/finger enclosure. The review proposed using the straight-rod Tripod tracking receipt to set a measured clearance budget and testing coupled Power roll/yaw around a retained palm witness.
