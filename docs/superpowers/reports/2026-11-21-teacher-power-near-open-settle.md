# Power near-open support settling

The far-parked support experiment showed both supports carrying the free rod for 2,856 solver substeps, with no hand or table contacts. The first near pose hit the rod early and disturbed a support. This independent reset moves the hands closer while leaving space for the rod to sag. It retains the same free rod, gravity, support locations, locked meshes, DH116 protection and 24D controls.

| Hand | World offset from candidate wrap pose (mm) | Minimum sampled rod-tube gap (mm) | Support gap (mm) | Table clearance (mm) |
| --- | --- | ---: | ---: | ---: |
| Left | +12 X, +2 Z | 1.029 | 13.245 | 7.370 |
| Right | +15 X, +4 Z | 1.454 | 27.418 | 31.832 |

Both starting palms rotate by +0.22 rad about World Y from their hand-specific candidate wrap orientations. Digits use a bounded 0.08 rad opening from the candidate flexion values, subject to the existing active limits. Across 44 hand envelopes the World Y intervals certify at least 257.24 mm between the hands at startup. The tube check includes all 238 measured archived 41-node samples, divided into consecutive overlapping station bands, plus the initial curve. It covers the sampled rod positions and capsule radius. Motion between samples and the live native pose require separate validation.

The CLI binds the natural-exit parent receipt, the complete geometry screen SHA, and the exact selected candidate row for each hand. A candidate missing from either geometry row is rejected. The no-hand-contact checks apply throughout this separate run. The new launch mode has two setup SDK steps, 238 measured SDK steps, 2,856 solver substeps and one environment. It issues zero control actions and performs zero PPO updates. The 180-second soft limit and natural-exit evidence remain.

86 CPU tests passed across the supported task, physical exchange, end-support geometry, archived rod reference and work-package registry. The CPU freeze was exercised with the real receipts and both actual locked SDF sources. Its pre-publication fingerprint was `e9f6bb0caf1a5ed0958aae0492a2e2f41d3a4a3814aba3f95407a30e0e6a7f2e`. The native launcher freezes again after publication. The native follow-up must confirm both supports, actual hand poses, absence of incidental hand/table contact, and natural process exit. A later dynamic approach still needs its own measured path and force evidence.
