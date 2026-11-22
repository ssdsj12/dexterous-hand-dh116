# Canonical supported reaction routes

Antigravity skill review R2 identified the ordering defect at HEAD44b0cdce82ebcf4646be9033c0dfe70aea4e0bce. Codex read and independently matched all cited relevant source SHAs, reproduced the defect and implemented this repair in Ubuntu. R3 is separately reviewing the full grasp progression; no R3 approval is claimed here.

## Root cause and actual change
FlexiblePhysicalExchange.__init__ selected per-side bodies by dictionary insertion order, while the supported rod's measured routes use contact_links canonical order. Alphabetically serialized bindings put base_link first. This changes the fresh commanded-zero route's body order and falsely rejects valid measured continuation routes. It does not prove earlier canonical-input native data had misapplied nonzero forces.

The constructor now selects supported fresh/continuation binding rows by canonical link names. The original default Pinch branch is byte-for-byte the same expression. PhysX prim/body identity, layout, force, same-clock and exclusive-owner checks remain. No solver, SDF, geometry, physics or action changes.

## Test evidence
Three new owner-boundary cases first failed on original code: fresh17 zero-to-measured route; supported6 continuation; supported7 continuation. All pass after repair. The continuation cases use distinct per-channel force and torque values and inspect the actual indexed composer writer. Fresh17 checks first-zero IDs and subsequent measured feedback over8 real adapter substeps with CPU solver/SDK doubles.

Primary regression:212 passed,0 failed,103.93s over21 files. Narrow33 tests overlap and are not added to212. Exact files and log SHA are in readiness. No Isaac was started in this package; these tests do not prove physical grasp success.

## Source integrity, not SHA bypass
The repaired file is outside the528 protection inventory but is pinned by the frozen native Reset/source chain. An actual old v11 recipe now rejects before launch with v2 source inventory SHA differs from actual scheduler source closure. That negative-control failure log is preserved; the historical B test now tests the correct rejection at the actual Repository boundary. Its previous positive evidence remains at commit0a5e85658471bcbf4e4de2c20b41cf38a8090258.

Old recipe pins, Reset artifacts, A receipt and checkpoint are unchanged. Historical A's diagnostic remains historical; current launch needs a new matching Reset/A/B chain. Unexecuted B is source-blocked as well as subject to its prior measured budget risk. No old SHA was refreshed to create compatibility.

## Limits and next work
Re-record a fresh17 Reset under this source using the existing recorder,2 SDK setup steps,0 control/solver/PPO,one environment and natural exit verification. Reuse existing owners rather than implementing another runtime. Full Teacher still needs measured contact/opposition, support-off, fixed wrist hold, phase and slip fields. Power's open-palm witness is not five-finger grasp qualification. R2 and successful BC/DAgger stay closed.

Fixed two RGB and their calibration remain unchanged. The protected528 files,3 historical checkpoints and unrelated user files remain preserved. Raw review text, failure logs and source evidence are private; only code,tests,readiness and this report are released.
