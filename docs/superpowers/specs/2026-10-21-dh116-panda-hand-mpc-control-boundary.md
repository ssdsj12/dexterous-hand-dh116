# DH116 Hand MPC with Panda Approach and Lift

## Goal

Restore the full Isaac Sim visualization with both Panda arms and mounted DH116 hands, while separating control responsibilities. “Decoupled” means separate action ownership, not removing Panda from the scene.

## Approved control boundary

- Keep the existing fixed dual Panda + DH116 USD assembly and tabletop wire visible in the Isaac scene.
- Panda arm joints own the approach and lift trajectory phases.
- The DH116 MPC owns only the 12 active hand coordinates (six per hand) during grasp closure/hold. It must not output Panda arm coordinates.
- The phase coordinator sequences Panda approach → hold arm pose while the hand MPC closes and establishes grasp evidence → Panda lift while the hand maintains the grasp command.
- Hand world pose and kinematic/contact observations come from the live integrated articulation. The hand-only collector remains a diagnostic tool and is not the user-facing play scene.

## Implementation outline

Extend the integrated Isaac path rather than replacing its scene with the hand-only probe. Reuse the existing dual Panda/DH116 asset, tabletop wire backend, and arm trajectory/sequence controls. Add a bounded hand-MPC adapter whose action dimension is 12 and route its output only to the active DH116 joints. Freeze Panda targets during the close/hold interval; during approach/lift, keep the current DH116 grasp target and do not allow the hand planner to move Panda joints. Record phase, arm targets, hand MPC actions, measured contact, and wire height in the existing run evidence.

## Success criteria

1. Isaac Sim opens with both Panda arms, both DH116 hands, table, and wire visible.
2. Logs show Panda-controlled approach, DH116-MPC-controlled close/hold, then Panda-controlled lift.
3. Hand MPC action/state dimensions are validated as 12 active hand coordinates, with no Panda joints in its action vector.
4. Lift success is reported only when measured wire displacement and sustained grasp/contact evidence meet the existing sequence criteria. A closed hand alone is not success.
5. A failed grasp/lift is reported as failed evidence; it does not silently fall back to a fabricated successful trajectory or synthetic checkpoint.

## Verification and limitations

Run unit tests for phase/action routing, then one Isaac Sim integration run in the full scene. Confirm the rendered scene and inspect per-phase evidence. Do not launch concurrent Isaac instances. Simulation results do not establish hardware calibration or real DH116 force ratings.

## Options considered

- **Extend the integrated Panda+DH116 scene (selected):** preserves the expected visualization and existing Panda approach/lift sequence while isolating the hand MPC action interface.
- **Keep hand-only collection and replay arm motion separately:** simpler to probe hand contacts, but does not demonstrate the requested integrated scene or end-to-end phase transitions.
- **Use one joint-space MPC for all arm and hand joints:** preserves scene integration but violates the requested ownership split and lets hand planning alter Panda motion.

