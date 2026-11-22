# Power contact and Lift use the assigned wire sides

Power's shared-rod runner counted a 12-substep force pattern before checking the
wire half. Its feedback and Lift consumers ran different follow-up checks.
A complete palm and five-finger load on the opposite wire half could increment
the shared diagnostic. A crossed proximal contact could also leave Lift eligible
when all six required links had load on the expected side.

The supported Power reducer now checks each hand pair against both the material
index and the current world Y half derived from the same 41 solver nodes. The
left hand uses the positive half and the right hand the negative half. This
reuses the Pinch and Tripod split. A wrong pair clears that hand's opposition
flag for the affected substep. Raw per-link forces remain available. Proximal
contacts stay additional evidence and cannot substitute for the palm or a
finger. The shared ledger, feedback, and Lift now read the same verdict. The
legacy 13-slot Power measurement retains its force-only behavior.

## Evidence

Four CPU red cases exposed crossed hands, a bent material station on the wrong
world half, missing node positions, and a crossed proximal contact receiving
40 mm of synthetic Lift credit. All four passed after the fix. The related CPU
suite passed 172 tests. The readiness file records the exact command and hashes.

The same reducer processed 6,986 SHA-verified historical Isaac exchanges from
three previous native runs on CPU. The source records included settling,
proximal action, and the staged palm probe. Old and new reductions show zero
complete Power-opposition rows. Per-link force and registered support load
agreed for every replayed row. The staged palm run measured 0.01090 N left and
0.01516 N right peak normal load, below the existing 0.05 N per-link bound. It
recorded zero support-free rows. No new simulation or training ran.

## Next native evidence

Verify each frozen pregrasp Reset in its own run. Establish measured opposite
contact, stabilize the wire with fixed wrists, then verify lift from the original
rod height after support separation. Power requires both palms and all five
finger pads. Continue using the same fixed stereo pair.

After a nominal grasp meets Lift ≥30 mm, Hold ≥5 s and Slip ≤0.5 mm, evaluate a
single frozen policy per grasp over the independent held-out diameter, length
and stiffness matrix. Record palm-local wire position and tangent, measured
per-link reaction, and issued corrections. The current fixtures and historical
replay verify the evidence path. Teacher qualification and generalization stay
pending.
