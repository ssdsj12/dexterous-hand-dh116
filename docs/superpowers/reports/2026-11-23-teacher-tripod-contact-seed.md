# Tripod contact-based Reset and local oscillation evidence

## What the verified nominal archive shows

A symmetric SDF endpoint decomposition retained each observed rod or hand endpoint in turn. Before the original wrist correction, the left index crossed zero gap 63 times in 127 adjacent SDK transitions. Hand-pose contribution was 18.90 µm RMS and rod motion 4.56 µm RMS. After the correction, the values were 63, 19.03 µm and 4.62 µm. In both intervals the hand-pose term dominated every gap crossing. Peak left index load was 1.19–1.21 N despite 989 of 1536 substeps below or at 0.05 N. This distinguishes endpoint motion contributions in the archive. It cannot establish the cause of motion or qualify force control. Instantaneous low load alone is a poor basis for another closing increment.

## CPU candidate and guards

SDK2066 of the nominal 10 mm route has all 12 valid opposing Tripod substeps on both hands. It is an isolated contact row. The generator takes the six measured active joints per hand and rigidly transfers each measured palm stance to the same material station on the free straight supported rod, rotating along the shortest arc between directed tangents. The left material anchor is 35 and the right is 3, each present in its own verified contact pairs. Original joint/mimic expansion, scene, gravity and 24D control are used.

The actual 9 mm × 450 mm candidate uses one authored straight rod pose and an additional eight actual early native sag frames from the same spec. It rechecks all 44 hand hulls, both supports and the table. The selected left retreat is 4.0 mm down in World Z and the right is a 14.5 mm XZ retreat. The table and support clearances retain their original protections.

- Left minimum rod gap 1.351 mm, minimum table gap 48.568 mm, and minimum support gap 28.913 mm.
- Right minimum rod gap 1.247 mm, minimum table gap 62.983 mm, and minimum support gap 6.493 mm.
- The right middle pad remains 7.241 mm from the conservative rod envelope. Reversing all 14.5 mm would take at least 1.45 s at the original wrist rate. There is no claimed safe straight path into full grasp.

Antigravity reviewed all three steps, including the actual generator, selected frame and original runtime path. The 77 CPU regressions passed. The first native plan preflight stopped on a noncanonical filename before Isaac. The next preflight resolved the same file and passed. The original runtime must still verify the resized native Reset, actual hulls and short support exchange.

## Next controlled phase

Measure approach against the actual pad SDF distances, then use multiple sample windows of measured force and geometric clearance to select closing or relief. Keep the existing target-only action path. Progress to Lift only from consecutive stable opposing contact. The separate Pinch and Power contact criteria and all off-support height, hold and slip requirements still apply.
