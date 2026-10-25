# DeformX bundled flexible object import

The user selected the paper project's bundled flexible object and an Isaac-supported import, with lift success measured near the grasp rather than over the entire rope. These choices authorize implementation.

Use DeformX's `wire_orange_s20_r0.005_l1_smooth.usdc`, retaining its original mesh, skin and 20 bones. It is already an Isaac-readable USD asset: no OBJ/STL conversion. Package a portable local USD entry with relative dependencies and provenance. Disable bundled lighting only within the imported object. Do not add a rigid-body schema to the wire. An explicit Cosserat runtime must drive the skeleton; USD alone does not supply flexibility.

Geometry is length 1 m, radius 0.005 m and 20 elements. Use example material density 700 kg/m^3, Young modulus 200000 Pa, shear modulus E/3, damping 0.1 and rod dt 1e-5 s. The example physics radius 0.00635 m conflicts with the mesh; intentionally align physics with the mesh and document this difference. These are example simulation parameters, not calibrated material measurements.

Keep the existing DH116 scene and controller. First verify asset portability, skeleton deformation, table contact and two-way hand contact in a single-environment Isaac import executable. No fixed attachment or scripted grasp success. Existing 512-environment rigid PPO remains running during offline preparation; save it before an Isaac validation, and never run two Isaac instances. Do not claim the existing rigid PPO has become a flexible 512-environment trainer. That requires a separate vectorized solver integration and measured throughput.

Record fixed material regions near each initial hand grasp; local surface height is evaluated there, so selection cannot jump to a different segment each step. Allow other parts to sag or remain on the table. Success remains 30 mm above the tabletop for 0.5 s with the existing stability constraints when connected to training.

All work is local, on the current dedicated branch. Preserve existing dirty work and run artifacts. Single numeric thread and CPU0 affinity apply. Report asset import, dynamic validation and trained grasp success separately.
