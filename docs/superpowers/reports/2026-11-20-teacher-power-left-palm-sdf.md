# Power left palm SDF on CUDA

The original left Power palm fan matches the cooked convex boundary. The bake consumes
the authenticated left GPU geometry from the existing parity receipt and rejects a
source with an unqualified left fan. It uses the existing SDK bake and one-mesh
RopeBatch registration path.

- Device 2, RTX 4090 (SM89). Grid 110 × 79 × 153, 1 mm spacing, 12 mm margin.
- Inside and outside checks read −20.064 mm and +10.030 mm. Maximum absolute
  interpolated field value at the hull vertices was 0.415 mm.
- The one-mesh batch exposed a [1, 1, 6] wrench buffer and exited naturally with
  code 0 in 104.5 s. The run executed zero solver and physics steps.
- Raw receipt SHA-256 `8316ca79897e63f12fb5c7908544d549312a93c9b3cb66beec64b78fd6bb9cdf`.
- Field SHA-256 `824df146c5bd50c3e878a577e4f91ebce66b7b76fc2873c54aacef11c6a9e961`.

Run the pinned inputs with `python -m scripts.bake_dh116_power_left_palm_sdf --help`
for the required raw parity evidence, backend lock, and independent output flags.
CPU regressions (100 passed) are declared in the work package registry.

Power still needs a qualified right palm surface, live palm and digit contact
under load, and independent Lift/Hold evaluation. The live assembly remains at
11 of 13 mesh slots. No grasp or Teacher admission follows from baking this one field.
