# Task 2 report — DH116 middle-finger contact geometry

Status: DONE

Calibration already exposed `finger33_link`; the builder and selector now
carry that digit through the same pinned source/layer SHA path as thumb and
index. A new `symmetric_grasp_middle_v1` bundle was generated from the fixed
dual-Panda DH116 USD with six references and six quantized convex-hull contact
meshes. The bundle explicitly remains simulation geometry only:
`physics_contact_qualified=false`, `hardware_calibrated=false`.

Verification:

```text
PYTHONPATH=src bash scripts/python.sh -m pytest -q \
  tests/test_dh116_middle_hook_geometry.py tests/test_dh116_usd_contract.py
3 passed
```

The generated middle entries use `finger33_link`, finite unit normals, and
record source mesh, derived mesh, reference, layer, and asset SHA values.
