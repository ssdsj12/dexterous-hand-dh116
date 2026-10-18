# DH116 structural USD conversion plan

Goal: generate two independent structural USD assets at the user-selected `/home/xk/coding/M1/assets/dh116_v2`, without replacing runtime O6.

Use installed IsaacLab UrdfConverter, not a new importer implementation. This is a generated-asset operation. Preserve source URDF linear mimic, fixed links, mass/inertia and joint limits. Explicitly disable joint drives; do not invent real hardware effort/velocity. Fixed base, non-instanceable meshes, convex-hull diagnostic colliders, self collision enabled. These colliders are not qualified fingertip contact surfaces.

- [ ] Confirm destination does not exist; do not overwrite existing assets.
- [ ] Launch installed IsaacLab headless on GPU0/core0 with single numerical threads. Convert desktop left/right URDFs using `fix_base=True`, `merge_fixed_joints=False`, `make_instanceable=False`, `convert_mimic_joints_to_normal_joints=False`, `self_collision=True`, `joint_drive.target_type='none'`, zero gains.
- [ ] Reopen each USD; count populated mesh points, rigid bodies and revolute joints, check metersPerUnit and unresolved dependencies. Fail structural acceptance on missing geometry or dependencies.
- [ ] Record source/output SHA and explicit qualification `structural_only`, `motion_authorized=False` in manifest. No physics stepping or control changes.
- [ ] Update notes with actual results and remaining mimic/limits/Jacobian work.

If importer or dependencies fail, preserve diagnostic artifacts and report the failure; do not advertise usable dynamic assets. No commits or uploads.
