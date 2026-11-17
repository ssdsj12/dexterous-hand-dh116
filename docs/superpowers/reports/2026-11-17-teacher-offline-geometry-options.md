# Offline testing of proposed geometry/device diagnostics

## Scope
Base8c0c0861c785508accea6ddb2d02c40f634f4494, branch11-17. Only Ubuntu CPU/offline work. No Isaac app, GPU solver, controls, Teacher/model inference or PPO was started. fast_shutdown remainsFalse. Original USD/limits/materials/control and30mm/5s/0.5mm success standards are unchanged.

## Static authored geometry
Executed existing scripts.audit_dh116_cooked_convex_hulls with the actual installed PhysX generatedSchema.usda,192 deterministic surface samples/direction, and an independent output path. Exit0,10 original distal pads decoded. Original USD hashes still match. Maximum link/world/link roundtrip error4.2350e-17m. Installed schema SHA829db2f41e2558fd5797fb3d211f7ec327e7468f377ff1316051ca2da51e3176.

This audits USD data and geometric estimates; it neither reads current native cooked hulls nor establishes that all34 current collision owners are attached. No evidence of source-asset modification was found. It is not a proof that every cached/native representation is valid.

## CPU/GPU historical hull parity
Executed existing scripts.audit_dh116_cooked_hull_device_parity using the actual historical CPU/GPU captures under flex-f4-parity-prep-Vu9H0f. Both original runs report matching source hashes, mesh-local frames and the same clock (SDK step4,2 commanded steps). Backend identity, all10 distal topology checks and vertex matching pass. Maximum CPU/GPU vertex difference0m. Recomputed JSON equals the prior parity result.

This is offline reanalysis of historical native data, not a newly performed native experiment, new initialization qualification or full34-owner validation. The10 distal surfaces must not be represented as all palm/proximal sources.

## Confirmed quaternion conversion defect and minimal repair
The worker accepted raw SDK XYZW quaternions with norm error <=1e-5 but constructed a matrix using the unnormalized values. The standard rotation formula presumes unit length.24 signed/oriented/scaled boundary cases failed the exact-rotation contract before repair;16 were also rejected by the unchanged protected rigid-matrix validator.

pose_matrix_xyzw now normalizes only its derived rotation after the original input guard. The raw SDK array is untouched and the previously added pre-conversion snapshot still stores raw values. Zero/nonfinite/out-of-tolerance inputs remain rejected. The protected collector and its1e-6 rigid-matrix tolerance remain unchanged. This is a coordinate conversion correction, not relaxed collision qualification or altered physical pose/control.

The24 boundary regressions failed before the two-line conversion repair and now pass.5 negative guard cases still pass. A complete34-owner CPU USD fixture with near-unit float32 poses traverses capture_once -> SDK clock adapter -> protected collector and reads68 fixture getter calls, with native_attested and live attachment proof bothFalse.

The2 actual saved base poses pass the old and new rigid validator. Their orthogonality residuals improve from2.30725e-7/5.82622e-8 to4.44e-16/3.33e-16. Because the spent native run did not preserve the remaining32 raw body poses, this proves the software defect, not the exact body or unique cause of the original native rejection.

## Verification and protection
414 CPU tests passed,0 failed,0 skipped in the full relevant regression. Earlier targeted pre-integration run:98 passed. The30 new cases cover24 valid boundaries,5 invalid inputs and1 end-to-end34-owner CPU capture. Readiness records exact commands, environment, input SHA, result SHA and failures-before-repair.

All528 protected hashes and3 original candidate model hashes match. No original data, checkpoint or protected audit script was overwritten. Raw audit outputs and logs stay on the data disk outside Git. Only worker conversion, tests and corresponding reports/readiness/registry/release metadata are committed via Ubuntu and published via Windows.

## Remaining work
The original34-owner native readback is still unqualified, and the exact failing body remains unknown. PhysX Tensor crash during Py_FinalizeEx remains unresolved; fast shutdown was not used to hide it. Next software work is explicit runtime/factory reference release. A fresh frozen, separately authorized initialization-only native test is required after the software checks; no old budget is reused. Teacher force closure/Lift admission remains blocked.
