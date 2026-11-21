# Power proximal Tensor View pose comparison

The earlier raw GPU rays differed from the Cooked fan at the rigid-interface pose by as much as 0.517 mm. An independent geometric fit reduced the mismatch and found approximately 13.8 mrad relative rotation, but leaves residuals and cannot identify an actual body pose. The native control therefore reads each original `finger52_link` twice, once through the Physics rigid interface and once from its exact single-body Tensor View, then generates the same fan ray pattern for each readback.

The view must identify the same original body and return finite metre position and unit XYZW rotation. The SDK clock must remain unchanged across all readings and rays. Both raw sources and all actual hit identities are retained. The relative-geometry comparison holds source topology fixed. It does not translate the raw report, adjust PhysX tolerances, or create an SDF.

66 related CPU tests passed. The distinct frozen CUDA diagnostic uses one Isaac process, only two SDK setup steps, zero control/solver/PPO steps and the existing cleanup and 180-second soft wall budget. Surface qualification still requires the actual native data.
