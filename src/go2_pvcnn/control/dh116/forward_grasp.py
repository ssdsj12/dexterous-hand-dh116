"""Stage-local forward/palm-down pose checks, not collision qualification."""
import numpy as np


def forward_pose_report(palm_world, wrist_world, grasp_world):
    """Check calibrated palm +Y forward, +Z down, and wrist behind grasp.

    Coordinates are world-space metres. This does not alter a robot pose,
    enforce global mirroring, or establish contact/lift success.
    """
    t = np.asarray(palm_world, dtype=float)
    wrist = np.asarray(wrist_world, dtype=float)
    grasp = np.asarray(grasp_world, dtype=float)
    if (t.shape != (4, 4) or wrist.shape != (3,) or grasp.shape != (3,)
            or not all(np.isfinite(x).all() for x in (t, wrist, grasp))):
        raise ValueError('expected finite palm (4,4), wrist (3,), grasp (3,)')
    r = t[:3, :3]
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-8, rtol=0)):
        raise ValueError('palm must be a proper rigid transform')
    finger_error = float(np.arccos(np.clip(r[0, 1], -1., 1.)))
    palm_error = float(np.arccos(np.clip(-r[2, 2], -1., 1.)))
    margin = float(grasp[0] - wrist[0])
    return dict(finger_error_rad=finger_error, palm_error_rad=palm_error,
                wrist_rear_margin_m=margin,
                passed=bool(max(finger_error, palm_error) <= np.deg2rad(15.) and margin > 0))
