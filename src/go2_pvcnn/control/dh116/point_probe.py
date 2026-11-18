"""Numerical comparison of PhysX COM Jacobians with explicit surface points."""
import numpy as np
from scipy.spatial.transform import Rotation
from .contact_frames import point_jacobian


def compare_point(raw,names,side,link_pose,com_position,com_velocity,local,
                  expected_point,expected_jacobian,active_velocity):
    pose=np.asarray(link_pose);com=np.asarray(com_position);v=np.asarray(com_velocity)
    rotation=Rotation.from_quat(np.r_[pose[4:7],pose[3]]).as_matrix()
    measured=pose[:3]+rotation@local
    offset=measured-com
    jacobian=point_jacobian(raw,offset,names,side)
    measured_velocity=v[:3]+np.cross(v[3:],offset)
    errors=dict(position_error_m=float(np.linalg.norm(measured-expected_point)),
                jacobian_error_m_rad=float(np.max(np.abs(jacobian-expected_jacobian))),
                velocity_error_m_s=float(np.linalg.norm(measured_velocity-jacobian@active_velocity)))
    if not np.isfinite(list(errors.values())).all():raise ValueError('nonfinite probe errors')
    return errors
