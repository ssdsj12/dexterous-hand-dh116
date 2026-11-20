"""Candidate DH116 table clearance for a commanded joint interpolation.

All transforms and passive starting angles come from one measured SDK state.
This kinematic certificate neither advances physics nor predicts drive tracking.
"""
from itertools import product
from collections import namedtuple

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_tripod_fixed_root_path import certify_path
from .linear_coupling import JOINTS
from .tabletop_deformable_wrist_projection import CLEARANCE_M, DT
from .tabletop_free_wrist_grasp import CONTRACT as WRIST
from .tabletop_surface_contract import ACTION_CONTRACT

LINEAR_STEP_M = ACTION_CONTRACT["translation_rate_m_s"] * DT
ANGULAR_STEP_RAD = ACTION_CONTRACT["rotation_rate_rad_s"] * DT
_SweepBoundary = namedtuple("_SweepBoundary", ("prove", "endpoint_upper_bound"))


def _finite(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("finite named sweep inputs with the expected shape required")
    return array


def _rotation(value):
    frame = _finite(value, (3, 3))
    if (not np.allclose(frame.T @ frame, np.eye(3), atol=1e-8, rtol=0)
            or np.linalg.det(frame) <= 0):
        raise ValueError("proper rigid rotation required for the travel bound")
    return frame


def _rigid(value):
    frame = _finite(value, (4, 4))
    _rotation(frame[:3, :3])
    if not np.array_equal(frame[3], (0., 0., 0., 1.)):
        raise ValueError("homogeneous rigid transform required for the travel bound")
    return frame


class WristSweep:
    """Bound the authored FK path, anchored at the actual link transforms."""

    def __init__(self, *, root, edges, vertices, body_paths, table_top_m):
        self.root = root
        self.edges = tuple(edges)
        self.paths = dict(body_paths)
        self.vertices = {name: np.array(points, dtype=float, copy=True)
                         for name, points in vertices.items()}
        self.table_top = float(table_top_m)
        if (not self.vertices or self.vertices.keys() != self.paths.keys()
                or not np.isfinite(self.table_top)):
            raise ValueError("complete table and collision envelope mapping required")
        by_child = {edge[2]: edge for edge in self.edges}
        known = {self.root}
        for _, parent, child, t0, t1, axis in self.edges:
            if parent not in known or child in known:
                raise ValueError("ordered acyclic source FK required")
            for frame in (t0, t1):
                _rigid(frame)
            if axis is not None and not np.isclose(np.linalg.norm(_finite(axis, (3,))), 1.,
                                                   rtol=0, atol=1e-8):
                raise ValueError("source FK joint axes must have unit length")
            known.add(child)
        if not set(self.paths.values()) <= known:
            raise ValueError("source collider body path is missing from FK")
        self.joint_radii, self.wrist_radii = {}, {}
        for name, points in self.vertices.items():
            if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                    or not np.isfinite(points).all()):
                raise ValueError("nonempty finite xyz collider required")
            lever = float(np.linalg.norm(points, axis=1).max())
            radii = np.zeros(len(JOINTS))
            path = self.paths[name]
            while path != self.root:
                joint, parent, _, t0, t1, axis = by_child[path]
                lever += np.linalg.norm(t1[:3, 3])
                if axis is not None:
                    radii[JOINTS.index(joint)] = lever
                lever += np.linalg.norm(t0[:3, 3])
                path = parent
            self.joint_radii[name], self.wrist_radii[name] = radii, lever

    def _frames(self, anchor_rotation, xyz, angles, q):
        base = np.eye(4)
        base[:3, :3] = anchor_rotation @ Rotation.from_euler("XYZ", angles).as_matrix()
        base[:3, 3] = xyz
        frames = {self.root: base}
        for joint, parent, child, t0, t1, axis in self.edges:
            motion = np.eye(4)
            if axis is not None:
                motion[:3, :3] = Rotation.from_rotvec(axis * q[JOINTS.index(joint)]).as_matrix()
            frames[child] = frames[parent] @ t0 @ motion @ t1
        return frames

    def boundary(self, *, measured_link_frames, anchor_rotation, start_xyz, start_angles, start_q, final_q):
        """Precompute the owned reference once per hand and SDK step."""
        if measured_link_frames.keys() != self.paths.keys():
            raise ValueError("same-step frame required for every collision envelope")
        anchor_rotation = _rotation(anchor_rotation)
        xyz = _finite(start_xyz, (3,))
        angles = _finite(start_angles, (3,))
        q = _finite(start_q, (len(JOINTS),))
        dq = _finite(final_q, (len(JOINTS),)) - q
        frames = self._frames(anchor_rotation, xyz, angles, q)
        measured = {name: _rigid(value) for name, value in measured_link_frames.items()}
        anchored = {name: measured[name] @ np.linalg.inv(frames[path])
                    for name, path in self.paths.items()}
        joint_travel = {name: float(radius @ abs(dq)) for name, radius in self.joint_radii.items()}

        def prove(target_xyz, target_angles):
            delta_xyz = _finite(target_xyz, (3,)) - xyz
            delta_angles = _finite(target_angles, (3,)) - angles
            # Intrinsic XYZ composes three revolute joints. Bound each axis
            # contribution individually, then sum. The per-link anchor is rigid.
            angle_travel = abs(delta_angles).sum()
            translation_travel = np.linalg.norm(delta_xyz)
            travel = {name: translation_travel + angle_travel * lever + joint_travel[name]
                      for name, lever in self.wrist_radii.items()}

            def heights(t):
                frames = self._frames(anchor_rotation, xyz + t * delta_xyz,
                                      angles + t * delta_angles, q + t * dq)
                lower = {}
                for name, path in self.paths.items():
                    row = anchored[name][2] @ frames[path]
                    lower[name] = float((self.vertices[name] @ row[:3]).min() + row[3] - self.table_top)
                return lower

            proof = certify_path(heights, travel, dict.fromkeys(travel, CLEARANCE_M),
                                 max_depth=8, max_samples=129)
            return dict(status=proof["status"], proved=proof["proved"],
                        start_m=min(proof["samples"][0.0].values()),
                        end_m=min(proof["samples"][1.0].values()),
                        margin_lower_m=proof["minimum_clearance_margin_m"],
                        sample_count=len(proof["samples"]))

        def endpoint_upper_bound(last_xyz, last_angles):
            """Upper bound over the full original wrist target rate balls."""
            origin = _finite(last_xyz, (3,))
            prior_angles = _finite(last_angles, (3,))
            rz_parent = anchor_rotation @ Rotation.from_euler("XY", prior_angles[:2]).as_matrix()
            ry_parent = anchor_rotation @ Rotation.from_rotvec([prior_angles[0], 0., 0.]).as_matrix()
            axes = np.stack((anchor_rotation[:, 0], ry_parent[:, 1], rz_parent[:, 2]))
            frames = self._frames(anchor_rotation, origin, prior_angles, q + dq)
            ceiling = float("inf")
            for name, path in self.paths.items():
                transform = frames[path]
                points = self.vertices[name] @ transform[:3, :3].T + transform[:3, 3]
                direction = anchored[name][2, :3]
                height = points @ direction + anchored[name][2, 3] - self.table_top
                dz = np.cross(axes[:, None, :], points[None, :, :] - origin) @ direction
                slope = np.linalg.norm(dz, axis=0)
                # Product of three intrinsic rotations has a directional
                # second derivative bounded by radius * (sum |delta_j|)^2.
                residual = 1.5 * self.wrist_radii[name] * ANGULAR_STEP_RAD ** 2
                bound = height + slope * ANGULAR_STEP_RAD + LINEAR_STEP_M + residual
                ceiling = min(ceiling, float(bound.min()))
            return ceiling + 1e-12

        return _SweepBoundary(prove, endpoint_upper_bound)

    def prove(self, *, target_xyz, target_angles, **boundary):
        return self.boundary(**boundary).prove(target_xyz, target_angles)


def _bounded_motion(delta, limit):
    return delta * min(1., limit / max(np.linalg.norm(delta), 1e-15))


def preaction_target(*, nominal_xyz, nominal_angles, last_xyz, last_angles,
                     measured_angles, seed_xyz, certify, terminal_reserve_m=0.):
    """Prefer the nominal pair. Submit only options within all original step limits.

    Reserve applies at the target. The entire path must clear the hard floor
    even when the observed starting pose already lies inside the reserve.
    """
    last_xyz, last_angles = _finite(last_xyz, (3,)), _finite(last_angles, (3,))
    seed = _finite(seed_xyz, (3,))
    xyz = _finite(nominal_xyz, (3,))
    angles = _finite(nominal_angles, (3,))
    measured_angles = _finite(measured_angles, (3,))
    if not np.isfinite(terminal_reserve_m) or terminal_reserve_m < 0:
        raise ValueError("explicit finite nonnegative terminal reserve required")
    low, high = seed + WRIST["palm_lower_m"], seed + WRIST["palm_upper_m"]
    if np.any(last_xyz < low - 1e-12) or np.any(last_xyz > high + 1e-12):
        raise ValueError("last submitted palm target outside original workspace")
    linear_cap = LINEAR_STEP_M
    angular_cap = ANGULAR_STEP_RAD
    if np.any(abs(last_angles) > np.pi / 2 + 1e-12):
        raise ValueError("last submitted wrist angles outside original limits")
    step = _bounded_motion(np.clip(xyz, low, high) - last_xyz, linear_cap)
    angle_step = _bounded_motion(np.clip(angles, -np.pi / 2, np.pi / 2) - last_angles, angular_cap)
    recovery = _bounded_motion(np.clip(measured_angles, -np.pi / 2, np.pi / 2) - last_angles, angular_cap)
    upward = min(high[2] - last_xyz[2], linear_cap)
    # Try the full in-limit command first. Further options sacrifice part of
    # the requested turn or XY movement to respect the same 3D drive budget.
    translations = [("nominal", step), ("up", np.array([0., 0., upward])), ("hold", np.zeros(3))]
    rotations = [(str(fraction), fraction * angle_step) for fraction in (1., .75, .5, .25, 0.)]
    rotations.append(("towards_measured", recovery))
    attempted = set()
    for (shift, ds), (turn, da) in product(translations, rotations):
        goal_xyz, goal_angles = last_xyz + ds, last_angles + da
        key = (goal_xyz.tobytes(), goal_angles.tobytes())
        if key in attempted:
            continue
        attempted.add(key)
        proof = certify(goal_xyz, goal_angles)
        if (proof["proved"] is True and _endpoint_passed(proof, terminal_reserve_m)):
            return dict(accepted=True, xyz=goal_xyz, angles=goal_angles, translation=shift,
                        rotation=turn, certificate=proof, attempts=len(attempted))
    return dict(accepted=False, reason="no_certified_original_limit_target", attempts=len(attempted))


def _endpoint_passed(proof, terminal_reserve_m):
    return (np.isfinite([proof["start_m"], proof["end_m"]]).all()
            and proof["start_m"] >= CLEARANCE_M and proof["end_m"] >= CLEARANCE_M + terminal_reserve_m)
