"""Opt-in DH116 physical mapping and read-only kinematics audit.

Each hand uses its own authored USD joint frames. The current 24D contract
already commands named finger speeds, world translation and intrinsic XYZ
wrist rates, so this adapter preserves those commands. It exposes their actual
per-hand physical interpretation without assuming reflection symmetry.
Geometry and affine follower predictions are not measured native contact or
hardware calibration. Existing checkpoints must explicitly declare this
adapter before inference may use it.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .linear_coupling import ACTIVE, JOINTS, MIMIC, coupling_matrix
from .tabletop_surface_contract import ACTION_CONTRACT


ADAPTER_VERSION = 'dh116-bimanual-physical-action-v1'
PAD_BODIES = ('finger14_link', 'finger23_link', 'finger33_link')
DIGITS = ('thumb', 'index', 'middle')


def _finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


def _pose(value):
    pose = _finite(value, (7,), 'xyz wxyz pose')
    if not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5, rtol=0):
        raise ValueError('unit wxyz quaternion required')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def _rigid(value):
    result = _finite(value, (4, 4), 'joint frame')
    r = result[:3, :3]
    if (not np.allclose(result[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-6, rtol=0)):
        raise ValueError('proper rigid joint frame required')
    return result


def _full_angles(active):
    """Affine predicted followers, with no clamping of measured states."""
    result = coupling_matrix() @ _finite(active, (6,), 'active angles')
    for follower, _, _, offset in MIMIC:
        result[JOINTS.index(follower)] += offset
    return result


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class HandPhysicalKinematics:
    """Full authored joint tree, including independently measured followers."""

    def __init__(self, side, root, edges, pads, source_sha256):
        if side not in ('left', 'right'):
            raise ValueError('left or right hand required')
        if (not isinstance(source_sha256, str) or len(source_sha256) != 64
                or any(char not in '0123456789abcdef' for char in source_sha256)):
            raise ValueError('source USD SHA-256 required')
        self.side = side
        self.root = root
        self.source_sha256 = source_sha256
        self.edges = []
        pending = list(edges)
        known = {root}
        while pending:
            ready = [row for row in pending if row['parent'] in known]
            if not ready:
                raise ValueError('disconnected or cyclic source joint tree')
            for row in ready:
                if row['child'] in known:
                    raise ValueError('duplicate child in source joint tree')
                axis = row['axis']
                if axis is not None:
                    axis = _finite(axis, (3,), 'joint axis')
                    if not np.isclose(np.linalg.norm(axis), 1., atol=1e-6, rtol=0):
                        raise ValueError('unit joint axis required')
                self.edges.append((row['name'], row['parent'], row['child'],
                                   _rigid(row['t0']), _rigid(row['t1']), axis))
                known.add(row['child'])
                pending.remove(row)
        moving = [row[0] for row in self.edges if row[-1] is not None]
        if len(moving) != 11 or set(moving) != set(JOINTS):
            raise ValueError('all eleven named source finger joints required')
        self.pads = []
        if set(pads) != set(PAD_BODIES):
            raise ValueError('thumb, index and middle source pad references required')
        for name in PAD_BODIES:
            body = next((body for body in known if body.rsplit('/', 1)[-1] == name), None)
            if body is None:
                raise ValueError(f'missing distal body {name}')
            point = _finite(pads[name]['center_local_m'], (3,), 'source pad point')
            normal = _finite(pads[name]['reference_outward_normal_local'], (3,), 'source pad normal')
            if not np.isclose(np.linalg.norm(normal), 1., atol=1e-6, rtol=0):
                raise ValueError('unit source pad normal required')
            self.pads.append((body, point, normal))

    @classmethod
    def from_usd(cls, source, side):
        """Read each side's own frames and audited collision-surface atlas."""
        from .tabletop_surface_kinematics import SurfaceHandModel
        from .tabletop_effective_pad import _patch
        source = Path(source)
        surface = SurfaceHandModel.from_usd(source, side)
        pads = {}
        for shape in surface.spec['shapes']:
            name = shape['body'].rsplit('/', 1)[-1]
            if name in PAD_BODIES:
                if name in pads:
                    raise ValueError('ambiguous distal source collision body')
                pads[name] = _patch(shape['vertices'], side, name)
        return cls(side, surface.spec['root'], surface.spec['edges'], pads,
                   hashlib.sha256(source.read_bytes()).hexdigest())

    def _fk(self, q11, base_pose_w7):
        full = _finite(q11, (11,), 'measured full joint angles')
        poses = {self.root: _pose(base_pose_w7)}
        axes = {}
        ancestors = {self.root: ()}
        for name, parent, child, t0, t1, axis in self.edges:
            before = poses[parent] @ t0
            motion = np.eye(4)
            if axis is not None:
                motion[:3, :3] = Rotation.from_rotvec(axis * full[JOINTS.index(name)]).as_matrix()
                axes[name] = (before[:3, 3], before[:3, :3] @ axis)
            poses[child] = before @ motion @ t1
            ancestors[child] = ancestors[parent] + ((name,) if axis is not None else ())
        return poses, axes, ancestors

    def fk_full(self, q11, base_pose_w7):
        return self._fk(q11, base_pose_w7)[0]

    def fk_active(self, q6, base_pose_w7):
        return self.fk_full(_full_angles(q6), base_pose_w7)

    def pad_kinematics(self, q11, base_pose_w7):
        """World pad references and analytic derivatives, not native contacts."""
        poses, axes, ancestors = self._fk(q11, base_pose_w7)
        points, normals, position_j, normal_j = [], [], [], []
        for body, local_point, local_normal in self.pads:
            transform = poses[body]
            point = transform[:3, :3] @ local_point + transform[:3, 3]
            normal = transform[:3, :3] @ local_normal
            jp, jn = np.zeros((3, 11)), np.zeros((3, 11))
            for name in ancestors[body]:
                origin, axis = axes[name]
                col = JOINTS.index(name)
                jp[:, col] = np.cross(axis, point - origin)
                jn[:, col] = np.cross(axis, normal)
            points.append(point)
            normals.append(normal)
            position_j.append(jp)
            normal_j.append(jn)
        position_j, normal_j = np.array(position_j), np.array(normal_j)
        return dict(points_w_m=np.array(points), normals_w=np.array(normals),
                    position_jacobian_full=position_j, normal_jacobian_full=normal_j,
                    position_jacobian_active=position_j @ coupling_matrix(),
                    normal_jacobian_active=normal_j @ coupling_matrix())

    def audit_jacobians(self, q6, base_pose_w7, epsilon=1e-6):
        """Central differences audit analytic position and normal derivatives."""
        active = _finite(q6, (6,), 'active angles')
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('positive finite difference epsilon required')
        actual = self.pad_kinematics(_full_angles(active), base_pose_w7)
        fd_position, fd_normal = [], []
        for direction in np.eye(6) * epsilon:
            positive = self.pad_kinematics(_full_angles(active + direction), base_pose_w7)
            negative = self.pad_kinematics(_full_angles(active - direction), base_pose_w7)
            fd_position.append((positive['points_w_m'] - negative['points_w_m']) / (2 * epsilon))
            fd_normal.append((positive['normals_w'] - negative['normals_w']) / (2 * epsilon))
        fd_position, fd_normal = np.stack(fd_position, -1), np.stack(fd_normal, -1)
        return dict(epsilon_rad=epsilon,
                    position_error_m_per_rad=np.abs(fd_position - actual['position_jacobian_active']),
                    normal_error_per_rad=np.abs(fd_normal - actual['normal_jacobian_active']),
                    position_jacobian_fd=fd_position, normal_jacobian_fd=fd_normal)


class BimanualPhysicalActionAdapter:
    """Named physical actions; no reflected joint axes or fitted coefficients.

    This version deliberately preserves original SDK semantics. Palm-local
    linear velocities and world angular velocities are explicit diagnostics.
    The original intrinsic XYZ wrist-rate command stays in its authored seed
    chart, including when the actual base differs from its commanded target.
    """

    def __init__(self, models):
        if set(models) != {'left', 'right'} or any(model.side != side for side, model in models.items()):
            raise ValueError('independently loaded left and right models required')
        self.models = models

    def contract(self):
        contract = dict(version=ADAPTER_VERSION, actor_size=3844, critic_size=371,
                        action=deepcopy(ACTION_CONTRACT),
                        active_joint_names=list(ACTIVE), full_joint_names=list(JOINTS),
                        affine_mimic=[list(row) for row in MIMIC],
                        source_asset_sha256={side: self.models[side].source_sha256 for side in ('left', 'right')},
                        adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        frames='actual_runtime_base_link_and_palm_link;wxyz;column_vectors',
                        sdk_mapping='named_ACTIVE_rates;world_translation;seed_intrinsic_XYZ_rates',
                        qualification='independent_native_validation_required', hardware_calibrated=False)
        contract['sha256'] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return contract

    def require_checkpoint_contract(self, checkpoint_contract):
        if not isinstance(checkpoint_contract, dict) or checkpoint_contract.get('physical_action_adapter') != self.contract():
            raise ValueError('checkpoint requires explicit matching physical_action_adapter contract')

    def decode(self, action24, base_poses_w, palm_poses_w, wrist_angles_xyz=None, *, wrist_seed_poses_w=None):
        """Explicit experiment API. Inputs are physical SDK contract actions."""
        action = _finite(action24, (24,), 'physical 24D action')
        if np.any(np.abs(action) > 1. + 1e-7):
            raise ValueError('normalized physical actions must be within [-1,1]')
        if set(base_poses_w) != {'left', 'right'} or set(palm_poses_w) != {'left', 'right'}:
            raise ValueError('both actual base and palm poses required')
        angles = np.zeros((2, 3)) if wrist_angles_xyz is None else _finite(wrist_angles_xyz, (2, 3), 'intrinsic wrist angles')
        if wrist_seed_poses_w is None and np.any(angles != 0):
            raise ValueError('authored wrist seed poses required for nonzero intrinsic angles')
        if wrist_seed_poses_w is not None and set(wrist_seed_poses_w) != {'left', 'right'}:
            raise ValueError('both authored wrist seed poses required')
        result = {}
        for hi, side in enumerate(('left', 'right')):
            base, palm = _pose(base_poses_w[side]), _pose(palm_poses_w[side])
            finger = action[hi * 6:(hi + 1) * 6] * ACTION_CONTRACT['joint_rate_rad_s']
            velocity_w = action[12 + hi * 3:15 + hi * 3] * ACTION_CONTRACT['translation_rate_m_s']
            intrinsic_rates = action[18 + hi * 3:21 + hi * 3] * ACTION_CONTRACT['rotation_rate_rad_s']
            # The physical chart is seed * Rx(a) * Ry(b) * Rz(c).
            a, b, c = angles[hi]
            rx, ry, rz = (Rotation.from_rotvec(np.eye(3)[i] * angle).as_matrix()
                          for i, angle in enumerate((a, b, c)))
            # An actual pose can lag a driven target. Never infer the authored
            # seed by removing commanded angles from that measured pose.
            seed = base[:3, :3] if wrist_seed_poses_w is None else _pose(wrist_seed_poses_w[side])[:3, :3]
            angular_axes = seed @ np.column_stack((np.eye(3)[0], rx @ np.eye(3)[1], rx @ ry @ np.eye(3)[2]))
            result[side] = dict(active_joint_names=ACTIVE, finger_velocity_rad_s=finger,
                                full_joint_velocity_rad_s=coupling_matrix() @ finger,
                                translation_world_m_s=velocity_w,
                                translation_palm_m_s=palm[:3, :3].T @ velocity_w,
                                intrinsic_xyz_rate_rad_s=intrinsic_rates,
                                angular_velocity_world_rad_s=angular_axes @ intrinsic_rates,
                                world_angular_axes_reference=('actual_base_at_zero_chart_angles' if wrist_seed_poses_w is None else 'authored_wrist_seed'),
                                world_angular_axes=angular_axes, base_from_palm=np.linalg.inv(base) @ palm)
        return result

    def to_sdk(self, decoded):
        """Return canonical normalized 24D, independent of articulation order."""
        action = np.zeros(24)
        for hi, side in enumerate(('left', 'right')):
            hand = decoded[side]
            if tuple(hand['active_joint_names']) != ACTIVE:
                raise ValueError('canonical named active joint order required')
            action[hi * 6:(hi + 1) * 6] = _finite(hand['finger_velocity_rad_s'], (6,), 'finger speeds') / ACTION_CONTRACT['joint_rate_rad_s']
            action[12 + hi * 3:15 + hi * 3] = _finite(hand['translation_world_m_s'], (3,), 'world velocity') / ACTION_CONTRACT['translation_rate_m_s']
            action[18 + hi * 3:21 + hi * 3] = _finite(hand['intrinsic_xyz_rate_rad_s'], (3,), 'intrinsic rates') / ACTION_CONTRACT['rotation_rate_rad_s']
        if np.any(np.abs(action) > 1. + 1e-7):
            raise ValueError('decoded action exceeds existing normalized limits')
        return action

    def adapt_for_checkpoint(self, action24, base_poses_w, palm_poses_w, checkpoint_contract, wrist_angles_xyz=None, *, wrist_seed_poses_w=None):
        self.require_checkpoint_contract(checkpoint_contract)
        return self.to_sdk(self.decode(action24, base_poses_w, palm_poses_w, wrist_angles_xyz,
                                       wrist_seed_poses_w=wrist_seed_poses_w))

    @staticmethod
    def sdk_joint_velocity(decoded_hand, sdk_joint_names):
        """Map named full finger rates to the SDK's actual articulation order."""
        if len(set(sdk_joint_names)) != len(sdk_joint_names) or not set(JOINTS).issubset(sdk_joint_names):
            raise ValueError('unambiguous complete SDK finger joint names required')
        full = _finite(decoded_hand['full_joint_velocity_rad_s'], (11,), 'full finger speeds')
        result = np.zeros(len(sdk_joint_names))
        for index, name in enumerate(JOINTS):
            result[sdk_joint_names.index(name)] = full[index]
        return result


def audit_native_snapshot(world, models, env_ids):
    """JSON-ready read-only native mapping evidence for TensorWorld snapshots.

    Caller supplies the episode/policy/physics/time identity at capture time.
    No simulation step, target, material, drive or limit is modified here.
    """
    rows = []
    env_ids = list(env_ids)
    if (not env_ids or any(type(index) not in (int, np.int64, np.int32) or index < 0 for index in env_ids)
            or len(set(env_ids)) != len(env_ids)):
        raise ValueError('distinct nonnegative integer environment IDs required')
    for hi, side in enumerate(('left', 'right')):
        model, hand = models[side], world.hands[hi]
        names = list(hand.joint_names)
        if any(names.count(name) != 1 for name in JOINTS):
            raise ValueError('unique complete named native finger joints required')
        joint_ids = [names.index(name) for name in JOINTS]
        active_ids = [JOINTS.index(name) for name in ACTIVE]
        q = _numpy(hand.data.joint_pos)[:, joint_ids]
        if max(env_ids) >= len(q):
            raise ValueError('environment ID exceeds actual native batch')
        poses = _numpy(hand.data.body_link_pose_w)
        base_id = hand.body_names.index('base_link')
        palm_name = 'palm_link' if 'palm_link' in hand.body_names else 'base_link'
        palm_id = hand.body_names.index(palm_name)
        for env_id in env_ids:
            full, base = q[env_id], poses[env_id, base_id]
            active = full[active_ids]
            measured = model.pad_kinematics(full, base)
            coupled = model.pad_kinematics(_full_angles(active), base)
            sdk_points, sdk_normals = [], []
            for body, point, normal in model.pads:
                body_name = body.rsplit('/', 1)[-1]
                t = _pose(poses[env_id, hand.body_names.index(body_name)])
                sdk_points.append(t[:3, :3] @ point + t[:3, 3])
                sdk_normals.append(t[:3, :3] @ normal)
            sdk_points, sdk_normals = np.array(sdk_points), np.array(sdk_normals)
            normal_cosine = np.clip((measured['normals_w'] * sdk_normals).sum(-1), -1., 1.)
            fd = model.audit_jacobians(active, base)
            follower_residual = [float(full[JOINTS.index(f)] - (full[JOINTS.index(m)] * ratio + offset))
                                 for f, m, ratio, offset in MIMIC]
            rows.append(dict(env_id=int(env_id), hand=side, source_asset_sha256=model.source_sha256,
                             base_pose_w=base.tolist(), palm_link_name=palm_name,
                             palm_pose_w=poses[env_id, palm_id].tolist(),
                             base_from_palm=(np.linalg.inv(_pose(base)) @ _pose(poses[env_id, palm_id])).tolist(),
                             measured_full_joint_rad11=full.tolist(),
                             passive_coupling_residual_rad5=follower_residual,
                             pad_body_names=list(PAD_BODIES), measured_pad_points_w_m=sdk_points.tolist(),
                             fk_pad_points_w_m=measured['points_w_m'].tolist(),
                             measured_pad_normals_w=sdk_normals.tolist(),
                             pad_position_error_measured_q_m3=np.linalg.norm(measured['points_w_m'] - sdk_points, axis=-1).tolist(),
                             pad_position_error_coupled_q_m3=np.linalg.norm(coupled['points_w_m'] - sdk_points, axis=-1).tolist(),
                             pad_normal_error_measured_q_rad3=np.arccos(normal_cosine).tolist(),
                             fd_position_jacobian_max_m_per_rad=float(fd['position_error_m_per_rad'].max()),
                             fd_normal_jacobian_max_per_rad=float(fd['normal_error_per_rad'].max()),
                             measured_native_contact=False, geometry_only=True))
    return rows
