"""Explicit authored geometry and provenance for the expanded grasp policy."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .linear_coupling import ACTIVE, JOINTS, expand


def validate_adaptive_options(*, adaptive_grasp, grasp_initialization,
                              close_reset_snapshot, initialize_from, physics_hz):
    if adaptive_grasp:
        if not grasp_initialization or close_reset_snapshot:
            raise ValueError('adaptive grasp requires grasp-initialization, without close-reset-snapshot')
        if initialize_from:
            raise ValueError('adaptive grasp changes observation/action dimensions; use fresh training or exact resume')
        if physics_hz != 480:
            raise ValueError('adaptive grasp requires 480 Hz physics')
    elif not close_reset_snapshot or grasp_initialization:
        raise ValueError('legacy mode requires close-reset-snapshot, without grasp-initialization')


def adaptive_initialization(payload, layout, provenance):
    """Construct a reset state; no claim of a physically settled checkpoint.

    Initial roots are absolute local-scene poses. The existing scalar root
    offset is retained solely for the timed lift, with no hidden extra raise.
    """
    if payload.get('format') != 'dh116-adaptive-grasp-initialization-v1':
        raise ValueError('unknown adaptive initialization format')
    if payload.get('source_contract') != provenance:
        raise ValueError('adaptive geometry provenance differs from current assets/pose/physics')

    def array(key, shape):
        value = np.asarray(payload[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'finite {key} with shape {shape} required')
        return value.copy()

    roots = array('roots', (2,7))
    joints = array('joint_positions', (2,11))
    grasp = array('grasp_command', (12,))
    wire = array('wire_pose', (7,))
    if (not np.allclose(np.linalg.norm(roots[:,3:],axis=1),1.,atol=1e-6,rtol=0)
            or abs(np.linalg.norm(wire[3:])-1.) > 1e-6):
        raise ValueError('unit root/object quaternions required')
    # This initializer is for the horizontal rigid cylinder resting on this table.
    if abs(wire[2]-.308) > .0005 or not np.allclose(wire[3:], [1,0,0,0],atol=1e-6,rtol=0):
        raise ValueError('initial object must be horizontal on the table')
    indices = [JOINTS.index(n) for n in ACTIVE]
    command = joints[:,indices].reshape(12)
    for hi in range(2):
        if not np.allclose(expand(command[hi*6:hi*6+6]),joints[hi],atol=1e-7,rtol=0):
            raise ValueError('initial joints must satisfy affine coupling')
        expand(grasp[hi*6:hi*6+6])
    configured = deepcopy(layout)
    offset = .004
    for hi,side in enumerate(('left','right')):
        configured['hand_roots'][side] = roots[hi].copy()
        configured['hand_roots'][side][2] -= offset
        configured['active_joints'][side] = grasp[hi*6:hi*6+6].copy()
    configured['wire_center'] = wire[:3].copy()
    state = dict(roots=roots, joint_positions=joints,joint_velocities=np.zeros((2,11)),
        initial_command=command,grasp_command=grasp,root_xy=np.zeros(2),root_offset=offset,
        wire_pose=wire,wire_velocity=np.zeros(6))
    return configured, state


def adaptive_contract_fields(initialization_sha256):
    """All changed control semantics are part of strict checkpoint equality."""
    return dict(observation_version='dh116-adaptive-grasp-v1',observation_size=125,
        action='20D_joint_palm_stiffness_v1',
        palm_motion='ppo_per_hand_xyz_target_v1',palm_xyz_limit_m=[.003,.003,.004],
        palm_rate_m_s=.002,palm_descent_clearance_m=.0021,
        grip_control='ppo_stiffness_target_v1',active_stiffness_range=[10.,20.],
        active_stiffness_reset=10.,active_stiffness_rate_per_s=10.,
        active_damping=.2,active_effort_limit_nm=.20,
        reset_initialization='authored_geometry_v1',
        grasp_initialization_sha256=initialization_sha256)


def object_spec(root, kind):
    """Load the actual hollow asset, keeping historical solid contracts intact."""
    if kind == 'solid_cylinder':
        return None, .12, {}
    if kind != 'hollow_straw':
        raise ValueError('unknown grasp object kind')
    directory = Path(root)/'assets/dh116_hollow_straw_v1'
    usd = directory/'straw.usda'
    manifest = json.loads((directory/'manifest.json').read_text())
    digest = hashlib.sha256(usd.read_bytes()).hexdigest()
    if manifest.get('usd_sha256') != digest:
        raise ValueError('hollow straw asset hash mismatch')
    expected = dict(outer_radius_m=.008,inner_radius_m=.0075,length_m=.5,
        wall_thickness_m=.0005,density_kg_m3=900.,axis='Y',collision_segments=32)
    if any(manifest.get(key) != value for key,value in expected.items()):
        raise ValueError('hollow straw geometry differs from supported outer envelope')
    mass = float(manifest['mass_kg'])
    if not np.isfinite(mass) or not np.isclose(mass,900*np.pi*(.008**2-.0075**2)*.5,rtol=1e-8,atol=0):
        raise ValueError('hollow straw mass differs from declared dimensions/density')
    return str(usd),mass,dict(object=dict(kind='hollow_straw',usd_sha256=digest,
        mass_kg=mass,rigid=True,collision='32_convex_annular_segments',**expected))
