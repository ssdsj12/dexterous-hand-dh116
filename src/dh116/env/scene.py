"""USD scene authoring, assets/material inputs, reset/readback and static scene validation."""
import copy
import hashlib as scene_input__hashlib
from copy import deepcopy as scene_reset__deepcopy
from pathlib import Path as scene_suppo__Path
import numpy as scene_valid__np
from copy import deepcopy as end_suppo_scene__deepcopy
from dataclasses import asdict
import numpy as mater__np
from dataclasses import dataclass as teach_pose_goal__dataclass
import hashlib as usd_surfa__hashlib
import numpy as scene_autho__np
import json as scene_input__json
import hashlib as scene_reset__hashlib, json as scene_reset__json, math as scene_reset__math
import numpy as scene_suppo__np
from dh116.physics.geometry import DISTAL_PAD_LINKS as scene_valid__DISTAL_PAD_LINKS
import numpy as end_suppo_scene__np
from dataclasses import dataclass as load_param__dataclass

def material_contract(profile):
    if profile == 'default':
        return {}
    if profile == 'extreme_grip':
        return dict(object_contact_material=dict(profile='extreme_grip_v1', static_friction=5.0, dynamic_friction=5.0, restitution=0.0, friction_combine_mode='max', restitution_combine_mode='min', scope='object_colliders', calibration='artificial_high_friction_experiment_not_measured'))
    if profile == 'rough_rod':
        return dict(object_contact_material=dict(profile='rough_rod_v1', static_friction=1.5, dynamic_friction=1.2, restitution=0.0, friction_combine_mode='max', restitution_combine_mode='min', scope='object_colliders', calibration='training_assumption_not_measured'))
    if profile != 'high_friction':
        raise ValueError('unknown object contact material profile')
    return dict(object_contact_material=dict(profile='high_friction_v1', static_friction=1.2, dynamic_friction=1.0, restitution=0.0, friction_combine_mode='max', restitution_combine_mode='min', scope='object_colliders'))
import hashlib as teach_pose_goal__hashlib
from pathlib import Path as usd_surfa__Path
from dh116.env.contracts import linea_coupl__ACTIVE as scene_autho__ACTIVE
from pathlib import Path as scene_input__Path
import numpy as scene_reset__np
from dh116.physics.geometry import DISTAL_PAD_LINKS as scene_suppo__DISTAL_PAD_LINKS
from dh116.hand.kinematics import JOINT_NAMES

def plan_end_support_scene(layout, clearance_m, *, wire_length_m=None, wire_radius_m=None):
    """Return an independent layout and JSON metadata (None for the legacy scene).

    clearance_m is rod-bottom clearance above the unchanged table. Explicit
    dimensions must come from the actual spawn configuration, not an assumed
    grasp span. The pure helper also accepts the reference layout's dimensions.
    """
    clearance = float(clearance_m)
    if not end_suppo_scene__np.isfinite(clearance) or clearance < 0:
        raise ValueError('end support clearance must be finite and nonnegative')
    configured = end_suppo_scene__deepcopy(layout)
    if clearance == 0:
        return (configured, None)
    length = float(layout['wire_length_m'] if wire_length_m is None else wire_length_m)
    radius = float(layout['wire_radius_m'] if wire_radius_m is None else wire_radius_m)
    if not end_suppo_scene__np.isfinite([length, radius]).all() or length <= 0 or radius <= 0:
        raise ValueError('end supports require finite positive wire length and radius')
    if layout.get('wire_axis', 'Y') != 'Y':
        raise ValueError('end supports require the native horizontal Y-axis rod')
    (center, table_center, table_size) = (end_suppo_scene__np.asarray(layout[k], dtype=float) for k in ('wire_center', 'table_center', 'table_size'))
    if any((v.shape != (3,) or not end_suppo_scene__np.isfinite(v).all() for v in (center, table_center, table_size))) or end_suppo_scene__np.any(table_size <= 0):
        raise ValueError('finite wire/table centers and positive table dimensions required')
    table_top = float(table_center[2] + table_size[2] / 2)
    bottom = table_top + clearance
    dz = bottom - (center[2] - radius)
    configured['wire_center'] = center.copy()
    configured['wire_center'][2] += dz
    for (side, pose) in configured['hand_roots'].items():
        pose = end_suppo_scene__np.asarray(pose, dtype=float).copy()
        if pose.shape != (7,) or not end_suppo_scene__np.isfinite(pose).all():
            raise ValueError('finite seven-component hand root poses required')
        pose[2] += dz
        configured['hand_roots'][side] = pose
    axial_width = min(0.01, 0.04 * length)
    inset = min(0.002, 0.01 * length)
    end_offset = length / 2 - inset - axial_width / 2
    transverse_width = max(0.024, 2 * radius + 0.008)
    supports = []
    for (name, sign) in (('negative_y', -1), ('positive_y', 1)):
        position = (float(center[0]), float(center[1] + sign * end_offset), table_top + clearance / 2)
        size = (transverse_width, axial_width, clearance)
        if any((abs(position[i] - table_center[i]) + size[i] / 2 > table_size[i] / 2 + 1e-09 for i in (0, 1))):
            raise ValueError('end support footprint falls outside the table')
        supports.append(dict(name=name, center_m=list(position), size_m=list(size)))
    metadata = dict(version='dh116-static-end-supports-v1', clearance_m=clearance, dz_m=float(dz), table_top_m=table_top, lift_reference_bottom_m=float(bottom), wire_length_m=length, wire_radius_m=radius, wire_axis='Y', central_gap_m=float(2 * end_offset - axial_width), coordinates='source_environment_without_origin', support_type='persistent_static_collision', supports=supports)
    return (configured, metadata)
import json as load_param__json

def hand_material_contract(profile):
    if profile == 'default':
        return {}
    if profile not in ('grip', 'grip_compliant'):
        raise ValueError('unknown hand contact material profile')
    spec = dict(profile='grip_v1', static_friction=1.0, dynamic_friction=0.8, restitution=0.0, friction_combine_mode='max', restitution_combine_mode='min', scope='all_hand_colliders', calibration='training_assumption_not_measured')
    if profile == 'grip_compliant':
        spec.update(profile='grip_compliant_pads_v1', pad_compliance=dict(links=['finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link'], stiffness_n_m=10000.0, damping_n_s_m=30.0, calibration='simulation_estimate_not_force_displacement_calibrated'))
    return dict(hand_contact_material=spec)
import json as teach_pose_goal__json
import numpy as usd_surfa__np
from dh116.env.contracts import JOINTS as scene_autho__JOINTS
import numpy as scene_input__np
from dh116.env.contracts import linea_coupl__ACTIVE as scene_reset__ACTIVE
scene_suppo__SIDES = ('left', 'right')
from dh116.hand.kinematics import ANGULAR_JOINT_NAMES

def translate_end_support_seed(seed, scene):
    """Translate a fresh copy of the original local seed once at each bind.

    root_offset/root_xy, rotations, velocities and joint state retain their
    existing meanings. Both GpuWorld and TensorWorld read these same seeds.
    """
    shifted = end_suppo_scene__deepcopy(seed)
    if scene is not None:
        dz = scene['dz_m']
        pose = end_suppo_scene__np.asarray(shifted['wire_pose'], dtype=float).copy()
        if pose.shape != (7,) or not end_suppo_scene__np.isfinite(pose).all():
            raise ValueError('finite seven-component seed wire pose required')
        pose[2] += dz
        shifted['wire_pose'] = pose
        if 'roots' in shifted:
            roots = end_suppo_scene__np.asarray(shifted['roots'], dtype=float).copy()
            if roots.shape != (2, 7) or not end_suppo_scene__np.isfinite(roots).all():
                raise ValueError('finite two-hand seed root poses required')
            roots[:, 2] += dz
            shifted['roots'] = roots
    return shifted
import math as load_param__math

def material_config_fields(spec, *, compliant=False):
    fields = {k: spec[k] for k in ('static_friction', 'dynamic_friction', 'restitution', 'friction_combine_mode', 'restitution_combine_mode')}
    if compliant:
        pad = spec['pad_compliance']
        fields.update(compliant_contact_stiffness=pad['stiffness_n_m'], compliant_contact_damping=pad['damping_n_s_m'])
    return fields
import math as teach_pose_goal__math
from dh116.env.contracts import expand
from dh116.env.contracts import runti_io__ROOT as ROOT
from dh116.env.contracts import JOINTS as scene_reset__JOINTS
PADS = tuple(scene_suppo__DISTAL_PAD_LINKS)

def wire_geometry_from_spawn(spawn):
    """Read native CylinderCfg dimensions, or the actual centered USD envelope.

    USD is imported lazily, only for the explicit external rigid-rod path.
    Unsupported off-center/non-Y envelopes fail before any scene is authored.
    """
    from pxr import Usd, UsdGeom
    if hasattr(spawn, 'radius') and hasattr(spawn, 'height'):
        if spawn.axis != 'Y':
            raise ValueError('end supports require a Y-axis cylinder spawn')
        (length, radius) = (float(spawn.height), float(spawn.radius))
    else:
        from pxr import Usd
        from pxr import UsdGeom
        stage = Usd.Stage.Open(str(spawn.usd_path))
        if stage is None or not end_suppo_scene__np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise ValueError('end supports require a readable meter-unit rod USD')
        root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
        bounds = bounds.ComputeWorldBound(root).ComputeAlignedRange()
        (low, high) = (end_suppo_scene__np.asarray(bounds.GetMin()), end_suppo_scene__np.asarray(bounds.GetMax()))
        if not end_suppo_scene__np.isfinite([low, high]).all() or end_suppo_scene__np.any(high <= low) or (not end_suppo_scene__np.allclose(low, -high, atol=1e-06, rtol=0)) or (not end_suppo_scene__np.isclose(high[0], high[2], atol=1e-06, rtol=0)) or (high[1] <= max(high[0], high[2])):
            raise ValueError('end supports require a centered Y-axis cylindrical USD envelope')
        (length, radius) = (float(high[1] - low[1]), float(high[2]))
    if not end_suppo_scene__np.isfinite([length, radius]).all() or length <= 0 or radius <= 0:
        raise ValueError('finite positive spawned rod dimensions required')
    return (length, radius)
from pathlib import Path as load_param__Path

def bind_hand_materials(stage, root_path, hard_material, spec, soft_material=None):
    paths = bind_object_colliders(stage, root_path, hard_material)
    if 'pad_compliance' not in spec:
        return (paths, verify_collider_bindings(stage, paths, spec))
    if soft_material is None:
        raise ValueError('compliant pad material required')
    pad = spec['pad_compliance']
    soft_paths = []
    for link in pad['links']:
        soft_paths.extend(bind_object_colliders(stage, root_path + '/' + link, soft_material))
    soft_spec = dict(spec, compliant_contact_stiffness=pad['stiffness_n_m'], compliant_contact_damping=pad['damping_n_s_m'])
    verify_collider_bindings(stage, soft_paths, soft_spec)
    hard_paths = [path for path in paths if path not in soft_paths]
    hard_spec = dict(spec, compliant_contact_stiffness=0.0, compliant_contact_damping=0.0)
    if hard_paths:
        verify_collider_bindings(stage, hard_paths, hard_spec)
    return (paths, dict(colliders=len(paths), compliant_pad_colliders=len(soft_paths), hard_colliders=len(hard_paths), pad_compliance=pad))
from pathlib import Path as teach_pose_goal__Path
from dh116.hand.kinematics import install_hand_mimic
from dh116.env.contracts import checked_file
from dh116.env.contracts import active_limits as scene_reset__active_limits
HAND_NAMES = tuple((f'{s}.{n}' for s in scene_suppo__SIDES for n in PADS))

def end_suppo_scene__author_end_supports(stage, prefix, scene):
    """Author static USD cubes under the source environment BEFORE cloning.

    No RigidBodyAPI (hence static), joints, materials or object edits. The
    existing table, gravity and rod mass/material configuration stay intact.
    """
    from pxr import UsdGeom, UsdPhysics
    if scene is None:
        return ()
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    parent = prefix + '/EndSupports'
    UsdGeom.Xform.Define(stage, parent)
    paths = []
    for support in scene['supports']:
        path = parent + '/' + support['name']
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
        cube.AddTranslateOp().Set(Gf.Vec3d(*support['center_m']))
        cube.AddScaleOp().Set(Gf.Vec3d(*support['size_m']))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        prim = cube.GetPrim()
        prim.AddAppliedSchema('PhysxCollisionAPI')
        prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(0.001)
        prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.0)
        paths.append(path)
    return tuple(paths)
import numpy as load_param__np

def verify_collider_bindings(stage, paths, spec):
    """Check resolved physics-purpose binding, including combination modes."""
    from pxr import UsdPhysics
    from pxr import UsdPhysics
    from pxr import UsdShade
    if not paths:
        raise ValueError('no collider bindings to verify')
    for path in paths:
        (material, _) = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')
        if not material:
            raise ValueError(f'missing physics material: {path}')
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        values = [physics.GetStaticFrictionAttr().Get(), physics.GetDynamicFrictionAttr().Get(), physics.GetRestitutionAttr().Get()]
        if not mater__np.allclose(values, [spec[k] for k in ('static_friction', 'dynamic_friction', 'restitution')], atol=1e-06, rtol=0):
            raise ValueError(f'bound collider material differs: {path}')
        prim = material.GetPrim()
        if prim.GetAttribute('physxMaterial:frictionCombineMode').Get() != spec['friction_combine_mode'] or prim.GetAttribute('physxMaterial:restitutionCombineMode').Get() != spec['restitution_combine_mode']:
            raise ValueError(f'bound material combine mode differs: {path}')
        for (key, attribute) in (('compliant_contact_stiffness', 'compliantContactStiffness'), ('compliant_contact_damping', 'compliantContactDamping')):
            if key in spec:
                value = prim.GetAttribute('physxMaterial:' + attribute).Get()
                if value is None or not mater__np.isclose(value, spec[key], atol=1e-06, rtol=0):
                    raise ValueError(f'bound material compliance differs: {path}')
    return dict(colliders=len(paths), friction_combine_mode=spec['friction_combine_mode'], restitution_combine_mode=spec['restitution_combine_mode'])
import numpy as teach_pose_goal__np
from dh116.hand.kinematics import configure_mimic_compliance
from dh116.env.contracts import ProjectPaths
from dh116.env.contracts import _SOURCE_LOWER
PALM_NAMES = tuple((f'{s}.base_link' for s in scene_suppo__SIDES))

@load_param__dataclass(frozen=True)
class LoadParameters:
    max_force_n: float = 1.0
    force_rate_n_s: float | None = 0.5
    deadband_n: float = 0.02
    stiffness_max: float = 20.0
    stiffness_rate: float = 10.0
    effort_limit_nm: float = 0.2
    calibration_status: str = 'simulation_estimate'

    def __post_init__(self):
        bounds = dict(max_force_n=(0.1, 32.0), force_rate_n_s=(0.0, 10.0), deadband_n=(0.0, 0.5), stiffness_max=(10.0, 200.0), stiffness_rate=(0.0, 200.0), effort_limit_nm=(0.0, 2.0))
        for (key, (low, high)) in bounds.items():
            value = getattr(self, key)
            if key == 'force_rate_n_s' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or (not load_param__math.isfinite(value)) or (not low < value <= high):
                raise ValueError(f'{key} must be finite within ({low},{high}]')
        if self.calibration_status != 'simulation_estimate':
            raise ValueError('this profile cannot assert hardware calibration')

    def as_dict(self):
        return asdict(self)

def verify_hand_material_properties(properties, profile, expected_envs, expected_shapes):
    values = mater__np.asarray(properties, dtype=float)
    if values.shape != (expected_envs, expected_shapes, 3) or not mater__np.isfinite(values).all():
        raise ValueError('invalid live hand material properties')
    settings = hand_material_contract(profile)
    report = dict(environments=expected_envs, shapes_per_hand=expected_shapes, unique_coefficients=mater__np.unique(values.reshape(-1, 3), axis=0).tolist())
    if settings:
        spec = settings['hand_contact_material']
        expected = [spec[k] for k in ('static_friction', 'dynamic_friction', 'restitution')]
        pad = spec.get('pad_compliance')
        if pad:
            soft = mater__np.isclose(values[..., 2], -pad['stiffness_n_m'], atol=1e-06, rtol=0)
            hard = mater__np.isclose(values[..., 2], 0.0, atol=1e-06, rtol=0)
            if not mater__np.allclose(values[..., :2], expected[:2], atol=1e-06, rtol=0) or not (soft | hard).all() or (not (soft.sum(1) == len(pad['links'])).all()):
                raise ValueError('live hand material differs from requested pad profile')
            report['compliant_shapes_per_hand'] = len(pad['links'])
        elif not mater__np.allclose(values, expected, atol=1e-06, rtol=0):
            raise ValueError('live hand material differs from requested profile')
    return report
from dh116.env.contracts import linea_coupl__ACTIVE as teach_pose_goal__ACTIVE

def layer_pins(stage):
    """Hash all used file layers; ignore only an empty anonymous session layer."""
    pins = {}
    for layer in stage.GetUsedLayers():
        if layer == stage.GetSessionLayer() and layer.anonymous and layer.empty:
            continue
        if layer.anonymous or not layer.realPath:
            raise ValueError('cannot pin anonymous or unresolved source layer')
        path = usd_surfa__Path(layer.realPath).resolve(strict=True)
        pins[str(path)] = usd_surfa__hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(pins.items()))

ANATOMICAL_SCENE_VERSION = 'dh116-anatomical-scene-v1'
ANATOMICAL_REFERENCE_VERSION = 'dh116-anatomical-wrist-reference-v1'
ANATOMICAL_SIDES = ('left', 'right')

def _anatomical_rod_nodes(value):
    nodes = _array(value, (41, 3), 'anatomical material nodes')
    if (not scene_input__np.all(scene_input__np.diff(nodes[:, 1]) > 0)
            or not scene_input__np.allclose(nodes[:, (0, 2)], nodes[20, (0, 2)], atol=2e-06, rtol=0)):
        raise ValueError('a straight rod in material node order along world Y is required')
    return nodes

def _require_anatomical_hand_positions(poses, center):
    if not (poses[0, 1] < center[1] < poses[1, 1]
            and scene_input__np.all(poses[:, 0] > center[0])):
        raise ValueError('original left and right wrists must flank the rod from world +X')

def load_anatomical_reference(grasp, reference_sha256, *, root=ROOT):
    """Return the source-pinned bilateral goal in material node order."""
    if grasp not in ('pinch', 'tripod', 'power'):
        raise ValueError('unknown anatomical grasp reference')
    path = scene_input__Path(root) / 'configs/tasks' / f'{grasp}_reference.json'
    data = scene_input__json.loads(checked_file(path, reference_sha256))
    ref = data.get('anatomical_reference', {})
    if (ref.get('version') != ANATOMICAL_REFERENCE_VERSION or ref.get('native_qualification') is not False
            or set(ref.get('hands', {})) != set(ANATOMICAL_SIDES)):
        raise ValueError('source-pinned anatomical wrist reference required')
    hands = [ref['hands'][side] for side in ANATOMICAL_SIDES]
    nodes = _anatomical_rod_nodes(ref['rod_nodes_w_m'])
    poses = _array([hand['root_pose_wxyz'] for hand in hands], (2, 7), 'anatomical wrist goal poses')
    _require_anatomical_hand_positions(poses, nodes[20])
    if not scene_input__np.allclose(scene_input__np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError('unit wrist goal quaternions required')
    stations = tuple(hand['station_node'] for hand in hands)
    if any(type(i) is not int or not 0 <= i < len(nodes) for i in stations) or not stations[0] < 20 < stations[1]:
        raise ValueError('left and right goals require material stations on their original anatomical sides')
    targets = tuple(tuple(hand['target_active_q_rad']) for hand in hands)
    for row in targets:
        expand(row)
    return dict(root_pose_wxyz=poses, station_nodes=stations, rod_nodes_w_m=nodes,
                target_active_q_rad=targets)

def anatomical_hand_axes(asset, pose):
    """Measure longitudinal and thumb-side axes from original USD joint origins."""
    from pxr import Usd, UsdPhysics
    from dh116.physics.geometry import pose_matrix

    stage = Usd.Stage.Open(str(asset))
    root = stage.GetDefaultPrim().GetPath()
    base = root.AppendChild('base_link')
    points = []
    for name in ('finger11', 'finger21', 'finger31', 'finger41', 'finger51'):
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root.AppendPath('joints/' + name)))
        if (not joint or joint.GetBody0Rel().GetTargets() != [base]
                or tuple(joint.GetLocalPos1Attr().Get()) != (0.0, 0.0, 0.0)):
            raise ValueError('original thumb and finger base joint origins required')
        points.append(joint.GetLocalPos0Attr().Get())
    points = _array(points, (5, 3), 'original USD hand joint origins')
    rotation = pose_matrix(pose)[:3, :3]
    knuckles = points[1:].mean(0)
    longitudinal = knuckles / scene_input__np.linalg.norm(knuckles)
    thumb_side = points[0] - knuckles
    thumb_side -= (thumb_side @ longitudinal) * longitudinal
    thumb_side /= scene_input__np.linalg.norm(thumb_side)
    return rotation @ longitudinal, rotation @ thumb_side

def anatomical_scene_inputs(source, spec, *, root=ROOT):
    """Apply anatomical station placement to a verified physical scene archive."""
    fields = {'schema', 'grasp', 'source_scene', 'source_station_nodes', 'reference_sha256'}
    if (set(spec) != fields or spec['schema'] != ANATOMICAL_SCENE_VERSION
            or spec['grasp'] != source['grasp'] or source.get('anatomical_placement') is not None
            or spec['source_scene']['sha256'] != source['manifest_sha256']):
        raise ValueError('explicit original scene and matching anatomical placement required')
    old_indices = spec['source_station_nodes']
    if (type(old_indices) is not list or len(old_indices) != 2
            or any(type(i) is not int or not 0 <= i <= 40 for i in old_indices)
            or not old_indices[1] < 20 < old_indices[0]):
        raise ValueError('two original material station indices required')
    archive = dict(source['archive'])
    pair, task = archive['pair'], archive['task']
    if task is None:
        original = _seed(pair['reset_seed'])
        nodes = _anatomical_rod_nodes(pair['rod_seed']['proposed_nodes_m'])
    else:
        original = _seed(source['authored_seed'])
        nodes = _anatomical_rod_nodes(source['authored_rod_nodes'])
    stations = tuple(40 - i for i in old_indices)
    reference = load_anatomical_reference(source['grasp'], spec['reference_sha256'], root=root)
    reference_columns = (0, 1) if task is None else (0, 1, 2)
    if (reference['station_nodes'] != stations or not scene_input__np.allclose(
            reference['rod_nodes_w_m'][:, reference_columns], nodes[:, reference_columns], atol=1e-7, rtol=0)):
        raise ValueError('goal and Reset must share the same mirrored material stations')
    shifts = nodes[list(stations), 1] - nodes[old_indices, 1]
    roots = _array(original['hand_roots'], (2, 7), 'original wrist poses')
    roots[:, 1] += shifts
    original_seed_sha = original['sha256']
    provenance = dict(source_scene=spec['source_scene'], original_seed_sha256=original_seed_sha,
                      original_reset_receipt_sha256=archive['reset_receipt_sha256'],
                      source_station_nodes=old_indices, station_nodes=list(stations),
                      hand_shifts_y_m=shifts.tolist(), reference_sha256=spec['reference_sha256'])
    seed = prepare_reset_state(hand_roots=dict(zip(ANATOMICAL_SIDES, roots)), origin=original['origin'],
                               root_xy=original['root_xy'], initial_offset_m=original['initial_offset_m'],
                               joint_positions=original['joint_positions'], joint_velocities=original['joint_velocities'],
                               source_description=_digest(provenance))
    wrists = _array(seed['reset_root_pose_w'], (2, 7), 'anatomical Reset wrist poses')
    _require_anatomical_hand_positions(wrists, nodes[20])
    layout = dict(source['author_layout'], wire_center=nodes[20].copy(), hand_roots={}, active_joints={})
    active_ids = [scene_reset__JOINTS.index(name) for name in scene_reset__ACTIVE]
    for i, side in enumerate(ANATOMICAL_SIDES):
        layout['hand_roots'][side] = scene_input__np.asarray(seed['authored_root_seed_w'][i]).copy()
        layout['hand_roots'][side][2] -= BUILDER_OFFSET_M
        layout['active_joints'][side] = scene_input__np.asarray(seed['joint_positions'][i])[active_ids]

    # Ropesim constructor poses are translated from the archived readback.
    # The native owner writes its own freshly measured mesh poses before stepping.
    archived_reset = source['reset']
    measured_nodes = archived_reset['rod_reset']['node_positions']
    if not scene_input__np.array_equal(scene_input__np.asarray(measured_nodes)[0], nodes.astype(measured_nodes.dtype)):
        raise ValueError('archived solver nodes differ from original authored material nodes')
    pad_poses = {}
    for name, (position, quaternion) in archived_reset['pad_poses_xyzw'].items():
        side = name.split('.', 1)[0]
        shifted = scene_input__np.array(position, copy=True)
        shifted[1] += shifts[ANATOMICAL_SIDES.index(side)]
        pad_poses[name] = (shifted, scene_input__np.array(quaternion, copy=True))
    reset = dict(rod_reset=dict(node_positions=measured_nodes.copy()), pad_poses_xyzw=pad_poses,
                 seed=seed, kind='authored_solver_constructor', native_qualification=False)

    if task is None:
        archive['pair'] = dict(version=ANATOMICAL_SCENE_VERSION, reset_seed=seed, rod_seed=pair['rod_seed'],
                               native_qualified=False)
    else:
        hands = {side: dict(task['hands'][side], pose_wxyz=seed['reset_root_pose_w'][i])
                 for i, side in enumerate(ANATOMICAL_SIDES)}
        archive['task'] = dict(task, hands=hands)
        if source['grasp'] == 'power':
            archive['task']['near_open_source_sha256'] = spec['reference_sha256']
    archive.update(task_seed_sha256=seed['sha256'], reset_receipt_sha256=None)
    return dict(source, schema=ANATOMICAL_SCENE_VERSION, archive=archive, author_layout=layout,
                authored_seed=seed, reset=reset, anatomical_placement=provenance, receipt_sha256=None)

def load_scene_inputs(path, expected_sha256, *, root=ROOT):
    root = scene_input__Path(root)
    paths = ProjectPaths(root)
    path = scene_input__Path(path)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or root not in path.parents:
        raise ValueError('project-owned scene manifest required')
    raw = checked_file(path, expected_sha256)
    tree = scene_input__json.loads(raw)
    if tree.get('schema') == ANATOMICAL_SCENE_VERSION:
        source_spec = tree['source_scene']
        source_path = paths._relative(source_spec['path'])
        source = load_scene_inputs(source_path, source_spec['sha256'], root=root)
        result = anatomical_scene_inputs(source, tree, root=root)
        result['manifest_sha256'] = expected_sha256
        return result
    if tree.get('schema') != 'dh116-materialized-scene-v1' or tree.get('grasp') not in ('pinch', 'tripod', 'power') or tree.get('qualification') != {'native': False, 'teacher': False, 'grasp': False}:
        raise ValueError('unqualified materialized scene schema required')
    arrays_path = path.parent / 'arrays.npz'
    checked_file(arrays_path, tree['arrays_sha256'])
    for (side, name) in tree['assets'].items():
        asset = paths._relative(name)
        checked_file(asset, tree['prepared']['hands'][side]['usd_sha256'])
    with scene_input__np.load(arrays_path, allow_pickle=False) as store:

        def decode(value):
            if isinstance(value, list):
                return [decode(v) for v in value]
            if not isinstance(value, dict):
                return value
            if '__array__' in value:
                if set(value) != {'__array__', 'dtype', 'shape'}:
                    raise ValueError('exact typed array reference required')
                data = store[value['__array__']]
                if data.dtype.hasobject or data.dtype.str != value['dtype'] or list(data.shape) != value['shape']:
                    raise ValueError('scene array dtype or shape differs')
                return scene_input__np.array(data, copy=True)
            if set(value) == {'__float__'}:
                if value['__float__'] not in ('nan', 'inf', '-inf'):
                    raise ValueError('known nonfinite provenance marker required')
                return float(value['__float__'])
            return {k: decode(v) for (k, v) in value.items()}
        result = decode(tree)
    result['manifest_sha256'] = expected_sha256
    return result
from dh116.env.contracts import _SOURCE_UPPER
PROXIMAL_LINK = 'finger52_link'
load_param__LEGACY_LOAD_PARAMETERS = LoadParameters()

def verify_hand_link_material(properties, profile, link, expected_envs):
    spec = hand_material_contract(profile)['hand_contact_material']
    pad = spec.get('pad_compliance', {})
    encoded = -pad['stiffness_n_m'] if link in pad.get('links', []) else spec['restitution']
    expected = [spec['static_friction'], spec['dynamic_friction'], encoded]
    values = mater__np.asarray(properties, dtype=float)
    if values.shape != (expected_envs, 1, 3) or not mater__np.isfinite(values).all() or (not mater__np.allclose(values, expected, atol=1e-06, rtol=0)):
        raise ValueError(f'live hand link material differs: {link}')
    return dict(link=link, environments=expected_envs, coefficients=expected)
from dh116.env.contracts import active_limits as teach_pose_goal__active_limits

def read_surface(stage, body_path, branch='collisions'):
    """One mesh under body's explicit visuals/collisions branch, in body metres."""
    from pxr import Usd, UsdGeom, UsdPhysics
    if branch not in ('visuals', 'collisions'):
        raise ValueError('branch must be visuals or collisions')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError('expected metre USD stage')
    body = stage.GetPrimAtPath(body_path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('body must exist and be a rigid body')
    prefix = str(body.GetPath()) + '/' + branch + '/'
    meshes = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith(prefix)]
    if len(meshes) != 1:
        raise ValueError(f'expected one {branch} mesh, found {len(meshes)}')
    prim = meshes[0]
    mesh = UsdGeom.Mesh(prim)
    points = usd_surfa__np.asarray(mesh.GetPointsAttr().Get(), dtype=usd_surfa__np.float64)
    counts = usd_surfa__np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = usd_surfa__np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or (not usd_surfa__np.isfinite(points).all()) or (counts.ndim != 1) or (len(counts) == 0) or (not usd_surfa__np.all(counts == 3)) or (indices.shape != (3 * len(counts),)) or usd_surfa__np.any(indices < 0) or usd_surfa__np.any(indices >= len(points)):
        raise ValueError('expected finite triangulated geometry with valid indices')
    orientation = str(mesh.GetOrientationAttr().Get())
    if orientation not in ('rightHanded', 'leftHanded'):
        raise ValueError('unknown mesh orientation')
    cache = UsdGeom.XformCache()
    transform = usd_surfa__np.array(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()).T
    rotation = transform[:3, :3]
    if not usd_surfa__np.isfinite(transform).all() or not usd_surfa__np.allclose(rotation.T @ rotation, usd_surfa__np.eye(3), atol=1e-08, rtol=0.0) or abs(usd_surfa__np.linalg.det(rotation) - 1.0) > 1e-08 or (not usd_surfa__np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-08, rtol=0.0)):
        raise ValueError('mesh-to-body transform must be rigid, in metres')
    with usd_surfa__np.errstate(over='ignore', invalid='ignore'):
        points_body = points @ rotation.T + transform[:3, 3]
    if not usd_surfa__np.isfinite(points_body).all():
        raise ValueError('transformed points overflowed')
    (owner, approximation) = (None, None)
    ancestor = prim
    while ancestor and ancestor.GetPath().HasPrefix(body.GetPath()):
        if ancestor.HasAPI(UsdPhysics.CollisionAPI):
            if not UsdPhysics.CollisionAPI(ancestor).GetCollisionEnabledAttr().Get():
                raise ValueError('collision owner is disabled')
            owner = str(ancestor.GetPath())
            if ancestor.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximation = str(UsdPhysics.MeshCollisionAPI(ancestor).GetApproximationAttr().Get())
            break
        ancestor = ancestor.GetParent()
    if branch == 'collisions' and owner is None:
        raise ValueError('collision branch has no collision owner')
    digest = usd_surfa__hashlib.sha256()
    digest.update(b'DH116-triangle-surface-v1\x00')
    digest.update(usd_surfa__np.asarray(points.shape, dtype='<i8').tobytes())
    digest.update(points.astype('<f8').tobytes())
    digest.update(indices.astype('<i8').tobytes())
    digest.update(orientation.encode('ascii'))
    return dict(body_path=str(body.GetPath()), mesh_path=str(prim.GetPath()), points_mesh=points.tolist(), points_body=points_body.tolist(), triangles=indices.reshape(-1, 3).tolist(), mesh_to_body=transform.tolist(), collision_owner=owner, collision_approximation=approximation, orientation=orientation, geometry_sha256=digest.hexdigest())

def prepared_scene(inputs):
    from copy import deepcopy
    result = deepcopy(inputs['prepared'])
    result['runtime_bundle_sha256'] = inputs['manifest_sha256']
    return result
scene_reset__SIDES = ('left', 'right')
PROXIMAL_NAMES = tuple((f'{s}.{PROXIMAL_LINK}' for s in scene_suppo__SIDES))
scene_valid__read_surface = read_surface

def read_load_parameters(path):
    if path is None:
        return None
    payload = load_param__json.loads(load_param__Path(path).read_text())
    if payload.get('version') != 'dh116-simulation-load-v1':
        raise ValueError('unknown simulation load profile')
    return LoadParameters(**payload['parameters'])

def bind_object_colliders(stage, root_path, material):
    """Bind physics purpose on each real collider, leaving visual material alone."""
    from pxr import Usd, UsdPhysics
    from pxr import Usd
    from pxr import UsdPhysics
    from pxr import UsdShade
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid() or not material.GetPrim().IsValid():
        raise ValueError('valid object collider root and material required')
    colliders = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    if not colliders:
        raise ValueError('no enabled object colliders found')
    for prim in colliders:
        target = prim
        while target.IsInstanceProxy():
            target = target.GetParent()
        UsdShade.MaterialBindingAPI.Apply(target).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
    return [str(prim.GetPath()) for prim in colliders]
teach_pose_goal__SIDES = ('left', 'right')

def spawn_parameters(inputs, project):
    from copy import deepcopy
    from types import SimpleNamespace
    parameters = deepcopy(inputs['author_args'])
    parameters['load_parameters'] = SimpleNamespace(**parameters['load_parameters'])
    args = SimpleNamespace(**parameters)
    layout = deepcopy(inputs['author_layout'])
    sources = {side: str(scene_input__Path(project) / name) for (side, name) in inputs['assets'].items()}
    return (args, layout, sources)
SEED_VERSION = 'dh116-authored-reset-state-v1'

def contact_links(values):
    names = set(values)
    if names == set(HAND_NAMES):
        return PADS
    if names == set((*HAND_NAMES, *PALM_NAMES)):
        return (*PADS, 'base_link')
    if names == set((*HAND_NAMES, *PALM_NAMES, *PROXIMAL_NAMES)):
        return (*PADS, 'base_link', PROXIMAL_LINK)
    raise ValueError('both hands must bind the same original pad, palm and optional proximal layout')

def verify_drive_readback(stiffness, damping, effort, parameters=None):
    """Fail before training if the simulator did not apply the requested limits."""
    parameters = parameters or load_param__LEGACY_LOAD_PARAMETERS
    (stiffness, damping, effort) = [load_param__np.asarray(x, dtype=float) for x in (stiffness, damping, effort)]
    if stiffness.ndim != 2 or stiffness.shape[1] != 6 or damping.shape != stiffness.shape or (effort.shape != stiffness.shape) or (not all((load_param__np.isfinite(x).all() for x in (stiffness, damping, effort)))):
        raise ValueError('finite N x 6 active drive readback required')
    if load_param__np.any(stiffness < 10.0 - 1e-05) or load_param__np.any(stiffness > parameters.stiffness_max + 1e-05):
        raise ValueError('simulator stiffness outside selected profile')
    if not load_param__np.allclose(damping, 0.2, atol=1e-06, rtol=0):
        raise ValueError('simulator damping differs from selected profile')
    if not load_param__np.allclose(effort, parameters.effort_limit_nm, atol=1e-06, rtol=0):
        raise ValueError('simulator effort limit differs from selected profile')

def verify_material_properties(properties, profile, expected_shapes):
    """Refuse to train if any cooked collider kept the wrong coefficients."""
    values = mater__np.asarray(properties, dtype=float)
    if values.shape != (1, expected_shapes, 3) or not mater__np.isfinite(values).all():
        raise ValueError('invalid live object material properties')
    settings = material_contract(profile)
    if settings:
        spec = settings['object_contact_material']
        expected = [spec[k] for k in ('static_friction', 'dynamic_friction', 'restitution')]
        if not mater__np.allclose(values, expected, atol=1e-06, rtol=0):
            raise ValueError('live object material differs from requested profile')
    return dict(shapes=expected_shapes, static_friction=values[0, :, 0].tolist(), dynamic_friction=values[0, :, 1].tolist(), restitution=values[0, :, 2].tolist())
ERROR_FIELDS = tuple(('finger_goal_max_error_' + side for side in teach_pose_goal__SIDES))

def load_task_scene(grasp, *, root=ROOT, full_grasp=False):
    from types import SimpleNamespace
    root = scene_input__Path(root)
    task = scene_input__json.loads((root / 'configs/tasks' / (grasp + '.json')).read_text())
    spec = task.get('full_scene_input', task['scene_input']) if full_grasp else task['scene_input']
    inputs = load_scene_inputs(root / spec['path'], spec['sha256'], root=root)
    if inputs['grasp'] != grasp:
        raise ValueError('independent grasp scene identity differs')
    archive = SimpleNamespace(**inputs['archive'])
    archive.scene_inputs = inputs
    return (archive, inputs['reset'])

def _array(value, shape, label):
    array = scene_reset__np.array(value, dtype=scene_reset__np.float64, copy=True)
    if array.shape != shape or not scene_reset__np.isfinite(array).all():
        raise ValueError(f'finite {label} {shape} required')
    return array

def scene_suppo__author_end_supports(stage, prefix, scene):
    """Author static USD cubes under the source environment BEFORE cloning.

    No RigidBodyAPI (hence static), joints, materials or object edits. The
    existing table, gravity and rod mass/material configuration stay intact.
    """
    from pxr import UsdGeom, UsdPhysics
    if scene is None:
        return ()
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    parent = prefix + '/EndSupports'
    UsdGeom.Xform.Define(stage, parent)
    paths = []
    for support in scene['supports']:
        path = parent + '/' + support['name']
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
        cube.AddTranslateOp().Set(Gf.Vec3d(*support['center_m']))
        cube.AddScaleOp().Set(Gf.Vec3d(*support['size_m']))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        prim = cube.GetPrim()
        prim.AddAppliedSchema('PhysxCollisionAPI')
        prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(0.001)
        prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.0)
        paths.append(path)
    return tuple(paths)
scene_valid__hand_material_contract = hand_material_contract
scene_valid__material_config_fields = material_config_fields
scene_valid__bind_hand_materials = bind_hand_materials
scene_valid__verify_collider_bindings = verify_collider_bindings
scene_valid__bind_object_colliders = bind_object_colliders

def _wrist_joint_ids(names):
    """Accept no support, a complete XYZ support, or complete XYZ + roll/pitch/yaw."""
    from dh116.hand.kinematics import JOINT_NAMES
    from dh116.hand.kinematics import ANGULAR_JOINT_NAMES
    linear = [names.index(n) for n in JOINT_NAMES if n in names]
    angular = [names.index(n) for n in ANGULAR_JOINT_NAMES if n in names]
    if len(linear) not in (0, 3) or len(angular) not in (0, 3) or (angular and (not linear)) or (len(names) != len(set(names))):
        raise ValueError('complete wrist carriage and optional complete angular joint set required')
    return (linear, angular)
GOAL_SCHEMA = 'dh116-teacher-finger-goal-v1'

def _digest(value):
    return scene_reset__hashlib.sha256(scene_reset__json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _bind_support_material(stage, env_prefix, paths, mu_s, mu_k):
    """Independent per-environment candidate material; never edit table/hand."""
    from pxr import UsdPhysics
    from pxr import Sdf
    from pxr import UsdPhysics
    from pxr import UsdShade
    material_path = env_prefix + '/EndSupports/PhysicsMaterial'
    material = UsdShade.Material.Define(stage, material_path)
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(float(mu_s))
    physical.CreateDynamicFrictionAttr(float(mu_k))
    physical.CreateRestitutionAttr(0.0)
    prim = material.GetPrim()
    prim.AddAppliedSchema('PhysxMaterialAPI')
    prim.CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('average')
    prim.CreateAttribute('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token).Set('average')
    prim.CreateAttribute('dh116:calibrationStatus', Sdf.ValueTypeNames.Token).Set('candidate_not_measured')
    for path in paths:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
    return dict(path=material_path, mu_s=float(mu_s), mu_k=float(mu_k), restitution=0.0, friction_combine_mode='average', restitution_combine_mode='average', calibration_status='candidate_not_measured')
scene_valid__SIDES = ('left', 'right')

from dh116.hand.kinematics import PASSIVE_VELOCITY_MIN_RAD_S, passive_velocity_limit

def _verify_velocity_readback(velocity, names, instances, passive_limit):
    from dh116.env.contracts import linea_coupl__ACTIVE as ACTIVE
    from dh116.env.contracts import JOINTS
    from dh116.hand.kinematics import veloc_limit__ACTIVE_VELOCITY_LIMIT_RAD_S as ACTIVE_VELOCITY_LIMIT_RAD_S
    from dh116.hand.kinematics import JOINT_NAMES
    from dh116.hand.kinematics import palm_mount__CONTRACT as WRIST
    from dh116.hand.kinematics import ANGULAR_JOINT_NAMES
    from dh116.hand.kinematics import ROTATION_CONTRACT
    _wrist_joint_ids(names)
    valid = set(names) in (set(JOINTS), set(JOINTS) | set(JOINT_NAMES), set(JOINTS) | set(JOINT_NAMES) | set(ANGULAR_JOINT_NAMES))
    if len(names) != len(set(names)) or not valid:
        raise ValueError('velocity readback requires eleven DH116 joints and optional complete wrist carriage')
    if instances < 1 or velocity.shape != (instances, len(names)) or not load_param__np.isfinite(velocity).all():
        raise ValueError('finite all-instance velocity limit readback in named joint order required')
    expected_by_joint = {
        **dict.fromkeys(JOINTS, passive_velocity_limit(passive_limit)),
        **dict.fromkeys(ACTIVE, ACTIVE_VELOCITY_LIMIT_RAD_S),
        **dict.fromkeys(JOINT_NAMES, WRIST['velocity_limit_m_s']),
        **dict.fromkeys(ANGULAR_JOINT_NAMES, ROTATION_CONTRACT['velocity_limit_rad_s']),
    }
    expected = load_param__np.asarray([expected_by_joint[name] for name in names])
    mismatch = ~load_param__np.isclose(velocity, expected[None, :], atol=1e-06, rtol=0)
    if mismatch.any():
        row, joint = load_param__np.argwhere(mismatch)[0]
        raise ValueError(f'simulator velocity limits differ at row {row}, {names[joint]}: '
                         f'{velocity[row, joint]} read, {expected[joint]} configured')

def velocity_limits_readback(hands, *, instances, passive_limit=PASSIVE_VELOCITY_MIN_RAD_S):
    """Verify configured speed ceilings against every native articulation row."""
    samples = []
    for hand in hands:
        names = tuple(hand.joint_names)
        velocity = hand.root_physx_view.get_dof_max_velocities().detach().cpu().numpy()
        _verify_velocity_readback(velocity, names, instances, passive_limit)
        samples.append(dict(joint_names=list(names), max_velocity=velocity[0].tolist()))
    if not samples:
        raise ValueError('at least one native hand is required for velocity readback')
    return dict(source='root_physx_view.get_dof_max_velocities', instances_per_hand=instances,
                units='m/s for wrist translations and rad/s for all revolute joints',
                passive_limit_rad_s=float(passive_limit), all_instance_velocity_limits_verified=True,
                hands=samples)

@teach_pose_goal__dataclass(frozen=True)
class FingerGoal:
    grasp: str
    reference_sha256: str
    target_rad: tuple
    participating: tuple
    tolerance_rad: float = 0.05
    scale_rad: float = 0.25
    arrival_steps: int = 16
    arrival_reward: float = 2.0

    def max_errors_rad(self, measured):
        q = teach_pose_goal__np.asarray(measured, dtype=float)
        if q.shape != (2, len(teach_pose_goal__ACTIVE)) or not teach_pose_goal__np.isfinite(q).all():
            raise ValueError('finite two-hand active radians required')
        error = teach_pose_goal__np.abs(q - self.target_rad)
        return tuple((float(value) for value in error[:, self.participating].max(axis=1)))

    def potential(self, errors):
        return teach_pose_goal__math.fsum((teach_pose_goal__math.exp(-max(0.0, error - self.tolerance_rad) / self.scale_rad) for error in errors)) / len(teach_pose_goal__SIDES)

def prepare_reset_state(*, hand_roots, origin, root_xy, initial_offset_m, joint_positions, joint_velocities, source_description):
    """Apply TensorWorld's authored seed arithmetic; all physical inputs explicit.

    root_seed excludes initial_offset. reset_root_pose includes it. The caller's
    provenance label is retained verbatim but is not verified source approval.
    Preserve all eleven supplied q/qd; do not infer follower measurements.
    """
    if set(hand_roots) != set(scene_reset__SIDES) or not isinstance(source_description, str) or (not source_description):
        raise ValueError('both authored hand roots and explicit source description required')
    roots = _array([hand_roots[s] for s in scene_reset__SIDES], (2, 7), 'authored wxyz roots')
    if not scene_reset__np.allclose(scene_reset__np.linalg.norm(roots[:, 3:], axis=1), 1.0, rtol=0, atol=1e-06):
        raise ValueError('normalized authored root quaternions required')
    origin = _array(origin, (3,), 'environment origin')
    xy = _array(root_xy, (2, 2), 'authored root_xy')
    if type(initial_offset_m) not in (float, int) or not scene_reset__math.isfinite(initial_offset_m):
        raise ValueError('explicit finite initial root offset required')
    q = _array(joint_positions, (2, 11), 'canonical joint positions')
    qd = _array(joint_velocities, (2, 11), 'canonical joint velocities')
    (lo, hi) = scene_reset__active_limits()
    active = [scene_reset__JOINTS.index(n) for n in scene_reset__ACTIVE]
    if scene_reset__np.any(q[:, active] < lo) or scene_reset__np.any(q[:, active] > hi):
        raise ValueError('authored active joints exceed original coupled limits')
    if scene_reset__np.any(q < scene_reset__np.asarray(_SOURCE_LOWER)) or scene_reset__np.any(q > scene_reset__np.asarray(_SOURCE_UPPER)):
        raise ValueError('authored full joints exceed original source limits')
    root_seed = roots.copy()
    root_seed[:, :3] += origin
    root_seed[:, :2] += xy
    reset_pose = root_seed.copy()
    reset_pose[:, 2] += float(initial_offset_m)
    payload = dict(version=SEED_VERSION, sides=list(scene_reset__SIDES), joint_names=list(scene_reset__JOINTS), hand_roots=roots.tolist(), origin=origin.tolist(), root_xy=xy.tolist(), initial_offset_m=float(initial_offset_m), joint_positions=q.tolist(), joint_velocities=qd.tolist(), authored_root_seed_w=root_seed.tolist(), reset_root_pose_w=reset_pose.tolist(), source_description=source_description, source_qualification=False, initial_pose_qualified=False)
    return dict(payload, sha256=_digest(payload))
BUILDER_OFFSET_M = 0.04

def physics_load_readback(hands, parameters=None, *, passive_velocity_limit_rad_s=None, force_command_channels=None):
    """Actual PhysX views; compact samples plus all-instance drive validation."""
    from dh116.env.contracts import linea_coupl__ACTIVE as ACTIVE
    from dh116.env.contracts import JOINTS
    from dh116.hand.kinematics import JOINT_NAMES
    from dh116.hand.kinematics import palm_mount__CONTRACT as WRIST
    from dh116.hand.kinematics import ROTATION_CONTRACT
    result = []
    for hand in hands:
        (wrist_ids, angular_ids) = _wrist_joint_ids(hand.joint_names)
        view = hand.root_physx_view
        read = lambda method: getattr(view, method)().detach().cpu().numpy()
        (stiffness, damping, effort) = [read(method) for method in ('get_dof_stiffnesses', 'get_dof_dampings', 'get_dof_max_forces')]
        velocity = read('get_dof_max_velocities')
        limits = read('get_dof_limits')
        if passive_velocity_limit_rad_s is not None:
            _verify_velocity_readback(velocity, hand.joint_names, stiffness.shape[0], passive_velocity_limit_rad_s)
        ids = [hand.joint_names.index(n) for n in ACTIVE]
        verify_drive_readback(stiffness[:, ids], damping[:, ids], effort[:, ids], parameters)
        for (support_ids, contract, keys) in ((wrist_ids, WRIST, ('stiffness_n_m', 'damping_n_s_m', 'effort_limit_n', 'velocity_limit_m_s')), (angular_ids, ROTATION_CONTRACT, ('stiffness_nm_rad', 'damping_nms_rad', 'effort_limit_nm', 'velocity_limit_rad_s'))):
            if not support_ids:
                continue
            for (values, key) in zip((stiffness, damping, effort, velocity), keys):
                if values.shape != (stiffness.shape[0], len(hand.joint_names)) or not load_param__np.allclose(values[:, support_ids], contract[key], rtol=0, atol=1e-05):
                    raise ValueError('wrist carriage drive/velocity readback mismatch: ' + key)
        if angular_ids:
            angle_limit = ROTATION_CONTRACT['angle_limit_rad']
            if limits.shape != (stiffness.shape[0], len(hand.joint_names), 2) or not load_param__np.allclose(limits[:, angular_ids], [-angle_limit, angle_limit], rtol=0, atol=1e-06):
                raise ValueError('angular wrist limits must read back in radians')
        (mass, inertia, com) = [read(method) for method in ('get_masses', 'get_inertias', 'get_coms')]
        result.append(dict(instances=int(stiffness.shape[0]), joint_names=hand.joint_names, body_names=hand.body_names, stiffness_nm_per_rad=stiffness[:1].tolist(), damping_nms_per_rad=damping[:1].tolist(), effort_limit_nm=effort[:1].tolist(), joint_limits_rad=limits[:1].tolist(), velocity_limits_rad_s=velocity[:1].tolist(), sample_body_masses_kg=mass[:1].tolist(), sample_body_inertias_kg_m2=inertia[:1].tolist(), sample_body_com_poses=com[:1].tolist(), total_mass_kg_range=[float(mass.sum(axis=1).min()), float(mass.sum(axis=1).max())], all_instance_drive_limits_verified=True))
        if wrist_ids:
            result[-1]['wrist_carriage'] = WRIST
            result[-1]['joint_units'] = {n: 'linear_m_N' if n in JOINT_NAMES else 'angular_rad_Nm' for n in hand.joint_names}
            result[-1]['all_instance_wrist_velocity_limits_verified'] = True
        if angular_ids:
            result[-1]['wrist_rotation'] = ROTATION_CONTRACT
            result[-1]['all_instance_wrist_rotation_limits_verified'] = True
        if passive_velocity_limit_rad_s is not None:
            result[-1]['all_instance_velocity_limits_verified'] = True
    report = dict(hands=result, hardware_calibration_complete=False, inertial_model='unchanged manufacturer CAD; finger32 distribution anomaly unresolved', coupling_model='source URDF affine approximation; not nonlinear MJCF linkage', torque_diagnostic='implicit PD estimate, not measured solver drive torque', contact_diagnostic='object-filtered normal forces on pads, not total contact wrench')
    if any(('wrist_rotation' in hand for hand in result)):
        external = sum((len(_wrist_joint_ids(hand['joint_names'])[0]) + len(_wrist_joint_ids(hand['joint_names'])[1]) for hand in result))
        channels = len(result) if force_command_channels is None else int(force_command_channels)
        if channels not in (0, len(result)):
            raise ValueError('unexpected force command channel count')
        report['dof_accounting'] = dict(finger_active=len(ACTIVE) * len(result), finger_passive=(len(JOINTS) - len(ACTIVE)) * len(result), external_wrist_drives=external, policy_command_size=len(ACTIVE) * len(result) + external + channels, force_command_channels=channels)
    return report

def load_finger_goal(spec):
    if type(spec) is not dict or set(spec) != {'schema', 'grasp', 'reference_sha256'} or spec['schema'] != GOAL_SCHEMA or (spec['grasp'] not in ('pinch', 'tripod', 'power')):
        raise ValueError('versioned native finger goal required')
    grasp = spec['grasp']
    goal = load_anatomical_reference(grasp, spec['reference_sha256'])
    targets = goal['target_active_q_rad']
    actual_sha = spec['reference_sha256']
    counts = {'pinch': 3, 'tripod': 4, 'power': 6}
    return FingerGoal(grasp, actual_sha, targets, tuple(range(counts[grasp])))
scene_autho__LEGACY_LOAD_PARAMETERS = load_param__LEGACY_LOAD_PARAMETERS

def _seed(value):
    packet = scene_reset__deepcopy(value)
    expected = packet.pop('sha256', None)
    if expected != _digest(packet):
        raise ValueError('reset seed content fingerprint differs')
    rebuilt = prepare_reset_state(hand_roots=dict(zip(scene_reset__SIDES, packet['hand_roots'])), origin=packet['origin'], root_xy=packet['root_xy'], initial_offset_m=packet['initial_offset_m'], joint_positions=packet['joint_positions'], joint_velocities=packet['joint_velocities'], source_description=packet['source_description'])
    if rebuilt != value:
        raise ValueError('authored reset arithmetic/schema differs')
    return rebuilt

def _material(stage, path, fields):
    """USD-only material authoring; the original binder verifies every resolved field."""
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import UsdPhysics
    from pxr import Sdf
    from pxr import PhysxSchema
    fields = {'compliant_contact_stiffness': 0.0, 'compliant_contact_damping': 0.0, **fields}
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(fields['static_friction']))
    api.CreateDynamicFrictionAttr(float(fields['dynamic_friction']))
    api.CreateRestitutionAttr(float(fields['restitution']))
    names = dict(friction_combine_mode='frictionCombineMode', restitution_combine_mode='restitutionCombineMode', compliant_contact_stiffness='compliantContactStiffness', compliant_contact_damping='compliantContactDamping')
    for (key, name) in names.items():
        if key in fields:
            value = fields[key]
            material.GetPrim().CreateAttribute('physxMaterial:' + name, Sdf.ValueTypeNames.Token if isinstance(value, str) else Sdf.ValueTypeNames.Float).Set(value)
    return material

def measured_active_angles(hands):
    """Extract active joint positions from single-environment SDK measurements."""
    angles = []
    for side in teach_pose_goal__SIDES:
        sample = hands[side]
        names = tuple(sample['joint_names'])
        q = teach_pose_goal__np.asarray(sample['q'])
        if q.shape != (1, len(names)) or len(set(names)) != len(names) or (not set(teach_pose_goal__ACTIVE).issubset(names)) or (not teach_pose_goal__np.isfinite(q).all()):
            raise ValueError('named finite SDK joint readback required')
        angles.append(tuple((float(q[0, names.index(name)]) for name in teach_pose_goal__ACTIVE)))
    return tuple(angles)
from dh116.hand.kinematics import veloc_limit__ACTIVE_VELOCITY_LIMIT_RAD_S as ACTIVE_VELOCITY_LIMIT_RAD_S
import re

def _bind_materials(stage, prepared):
    prefix = prepared['prefix']
    spec = scene_valid__hand_material_contract(prepared['candidate_physics']['hand']['material_profile'])['hand_contact_material']
    hard = _material(stage, prefix + '/HandContactMaterial', scene_valid__material_config_fields(spec))
    soft = _material(stage, prefix + '/PadContactMaterial', scene_valid__material_config_fields(spec, compliant=True))
    reports = {}
    for side in scene_valid__SIDES:
        (_, reports[side]) = scene_valid__bind_hand_materials(stage, prefix + '/' + side + 'Hand', hard, spec, soft)
    table = prepared['candidate_physics']['table']
    fields = dict(static_friction=table['mu_s'], dynamic_friction=table['mu_k'], restitution=0.0, friction_combine_mode='average', restitution_combine_mode='average')
    material = _material(stage, prefix + '/TableContactMaterial', fields)
    paths = scene_valid__bind_object_colliders(stage, prefix + '/Table', material)
    scene_valid__verify_collider_bindings(stage, paths, fields)
    return reports
SHA = re.compile('[0-9a-f]{64}')

def inspect_source_stage(stage, prepared):
    """Check actual referenced original geometry after authoring, including proxies."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    count = 0
    for side in scene_valid__SIDES:
        root = prepared['prefix'] + '/' + side + 'Hand'
        for (name, row) in prepared['hands'][side]['source_colliders'].items():
            actual = scene_valid__read_surface(stage, root + '/' + name)
            if actual['geometry_sha256'] != row['geometry_sha256'] or not scene_valid__np.allclose(actual['mesh_to_body'], row['mesh_to_body'], rtol=0, atol=1e-08):
                raise ValueError('assembled source collision geometry changed')
            count += 1
        if stage.GetPrimAtPath(root + '/root_joint').IsActive() or not stage.GetPrimAtPath(root + '/wrist_fixed').HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError('original driven mount topology differs')
        for name in JOINT_NAMES + ANGULAR_JOINT_NAMES:
            if not stage.GetPrimAtPath(root + '/' + name):
                raise ValueError('complete original six-axis wrist required')
    table_root = stage.GetPrimAtPath(prepared['prefix'] + '/Table')
    if not table_root:
        raise ValueError('actual finite table required')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(table_root).ComputeAlignedRange()
    (lower, upper) = (scene_valid__np.asarray(bounds.GetMin()), scene_valid__np.asarray(bounds.GetMax()))
    if not scene_valid__np.allclose((upper + lower) / 2, prepared['table']['center_m'], rtol=0, atol=1e-06) or not scene_valid__np.allclose(upper - lower, prepared['table']['size_m'], rtol=0, atol=1e-06):
        raise ValueError('actual finite table bounds differ')
    table_paths = [str(p.GetPath()) for p in Usd.PrimRange(table_root) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    material = prepared['candidate_physics']['table']
    scene_valid__verify_collider_bindings(stage, table_paths, dict(static_friction=material['mu_s'], dynamic_friction=material['mu_k'], restitution=0.0, friction_combine_mode='average', restitution_combine_mode='average'))
    if count != 34:
        raise ValueError('both original seventeen-collider hands required')
    spec = scene_valid__hand_material_contract(prepared['candidate_physics']['hand']['material_profile'])['hand_contact_material']
    for side in scene_valid__SIDES:
        for name in prepared['hands'][side]['source_colliders']:
            surface = scene_valid__read_surface(stage, prepared['prefix'] + '/' + side + 'Hand/' + name)
            fields = dict(spec, compliant_contact_stiffness=10000.0 if name in scene_valid__DISTAL_PAD_LINKS else 0.0, compliant_contact_damping=30.0 if name in scene_valid__DISTAL_PAD_LINKS else 0.0)
            scene_valid__verify_collider_bindings(stage, [surface['collision_owner']], fields)
    return dict(original_colliders=count, compliant_pad_colliders=10, wrist_joints=12, usd_only=True, sdk_readback_performed=False, initial_joint_state_applied=False, native_qualified=False)

def author_scene(sim, args, original_layout, sources, prefix, origin):
    from pxr import UsdGeom
    if args.control_mode != 'adaptive_grasp' or args.palm_transport != 'driven_carriage' or (not args.free_wrist_grasp) or (args.load_parameters is None):
        raise ValueError('explicit current mounted-hand authoring parameters required')
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from pxr import UsdGeom

    def _author_cfg(cfg):
        cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=cfg.init_state.pos, orientation=cfg.init_state.rot)
        return cfg
    layout = copy.deepcopy(original_layout)
    origin = scene_autho__np.asarray(origin, dtype=float)
    for pose in layout['hand_roots'].values():
        pose[:3] += origin
    for key in ('wire_center', 'table_center'):
        layout[key] = scene_autho__np.asarray(layout[key]) + origin
    UsdGeom.Xform.Define(sim.stage, prefix)
    table = sim_utils.CuboidCfg(size=layout['table_size'], collision_props=sim_utils.CollisionPropertiesCfg())
    table.func(prefix + '/Table', table, translation=layout['table_center'])
    from dh116.env.scene import hand_material_contract as hand_material_contract
    from dh116.env.scene import bind_object_colliders as bind_object_colliders
    from dh116.env.scene import verify_collider_bindings as verify_collider_bindings
    from dh116.env.scene import material_config_fields as material_config_fields
    from dh116.env.scene import bind_hand_materials as bind_hand_materials
    hand_profile = hand_material_contract(getattr(args, 'hand_material', 'default'))
    (hand_colliders, hand_bindings) = ({}, {})
    hand_material = pad_material = None
    if hand_profile:
        from pxr import UsdShade
        settings = hand_profile['hand_contact_material']
        cfg = sim_utils.RigidBodyMaterialCfg(**material_config_fields(settings))
        cfg.func(prefix + '/HandContactMaterial', cfg)
        hand_material = UsdShade.Material(sim.stage.GetPrimAtPath(prefix + '/HandContactMaterial'))
        if 'pad_compliance' in settings:
            cfg = sim_utils.RigidBodyMaterialCfg(**material_config_fields(settings, compliant=True))
            cfg.func(prefix + '/PadContactMaterial', cfg)
            pad_material = UsdShade.Material(sim.stage.GetPrimAtPath(prefix + '/PadContactMaterial'))
    (hands, sensors) = ([], [])
    for side in ('left', 'right'):
        pose = layout['hand_roots'][side].copy()
        pose[2] += 0.04
        cfg = ArticulationCfg(prim_path=f'{prefix}/{side}Hand', spawn=sim_utils.UsdFileCfg(usd_path=sources[side], activate_contact_sensors=True, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True), articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False, solver_position_iteration_count=args.position_iterations, solver_velocity_iteration_count=getattr(args, 'velocity_iterations', 4))), init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]), joint_pos=dict(zip(scene_autho__JOINTS, map(float, expand(layout['active_joints'][side]))))), actuators={'active': ImplicitActuatorCfg(joint_names_expr=list(scene_autho__ACTIVE), stiffness=10.0, damping=0.2, effort_limit_sim=(getattr(args, 'load_parameters', None) or scene_autho__LEGACY_LOAD_PARAMETERS).effort_limit_nm, velocity_limit_sim=ACTIVE_VELOCITY_LIMIT_RAD_S), 'passive': ImplicitActuatorCfg(joint_names_expr=[n for n in scene_autho__JOINTS if n not in scene_autho__ACTIVE], stiffness=0.0, damping=0.0, effort_limit_sim=0.0, velocity_limit_sim=passive_velocity_limit(getattr(args, 'passive_velocity_limit_rad_s', PASSIVE_VELOCITY_MIN_RAD_S)))})
        if getattr(args, 'palm_transport', 'fixed_pose') == 'driven_carriage':
            from dh116.hand.kinematics import JOINT_NAMES
            from dh116.hand.kinematics import palm_mount__CONTRACT as WRIST
            from dh116.hand.kinematics import ANGULAR_JOINT_NAMES
            from dh116.hand.kinematics import ROTATION_CONTRACT
            cfg.actuators['wrist'] = ImplicitActuatorCfg(joint_names_expr=list(JOINT_NAMES), stiffness=WRIST['stiffness_n_m'], damping=WRIST['damping_n_s_m'], effort_limit_sim=WRIST['effort_limit_n'], velocity_limit_sim=WRIST['velocity_limit_m_s'])
            if getattr(args, 'free_wrist_grasp', False):
                cfg.actuators['wrist_rotation'] = ImplicitActuatorCfg(joint_names_expr=list(ANGULAR_JOINT_NAMES), stiffness=ROTATION_CONTRACT['stiffness_nm_rad'], damping=ROTATION_CONTRACT['damping_nms_rad'], effort_limit_sim=ROTATION_CONTRACT['effort_limit_nm'], velocity_limit_sim=ROTATION_CONTRACT['velocity_limit_rad_s'])
        hands.append(_author_cfg(cfg))
        if getattr(args, 'palm_transport', 'fixed_pose') == 'driven_carriage':
            from dh116.hand.kinematics import author_mount
            author_mount(sim.stage, f'{prefix}/{side}Hand', rotation=bool(getattr(args, 'free_wrist_grasp', False)))
        if hand_material is not None:
            (paths, binding_report) = bind_hand_materials(sim.stage, f'{prefix}/{side}Hand', hand_material, hand_profile['hand_contact_material'], pad_material)
            hand_colliders[side] = paths
            hand_bindings[side] = binding_report
        install_hand_mimic(sim.stage, f'{prefix}/{side}Hand')
        configure_mimic_compliance(sim.stage, f'{prefix}/{side}Hand', float(args.mimic_frequency), 1.0, allow_rigid=getattr(args, 'load_parameters', None) is not None)
    return dict(hands=hands, sensors=[], wire=None, layout=layout, prefix=prefix, origin=origin, object_material_colliders=[], hand_material_colliders=hand_colliders, hand_material_bindings=hand_bindings)

def derive_height_seed(seed, *, grasp, delta_z_m, evidence_sha256):
    original = _seed(seed)
    delta = scene_reset__np.asarray(delta_z_m, dtype=float)
    if grasp not in ('pinch', 'tripod', 'power') or delta.shape != (2,) or (not scene_reset__np.isfinite(delta).all()) or scene_reset__np.any(delta < 0) or scene_reset__np.any(delta > 0.02) or (not isinstance(evidence_sha256, str)) or (not SHA.fullmatch(evidence_sha256)):
        raise ValueError('grasp-specific independent height candidate and actual failure SHA required')
    roots = scene_reset__np.array(original['hand_roots'], dtype=float, copy=True)
    roots[:, 2] += delta
    description = 'shared_training_height_v1;grasp=' + grasp + ';parent=' + seed['sha256'] + ';dz=' + ','.join((format(float(x), '.17g') for x in delta)) + ';failure=' + evidence_sha256
    result = prepare_reset_state(hand_roots=dict(zip(scene_reset__SIDES, roots.tolist(), strict=True)), origin=original['origin'], root_xy=original['root_xy'], initial_offset_m=original['initial_offset_m'], joint_positions=original['joint_positions'], joint_velocities=original['joint_velocities'], source_description=description)
    return (result, dict(schema='dh116-shared-training-height-v1', grasp=grasp, parent_seed_sha256=seed['sha256'], derived_seed_sha256=result['sha256'], delta_z_m=delta.tolist(), failure_evidence_sha256=evidence_sha256, rod_translation_m=[0.0, 0.0, 0.0], joint_or_asset_or_physics_parameter_changed=False, native_reset_qualified=False, teacher_admission=False))

def validate_simulation_settings(sim, prepared):
    """Read actual initialized physics dt/gravity; no setters or stepping."""
    dt = float(sim.get_physics_dt())
    value = sim.physics_sim_view.get_gravity()
    gravity = scene_valid__np.asarray(list(value), dtype=float) if value is not None else scene_valid__np.asarray([])
    expected = prepared['candidate_physics']['gravity_world_m_s2']
    if dt != 1 / 480 or gravity.shape != (3,) or (not scene_valid__np.allclose(gravity, expected, rtol=0, atol=1e-06)):
        raise ValueError('actual physics dt/gravity differs from explicit candidate')
    return dict(dt_s=dt, gravity_world_m_s2=gravity.tolist(), source='SimulationContext.get_physics_dt/SimulationView.get_gravity')

def isolate_scenes(stage, scenes, *, physics_scene_path=None, build_mode='clone'):
    """Exclude cross-environment contacts without changing within-env contacts.

    reference preserves the original pairwise exclusion graph. The Cloner
    graph uses PhysX's inverted filter with one self target per environment.
    Inversion is global to the physics scene, so pre-existing groups must never
    be silently reinterpreted or deleted by this fresh-scene owner.
    """
    from pxr import UsdPhysics
    if build_mode == 'reference':
        groups = []
        for i, scene in enumerate(scenes):
            group = UsdPhysics.CollisionGroup.Define(stage, f'/World/collisionGroups/env_{i}')
            group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(scene['prefix'])
            groups.append(group)
        for i, group in enumerate(groups):
            group.CreateFilteredGroupsRel().SetTargets([g.GetPath() for j, g in enumerate(groups) if j != i])
        return
    if build_mode not in ('clone', 'replicated'):
        raise ValueError('explicit reference, clone or replicated scene build mode required')
    from pxr import PhysxSchema
    from isaacsim.core.cloner import Cloner
    if not physics_scene_path or not UsdPhysics.Scene(stage.GetPrimAtPath(physics_scene_path)):
        raise ValueError('actual SimulationContext physics scene path required')
    physics = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(physics_scene_path))
    if not physics or physics.GetInvertCollisionGroupFilterAttr().Get():
        raise ValueError('fresh non-inverted physics scene required')
    prefixes = [scene['prefix'] for scene in scenes]
    if not prefixes or len(set(prefixes)) != len(prefixes) or any(not stage.GetPrimAtPath(prefix) for prefix in prefixes):
        raise ValueError('distinct existing environment roots required')
    if stage.GetPrimAtPath('/World/collisionGroups'):
        raise ValueError('existing collision groups cannot be reinterpreted by inverted filtering')
    from pxr import Usd
    prefix_set = set(prefixes)
    global_paths = []
    # One hierarchy walk also preserves collisions with ungrouped external
    # objects. Ancestor membership is O(depth), independent of the env count.
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.CollisionGroup):
            raise ValueError('existing collision groups cannot be reinterpreted by inverted filtering')
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            if not any(str(path) in prefix_set for path in prim.GetPath().GetPrefixes()):
                global_paths.append(str(prim.GetPath()))
    Cloner(stage=stage).filter_collisions(physicsscene_path=physics_scene_path,
        collision_root_path='/World/collisionGroups', prim_paths=prefixes, global_paths=global_paths)
