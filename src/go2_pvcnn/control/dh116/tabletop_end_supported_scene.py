"""Optional static end supports; geometry/seed transforms do not import Isaac.

All metadata uses the source environment's coordinates, without clone origins.
Only initialization/reset poses are translated. Supports remain ordinary static
colliders throughout the rollout; no attachment or object force is introduced.
"""
from copy import deepcopy

import numpy as np


def plan_end_support_scene(layout, clearance_m, *, wire_length_m=None, wire_radius_m=None):
    """Return an independent layout and JSON metadata (None for the legacy scene).

    clearance_m is rod-bottom clearance above the unchanged table. Explicit
    dimensions must come from the actual spawn configuration, not an assumed
    grasp span. The pure helper also accepts the reference layout's dimensions.
    """
    clearance = float(clearance_m)
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('end support clearance must be finite and nonnegative')
    configured = deepcopy(layout)
    if clearance == 0:
        return configured, None
    length = float(layout['wire_length_m'] if wire_length_m is None else wire_length_m)
    radius = float(layout['wire_radius_m'] if wire_radius_m is None else wire_radius_m)
    if not np.isfinite([length, radius]).all() or length <= 0 or radius <= 0:
        raise ValueError('end supports require finite positive wire length and radius')
    if layout.get('wire_axis', 'Y') != 'Y':
        raise ValueError('end supports require the native horizontal Y-axis rod')
    center, table_center, table_size = (np.asarray(layout[k], dtype=float) for k in
                                      ('wire_center', 'table_center', 'table_size'))
    if (any(v.shape != (3,) or not np.isfinite(v).all()
            for v in (center, table_center, table_size)) or np.any(table_size <= 0)):
        raise ValueError('finite wire/table centers and positive table dimensions required')
    table_top = float(table_center[2] + table_size[2] / 2)
    bottom = table_top + clearance
    dz = bottom - (center[2] - radius)
    configured['wire_center'] = center.copy()
    configured['wire_center'][2] += dz
    for side, pose in configured['hand_roots'].items():
        pose = np.asarray(pose, dtype=float).copy()
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('finite seven-component hand root poses required')
        pose[2] += dz
        configured['hand_roots'][side] = pose

    # Narrow axial strips at the physical ends, outside the +/-0.2 m grasp
    # region of the current 0.5 m rod. No rail spans the central workspace.
    axial_width = min(.01, .04 * length)
    inset = min(.002, .01 * length)
    end_offset = length / 2 - inset - axial_width / 2
    transverse_width = max(.024, 2 * radius + .008)
    supports = []
    for name, sign in (('negative_y', -1), ('positive_y', 1)):
        position = (float(center[0]), float(center[1] + sign * end_offset),
                    table_top + clearance / 2)
        size = (transverse_width, axial_width, clearance)
        # A pedestal must actually stand on the retained table.
        if any(abs(position[i] - table_center[i]) + size[i] / 2 > table_size[i] / 2 + 1e-9
               for i in (0, 1)):
            raise ValueError('end support footprint falls outside the table')
        supports.append(dict(name=name, center_m=list(position), size_m=list(size)))
    metadata = dict(version='dh116-static-end-supports-v1', clearance_m=clearance,
                    dz_m=float(dz), table_top_m=table_top,
                    lift_reference_bottom_m=float(bottom),
                    wire_length_m=length, wire_radius_m=radius, wire_axis='Y',
                    central_gap_m=float(2 * end_offset - axial_width),
                    coordinates='source_environment_without_origin',
                    support_type='persistent_static_collision', supports=supports)
    return configured, metadata


def translate_end_support_seed(seed, scene):
    """Translate a fresh copy of the original local seed once at each bind.

    root_offset/root_xy, rotations, velocities and joint state retain their
    existing meanings. Both GpuWorld and TensorWorld read these same seeds.
    """
    shifted = deepcopy(seed)
    if scene is not None:
        dz = scene['dz_m']
        pose = np.asarray(shifted['wire_pose'], dtype=float).copy()
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('finite seven-component seed wire pose required')
        pose[2] += dz
        shifted['wire_pose'] = pose
        if 'roots' in shifted:
            roots = np.asarray(shifted['roots'], dtype=float).copy()
            if roots.shape != (2, 7) or not np.isfinite(roots).all():
                raise ValueError('finite two-hand seed root poses required')
            roots[:, 2] += dz
            shifted['roots'] = roots
    return shifted


def wire_geometry_from_spawn(spawn):
    """Read native CylinderCfg dimensions, or the actual centered USD envelope.

    USD is imported lazily, only for the explicit external rigid-rod path.
    Unsupported off-center/non-Y envelopes fail before any scene is authored.
    """
    if hasattr(spawn, 'radius') and hasattr(spawn, 'height'):
        if spawn.axis != 'Y':
            raise ValueError('end supports require a Y-axis cylinder spawn')
        length, radius = float(spawn.height), float(spawn.radius)
    else:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(spawn.usd_path))
        if stage is None or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
            raise ValueError('end supports require a readable meter-unit rod USD')
        root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
        bounds = bounds.ComputeWorldBound(root).ComputeAlignedRange()
        low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        if (not np.isfinite([low, high]).all() or np.any(high <= low)
                or not np.allclose(low, -high, atol=1e-6, rtol=0)
                or not np.isclose(high[0], high[2], atol=1e-6, rtol=0)
                or high[1] <= max(high[0], high[2])):
            raise ValueError('end supports require a centered Y-axis cylindrical USD envelope')
        length, radius = float(high[1] - low[1]), float(high[2])
    if not np.isfinite([length, radius]).all() or length <= 0 or radius <= 0:
        raise ValueError('finite positive spawned rod dimensions required')
    return length, radius


def author_end_supports(stage, prefix, scene):
    """Author static USD cubes under the source environment BEFORE cloning.

    No RigidBodyAPI (hence static), joints, materials or object edits. The
    existing table, gravity and rod mass/material configuration stay intact.
    """
    if scene is None:
        return ()
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    parent = prefix + '/EndSupports'
    UsdGeom.Xform.Define(stage, parent)
    paths = []
    for support in scene['supports']:
        path = parent + '/' + support['name']
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.)
        cube.CreateExtentAttr([Gf.Vec3f(-.5), Gf.Vec3f(.5)])
        cube.AddTranslateOp().Set(Gf.Vec3d(*support['center_m']))
        cube.AddScaleOp().Set(Gf.Vec3d(*support['size_m']))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        # Author the native schema/attributes directly so offline USD checks
        # do not need to load Kit's PhysxSchema Python extension.
        prim = cube.GetPrim()
        prim.AddAppliedSchema('PhysxCollisionAPI')
        prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.001)
        prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
        paths.append(path)
    return tuple(paths)
