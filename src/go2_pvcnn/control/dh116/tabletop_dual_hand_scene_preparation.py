"""Source-bound preparation for two original mounted hands and one free rod.

CPU preparation uses real USD authoring and locked F4 SDF bytes. The optional
scene assembly function delegates to the original builder after an external
owner has created an approved SimulationContext; this module never launches,
steps, resets, projects wrists, or grants native/Teacher qualification.
"""
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .deformx_dh116_colliders import DISTAL_PAD_LINKS, read_pad_triangles
from .linear_coupling import expand
from .tabletop_deformable_grasp_task import load_flexible_grasp_task
from .tabletop_deformable_grasp_runtime import load_candidate
from .tabletop_deformable_bimanual_step import _clock, _owned
from .tabletop_batch_geometry import collision_clearance
from .tabletop_load_parameters import LoadParameters
from .tabletop_palm_mount import author_mount, JOINT_NAMES, ANGULAR_JOINT_NAMES
from .runtime_mimic import install_hand_mimic, configure_mimic_compliance
from .tabletop_material import (hand_material_contract, material_config_fields,
    bind_hand_materials, verify_collider_bindings, bind_object_colliders)
from .usd_surface import read_surface
from scripts.audit_dh116_pinch_initial_pose import load_geometry, CAPTURE
from scripts.bake_dh116_deformx_pad_sdfs import checked_bytes, checked_json

VERSION = 'dh116-dual-hand-scene-preparation-v1'
SIDES = ('left', 'right')
SDF_DIR = Path('/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-sdf-register-rK7gw0/bake')
SDF_SHA = 'db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab'
PREFIX = '/World/DeformXDualHand'
BUILDER_OFFSET_M = .04
SCHEMA_ROOT = Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaacsim/extsPhysics/omni.usd.schema.physx'
SCHEMA_FILES = ('bin/libphysxSchema.so','pxr/PhysxSchema/__init__.py','pxr/PhysxSchema/_physxSchema.so',
    'plugins/PhysxSchema/resources/plugInfo.json','plugins/PhysxSchema/resources/generatedSchema.usda',
    'plugins/PhysxSchema/resources/schema.usda')
LOCAL_FILES = (
    'configs/dh116_tasks/flexible_dual_hand_grasp_v1.json',
    'configs/dh116_tasks/flexible_pinch_candidate_v1.json',
    'configs/dh116_training/pose_reference.json',
    'assets/fixed_dual_panda_dh116_v1/manifest.json',
    'docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json',
    'src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py',
    'src/go2_pvcnn/control/dh116/tabletop_dual_hand_scene_preparation.py',
    'src/go2_pvcnn/control/dh116/tabletop_palm_mount.py',
    'src/go2_pvcnn/control/dh116/runtime_mimic.py',
    'src/go2_pvcnn/control/dh116/tabletop_material.py',
    'src/go2_pvcnn/control/dh116/tabletop_load_parameters.py',
    'src/go2_pvcnn/control/dh116/tabletop_batch_geometry.py',
    'src/go2_pvcnn/control/dh116/tabletop_deformable_bimanual_step.py',
    'src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_task.py',
    'src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py',
    'src/go2_pvcnn/control/dh116/hand_point_kinematics.py',
    'src/go2_pvcnn/control/dh116/usd_contract.py',
    'src/go2_pvcnn/control/dh116/linear_coupling.py',
    'src/go2_pvcnn/control/dh116/usd_surface.py',
    'src/go2_pvcnn/control/dh116/deformx_dh116_colliders.py',
    'scripts/audit_dh116_pinch_initial_pose.py',
    'scripts/prepare_dh116_dual_hand_scene.py',
    'scripts/bake_dh116_deformx_pad_sdfs.py',
    'scripts/verify_dh116_cooked_convex_hulls_native.py',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def prepare_scene(project, *, capture_dir=CAPTURE, sdf_dir=SDF_DIR):
    project = Path(project).resolve()
    task = load_flexible_grasp_task(project/LOCAL_FILES[0], project)
    candidate = load_candidate(project/LOCAL_FILES[1], project)
    summary = checked_json(Path(sdf_dir)/'result.json', SDF_SHA)
    result = dict(version=VERSION, prefix=PREFIX, env_id=0,
        task_unknown_fields=dict(scene={k:v for k,v in task['scene'].items() if v is None},
                                 rod={k:v for k,v in task['rod'].items() if v is None}),
        candidate_physics=dict(gravity_world_m_s2=candidate['rod']['gravity_world_m_s2'],
            physics_hz=480, hand=deepcopy(candidate['hand']), table=deepcopy(candidate['table']),
            material_status=candidate['material_status']),
        table=dict(center_m=task['scene']['table_center_m'],size_m=task['scene']['table_size_m'],
                   top_m=task['scene']['table_top_m']),
        hands={}, pads={}, local_sources={p:_sha(project/p) for p in LOCAL_FILES},
        external_sources={str(Path(sdf_dir)/'result.json'):SDF_SHA},
        builder=dict(function='tabletop_vec_isaac.spawn_scene',external_wire=True,
            authoring_z_offset_m=BUILDER_OFFSET_M,original_default_effort_limit_nm=.2,
            explicit_candidate_effort_limit_nm=candidate['hand']['active_effort_limit_nm']),
        native_qualified=False,teacher_admitted=False,initial_pose_qualified=False,
        all_34_live_cooked_available=False,
        blockers=['source reference pose is not a qualified flexible reset',
                  'twelve non-pad cooked shapes per hand remain unavailable',
                  'native lifecycle and same-clock readback not executed',
                  'original runtime wrist safety projection must consume fresh geometry'])
    result['external_sources'].update({str(SCHEMA_ROOT/p):_sha(SCHEMA_ROOT/p) for p in SCHEMA_FILES})
    lock = json.loads((project/'docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json').read_text())
    capture_inputs = {**lock['outer_exit_sha256'],'parity.json':lock['parity_sha256'],
                      'gpu/result.json':lock['gpu_result_sha256']}
    for name,sha in capture_inputs.items():
        checked_bytes(Path(capture_dir)/name,sha)
        result['external_sources'][str(Path(capture_dir)/name)] = sha
    for side in SIDES:
        geometry = load_geometry(project,Path(capture_dir),side=side)
        usd = geometry['usd']
        relative = str(Path(usd['path']).relative_to(project))
        result['local_sources'][relative] = usd['sha256']
        pose = list(geometry['pose'])
        initial = pose.copy(); initial[2] += BUILDER_OFFSET_M
        expand(geometry['active_q'])
        shapes = {name:dict(vertices=row['vertices'].tolist(),
                    geometry_sha256=row['source']['geometry_sha256'],
                    mesh_to_body=row['source']['mesh_to_body'],
                    source_kind=row['kind'])
                  for name,row in geometry['source_colliders'].items()}
        result['hands'][side] = dict(usd_path=relative,usd_sha256=usd['sha256'],
            source_root_pose_w=pose,builder_initial_root_pose_w=initial,
            source_active_q=list(geometry['active_q']),source_colliders=shapes)
        for link in DISTAL_PAD_LINKS:
            key = side+'.'+link
            record = summary['pads'][key]
            cooked = geometry['cooked_pads'][link]
            if (record['gpu_row_sha256'] != cooked['capture_sha256']
                    or record['rod_radius_m'] != candidate['rod']['radius_m']
                    or record['closed_welded_hull'] is not True
                    or record['frame'] != 'original_pad_coincident_local_basis_m'):
                raise ValueError('locked SDF/cooked/source frame or radius differs')
            sdf_path = Path(sdf_dir)/(key+'.npz')
            checked_bytes(sdf_path,record['sdf_file_sha256'])
            capture_path = Path(capture_dir)/'gpu'/f'{side}_{link}.json'
            result['external_sources'][str(sdf_path)] = record['sdf_file_sha256']
            result['external_sources'][str(capture_path)] = cooked['capture_sha256']
            result['pads'][key] = dict(side=side,body_name=link,
                source_usd_sha256=usd['sha256'],
                source_geometry_sha256=cooked['geometry_sha256'],
                cooked_geometry_sha256=_digest(dict(vertices=cooked['vertices'].tolist(),
                                                     faces=cooked['faces'].tolist())),
                cooked_hash_encoding='canonical JSON {faces,vertices} from locked F4 hull',
                capture_sha256=cooked['capture_sha256'],sdf_path=str(sdf_path),
                sdf_sha256=record['sdf_file_sha256'],
                frame='PhysX_link_origin',unit='m')
    result['fingerprint_sha256'] = _digest(result)
    load_sdf_arrays(result)
    return result


def _verify(project, prepared):
    payload = deepcopy(prepared); claimed = payload.pop('fingerprint_sha256',None)
    if prepared.get('version') != VERSION or _digest(payload) != claimed:
        raise ValueError('prepared source fingerprint changed')
    for path,sha in prepared['local_sources'].items():
        checked_bytes(Path(project)/path,sha)
    for path,sha in prepared['external_sources'].items():
        checked_bytes(Path(path),sha)


def load_sdf_arrays(prepared):
    """Owned constructor inputs for existing ropesim.sdf.SdfGrid; no ropesim import."""
    result = {}
    for key,row in prepared['pads'].items():
        checked_bytes(Path(row['sdf_path']),row['sdf_sha256'])
        with np.load(row['sdf_path'],allow_pickle=False) as data:
            values,origin,spacing = data['values'].copy(),data['origin'].copy(),float(data['spacing'])
        if (values.dtype != np.float32 or values.ndim != 3 or min(values.shape)<2
                or origin.shape != (3,) or not np.isfinite(values).all()
                or not np.isfinite(origin).all() or spacing != .001):
            raise ValueError('finite original signed-grid constructor inputs required')
        result[key] = dict(values=values,origin=origin,spacing=spacing)
    return result


def spawn_parameters(project, prepared):
    """Arguments for original external_wire builder, with explicit candidate choices."""
    _verify(project,prepared)
    hand = prepared['candidate_physics']['hand']
    if (hand['active_stiffness'],hand['active_damping'],hand['mimic_damping'],
            hand['fixed_base'],hand['self_collision']) != (10.,.2,1.,True,False):
        raise ValueError('original scene builder cannot represent this candidate hand configuration')
    args = SimpleNamespace(control_mode='adaptive_grasp',position_iterations=64,
        velocity_iterations=16,palm_transport='driven_carriage',free_wrist_grasp=True,
        hand_material=hand['material_profile'],mimic_frequency=hand['mimic_stiffness'],
        passive_velocity_limit_rad_s=3.,
        load_parameters=LoadParameters(effort_limit_nm=hand['active_effort_limit_nm']))
    reference = json.loads((Path(project)/'configs/dh116_training/pose_reference.json').read_text())['reference_layout']
    layout = deepcopy(reference)
    layout['hand_roots'] = {s:np.asarray(prepared['hands'][s]['source_root_pose_w']).copy() for s in SIDES}
    layout['active_joints'] = {s:np.asarray(prepared['hands'][s]['source_active_q']).copy() for s in SIDES}
    return args,layout,{s:str(Path(project)/prepared['hands'][s]['usd_path']) for s in SIDES}


def _material(stage,path,fields):
    """USD-only material authoring; the original binder verifies every resolved field."""
    from pxr import UsdShade,UsdPhysics,Sdf,PhysxSchema
    fields = {'compliant_contact_stiffness':0.,'compliant_contact_damping':0.,**fields}
    material = UsdShade.Material.Define(stage,path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(fields['static_friction']))
    api.CreateDynamicFrictionAttr(float(fields['dynamic_friction']))
    api.CreateRestitutionAttr(float(fields['restitution']))
    names = dict(friction_combine_mode='frictionCombineMode',restitution_combine_mode='restitutionCombineMode',
                 compliant_contact_stiffness='compliantContactStiffness',compliant_contact_damping='compliantContactDamping')
    for key,name in names.items():
        if key in fields:
            value = fields[key]
            material.GetPrim().CreateAttribute('physxMaterial:'+name,
                Sdf.ValueTypeNames.Token if isinstance(value,str) else Sdf.ValueTypeNames.Float).Set(value)
    return material


def _bind_materials(stage,prepared):
    prefix = prepared['prefix']
    spec = hand_material_contract(prepared['candidate_physics']['hand']['material_profile'])['hand_contact_material']
    hard = _material(stage,prefix+'/HandContactMaterial',material_config_fields(spec))
    soft = _material(stage,prefix+'/PadContactMaterial',material_config_fields(spec,compliant=True))
    reports = {}
    for side in SIDES:
        _,reports[side] = bind_hand_materials(stage,prefix+'/'+side+'Hand',hard,spec,soft)
    table = prepared['candidate_physics']['table']
    fields = dict(static_friction=table['mu_s'],dynamic_friction=table['mu_k'],restitution=0.,
                  friction_combine_mode='average',restitution_combine_mode='average')
    material = _material(stage,prefix+'/TableContactMaterial',fields)
    paths = bind_object_colliders(stage,prefix+'/Table',material)
    verify_collider_bindings(stage,paths,fields)
    return reports




def _load_offline_usd_schema():
    # Only the installed USD schema metadata library, not Kit/PhysX simulation.
    # Its transitive dependencies are USD/Python/C++ libraries. No app or solver.
    import ctypes
    import pxr
    from pxr import Plug
    if 'pxr.PhysxSchema' not in sys.modules:
        ctypes.CDLL(str(SCHEMA_ROOT/'bin/libphysxSchema.so'),mode=ctypes.RTLD_GLOBAL)
        path = str(SCHEMA_ROOT/'pxr')
        if path not in pxr.__path__:
            pxr.__path__.append(path)
        Plug.Registry().RegisterPlugins(str(SCHEMA_ROOT/'plugins/PhysxSchema/resources'))


def author_source_stage(project,prepared):
    """Real USD topology/material preflight; initial SDK joint state is not applied."""
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
    _verify(project,prepared)
    _load_offline_usd_schema()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.SetTimeCodesPerSecond(480.)
    UsdGeom.Xform.Define(stage,'/World')
    prefix = prepared['prefix']; UsdGeom.Xform.Define(stage,prefix)
    scene = UsdPhysics.Scene.Define(stage,'/World/physicsScene')
    gravity = np.asarray(prepared['candidate_physics']['gravity_world_m_s2'])
    scene.CreateGravityDirectionAttr(Gf.Vec3f(*(gravity/np.linalg.norm(gravity))))
    scene.CreateGravityMagnitudeAttr(float(np.linalg.norm(gravity)))
    table = UsdGeom.Cube.Define(stage,prefix+'/Table')
    table.CreateSizeAttr(1.)
    table.AddTranslateOp().Set(Gf.Vec3d(*prepared['table']['center_m']))
    table.AddScaleOp().Set(Gf.Vec3f(*prepared['table']['size_m']))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim()).CreateCollisionEnabledAttr(True)
    for side in SIDES:
        row = prepared['hands'][side]; root = prefix+'/'+side+'Hand'
        prim = UsdGeom.Xform.Define(stage,root).GetPrim()
        prim.GetReferences().AddReference(str(Path(project)/row['usd_path']))
        pose = row['builder_initial_root_pose_w']
        xform = UsdGeom.Xform(prim); xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(pose[3]),Gf.Vec3d(*pose[4:])))
        for name in row['source_colliders']:
            stage.GetPrimAtPath(root+'/'+name).CreateAttribute(
                'physxRigidBody:disableGravity',Sdf.ValueTypeNames.Bool).Set(True)
        author_mount(stage,root,rotation=True)
        install_hand_mimic(stage,root)
        hand = prepared['candidate_physics']['hand']
        configure_mimic_compliance(stage,root,hand['mimic_stiffness'],hand['mimic_damping'])
    _bind_materials(stage,prepared)
    inspect_source_stage(stage,prepared)
    return stage


def inspect_source_stage(stage,prepared):
    """Check actual referenced original geometry after authoring, including proxies."""
    from pxr import Usd,UsdGeom,UsdPhysics
    count = 0
    for side in SIDES:
        root = prepared['prefix']+'/'+side+'Hand'
        for name,row in prepared['hands'][side]['source_colliders'].items():
            actual = read_surface(stage,root+'/'+name)
            if (actual['geometry_sha256'] != row['geometry_sha256']
                    or not np.allclose(actual['mesh_to_body'],row['mesh_to_body'],rtol=0,atol=1e-8)):
                raise ValueError('assembled source collision geometry changed')
            count += 1
        if (stage.GetPrimAtPath(root+'/root_joint').IsActive()
                or not stage.GetPrimAtPath(root+'/wrist_fixed').HasAPI(UsdPhysics.ArticulationRootAPI)):
            raise ValueError('original driven mount topology differs')
        for name in JOINT_NAMES+ANGULAR_JOINT_NAMES:
            if not stage.GetPrimAtPath(root+'/'+name):
                raise ValueError('complete original six-axis wrist required')
    table_root = stage.GetPrimAtPath(prepared['prefix']+'/Table')
    if not table_root:
        raise ValueError('actual finite table required')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]).ComputeWorldBound(table_root).ComputeAlignedRange()
    lower,upper = np.asarray(bounds.GetMin()),np.asarray(bounds.GetMax())
    if (not np.allclose((upper+lower)/2,prepared['table']['center_m'],rtol=0,atol=1e-6)
            or not np.allclose(upper-lower,prepared['table']['size_m'],rtol=0,atol=1e-6)):
        raise ValueError('actual finite table bounds differ')
    table_paths = [str(p.GetPath()) for p in Usd.PrimRange(table_root)
                   if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    material = prepared['candidate_physics']['table']
    verify_collider_bindings(stage,table_paths,dict(static_friction=material['mu_s'],
        dynamic_friction=material['mu_k'],restitution=0.,friction_combine_mode='average',
        restitution_combine_mode='average'))
    if count != 34:
        raise ValueError('both original seventeen-collider hands required')
    # Read-only material validation; do not repair a corrupted stage in an audit.
    spec = hand_material_contract(prepared['candidate_physics']['hand']['material_profile'])['hand_contact_material']
    for side in SIDES:
        for name in prepared['hands'][side]['source_colliders']:
            surface = read_surface(stage,prepared['prefix']+'/'+side+'Hand/'+name)
            fields = dict(spec,compliant_contact_stiffness=10000. if name in DISTAL_PAD_LINKS else 0.,
                          compliant_contact_damping=30. if name in DISTAL_PAD_LINKS else 0.)
            verify_collider_bindings(stage,[surface['collision_owner']],fields)
    return dict(original_colliders=count,compliant_pad_colliders=10,wrist_joints=12,
                usd_only=True,sdk_readback_performed=False,initial_joint_state_applied=False,
                native_qualified=False)


def validate_simulation_settings(sim,prepared):
    """Read actual initialized physics dt/gravity; no setters or stepping."""
    dt = float(sim.get_physics_dt())
    value = sim.physics_sim_view.get_gravity()
    gravity = np.asarray(list(value),dtype=float) if value is not None else np.asarray([])
    expected = prepared['candidate_physics']['gravity_world_m_s2']
    if dt != 1/480 or gravity.shape != (3,) or not np.allclose(gravity,expected,rtol=0,atol=1e-6):
        raise ValueError('actual physics dt/gravity differs from explicit candidate')
    return dict(dt_s=dt,gravity_world_m_s2=gravity.tolist(),
                source='SimulationContext.get_physics_dt/SimulationView.get_gravity')


def assemble_prepared_scene(sim,project,prepared):
    """Future approved owner only: delegate original authoring, no launch/reset/step."""
    args,layout,sources = spawn_parameters(project,prepared)
    if sim.stage.GetPrimAtPath(prepared['prefix']):
        raise ValueError('isolated empty scene prefix required')
    from .tabletop_vec_isaac import spawn_scene
    scene = spawn_scene(sim,args,layout,sources,prepared['prefix'],np.zeros(3),external_wire=True)
    _bind_materials(sim.stage,prepared)
    inspect_source_stage(sim.stage,prepared)
    # Backend gravity is available only after the caller's counted setup/reset.
    scene['scene_preparation'] = dict(fingerprint_sha256=prepared['fingerprint_sha256'],
        pending_actual_physics_readback=True,initial_pose_qualified=False,native_qualified=False)
    return scene


def bind_actual_geometry(project,prepared,stage,hands,*,read_clock):
    """Read actual body paths/actor-frame poses for the unchanged clearance kernel.

    Caller must update_articulations_kinematic after any reset and before this
    read. No joint state is set and no geometry is inferred from USD world poses.
    """
    _verify(project,prepared); inspect_source_stage(stage,prepared)
    if (len(hands)!=2 or hands[0] is hands[1]
            or hands[0].root_physx_view is hands[1].root_physx_view):
        raise ValueError('two distinct original articulation views required')
    before = _clock(read_clock,(prepared['env_id'],))
    poses,shapes,bindings = [],[],{}
    names_before,paths_before,view_paths_before = [],[],[]
    articulation_paths = {}
    for hi,(side,hand) in enumerate(zip(SIDES,hands)):
        root = prepared['prefix']+'/'+side+'Hand'
        names = list(hand.body_names); paths = deepcopy(hand.root_physx_view.link_paths)
        view_paths = list(hand.root_physx_view.prim_paths)
        # IsaacLab finds the authored ArticulationRootAPI at wrist_fixed.
        # Keep the actual backend path distinct from its hand container; some
        # backend views report that container as their canonical instance path.
        if (len(view_paths)!=1 or view_paths[0] not in (root,root+'/wrist_fixed')
                or len(set(names)) != len(names)
                or len(names)!=23 or len(paths)!=1 or len(paths[0])!=len(names)
                or paths[0] != [root+'/'+name for name in names]):
            raise ValueError('actual PhysX instance/link paths differ from original mounted hand')
        raw = _owned(hand.root_physx_view.get_link_transforms())
        if (raw.shape!=(1,len(names),7) or not np.isfinite(raw).all()
                or not np.allclose(np.linalg.norm(raw[...,3:],axis=-1),1.,atol=1e-5,rtol=0)):
            raise ValueError('finite normalized actual PhysX link transforms required')
        pose = raw[..., [0,1,2,6,3,4,5]].copy()  # native xyzw -> original kernel wxyz
        poses.append(pose)
        for link,row in prepared['hands'][side]['source_colliders'].items():
            if link not in names:
                raise ValueError('missing original collision body')
            body = names.index(link)
            shapes.append((hi,body,np.asarray(row['vertices'],dtype=float).copy()))
            key = side+'.'+link
            if key in prepared['pads']:
                _,_,geometry = read_pad_triangles(stage,paths[0][body])
                if geometry['geometry_sha256'] != prepared['pads'][key]['source_geometry_sha256']:
                    raise ValueError('runtime pad source is not the locked SDF geometry')
                bindings[key] = dict(prepared['pads'][key],prim_path=paths[0][body],body_index=body)
        names_before.append(names);paths_before.append(paths);view_paths_before.append(view_paths)
        articulation_paths[side] = dict(actual_articulation_view_path=view_paths[0],
            authored_articulation_root_path=root+'/wrist_fixed',hand_container_path=root)
    if (_clock(read_clock,(prepared['env_id'],)) != before
            or any(names_before[i]!=list(h.body_names) or paths_before[i]!=h.root_physx_view.link_paths
                   or view_paths_before[i]!=list(h.root_physx_view.prim_paths)
                   for i,h in enumerate(hands))):
        raise ValueError('clock or native link mapping changed during geometry capture')
    body_poses = np.stack(poses,axis=1)
    clearance,lowest = collision_clearance(body_poses,shapes)
    return dict(identity=before,body_poses_wxyz=body_poses,shapes=shapes,pad_bindings=bindings,
        articulation_paths=articulation_paths,actual_physics_settings_verified=False,
        clearance_m=clearance,lowest_shape_index=lowest,table_top_m=prepared['table']['top_m'],
        source='PhysX get_link_transforms + original USD link-local convex envelopes',
        all_34_live_cooked_available=False,native_qualified=False,teacher_admitted=False)
