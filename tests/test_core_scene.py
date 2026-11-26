import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pytest

def test_current_speed_action_binds_wrist_z_feedback_and_keeps_observation_width():
    from dh116.env.contracts import ACTION_NAME, LEGACY_OBSERVATION_VERSION, bind_contract
    from dh116.env.contracts import state_contract, surfa_contr__validate_contract
    from dh116.env.contracts import verify_estimator_state_contract_file

    rates = dict(joint_rate_rad_s=1.2, wrist_rate_m_s=0.05, rotation_rate_rad_s=1.0)
    motion = dict(action=ACTION_NAME, action_size=24, observation_size=351,
                  observation_version=LEGACY_OBSERVATION_VERSION, policy_control=rates)
    contract = bind_contract(motion)
    assert surfa_contr__validate_contract(contract) is contract
    assert contract['observation_size'] == state_contract()['size'] == 371
    action = contract['surface_contract']['action']
    assert action['size'] == 24
    assert (action['joint_rate_rad_s'], action['translation_rate_m_s'],
            action['rotation_rate_rad_s']) == (1.2, 0.05, 1.0)
    wrist_z = next(row for row in contract['surface_contract']['feedback_fields']
                   if row['name'] == 'wrist_z_request')
    assert wrist_z['slice'] == [367, 369]
    assert wrist_z['scale'] == 0.05
    for row in state_contract()['dimensions'][367:369]:
        assert row['normalization'] == 'mean(task.actions[:,[14,17]]*0.05)/0.05'
    manifest = Path(__file__).resolve().parents[1] / 'configs/contracts/privileged_state.json'
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify_estimator_state_contract_file(manifest, manifest_sha) == manifest_sha
    with pytest.raises(ValueError, match='action scale mismatch'):
        bind_contract(dict(motion, policy_control=dict(rates, wrist_rate_m_s=0.01)))

def test_old_speed_action_cannot_resume_a_checkpoint_even_with_consistent_hashes():
    from dh116.env.contracts import make_contract, stere_contr__digest, stere_contr__validate_contract
    from dh116.env.contracts import surfa_contr__ACTION_CONTRACT

    current = make_contract('teacher', {})
    assert stere_contr__validate_contract(current) is current
    old_action = dict(surfa_contr__ACTION_CONTRACT, version='dh116-surface-motion24-v1',
                      joint_rate_rad_s=0.3, translation_rate_m_s=0.01, rotation_rate_rad_s=0.5)
    old_sha = stere_contr__digest(old_action)
    assert old_sha == '498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9'
    old = dict(current, action_contract=old_action, action_contract_sha256=old_sha)
    old.pop('contract_sha256')
    old['contract_sha256'] = stere_contr__digest(old)
    with pytest.raises(ValueError, match='canonical action contract mismatch'):
        stere_contr__validate_contract(old)

def test_velocity_limit_readback_checks_every_lane_with_the_exact_configured_passive_cap():
    from types import SimpleNamespace

    import torch

    from dh116.env import scene
    from dh116.env.contracts import JOINTS, linea_coupl__ACTIVE
    from dh116.hand.kinematics import ANGULAR_JOINT_NAMES, JOINT_NAMES

    names = ANGULAR_JOINT_NAMES + JOINTS + JOINT_NAMES
    passive = 3.25
    speeds = [1.0 if name in ANGULAR_JOINT_NAMES else 0.10 if name in JOINT_NAMES
              else 1.2 if name in linea_coupl__ACTIVE else passive for name in names]
    velocities = torch.tensor(speeds).repeat(2, 1)
    view = SimpleNamespace(get_dof_max_velocities=lambda: velocities)
    hand = SimpleNamespace(joint_names=names, root_physx_view=view)
    receipt = scene.velocity_limits_readback((hand,), instances=2, passive_limit=passive)
    assert receipt['instances_per_hand'] == 2
    assert receipt['hands'][0]['joint_names'] == list(names)
    assert receipt['hands'][0]['max_velocity'] == pytest.approx(speeds)

    velocities[1, names.index('finger13')] = 3.0
    with pytest.raises(ValueError, match='simulator velocity limits differ'):
        scene.velocity_limits_readback((hand,), instances=2, passive_limit=passive)
    with pytest.raises(ValueError, match='at least 3.0'):
        scene.velocity_limits_readback((hand,), instances=2, passive_limit=2.97)

def test_measured_wrist_guard_accepts_speed_through_twelve_centimeters_per_second():
    import torch

    from dh116.hand.kinematics import transport_safe

    measured = torch.zeros((2, 2, 7))
    measured[..., 3] = 1.0
    velocity = torch.zeros((2, 2, 3))
    velocity[:, 0, 0] = torch.tensor([0.12, 0.1201])
    assert transport_safe(measured, measured, velocity).tolist() == [True, False]

def test_tensor_wrist_projection_reuses_checked_inputs_without_changing_targets():
    import torch

    from dh116.hand.kinematics import project_tensor_policy_wrist_targets

    seed = torch.zeros((2, 2, 7), dtype=torch.float64)
    seed[..., 3] = 1.0
    zero = seed.new_zeros((2, 2, 3))
    live = torch.tensor([True, False])
    inputs = dict(root_seed_w=seed, measured_base_w=seed, common_lift_offset_m=seed.new_zeros(2),
                  measured_clearance_m=seed.new_full((2, 2), 0.004),
                  palm_offsets_m=zero, requested_palm_offsets_m=zero + 0.012,
                  angles_rad=zero, requested_angles_rad=zero + 0.25, allowed=live, live=live)
    checked = project_tensor_policy_wrist_targets(**inputs)
    fast = project_tensor_policy_wrist_targets(**inputs, validate_values=False)
    assert checked.keys() == fast.keys()
    for name, expected in checked.items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(fast[name], expected, rtol=0, atol=0)
        else:
            assert fast[name] == expected
    with pytest.raises(ValueError, match='finite wrist inputs'):
        project_tensor_policy_wrist_targets(**dict(inputs, measured_base_w=seed + float('nan')))


@pytest.mark.parametrize('grasp,full_grasp', [('pinch', False), ('pinch', True),
                                            ('tripod', True), ('power', True)])
def test_task_scene_uses_original_hand_assets_on_anatomical_sides(grasp, full_grasp):
    from dh116.env.scene import anatomical_hand_axes, load_anatomical_reference, load_scene_inputs, load_task_scene

    archive, reset = load_task_scene(grasp, full_grasp=full_grasp)
    scene = archive.scene_inputs
    poses = scene['author_layout']['hand_roots']
    center = scene['author_layout']['wire_center']
    assert poses['left'][1] < center[1] < poses['right'][1]
    assert all(poses[side][0] > center[0] for side in ('left', 'right'))
    assert Path(scene['assets']['left']).name.endswith('DH116_left.usd')
    assert Path(scene['assets']['right']).name.endswith('DH116_right.usd')
    seed = archive.pair['reset_seed'] if archive.task is None else scene['authored_seed']
    np.testing.assert_allclose(np.array(seed['authored_root_seed_w'])[:, 1],
                               [poses['left'][1], poses['right'][1]], atol=0, rtol=0)
    np.testing.assert_allclose(reset['rod_reset']['node_positions'][0, 20], center,
                               atol=1e-7, rtol=0)
    placement = scene['anatomical_placement']
    assert placement['station_nodes'] == [40 - i for i in placement['source_station_nodes']]
    reference = load_anatomical_reference(grasp, placement['reference_sha256'])
    assert reference['station_nodes'] == tuple(placement['station_nodes'])
    root = Path(__file__).resolve().parents[1]
    source = placement['source_scene']
    original = load_scene_inputs(source['path'], source['sha256'], root=root)
    old_seed = original['archive']['pair']['reset_seed'] if archive.task is None else original['authored_seed']
    assert seed['sha256'] != old_seed['sha256']
    for key in ('joint_positions', 'joint_velocities'):
        assert seed[key] == old_seed[key]
    assert scene['assets'] == original['assets']
    for i, side in enumerate(('left', 'right')):
        pose = seed['reset_root_pose_w'][i]
        np.testing.assert_array_equal(np.delete(pose, 1), np.delete(old_seed['reset_root_pose_w'][i], 1))
        for candidate in (pose, reference['root_pose_wxyz'][i]):
            axis, thumb = anatomical_hand_axes(root / scene['assets'][side], candidate)
            assert axis[0] < 0
            assert thumb[1] * (1 if side == 'left' else -1) > 0


def package(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    hand = root / 'assets/left.usd'
    hand.parent.mkdir()
    hand.write_bytes(b'asset-original')
    bundle = root / 'build/scene'
    bundle.mkdir(parents=True)
    np.savez_compressed(bundle / 'arrays.npz', nodes=np.zeros((41, 3), dtype=np.float32))
    data = {'schema': 'dh116-materialized-scene-v1', 'grasp': 'pinch', 'assets': {'left': 'assets/left.usd'}, 'prepared': {'hands': {'left': {'usd_sha256': hashlib.sha256(hand.read_bytes()).hexdigest()}}}, 'qualification': {'native': False, 'teacher': False, 'grasp': False}, 'reset': {'nodes': {'__array__': 'nodes', 'dtype': '<f4', 'shape': [41, 3]}}, 'arrays_sha256': hashlib.sha256((bundle / 'arrays.npz').read_bytes()).hexdigest()}
    manifest = bundle / 'scene.json'
    manifest.write_text(json.dumps(data))
    return (root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())

def test_local_materialized_arrays_keep_dtype_shape_and_asset_binding(tmp_path):
    module = importlib.import_module('dh116.env.scene')
    (root, path, digest) = package(tmp_path)
    bundle = module.load_scene_inputs(path, digest, root=root)
    assert bundle['reset']['nodes'].dtype == np.float32
    assert bundle['reset']['nodes'].shape == (41, 3)
    (root / 'assets/left.usd').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        module.load_scene_inputs(path, digest, root=root)

@pytest.mark.parametrize('damage', ['manifest', 'arrays', 'qualification', 'escape'])
def test_materialized_input_cannot_override_identity_or_qualification(tmp_path, damage):
    module = importlib.import_module('dh116.env.scene')
    (root, path, digest) = package(tmp_path)
    data = json.loads(path.read_text())
    if damage == 'manifest':
        path.write_text('{}')
    if damage == 'arrays':
        (path.parent / 'arrays.npz').write_bytes(b'broken')
    if damage == 'qualification':
        data['qualification']['grasp'] = True
        path.write_text(json.dumps(data))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if damage == 'escape':
        data['assets']['left'] = '../outside.usd'
        path.write_text(json.dumps(data))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module.load_scene_inputs(path, digest, root=root)

def test_current_pinch_bundle_keeps_authored_and_measured_reset_distinct(monkeypatch):
    """Preserve the archived packet and use a separate anatomical startup seed."""
    from types import SimpleNamespace
    from dh116.env.scene import load_scene_inputs, load_task_scene, prepared_scene
    from dh116.env.lifecycle import NativeOwners
    import dh116.hand.articulation as tabletop_regex_articulation
    import dh116.hand.articulation as tabletop_regex_task_setup
    root = Path(__file__).resolve().parents[1]
    archive, constructor = load_task_scene('pinch', root=root)
    inputs = archive.scene_inputs
    assert inputs['grasp'] == archive.style == 'pinch'
    assert inputs['qualification'] == {'native': False, 'teacher': False, 'grasp': False}
    assert constructor is inputs['reset'] and constructor['kind'] == 'authored_solver_constructor'
    source = inputs['anatomical_placement']['source_scene']
    original = load_scene_inputs(source['path'], source['sha256'], root=root)
    measured = original['reset']
    proposed = original['archive']['pair']['reset_seed']
    assert proposed['sha256'] == 'c19b0c971d76e4e613b7ca3ed12d06012514fd3e7a08a3823daee927e634ce81'
    assert measured['seed']['sha256'] == '9ced76db3e4444107271296993f985b97dc034c40e5140f483288de5546b3bb5'
    assert archive.reset_receipt_sha256 is None
    assert archive.pair['reset_seed'] == constructor['seed']
    assert len({proposed['sha256'], measured['seed']['sha256'], constructor['seed']['sha256']}) == 3
    np.testing.assert_array_equal(constructor['rod_reset']['node_positions'],
                                  measured['rod_reset']['node_positions'])
    thumb = 'left.finger14_link'
    source_pos, source_quat = measured['pad_poses_xyzw'][thumb]
    pos, quat = constructor['pad_poses_xyzw'][thumb]
    np.testing.assert_allclose(pos, np.asarray(source_pos) + [0, -0.4, 0], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(quat, source_quat)

    class ReachedNativeAuthoring(Exception):
        pass

    def stop_before_sdk(*args, **kwargs):
        raise ReachedNativeAuthoring
    monkeypatch.setattr(tabletop_regex_articulation, 'spawn_regex_articulations', stop_before_sdk)
    lanes = [('/review/env_0', (0.0, 0.0, 0.0)), ('/review/env_1', (2.2, 0.0, 0.0))]
    with pytest.raises(ReachedNativeAuthoring):
        tabletop_regex_task_setup.prepare_regex_task(SimpleNamespace(current_time_step_index=0), NativeOwners(), None, root, prepared_scene(inputs), archive, lanes)

def test_pinch_required_sdf_pads_preserve_two_targets_and_three_non_target_roles():
    """Five reaction meshes remain; only the original Pinch pair requires SDF coverage."""
    from dh116.env.scene import load_task_scene
    from dh116.physics.geometry import TensorGeometry as TensorGeometry
    import dh116.physics.runtime as tabletop_tensor_physics_runtime
    root = Path(__file__).resolve().parents[1]
    (archive, _) = load_task_scene('pinch', root=root)
    materialized = archive.scene_inputs['geometry']
    inputs = tabletop_tensor_physics_runtime.pinch_geometry_inputs(materialized)
    required = ('finger14_link', 'finger23_link')
    assert inputs['required_pads'] == required
    assert all((len(inputs['grids'][side]) == 5 for side in ('left', 'right')))
    wrist = ('wrist_anchor', 'wrist_slide_x', 'wrist_slide_y', 'wrist_slide_z', 'wrist_rotate_x', 'wrist_rotate_y')
    names = tuple((tuple(inputs['geometry'][side]['source']) + wrist for side in ('left', 'right')))
    owner = TensorGeometry.from_materialized_geometry(inputs['geometry'], inputs['grids'], inputs['sources'], names, required_pads=inputs['required_pads'], device='cpu')
    assert len(owner.columns) == 44 and len(owner.pad_columns) == 4
    for (side_id, side) in enumerate(('left', 'right')):
        for (kind, entries) in inputs['geometry'][side].items():
            for link in entries:
                assert owner.non_target[owner.columns[side_id, kind + ':' + link]].item() is (link not in required)


def test_usd_template_clone_preserves_precise_origins_and_local_physics():
    """Installed Cloner and real USD composition; no native physics claim."""
    import os
    import subprocess
    import sys
    import sysconfig
    import textwrap
    isaac = Path(sysconfig.get_paths()['purelib']) / 'isaacsim'
    cloner = isaac / 'exts/isaacsim.core.cloner/isaacsim/core/cloner/cloner.py'
    schema = isaac / 'extsPhysics/omni.usd.schema.physx'
    if not cloner.is_file() or not schema.is_dir():
        pytest.skip('installed Isaac Cloner and PhysX USD schemas required')
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(schema / 'bin') + ':' + env.get('LD_LIBRARY_PATH', '')
    env['PXR_PLUGINPATH_NAME'] = str(schema / 'plugins/PhysxSchema/resources') + ':' + env.get('PXR_PLUGINPATH_NAME', '')
    script = textwrap.dedent(r"""
        import ast, copy, importlib.util, sys, types
        import numpy as np
        import pxr
        pxr.__path__.append(sys.argv[2])
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        from dh116.hand import articulation as art
        from dh116.hand.kinematics import author_mount, install_hand_mimic, ANGULAR_JOINT_NAMES
        from dh116.env.contracts import JOINTS
        from dh116.env import scene as scenes
        # Only App runtime dependencies are replaced. Cloner and USD are real.
        def module(name):
            result = types.ModuleType(name); sys.modules[name] = result
            return result
        carb = module('carb'); carb.settings = module('carb.settings')
        carb.settings.get_settings = lambda: types.SimpleNamespace(get_as_string=lambda _: '')
        omni = module('omni'); omni.usd = module('omni.usd'); omni.physx = module('omni.physx')
        callbacks = {}
        class Replicator:
            def register_replicator(self, stage, *functions): callbacks[stage] = functions
            def unregister_replicator(self, stage): callbacks.pop(stage, None)
        omni.physx.get_physx_replicator_interface = lambda: Replicator()
        omni.physx.get_physx_simulation_interface = lambda: None
        module('isaacsim'); module('isaacsim.core')
        module('isaacsim.core.simulation_manager').SimulationManager = types.SimpleNamespace(enable_usd_notice_handler=lambda _: None)
        spec = importlib.util.spec_from_file_location('installed_cloner', sys.argv[1])
        installed = importlib.util.module_from_spec(spec); spec.loader.exec_module(installed)
        module('isaacsim.core.cloner').Cloner = installed.Cloner
        module('isaaclab')
        module('isaaclab.sim').get_all_matching_child_prims = lambda path, predicate, stage, **_: [
            p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if predicate(p)]
        module('isaaclab.assets').Articulation = lambda cfg: types.SimpleNamespace(cfg=cfg)
        # Exercise the installed create_prim pose conversion. Plain cfg tuples
        # become torch.float32; ndarray table positions retain their dtype.
        import torch, sysconfig
        finder = next(__import__('pathlib').Path(sysconfig.get_paths()['purelib']).glob('__editable___isaaclab_*_finder.py'))
        tree = ast.parse(finder.read_text())
        mapping = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', None) == 'MAPPING')
        lab_root = __import__('pathlib').Path(ast.literal_eval(mapping)['isaaclab'])
        source = lab_root / 'sim/utils/prims.py'
        tree = ast.parse(source.read_text())
        converter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_to_tuple')
        namespace = dict(torch=torch, Any=object)
        exec(compile(ast.Module(body=[converter], type_ignores=[]), str(source), 'exec'), namespace)
        to_tuple = namespace['_to_tuple']
        module('isaaclab.sim.utils')
        module('isaaclab.sim.utils.prims')._to_tuple = to_tuple
        class Config:
            def __init__(self, path, position):
                self.prim_path, self.spawn = path, object()
                self.init_state = types.SimpleNamespace(pos=tuple(position))
            def copy(self): return copy.deepcopy(self)
        def author(sim, args, layout, sources, prefix, origin):
            stage = sim.stage
            UsdGeom.Xform.Define(stage, prefix)
            material = scenes._material(stage, prefix + '/Material', dict(
                static_friction=1., dynamic_friction=.8, restitution=0.,
                friction_combine_mode='max', restitution_combine_mode='min'))
            for side in ('left', 'right'):
                path = prefix + '/' + side + 'Hand'
                pose = layout['hand_roots'][side].copy()
                pose[:3] += origin
                pose[2] += .04
                UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(*to_tuple(tuple(pose[:3]))))
                base = UsdGeom.Cube.Define(stage, path + '/base_link')
                UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
                UsdPhysics.CollisionAPI.Apply(base.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdShade.MaterialBindingAPI.Apply(base.GetPrim()).Bind(material, materialPurpose='physics')
                root = UsdPhysics.FixedJoint.Define(stage, path + '/root_joint')
                root.CreateBody1Rel().SetTargets([base.GetPath()])
                UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
                for name in JOINTS:
                    joint = UsdPhysics.RevoluteJoint.Define(stage, path + '/joints/' + name)
                    joint.CreateAxisAttr('X'); joint.CreateBody1Rel().SetTargets([base.GetPath()])
                author_mount(stage, path, rotation=True); install_hand_mimic(stage, path)
            cube = UsdGeom.Cube.Define(stage, prefix + '/Table')
            cube.AddTranslateOp().Set(Gf.Vec3d(*to_tuple(origin + layout['table_center'])))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            paths = scenes.scene_suppo__author_end_supports(stage, prefix, dict(supports=[
                dict(name='end', center_m=(origin+[0,.2,.1]).tolist(), size_m=[.02,.01,.1])]))
            scenes._bind_support_material(stage, prefix, paths, .7, .6)
            configs = []
            for side in ('left','right'):
                pose = layout['hand_roots'][side].copy()
                pose[:3] += origin; pose[2] += .04
                configs.append(Config(prefix+'/'+side+'Hand', pose[:3]))
            return dict(prefix=prefix, origin=np.array(origin), layout=copy.deepcopy(layout),
                        hands=configs, sensors=[], wire=None)
        art.author_original_scene = author
        def simulation():
            stage = Usd.Stage.CreateInMemory()
            physics = UsdPhysics.Scene.Define(stage, '/Actual/PhysicsScene')
            PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
            external = UsdGeom.Cube.Define(stage, '/GlobalCube')
            UsdPhysics.CollisionAPI.Apply(external.GetPrim()).CreateCollisionEnabledAttr(True)
            return types.SimpleNamespace(stage=stage, cfg=types.SimpleNamespace(physics_prim_path='/Actual/PhysicsScene'))
        origins = np.array([[0.,0.,0.], [123.1234567890123,-51.2345678901234,0.]], dtype=np.float64)
        layout = dict(hand_roots={side: np.array([.548675123456,-.05123456789,.2,1,0,0,0]) for side in ('left','right')},
                      table_center=np.array([0.,0.,-.1]))
        support_scene = dict(supports=[dict(name='end', center_m=[0.,.2,.1], size_m=[.02,.01,.1])])
        sim = simulation()
        batch = art.spawn_regex_articulations(sim, None, layout, {}, origins, build_mode='clone', support_scene=support_scene)
        assert len(batch.hands) == 2 and batch.env_ids == (0,1)
        reference = simulation()
        baseline = art.spawn_regex_articulations(reference, None, layout, {}, origins, build_mode='reference', support_scene=support_scene)
        for i, prefix in enumerate(batch.prefixes):
            prim = sim.stage.GetPrimAtPath(prefix)
            assert [cfg.prim_path for cfg in batch.scenes[i]['hands']] == [prefix+'/'+side+'Hand' for side in ('left','right')]
            reference_prims = {str(p.GetPath()): p for p in Usd.PrimRange(reference.stage.GetPrimAtPath(prefix))}
            actual_prims = {str(p.GetPath()): p for p in Usd.PrimRange(prim)}
            assert actual_prims.keys() == reference_prims.keys()
            cache = UsdGeom.XformCache()
            for path, actual in actual_prims.items():
                expected = reference_prims[path]
                assert actual.GetTypeName() == expected.GetTypeName()
                assert actual.GetAppliedSchemas() == expected.GetAppliedSchemas()
                for attr in expected.GetAttributes():
                    if not attr.GetName().startswith('xformOp'):
                        assert actual.GetAttribute(attr.GetName()).Get() == attr.Get()
                if path != prefix and actual.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(expected).GetOrderedXformOps():
                    assert cache.GetLocalToWorldTransform(actual) == cache.GetLocalToWorldTransform(expected), (path, cache.GetLocalToWorldTransform(actual), cache.GetLocalToWorldTransform(expected))
                for rel in expected.GetRelationships():
                    assert actual.GetRelationship(rel.GetName()).GetTargets() == rel.GetTargets()
            assert np.array_equal(batch.scenes[i]['origin'], origins[i])
            assert np.array_equal(np.asarray(prim.GetAttribute('xformOp:translate').Get()), np.zeros(3))
            assert prim.GetAttribute('xformOp:translate').GetTypeName() == Sdf.ValueTypeNames.Double3
            for descendant in Usd.PrimRange(prim):
                for rel in descendant.GetRelationships():
                    assert all(str(p).startswith(prefix + '/') for p in rel.GetTargets()), (descendant, rel.GetName())
            for side in ('left','right'):
                hand = prefix + '/' + side + 'Hand'
                assert sim.stage.GetPrimAtPath(hand+'/wrist_fixed').HasAPI(UsdPhysics.ArticulationRootAPI)
                assert not sim.stage.GetPrimAtPath(hand+'/root_joint').IsActive()
                for name in ('wrist_x','wrist_y','wrist_z') + ANGULAR_JOINT_NAMES:
                    assert sim.stage.GetPrimAtPath(hand+'/'+name)
                base = sim.stage.GetPrimAtPath(hand+'/base_link')
                material, _ = UsdShade.MaterialBindingAPI(base).ComputeBoundMaterial('physics')
                assert str(material.GetPath()) == prefix+'/Material'
                position = UsdGeom.XformCache().GetLocalToWorldTransform(base).ExtractTranslation()
                expected = layout['hand_roots'][side][:3].copy()
                expected += origins[i]; expected[2] += .04
                assert np.array_equal(np.asarray(position), np.array(to_tuple(tuple(expected))))
            support = sim.stage.GetPrimAtPath(prefix+'/EndSupports/end')
            assert support.HasAPI(UsdPhysics.CollisionAPI) and not support.HasAPI(UsdPhysics.RigidBodyAPI)
            material, _ = UsdShade.MaterialBindingAPI(support).ComputeBoundMaterial('physics')
            assert str(material.GetPath()) == prefix+'/EndSupports/PhysicsMaterial'
        scenes.isolate_scenes(sim.stage, batch.scenes, physics_scene_path=sim.cfg.physics_prim_path)
        physics = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
        assert physics.GetInvertCollisionGroupFilterAttr().Get()
        groups = list(sim.stage.GetPrimAtPath('/World/collisionGroups').GetChildren())
        assert len(groups) == 3
        global_group = next(g for g in groups if g.GetName() == 'global_group')
        global_api = UsdPhysics.CollisionGroup(global_group)
        assert global_api.GetCollidersCollectionAPI().GetIncludesRel().GetTargets() == [Sdf.Path('/GlobalCube')]
        for group, prefix in zip([g for g in groups if g != global_group], batch.prefixes):
            api = UsdPhysics.CollisionGroup(group)
            assert api.GetCollidersCollectionAPI().GetIncludesRel().GetTargets() == [Sdf.Path(prefix)]
            assert api.GetFilteredGroupsRel().GetTargets() == [group.GetPath(), global_group.GetPath()]
            assert group.GetPath() in global_api.GetFilteredGroupsRel().GetTargets()
        try: scenes.isolate_scenes(sim.stage, batch.scenes, physics_scene_path=sim.cfg.physics_prim_path)
        except ValueError: pass
        else: raise AssertionError('existing collision groups were silently reinterpreted')
        batch = art.spawn_regex_articulations(simulation(), None, layout, {}, origins, build_mode='replicated', support_scene=support_scene)
        assert callbacks
        batch.release_replication()
        assert not callbacks
        try: art.spawn_regex_articulations(simulation(), None, {}, {}, origins, env_ids=(3,7), build_mode='replicated')
        except ValueError: pass
        else: raise AssertionError('noncontiguous replication IDs accepted')
        occupied = simulation()
        UsdGeom.Cube.Define(occupied.stage, '/World/envs/Existing')
        try: art.spawn_regex_articulations(occupied, None, {}, {}, origins, build_mode='replicated')
        except ValueError: pass
        else: raise AssertionError('extra child under replication exclusion root accepted')
        assert occupied.stage.GetPrimAtPath('/World/envs/Existing')
        print('real USD clone/filter contracts passed; native physics not exercised')
    """)
    result = subprocess.run([sys.executable, '-B', '-c', script, str(cloner), str(schema / 'pxr')],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr



def test_near_wire_overlay_preserves_rod_and_binds_reset_pose_to_scene():
    from dh116.env.contracts import linea_coupl__ACTIVE, expand
    from dh116.env.scene import load_task_scene, near_wire_scene_inputs

    base, reset = load_task_scene('pinch', full_grasp=True)
    inputs = base.scene_inputs
    seed = inputs['authored_seed']
    poses = np.asarray(seed['reset_root_pose_w']).copy()
    poses[:, 2] -= 0.003
    indices = [seed['joint_names'].index(name) for name in linea_coupl__ACTIVE]
    active = np.asarray(seed['joint_positions'])[:, indices]
    candidate = dict(source_scene_sha256=inputs['manifest_sha256'], root_pose_wxyz=poses.tolist(),
                     active_q_rad=active.tolist())
    near = near_wire_scene_inputs(inputs, candidate)
    np.testing.assert_allclose(near['authored_seed']['reset_root_pose_w'], poses, rtol=0, atol=1e-12)
    np.testing.assert_allclose(near['authored_seed']['joint_positions'], [expand(row) for row in active])
    np.testing.assert_array_equal(near['reset']['rod_reset']['node_positions'], reset['rod_reset']['node_positions'])
    assert near['archive']['task']['support_scene'] == base.task['support_scene']
    assert near['manifest_sha256'] != inputs['manifest_sha256']
    assert near['archive']['task_seed_sha256'] == near['authored_seed']['sha256']
    candidate['source_scene_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='source scene'):
        near_wire_scene_inputs(inputs, candidate)
