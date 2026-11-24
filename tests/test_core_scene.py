import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pytest

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
    """The native author uses the proposal; solver Reset keeps its measured packet."""
    from types import SimpleNamespace
    from dh116.env.scene import load_task_scene
    from dh116.env.scene import prepared_scene
    from dh116.env.lifecycle import NativeOwners
    import dh116.hand.articulation as tabletop_regex_articulation
    import dh116.hand.articulation as tabletop_regex_task_setup
    root = Path(__file__).resolve().parents[1]
    (archive, measured) = load_task_scene('pinch', root=root)
    inputs = archive.scene_inputs
    assert inputs['grasp'] == archive.style == 'pinch' and inputs['qualification'] == {'native': False, 'teacher': False, 'grasp': False} and (measured is inputs['reset'])
    pair = archive.pair
    assert pair['reset_seed']['sha256'] == 'c19b0c971d76e4e613b7ca3ed12d06012514fd3e7a08a3823daee927e634ce81' and measured['seed']['sha256'] == '9ced76db3e4444107271296993f985b97dc034c40e5140f483288de5546b3bb5' and (pair['reset_seed']['reset_root_pose_w'] != measured['seed']['reset_root_pose_w'])
    proposed = np.asarray(pair['rod_seed']['proposed_nodes_m'], dtype=np.float64)
    observed = np.asarray(measured['rod_reset']['node_positions'])
    assert proposed.shape == (41, 3) and observed.shape == (1, 41, 3) and np.array_equal(proposed.astype(np.float32), observed[0]) and (not np.array_equal(proposed, observed[0]))

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
