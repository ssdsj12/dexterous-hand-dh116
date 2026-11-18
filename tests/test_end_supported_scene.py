"""Offline scene contracts. Never import Isaac or launch a simulation.

The module can be copied beside this test into /tmp/agent_A for isolated tests.
Native USD checks use an in-memory stage only; GpuWorld checks execute its AST.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DH116 = ROOT / 'src/go2_pvcnn/control/dh116'
MODULE = DH116 / 'tabletop_end_supported_scene.py'
if not MODULE.exists() and (Path(__file__).parent / MODULE.name).exists():
    MODULE = Path(__file__).parent / MODULE.name
GPU = DH116 / 'tabletop_gpu_isaac.py'


def scene_module():
    if not MODULE.exists():
        raise AssertionError('end support scene module is not implemented')
    spec = importlib.util.spec_from_file_location('agent_A_end_support_scene', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layout():
    return dict(hand_roots={s: np.array([.52, y, .414, 1., 0., 0., 0.])
                           for s, y in (('left', .197), ('right', -.197))},
                active_joints={s: np.arange(6.) for s in ('left', 'right')},
                wire_center=(.43, 0., .308), wire_axis='Y',
                wire_length_m=.5, wire_radius_m=.008,
                table_center=(.6, 0., .25), table_size=(.8, 1.2, .1))


def seed():
    return dict(roots=np.array([layout()['hand_roots'][s] for s in ('left', 'right')]),
                wire_pose=np.array([.43, 0., .308, 1., 0., 0., 0.]),
                wire_velocity=np.arange(6.), root_xy=np.array([.01, -.02]),
                root_offset=.004, joint_positions=np.zeros((2, 6)),
                joint_velocities=np.zeros((2, 6)), initial_command=np.zeros(12))


class GeometryTests(unittest.TestCase):
    def test_zero_is_exact_deep_copy_without_geometry_validation(self):
        m = scene_module()
        original = layout()
        original['hand_roots']['left'] = original['hand_roots']['left'].astype(np.float32)
        configured, metadata = m.plan_end_support_scene(original, 0.)
        self.assertIsNone(metadata)
        self.assertEqual(type(configured['wire_center']), tuple)
        self.assertEqual(configured['wire_center'], original['wire_center'])
        for side in ('left', 'right'):
            self.assertEqual(configured['hand_roots'][side].dtype,
                             original['hand_roots'][side].dtype)
            self.assertEqual(configured['hand_roots'][side].tobytes(),
                             original['hand_roots'][side].tobytes())
            self.assertIsNot(configured['hand_roots'][side], original['hand_roots'][side])
        self.assertEqual(m.plan_end_support_scene({'legacy': 1}, 0.), ({'legacy': 1}, None))

    def test_sixty_mm_translates_hands_and_wire_but_keeps_table_and_joints(self):
        m = scene_module()
        original = layout()
        configured, info = m.plan_end_support_scene(original, .06)
        np.testing.assert_allclose(np.asarray(configured['wire_center']) - original['wire_center'],
                                   [0., 0., .06], atol=1e-15)
        for side in ('left', 'right'):
            np.testing.assert_allclose(configured['hand_roots'][side] - original['hand_roots'][side],
                                       [0., 0., .06, 0., 0., 0., 0.], atol=1e-15)
            np.testing.assert_array_equal(configured['active_joints'][side], original['active_joints'][side])
        self.assertEqual(configured['table_center'], original['table_center'])
        self.assertEqual(configured['table_size'], original['table_size'])
        self.assertEqual(original['wire_center'][2], .308)
        self.assertAlmostEqual(info['lift_reference_bottom_m'], .36)
        self.assertAlmostEqual(info['dz_m'], .06)
        json.dumps(info, allow_nan=False)

    def test_two_narrow_supports_touch_bottom_and_clear_existing_grasp_region(self):
        _, info = scene_module().plan_end_support_scene(layout(), .06)
        self.assertEqual(len(info['supports']), 2)
        intervals = []
        for support in info['supports']:
            x, y, z = support['center_m']
            sx, sy, sz = support['size_m']
            self.assertAlmostEqual(x, .43)
            self.assertAlmostEqual(z + sz / 2, .36)
            self.assertAlmostEqual(z - sz / 2, .30)
            self.assertLessEqual(sy, .012)
            self.assertLessEqual(sx, .04)
            self.assertGreater(abs(y) - sy / 2, .22)
            self.assertLessEqual(abs(y) + sy / 2, .25 + 1e-12)
            intervals.append((y - sy / 2, y + sy / 2))
        intervals.sort()
        self.assertAlmostEqual(info['central_gap_m'], intervals[1][0] - intervals[0][1])
        self.assertGreaterEqual(info['central_gap_m'], .46)

    def test_dimensions_and_center_are_parameters_not_assumed_point_four_m(self):
        original = layout()
        original['wire_center'] = (.5, .1, .32)
        configured, info = scene_module().plan_end_support_scene(
            original, .06, wire_length_m=.7, wire_radius_m=.02)
        self.assertEqual(info['wire_length_m'], .7)
        self.assertEqual(info['wire_radius_m'], .02)
        self.assertAlmostEqual(configured['wire_center'][2], .38)
        centers = [s['center_m'][1] for s in info['supports']]
        self.assertLess(min(centers), -.23)
        self.assertGreater(max(centers), .43)

    def test_clearance_is_measured_from_table_even_if_original_bottom_has_small_gap(self):
        original = layout()
        original['wire_center'] = (.43, 0., .3083)
        configured, info = scene_module().plan_end_support_scene(original, .06)
        self.assertAlmostEqual(info['dz_m'], .0597)
        self.assertAlmostEqual(configured['wire_center'][2] - .008, .36)

    def test_invalid_clearance_or_geometry_fails_before_authoring(self):
        m = scene_module()
        for value in (-.01, float('nan'), float('inf')):
            with self.subTest(clearance=value), self.assertRaises(ValueError):
                m.plan_end_support_scene(layout(), value)
        for key, value in (('wire_length_m', 0.), ('wire_radius_m', -.1),
                           ('wire_length_m', float('inf'))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.plan_end_support_scene(layout(), .06, **{key: value})
        original = layout()
        original['wire_axis'] = 'X'
        with self.assertRaises(ValueError):
            m.plan_end_support_scene(original, .06)

    def test_seed_translation_is_once_per_bind_and_does_not_mutate_input(self):
        m = scene_module()
        _, info = m.plan_end_support_scene(layout(), .06)
        original = seed()
        for _ in range(2):
            shifted = m.translate_end_support_seed(original, info)
            np.testing.assert_allclose(shifted['wire_pose'] - original['wire_pose'],
                                       [0., 0., .06, 0., 0., 0., 0.], atol=1e-15)
            np.testing.assert_allclose(shifted['roots'][:, 2] - original['roots'][:, 2], .06)
            for key in ('wire_velocity', 'root_xy', 'root_offset', 'joint_positions',
                        'joint_velocities', 'initial_command'):
                np.testing.assert_array_equal(shifted[key], original[key])
        self.assertEqual(original['wire_pose'][2], .308)
        np.testing.assert_array_equal(m.translate_end_support_seed(original, None)['wire_pose'],
                                      original['wire_pose'])

    def test_geometry_comes_from_actual_spawn_configuration(self):
        m = scene_module()
        self.assertEqual(m.wire_geometry_from_spawn(SimpleNamespace(
            radius=.02, height=.7, axis='Y')), (.7, .02))
        with self.assertRaises(ValueError):
            m.wire_geometry_from_spawn(SimpleNamespace(radius=.008, height=.5, axis='X'))


def gpu_scope():
    if not GPU.exists():
        raise unittest.SkipTest('GpuWorld source deliberately not uploaded to isolated /tmp/agent_A')
    module = scene_module()
    tree = ast.parse(GPU.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GpuWorld')
    scope = dict(np=np, copy=copy, LEGACY_LOAD_PARAMETERS=object(),
                 torch=SimpleNamespace(float32=np.float32))
    scope.update({name: getattr(module, name) for name in (
        'plan_end_support_scene', 'translate_end_support_seed',
        'wire_geometry_from_spawn', 'author_end_supports')})
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(GPU), 'exec'), scope)
    return scope, cls


class WorldIntegrationTests(unittest.TestCase):
    def make_world(self, clearance=None):
        scope, _ = gpu_scope()
        events = []
        class Cloner:
            def __init__(self, **kwargs): pass
            def define_base_env(self, path): pass
            def generate_paths(self, prefix, n): return [f'{prefix}_{i}' for i in range(n)]
            def clone(self, **kwargs):
                events.append('clone')
                return np.array([[0., 0., .5], [2., 0., 1.]])
        cfg = SimpleNamespace(replace=lambda **kw: SimpleNamespace(**kw))
        def spawn(sim, args, configured, sources, prefix, origin):
            events.append(('spawn', copy.deepcopy(configured)))
            return dict(hands=[SimpleNamespace(cfg=cfg)] * 2, wire=SimpleNamespace(cfg=cfg))
        scope.update(GridCloner=Cloner, spawn_scene=spawn,
                     wire_spawn_configuration=lambda args: SimpleNamespace(radius=.008, height=.5, axis='Y'),
                     author_end_supports=lambda stage, prefix, info: events.append(('supports', info)),
                     Usd=SimpleNamespace(PrimRange=lambda prim: []),
                     Articulation=lambda cfg: cfg, RigidObject=lambda cfg: cfg,
                     ContactSensor=lambda cfg: cfg, ContactSensorCfg=lambda **kw: kw)
        args = SimpleNamespace(control_mode='adaptive_grasp', num_envs=2, physics_hz=480,
                               device='cpu', env_spacing_m=2.)
        if clearance is not None:
            args.end_support_clearance_m = clearance
        sim = SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=lambda path: path))
        return scope['GpuWorld'](sim, args, layout(), {}), events, scope

    def test_supports_and_translated_spawn_exist_before_clone(self):
        world, events, _ = self.make_world(.06)
        self.assertEqual([e[0] if isinstance(e, tuple) else e for e in events],
                         ['spawn', 'supports', 'clone'])
        self.assertAlmostEqual(events[0][1]['wire_center'][2], .368)
        self.assertAlmostEqual(world.layout['hand_roots']['left'][2], .474)
        np.testing.assert_allclose(world.lift_reference_bottom_m, [.36, .36])
        self.assertEqual(world.end_support_scene['clearance_m'], .06)

    def test_missing_argument_and_zero_preserve_old_scene_without_support_authoring(self):
        for value in (None, 0.):
            world, events, _ = self.make_world(value)
            self.assertEqual(len(events), 2)
            self.assertIsNone(world.end_support_scene)
            self.assertEqual(world.layout['wire_center'], layout()['wire_center'])
            np.testing.assert_array_equal(world.layout['hand_roots']['left'],
                                          layout()['hand_roots']['left'])
            np.testing.assert_allclose(world.lift_reference_bottom_m, [.30, .30])

    def test_bind_uses_translated_seed_without_accumulating_height(self):
        world, _, scope = self.make_world(.06)
        # Execute the production bind prefix, stopping before native view access.
        tree = ast.parse(GPU.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GpuWorld')
        bind = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'bind')
        prefix = []
        for node in bind.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and
                                                    t.attr == 'mounts' for t in node.targets):
                break
            prefix.append(node)
        for _ in range(2):
            original = seed()
            exec(compile(ast.Module(body=prefix, type_ignores=[]), str(GPU), 'exec'),
                 dict(scope, self=world, seed=original))
            self.assertAlmostEqual(world.seed['wire_pose'][2], .368)
            self.assertEqual(original['wire_pose'][2], .308)

    def test_production_partial_reset_keeps_raised_roots_wire_and_local_reference(self):
        world, _, scope = self.make_world(.06)
        world.seed = scene_module().translate_end_support_seed(seed(), world.end_support_scene)
        class Array(np.ndarray):
            def clone(self): return self.copy()
            def cpu(self): return self
            def numpy(self): return np.asarray(self)
        def tensor(value, dtype=None): return np.asarray(value, dtype=dtype).view(Array)
        scope['torch'] = SimpleNamespace(long=np.int64, zeros_like=np.zeros_like)
        world.tensor = tensor
        world.mounts = None
        world.ids = world.named_ids = [list(range(6)), list(range(6))]
        world.hands = [Mock() for _ in range(2)]
        for hand in world.hands:
            hand.data.default_joint_pos = tensor(np.zeros((2, 6)))
            hand.actuators = {'active': SimpleNamespace(stiffness=np.zeros((2, 6)))}
        for name, shape in (('root_targets', (2, 2, 7)), ('commands', (2, 2, 6)),
                            ('gains', (2, 2)), ('previous_gains', (2, 2)),
                            ('palms', (2, 2, 3)), ('force', (2, 4)), ('force_vectors', (2, 4, 3))):
            setattr(world, name, np.full(shape, -9.))
        world.wire = Mock()
        world.sensors = []
        world.sim = SimpleNamespace(physics_sim_view=Mock())
        world.refresh = Mock()
        for _ in range(2):
            world.reset_slots([1], [.004])
            self.assertAlmostEqual(world.wire.write_root_pose_to_sim.call_args.args[0][0, 2], 1.368)
            for hand in world.hands:
                self.assertAlmostEqual(hand.write_root_pose_to_sim.call_args.args[0][0, 2], 1.478)
            np.testing.assert_array_equal(world.root_targets[0], -9.)
            np.testing.assert_allclose(world.lift_reference_bottom_m, [.36, .36])


class NativeUsdTests(unittest.TestCase):
    def test_external_usd_geometry_uses_actual_envelope_and_rejects_off_center_assets(self):
        m = scene_module()
        try:
            from pxr import Gf, Usd, UsdGeom
        except ImportError:
            self.skipTest('pxr unavailable; no Isaac import attempted')
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        rod = UsdGeom.Cylinder.Define(stage, '/Rod')
        rod.CreateRadiusAttr(.02)
        rod.CreateHeightAttr(.7)
        rod.CreateAxisAttr('Y')
        stage.SetDefaultPrim(rod.GetPrim())
        with patch.object(Usd.Stage, 'Open', return_value=stage):
            length, radius = m.wire_geometry_from_spawn(SimpleNamespace(usd_path='in-memory-only'))
            self.assertAlmostEqual(length, .7)
            self.assertAlmostEqual(radius, .02)
            rod.AddTranslateOp().Set(Gf.Vec3d(.1, 0., 0.))
            with self.assertRaisesRegex(ValueError, 'centered Y-axis'):
                m.wire_geometry_from_spawn(SimpleNamespace(usd_path='in-memory-only'))

    def test_supports_are_persistent_static_colliders_and_do_not_edit_table_or_wire(self):
        m = scene_module()
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest('pxr unavailable; no Isaac import attempted')
        stage = Usd.Stage.CreateInMemory()
        table = UsdGeom.Cube.Define(stage, '/World/env_0/Table').GetPrim()
        UsdPhysics.CollisionAPI.Apply(table)
        wire = UsdGeom.Cylinder.Define(stage, '/World/env_0/Wire').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(wire).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(wire).CreateMassAttr(.12)
        def attributes(prim):
            return {a.GetName(): a.Get() for a in prim.GetAttributes()}
        before = {name: attributes(stage.GetPrimAtPath('/World/env_0/' + name))
                  for name in ('Table', 'Wire')}
        _, info = m.plan_end_support_scene(layout(), .06)
        paths = m.author_end_supports(stage, '/World/env_0', info)
        self.assertEqual(len(paths), 2)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertTrue(prim.GetAttribute('physxCollision:contactOffset').HasAuthoredValue())
            self.assertAlmostEqual(prim.GetAttribute('physxCollision:contactOffset').Get(), .001)
            self.assertAlmostEqual(prim.GetAttribute('physxCollision:restOffset').Get(), 0.)
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            self.assertAlmostEqual(bounds.GetMin()[2], .30)
            self.assertAlmostEqual(bounds.GetMax()[2], .36)
        for name in before:
            self.assertEqual(attributes(stage.GetPrimAtPath('/World/env_0/' + name)), before[name])
        self.assertFalse(any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
