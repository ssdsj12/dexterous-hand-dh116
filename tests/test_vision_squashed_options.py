"""Launcher validation executes without starting Isaac or touching outputs."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', ['run_dh116_surface_student.py', 'train_dh116_tabletop_gpu_ppo.py'])
def test_launcher_exposes_opt_in_distribution_and_runs_validation_before_writes(script):
    tree = ast.parse((ROOT/'scripts'/script).read_text())
    additions = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
        and n.args and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == '--vision-action-distribution']
    assert len(additions) == 1
    opts = {k.arg: ast.literal_eval(k.value) for k in additions[0].keywords}
    assert opts['default'] == 'gaussian_clip'
    assert opts['choices'] == ('gaussian_clip', 'tanh_gaussian')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'validate_vision_action_options']
    assert len(calls) >= 1
    first_write = min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Attribute) and n.func.attr == 'mkdir')
             or (isinstance(n.func, ast.Name) and n.func.id == 'AppLauncher')))
    assert calls[0].lineno < first_write


@pytest.fixture
def v3_launch_checkpoint(tmp_path, monkeypatch):
    import os
    import runpy
    import torch
    from types import SimpleNamespace
    from test_vision_training import fixture, Camera
    from go2_pvcnn.control.dh116.tabletop_vision_training import run_vision_training
    from go2_pvcnn.control.dh116.tabletop_load_parameters import read_load_parameters
    cfg = ROOT/'configs/dh116_training'
    script = ROOT/'scripts/run_dh116_surface_student.py'
    calls = []
    common = ['--num-envs', '2', '--updates', '1', '--rollout-steps', '8',
        '--vision-student', '--surface-feedback', '--vision-width', '160', '--vision-height', '120',
        '--depth-camera-config', str(cfg/'front_depth_camera_v2.json'),
        '--pregrasp-candidates', str(cfg/'effective_pad_center_pregrasp_v1.json'),
        '--mixed-pregrasp-candidates', str(cfg/'effective_pad_triad_pregrasp_v1.json')]
    monkeypatch.setattr(os, 'execv', lambda program, command: calls.append(command))
    seed_output = tmp_path/'materialize'
    monkeypatch.setattr(sys, 'argv', [str(script), '--output', str(seed_output), *common])
    runpy.run_path(str(script), run_name='__main__')
    command = calls[0]
    dw, task, env, args = fixture(tmp_path, maximum=0)
    args.rollout_steps = 8
    args.vision_action_distribution = 'tanh_gaussian'
    args.surface_reward_profile = 'baseline'
    args.depth_camera_config = cfg/'front_depth_camera_v2.json'
    args.grasp_initialization = Path(command[command.index('--grasp-initialization')+1])
    args.mixed_grasp_initialization = Path(command[command.index('--mixed-grasp-initialization')+1])
    args.load_profile = Path(command[command.index('--load-profile')+1])
    torch.set_num_threads(1)
    run_vision_training(SimpleNamespace(), dw, task, env, args,
        {'observation_size':371,'load_parameters':read_load_parameters(args.load_profile).as_dict()},camera=Camera(2))
    return args.output/'vision_student_ppo.pt', args, common


@pytest.mark.parametrize('name,value', [
    ('num_envs',4), ('rollout_steps',16), ('vision_width',159), ('vision_height',119),
    ('vision_camera_mode','fixed'), ('vision_camera_batch_size',1),
    ('vision_visibility_mode','batch'), ('vision_ppo_microbatch_size',64),
    ('vision_inference_batch_size',64), ('depth_camera_config','changed'),
    ('grasp_initialization','changed'), ('mixed_grasp_initialization','changed')])
def test_resume_options_reject_every_known_static_change(v3_launch_checkpoint, tmp_path, name, value):
    import copy
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    source, args, _ = v3_launch_checkpoint
    current = copy.copy(args); current.resume = source
    if value == 'changed':
        value = tmp_path/'changed.json'; value.write_text('{}')
    setattr(current,name,value)
    with pytest.raises(ValueError, match='contract'):
        validate_vision_action_options(current)


def test_resume_preflight_accepts_unchanged_and_native_default_rollout(v3_launch_checkpoint):
    import copy
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    source, args, _ = v3_launch_checkpoint
    current = copy.copy(args); current.resume = source
    validate_vision_action_options(current)


@pytest.mark.parametrize('field,value', [('physics_hz',240), ('solver_type',1),
    ('position_iterations',16), ('velocity_iterations',4),
    ('passive_velocity_limit_rad_s',3.), ('env_spacing_m',3.),
    ('reset_height_jitter_m',.0001), ('object_kind','hollow_straw'),
    ('mimic_frequency',200), ('pose_summary','changed'), ('load_profile','changed'),
    ('surface_reward_profile','baseline'), ('object_material','default'), ('hand_material','default')])
def test_resume_preflight_rejects_known_physics_and_authored_config_changes(v3_launch_checkpoint, tmp_path, field, value):
    import copy
    import hashlib
    import torch
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    from go2_pvcnn.control.dh116.tabletop_vision_training import _contract
    from go2_pvcnn.control.dh116.tabletop_material import material_contract, hand_material_contract
    from go2_pvcnn.control.dh116.tabletop_load_parameters import read_load_parameters
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import ablation_contract
    source, args, _ = v3_launch_checkpoint
    current = copy.copy(args)
    current.pose_summary = ROOT/'configs/dh116_training/pose_reference.json'
    current.load_profile = ROOT/'configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json'
    expected = dict(physics_hz=480,solver_type=0,position_iterations=64,velocity_iterations=16,
        passive_velocity_limit_rad_s=2.97,env_spacing_m=2.,reset_height_jitter_m=0.,object_kind='solid_cylinder')
    for name,item in expected.items():setattr(current,name,item)
    current.mimic_frequency = 400
    current.surface_reward_profile = 'support_deficit'
    current.object_material = 'extreme_grip'; current.hand_material = 'grip_compliant'
    critic_fields = dict(expected); critic_fields.pop('object_kind')
    critic = dict(critic_fields,observation_size=371,mimic_natural_frequency_s_inv=400.,
        pose_sha256=hashlib.sha256(current.pose_summary.read_bytes()).hexdigest(),
        load_parameters=read_load_parameters(current.load_profile).as_dict(),
        surface_reward_ablation=ablation_contract(current.surface_reward_profile),
        **material_contract(current.object_material),**hand_material_contract(current.hand_material))
    saved = torch.load(source,weights_only=False)
    saved['contract'] = _contract(current,critic)
    torch.save(saved,source)
    current.resume = source
    validate_vision_action_options(current)  # A native-like authored contract is accepted first.
    if value == 'changed':
        value = tmp_path/'changed.json'
        if field == 'load_profile':
            payload = __import__('json').loads(current.load_profile.read_text())
            payload['parameters']['effort_limit_nm'] = .19
            value.write_text(__import__('json').dumps(payload))
        else:value.write_text('{}')
    setattr(current,field,value)
    with pytest.raises(ValueError,match='contract'):
        validate_vision_action_options(current)


@pytest.mark.parametrize('case', ['radius_added', 'density_added', 'radius_changed', 'density_changed',
    'object_removed', 'solver_added', 'solver_removed', 'support_added', 'support_removed',
    'support_clearance_changed', 'support_radius_changed', 'object_material_added',
    'hand_material_added', 'reward_added', 'reward_metadata_changed', 'load_added', 'load_removed'])
def test_native_authored_field_additions_removals_and_changes_are_rejected(v3_launch_checkpoint, case):
    import copy
    import torch
    from go2_pvcnn.control.dh116.tabletop_vision_training import _contract
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import object_spec
    from go2_pvcnn.control.dh116.tabletop_solver_order import CONTRACT as SOLVER_ORDER
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import ablation_contract
    from go2_pvcnn.control.dh116.tabletop_load_parameters import read_load_parameters
    source, args, _ = v3_launch_checkpoint
    current = copy.copy(args)
    current.object_kind = 'solid_cylinder'
    current.solid_cylinder_radius_m = .008
    current.solid_cylinder_density_kg_m3 = None
    current.solve_articulation_contact_last = False
    current.end_support_clearance_m = 0.
    current.object_material = current.hand_material = 'default'
    current.surface_reward_profile = 'baseline'
    current.load_profile = None
    critic = dict(observation_size=371,end_support_scene=None)
    if case in ('radius_changed', 'density_changed', 'object_removed'):
        current.solid_cylinder_radius_m = .009
        _, _, physical = object_spec(ROOT,current.object_kind,solid_radius_m=.009)
        critic.update(physical)
    if case == 'solver_removed':
        current.solve_articulation_contact_last = True
        critic['articulation_solver_order'] = copy.deepcopy(SOLVER_ORDER)
    if case in ('support_removed', 'support_clearance_changed', 'support_radius_changed'):
        current.end_support_clearance_m = .06
        critic['end_support_scene'] = dict(clearance_m=.06,wire_radius_m=.008,wire_length_m=.5)
    if case == 'reward_metadata_changed':
        current.surface_reward_profile = 'support_deficit'
        critic['surface_reward_ablation'] = ablation_contract('support_deficit')
    if case == 'load_removed':
        current.load_profile = ROOT/'configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json'
        critic['load_parameters'] = read_load_parameters(current.load_profile).as_dict()
    saved = torch.load(source,weights_only=False)
    saved['contract'] = _contract(current,critic)
    torch.save(saved,source)
    current.resume = source
    validate_vision_action_options(current)  # Preserve the old, valid authored configuration.
    if case == 'radius_added': current.solid_cylinder_radius_m = .009
    elif case == 'density_added': current.solid_cylinder_density_kg_m3 = 200.
    elif case == 'radius_changed': current.solid_cylinder_radius_m = .01
    elif case == 'density_changed': current.solid_cylinder_density_kg_m3 = 250.
    elif case == 'object_removed': current.solid_cylinder_radius_m = .008
    elif case == 'solver_added': current.solve_articulation_contact_last = True
    elif case == 'solver_removed': current.solve_articulation_contact_last = False
    elif case == 'support_added': current.end_support_clearance_m = .06
    elif case == 'support_removed': current.end_support_clearance_m = 0.
    elif case == 'support_clearance_changed': current.end_support_clearance_m = .05
    elif case == 'support_radius_changed':
        # Keep object provenance valid: isolate the independently authored support radius.
        saved['contract']['privileged_critic_contract']['end_support_scene']['wire_radius_m'] = .009
        saved['contract'] = _contract(current,saved['contract']['privileged_critic_contract'])
        torch.save(saved,source)
    elif case == 'object_material_added': current.object_material = 'extreme_grip'
    elif case == 'hand_material_added': current.hand_material = 'grip_compliant'
    elif case == 'reward_added': current.surface_reward_profile = 'support_deficit'
    elif case == 'reward_metadata_changed':
        saved['contract']['privileged_critic_contract']['surface_reward_ablation']['weights']['surface_support_deficit'] = -99.
        saved['contract'] = _contract(current,saved['contract']['privileged_critic_contract'])
        torch.save(saved,source)
    elif case == 'load_added': current.load_profile = ROOT/'configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json'
    elif case == 'load_removed': current.load_profile = None
    with pytest.raises(ValueError,match='contract'):
        validate_vision_action_options(current)


@pytest.mark.parametrize('old_output', [False, True])
def test_native_cli_known_mismatch_rejects_before_output_and_app(v3_launch_checkpoint, tmp_path, monkeypatch, capsys, old_output):
    import runpy
    from types import ModuleType
    source, args, _ = v3_launch_checkpoint
    output = tmp_path/'native_rejected'
    if old_output:
        output.mkdir(); (output/'monitor.json').write_bytes(b'original monitor')
    script = ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py'
    isaaclab, app = ModuleType('isaaclab'), ModuleType('isaaclab.app')
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--device', default='cuda:0')
            parser.add_argument('--headless', action='store_true')
        def __init__(self, *a, **kw): raise AssertionError('Isaac must not launch')
    app.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules,'isaaclab',isaaclab)
    monkeypatch.setitem(sys.modules,'isaaclab.app',app)
    monkeypatch.setattr(sys,'argv',[str(script),'--pose-summary',str(tmp_path/'pose.json'),
        '--output',str(output),'--num-envs','4','--rollout-steps','8','--vision-width','160',
        '--vision-height','120','--vision-student','--vision-action-distribution','tanh_gaussian',
        '--depth-camera-config',str(args.depth_camera_config),'--grasp-initialization',str(args.grasp_initialization),
        '--mixed-grasp-initialization',str(args.mixed_grasp_initialization),'--resume',str(source)])
    with pytest.raises(SystemExit): runpy.run_path(str(script),run_name='__main__')
    assert 'contract field: num_envs' in capsys.readouterr().err
    if old_output:
        assert (output/'monitor.json').read_bytes() == b'original monitor'
        assert sorted(p.name for p in output.iterdir()) == ['monitor.json']
    else: assert not output.exists()


def test_front_resume_checks_actual_materialized_reset_bytes_before_any_write(v3_launch_checkpoint, tmp_path, monkeypatch):
    import os
    import runpy
    source, args, common = v3_launch_checkpoint
    script = ROOT/'scripts/run_dh116_surface_student.py'
    output = tmp_path/'front_valid'
    calls = []
    monkeypatch.setattr(os,'execv',lambda program,command:calls.append(command))
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(output),*common,
        '--vision-action-distribution','tanh_gaussian','--resume',str(source)])
    runpy.run_path(str(script),run_name='__main__')
    assert len(calls) == 1
    command = calls[0]
    assert Path(command[command.index('--grasp-initialization')+1]).read_bytes() == args.grasp_initialization.read_bytes()
    assert Path(command[command.index('--mixed-grasp-initialization')+1]).read_bytes() == args.mixed_grasp_initialization.read_bytes()
    changed = [*common]
    index = changed.index('--pregrasp-candidates')
    changed[index+1] = changed[changed.index('--mixed-pregrasp-candidates')+1]
    rejected = tmp_path/'front_rejected'
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(rejected),*changed,
        '--vision-action-distribution','tanh_gaussian','--resume',str(source)])
    with pytest.raises(SystemExit): runpy.run_path(str(script),run_name='__main__')
    assert not rejected.exists()
    assert not rejected.with_name(rejected.name+'_initialization.json').exists()
    assert not rejected.with_name(rejected.name+'_mixed_initialization.json').exists()
    assert len(calls) == 1


def test_front_launcher_forwards_distribution():
    source = (ROOT/'scripts/run_dh116_surface_student.py').read_text()
    assert "'--vision-action-distribution',args.vision_action_distribution" in source


def test_shared_option_validator_rejects_unsupported_combination_without_files(tmp_path):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    for changes in (dict(vision_student=False), dict(vision_resume_downsize=True)):
        options = dict(vision_student=True, vision_action_distribution='tanh_gaussian',
                       vision_resume_downsize=False, resume=None)
        options.update(changes)
        with pytest.raises(ValueError): validate_vision_action_options(SimpleNamespace(**options))
    assert not list(tmp_path.iterdir())


def test_native_cli_rejects_v2_resume_before_app_or_old_output_write(tmp_path, monkeypatch):
    import runpy
    from types import ModuleType
    import torch
    script = ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py'
    old = tmp_path/'old'; old.mkdir()
    marker = old/'monitor.json'; marker.write_bytes(b'preserve old run')
    source = tmp_path/'v2.pt'
    torch.save({'version': 'dh116-vision-student-ppo-v2-bounded'}, source)
    isaaclab, app = ModuleType('isaaclab'), ModuleType('isaaclab.app')
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--device', default='cuda:0')
            parser.add_argument('--headless', action='store_true')
        def __init__(self, *a, **kw): raise AssertionError('Isaac must not launch')
    app.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, 'isaaclab', isaaclab)
    monkeypatch.setitem(sys.modules, 'isaaclab.app', app)
    monkeypatch.setattr(sys, 'argv', [str(script), '--pose-summary', str(tmp_path/'pose.json'),
        '--output', str(old), '--vision-student', '--vision-action-distribution', 'tanh_gaussian',
        '--resume', str(source)])
    with pytest.raises(SystemExit) as error: runpy.run_path(str(script), run_name='__main__')
    assert error.value.code == 2
    assert marker.read_bytes() == b'preserve old run'
    assert sorted(p.name for p in old.iterdir()) == ['monitor.json']
