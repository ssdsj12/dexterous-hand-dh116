"""Contact probes must remain small, inference-only and uncertifiable."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import pytest
import torch
from test_vision_action_derivation import source_checkpoint
from test_vision_evaluation_trials import manifest
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import _digest
from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import validate_vision_evaluation_options


def arguments(tmp_path, *, probe=True, split='test', n=4):
    source, saved = source_checkpoint(tmp_path)
    saved['contract'].update(camera_mode='fixed_per_env', camera_batch_size=2,
        camera_batching='fixed_per_environment_real_RGBD', camera_resolution_wh=[224, 96])
    torch.save(saved, source)
    value = manifest(n, split)
    value['metadata']['camera_sha256'] = saved['contract']['camera_config_sha256']
    value['metadata']['physics_sha256'] = _digest(saved['contract']['privileged_critic_contract'])
    path = tmp_path/'probe_manifest.json'; path.write_text(json.dumps(value))
    return SimpleNamespace(vision_evaluation_checkpoint=source, vision_evaluation_manifest=path,
        vision_evaluation_max_control_steps=120, output=tmp_path/'probe_output', vision_student=True,
        tensor_pipeline=True, headless=True, num_envs=n*2, vision_camera_mode='fixed',
        vision_width=224, vision_height=96, vision_visibility_mode='all', resume=None,
        grasp_contact_diagnostics=True, grasp_contact_probe=probe)


def test_small_probe_is_explicit_inference_only_without_lowering_production_minimum(tmp_path):
    args = arguments(tmp_path)
    validate_vision_evaluation_options(args)
    assert not args.output.exists()
    args.grasp_contact_probe = False
    with pytest.raises(ValueError): validate_vision_evaluation_options(args)


def test_probe_cannot_reduce_validation_split_trial_count(tmp_path):
    with pytest.raises(ValueError): validate_vision_evaluation_options(arguments(tmp_path, split='validation'))


def test_contact_flags_cannot_silently_start_ppo_when_checkpoint_missing(tmp_path):
    args = SimpleNamespace(vision_evaluation_checkpoint=None, vision_evaluation_manifest=None,
        grasp_contact_diagnostics=True, grasp_contact_probe=True)
    with pytest.raises(ValueError): validate_vision_evaluation_options(args)


@pytest.mark.parametrize('field,value', [('resume', 'historical.pt'),
    ('grasp_contact_diagnostics', False), ('vision_evaluation_max_control_steps', 181),
    ('num_envs', 200)])
def test_invalid_probe_preflight_rejects_before_output_write(tmp_path, field, value):
    args = arguments(tmp_path); setattr(args, field, value)
    with pytest.raises(ValueError): validate_vision_evaluation_options(args)
    assert not args.output.exists()


def test_probe_parser_flags_are_explicit():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    flags = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
        and n.args and isinstance(n.args[0], ast.Constant)}
    assert {'--grasp-contact-diagnostics', '--grasp-contact-probe'} <= flags
