"""Evaluation preflight is read-only, excludes training, and precedes Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import pytest
from test_vision_action_derivation import source_checkpoint
from test_vision_evaluation_trials import manifest
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import _digest


def test_preflight_uses_evaluation_projection_without_exact_resume_guard(tmp_path):
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract')
    source,saved=source_checkpoint(tmp_path)
    import torch
    saved['contract'].update(camera_mode='fixed_per_env',camera_batch_size=2,
        camera_batching='fixed_per_environment_real_RGBD',camera_resolution_wh=[224,96])
    torch.save(saved,source)
    value=manifest(100,'validation')
    value['metadata']['camera_sha256']=saved['contract']['camera_config_sha256']
    value['metadata']['physics_sha256']=_digest(saved['contract']['privileged_critic_contract'])
    frozen=tmp_path/'frozen.json'; frozen.write_text(json.dumps(value))
    args=SimpleNamespace(vision_evaluation_checkpoint=source,vision_evaluation_manifest=frozen,
        vision_evaluation_max_control_steps=600,output=tmp_path/'independent',vision_student=True,
        tensor_pipeline=True,headless=True,num_envs=200,vision_camera_mode='fixed',
        vision_width=224,vision_height=96,vision_visibility_mode='all',resume=None)
    assert hasattr(module,'validate_vision_evaluation_options'), 'evaluation preflight missing'
    module.validate_vision_evaluation_options(args)
    assert not args.output.exists()
    for field in ('resume','surface_mpc','teacher_search_cases','actor_initialization'):
        setattr(args,field,'forbidden')
        with pytest.raises(ValueError,match='exclusive'):
            module.validate_vision_evaluation_options(args)
        setattr(args,field,None)


def test_native_cli_exposes_eval_and_calls_preflight_before_output_or_isaac():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    flags={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args
        and isinstance(n.args[0],ast.Constant)}
    assert {'--vision-evaluation-checkpoint','--vision-evaluation-manifest',
            '--vision-evaluation-max-control-steps'} <= flags
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='validate_vision_evaluation_options']
    assert calls
    writes=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and
        ((isinstance(n.func,ast.Attribute) and n.func.attr=='mkdir') or
         (isinstance(n.func,ast.Name) and n.func.id=='AppLauncher'))]
    assert calls[0].lineno<min(writes)
