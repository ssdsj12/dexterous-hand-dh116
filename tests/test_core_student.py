"""Pure current Student contracts; CPU checks do not qualify native grasping."""
import importlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from dh116.rl.student_model import SensorBatch
from dh116.rl.student_model import StereoHistory
from dh116.rl.student_model import make_stereo_actor
from dh116.rl.student_model import StereoAsymmetricPolicy
torch.set_num_threads(1)
PREFIX = 'dh116.'
NETWORK = dict(embed_dim=16, history=2, num_heads=4, num_layers=1, image_resolution_wh=[16, 16], image_processing_batch_size=1)

def history(batch=1):
    p = torch.zeros(batch, 62)
    p[:, 27] = p[:, 34] = 1.0
    frame = SensorBatch(left_rgb=torch.rand(batch, 3, 16, 16), right_rgb=torch.rand(batch, 3, 16, 16), depth_m=torch.ones(batch, 16, 16), depth_valid=torch.ones(batch, 16, 16, dtype=torch.bool), depth_confidence=torch.ones(batch, 16, 16), points_world=torch.rand(batch, 128, 3) * 0.01, point_valid=torch.ones(batch, 128, dtype=torch.bool), proprio=p, dt=torch.full((batch,), 1 / 60), image_age=torch.zeros(batch), image_valid=torch.ones(batch, 2, dtype=torch.bool))
    store = StereoHistory(batch, history=2)
    store.update(frame)
    return store.update(frame)

def test_student_imports_are_pure_current_dependencies():
    code = "import importlib,json,sys;[importlib.import_module('" + PREFIX + "'+n) for n in ['rl.training']];print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    names = json.loads(result.stdout)
    bad = [name for name in names if name.startswith(('scripts.', 'scripts')) or (name.startswith(PREFIX) and any((x in name.lower() for x in ('mpc', 'adaptive', 'tensor_world', 'vision_command_shadow', 'privileged_cli'))))]
    assert bad == []

@pytest.mark.parametrize('mode', ['latent', 'estimated_task_state', 'hybrid'])
def test_three_actor_modes_preserve_sensor_gradients_masks_and_unknowns(mode):
    from dh116.env.contracts import state_contract
    row = state_contract()['dimensions'][116]
    actor = make_stereo_actor(NETWORK | dict(actor_input_mode=mode, task_state_config=dict(allowlist=[116], source_qualification={key: True for key in row['required_validity_keys']})))
    h = history()
    h.left_rgb.requires_grad_()
    h.right_rgb.requires_grad_()
    h.depth_valid[:, :, 0, 0] = False
    h.depth_m[:, :, 0, 0] = float('nan')
    h.point_valid[:, :, 0] = False
    h.points_world[:, :, 0] = float('nan')
    h.point_valid_left[:, :, 0] = h.point_valid_right[:, :, 0] = False
    h.points_left_palm[:, :, 0] = h.points_right_palm[:, :, 0] = float('nan')
    h.validate()
    action = actor.deterministic(h)
    assert action.shape == (1, 24) and torch.isfinite(action).all() and (action.abs().max() <= 1)
    action.square().sum().backward()
    for rgb in (h.left_rgb, h.right_rgb):
        assert torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0
    assert torch.isnan(h.points_world[:, :, 0]).all()
    if mode != 'latent':
        output = actor.task_state_adapter(actor.estimator(h), h)
        assert torch.isnan(output['values'][~output['valid']]).all()
        assert (output['uncertainty'] == 1).all()
    invalid = h.clone()
    invalid.point_valid[:, :, 0] = True
    with pytest.raises(ValueError, match='finite'):
        invalid.validate()

def test_visual_actor_does_not_call_or_read_privileged_critic():
    model = StereoAsymmetricPolicy(NETWORK)

    def forbidden(*args, **kwargs):
        raise AssertionError('visual action touched critic')
    model.critic.forward = forbidden
    h = history()
    expected = model.deterministic(h)
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.fill_(float('nan'))
    torch.testing.assert_close(model.deterministic(h), expected)
    with pytest.raises((ValueError, TypeError)):
        model.deterministic(h.as_dict() | {'critic': torch.zeros(1, 371)})

def test_actor_export_is_strict_actor_only(tmp_path):
    from dh116.rl.student_data import save_actor_package
    from dh116.rl.student_data import load_deployment_actor
    from dh116.env.contracts import make_contract
    actor = make_stereo_actor(NETWORK).eval()
    path = tmp_path / 'actor.pt'
    save_actor_package(path, actor, make_contract('bc', dict(schema_only=True, network=NETWORK), calibration_sha256='c' * 64), network_config=NETWORK)
    loaded = load_deployment_actor(path, expected_calibration_sha256='c' * 64)
    h = history()
    torch.testing.assert_close(loaded.deterministic(h), actor.deterministic(h))
    payload = torch.load(path, weights_only=False)
    assert not any((key.startswith(('critic', 'teacher')) for key in payload['actor']))
    payload['critic'] = {}
    torch.save(payload, path)
    with pytest.raises(ValueError, match='actor-only'):
        load_deployment_actor(path)
