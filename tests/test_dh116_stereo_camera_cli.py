"""Depth-reference arithmetic and CLI import isolation, not native validation."""
import json
import torch
import pytest

from go2_pvcnn.control.dh116 import tabletop_stereo_camera_cli as cli


def test_reference_depth_does_not_fill_invalid_prediction():
    predicted = torch.tensor([[[.4, float('nan')]]])
    reference = torch.tensor([[[.5, .7]]])
    report = cli.depth_metrics(predicted, reference, torch.tensor([[[True, False]]]),
        torch.ones(1, 1, 2, dtype=torch.bool))
    assert report['count'] == 1
    assert report['mae_m'] == pytest.approx(.1)
    assert report['validity_coverage'] == .5


def test_config_probe_only_validates_without_native(tmp_path, capsys):
    output = tmp_path / 'never'
    assert cli.main('probe', ['--check-config', '--output', str(output)]) == 0
    assert not json.loads(capsys.readouterr().out)['native_started']
    assert not output.exists()


def test_mismatched_reference_rejected_before_output(tmp_path):
    capture = tmp_path / 'capture.pt'; reference = tmp_path / 'reference.pt'
    torch.save(dict(version='dh116-stereo-rgb-probe-v1', frames=[dict(
        calibration_sha256='a'*64, frame_id=2, simulation_time=.1)]), capture)
    torch.save(dict(version='dh116-stereo-depth-reference-v1',
        calibration_sha256='a'*64, frame_id=1, simulation_time=.1), reference)
    with pytest.raises(ValueError, match='mismatch'):
        cli.evaluate_capture(capture, reference)
