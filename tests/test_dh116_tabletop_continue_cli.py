"""Continuation argument/provenance tests; no Isaac process is started."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.continue_dh116_tabletop_ppo import (
    build_training_command, continuation_sources, load_continuation_contract,
)


def make_inputs(tmp_path, *, direct=True):
    checkpoint = tmp_path/'model.zip'
    checkpoint.write_text('synthetic checkpoint')
    pose = tmp_path/'pose.json'
    pose.write_text('{}')
    snapshot = tmp_path/'close seed.json'
    snapshot.write_text('{"synthetic": true}')
    contract = dict(action='12D_direct_joint_target_v1' if direct else
        '12D_bounded_nominal_residual_v2', residual_scale_rad=.02,
        reset_height_jitter_m=.0003, solver_type=0, position_iterations=16,
        physics_hz=240, pose_sha256=hashlib.sha256(pose.read_bytes()).hexdigest())
    if direct:
        contract.update(target_radius_rad=.08,
            close_seed_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    checkpoint.with_suffix('.json').write_text(json.dumps({'contract': contract}))
    return checkpoint, pose, snapshot, contract


def build(tmp_path, contract, checkpoint, pose, snapshot=None):
    return build_training_command(tmp_path, contract, checkpoint, tmp_path/'new round',
        pose_summary=pose, timesteps=8192, checkpoint_every=1024, eval_episodes=5,
        close_reset_snapshot=snapshot)


def value(argv, flag):
    return argv[argv.index(flag)+1]


def test_direct_checkpoint_adds_required_control_flags_and_preserves_physics(tmp_path):
    checkpoint, pose, snapshot, expected = make_inputs(tmp_path)
    contract = load_continuation_contract(checkpoint, pose, snapshot)
    assert contract == expected
    argv = build(tmp_path, contract, checkpoint, pose, snapshot)
    assert value(argv, '--control-mode') == 'joint_target'
    assert value(argv, '--close-reset-snapshot') == str(snapshot.resolve())
    assert float(value(argv, '--joint-target-radius-rad')) == .08
    assert value(argv, '--reset-height-jitter-m') == '0.0003'
    assert value(argv, '--solver-type') == '0'
    assert value(argv, '--position-iterations') == '16'
    assert value(argv, '--physics-hz') == '240'
    assert value(argv, '--resume') == str(checkpoint.resolve())
    assert '--curriculum' in argv and '--headless' in argv
    assert value(argv, '--device') == 'cpu'


def test_legacy_residual_checkpoint_keeps_original_command(tmp_path):
    checkpoint, pose, _, expected = make_inputs(tmp_path, direct=False)
    contract = load_continuation_contract(checkpoint, pose)
    assert contract == expected
    argv = build(tmp_path, contract, checkpoint, pose)
    assert '--control-mode' not in argv and '--close-reset-snapshot' not in argv
    assert '--joint-target-radius-rad' not in argv
    assert value(argv, '--residual-scale-rad') == '0.02'


def test_direct_missing_seed_fails_during_preflight_and_command_build(tmp_path):
    checkpoint, pose, _, contract = make_inputs(tmp_path)
    with pytest.raises(ValueError, match='snapshot'):
        load_continuation_contract(checkpoint, pose)
    with pytest.raises(ValueError, match='snapshot'):
        build(tmp_path, contract, checkpoint, pose)


def test_changed_snapshot_or_pose_is_rejected_before_launch(tmp_path):
    checkpoint, pose, snapshot, _ = make_inputs(tmp_path)
    original = snapshot.read_text()
    snapshot.write_text('changed')
    with pytest.raises(ValueError, match='snapshot'):
        load_continuation_contract(checkpoint, pose, snapshot)
    snapshot.write_text(original)
    pose.write_text('changed')
    with pytest.raises(ValueError, match='pose'):
        load_continuation_contract(checkpoint, pose, snapshot)


def test_residual_does_not_silently_ignore_close_seed(tmp_path):
    checkpoint, pose, snapshot, _ = make_inputs(tmp_path, direct=False)
    with pytest.raises(ValueError, match='residual'):
        load_continuation_contract(checkpoint, pose, snapshot)


@pytest.mark.parametrize('change', [
    {'action': 'unknown-action'}, {'target_radius_rad': float('nan')},
    {'target_radius_rad': .2}, {'close_seed_sha256': None},
])
def test_invalid_direct_contract_fails_before_launch(tmp_path, change):
    checkpoint, pose, snapshot, contract = make_inputs(tmp_path)
    contract.update(change)
    checkpoint.with_suffix('.json').write_text(json.dumps({'contract': contract}))
    with pytest.raises(ValueError):
        load_continuation_contract(checkpoint, pose, snapshot)


def test_source_pins_cover_direct_policy_and_immutable_reset_artifacts(tmp_path):
    checkpoint, pose, snapshot, _ = make_inputs(tmp_path)
    root = Path(__file__).resolve().parents[1]
    sources = continuation_sources(root, pose, snapshot)
    assert root/'src/go2_pvcnn/control/dh116/tabletop_direct.py' in sources
    assert root/'src/go2_pvcnn/control/dh116/tabletop_policy.py' in sources
    assert root/'scripts/train_dh116_tabletop_ppo.py' in sources
    assert pose.resolve() in sources and snapshot.resolve() in sources
    assert all(p.is_file() for p in sources)
    residual_sources = continuation_sources(root, pose)
    assert pose.resolve() in residual_sources and snapshot.resolve() not in residual_sources


@pytest.mark.parametrize('wrong_snapshot', [False, True])
def test_cli_rejects_missing_or_wrong_snapshot_before_supervisor(tmp_path, monkeypatch, capsys,
                                                               wrong_snapshot):
    from scripts import continue_dh116_tabletop_ppo as cli
    checkpoint, pose, snapshot, _ = make_inputs(tmp_path)
    argv = ['continue_dh116_tabletop_ppo.py', '--resume', str(checkpoint),
            '--pose-summary', str(pose), '--output', str(tmp_path/'output')]
    if wrong_snapshot:
        snapshot.write_text('different snapshot')
        argv += ['--close-reset-snapshot', str(snapshot)]
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.chdir(tmp_path)
    def unexpected_supervisor(*args, **kwargs):
        raise AssertionError('invalid reset input reached process supervisor')
    monkeypatch.setattr(cli, 'continue_training', unexpected_supervisor)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert 'snapshot' in capsys.readouterr().err
    assert not (tmp_path/'output').exists()
