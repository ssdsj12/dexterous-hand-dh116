import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def test_launcher_forwards_two_distinct_qualified_initializations(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    configs = root / 'configs/dh116_training'
    launcher = root / 'scripts/run_dh116_surface_student.py'
    output = tmp_path / 'training'
    argv = [
        str(launcher), '--output', str(output), '--num-envs', '4096', '--updates', '2048',
        '--pregrasp-candidates', str(configs / 'effective_pad_center_pregrasp_v1.json'),
        '--mixed-pregrasp-candidates', str(configs / 'effective_pad_triad_pregrasp_v1.json'),
    ]
    calls = []
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(os, 'execv', lambda program, command: calls.append((program, command)))
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(launcher), run_name='__main__')

    _, cmd = calls[0]
    primary = Path(cmd[cmd.index('--grasp-initialization') + 1])
    secondary = Path(cmd[cmd.index('--mixed-grasp-initialization') + 1])
    assert hashlib.sha256(primary.read_bytes()).hexdigest() == (
        '500e9f25f6b05f8a07d68358f8fd38c0d8d5a7172213d31b769bd38846484a68'
    )
    assert hashlib.sha256(secondary.read_bytes()).hexdigest() == (
        'c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b'
    )
    assert json.loads(primary.read_text())['roots'] != json.loads(secondary.read_text())['roots']
    assert cmd[cmd.index('--max-iterations') + 1] == '2048'
