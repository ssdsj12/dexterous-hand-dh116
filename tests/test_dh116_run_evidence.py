import ast
import importlib.util
import json
from pathlib import Path

import pytest


def implementation():
    name = 'go2_pvcnn.control.dh116.run_evidence'
    assert importlib.util.find_spec(name) is not None, 'Missing durable run evidence'
    return __import__(name, fromlist=['write_start_manifest'])


def test_start_manifest_is_explicitly_not_success(tmp_path):
    api = implementation()
    api.write_start_manifest(tmp_path, {'dt': .0001}, {'asset_sha256': 'abc'})
    data = json.loads((tmp_path/'run_manifest.json').read_text())
    assert data['status'] == 'initialized_not_qualified'
    assert data['grasp_qualified'] is False
    assert data['run_config'] == {'dt': .0001}
    assert data['asset_manifest'] == {'asset_sha256': 'abc'}


def test_start_manifest_never_overwrites(tmp_path):
    api = implementation()
    api.write_start_manifest(tmp_path, {'dt': .0001}, {})
    before = (tmp_path/'run_manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        api.write_start_manifest(tmp_path, {'dt': .001}, {})
    assert (tmp_path/'run_manifest.json').read_bytes() == before


def test_invalid_json_not_published(tmp_path):
    with pytest.raises(ValueError):
        implementation().write_start_manifest(tmp_path, {'dt': float('nan')}, {})
    assert not (tmp_path/'run_manifest.json').exists()


def test_stop_request_raises_and_is_preserved(tmp_path):
    api = implementation()
    api.check_stop_request(tmp_path)
    request = tmp_path/'STOP_REQUESTED'
    request.touch()
    with pytest.raises(RuntimeError, match='operator stop requested'):
        api.check_stop_request(tmp_path)
    assert request.exists()


def test_runner_checks_stop_before_publishing_actions():
    source = Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py'
    tree = ast.parse(source.read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and ast.unparse(n.target) == 'step']
    assert ast.unparse(loops[0].body[0]) == 'check_stop_request(args.output)'
    calls = {ast.unparse(n.func): n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert 'write_start_manifest' in calls
    assert calls['write_start_manifest'] < calls['su.SimulationContext']
