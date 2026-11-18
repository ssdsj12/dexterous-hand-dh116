"""Synthetic CPU identities only; none of these fixtures are native trials."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from go2_pvcnn.control.dh116.tabletop_evaluation_trials import (
    GRASP_STYLES, state_sha256, validate_manifest,
)


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/generate_stage5_student_manifest.py'


def api():
    spec = importlib.util.spec_from_file_location('student_manifest_generator', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_payload(n=2, split='test'):
    names = ['finger11', 'finger12', 'finger13', 'finger21', 'finger22',
             'finger31', 'finger32', 'finger41', 'finger42', 'finger51', 'finger52']
    rows = []
    for si, style in enumerate(GRASP_STYLES):
        for i in range(n):
            k = si * n + i
            # Nonzero XYZ quaternion components distinguish world left multiply
            # from the incorrect local right multiplication.
            q = [math.cos(.3), math.sin(.3), 0., 0.]
            state = dict(joint_names=[names.copy(), names.copy()],
                joint_positions=[[.1 + k * .00001] * 11, [.2] * 11],
                joint_velocities=[[.01] * 11, [-.02] * 11],
                roots=[[0., -.1, .4, 1., 0., 0., 0.], [0., .1, .4, 1., 0., 0., 0.]],
                root_velocities=[[.01] * 6, [-.01] * 6],
                wire_pose=[.02, -.03, .308] + q,
                wire_velocity=[.01, .02, .03, .04, .05, .06])
            rows.append(dict(trial_id=f'{style}:{i}', grasp_style=style,
                physical_state=state, state_sha256=state_sha256(state),
                environment_origin_m=[k * 2., 0., 0.]))
    return dict(schema_version='dh116-fixed-trials-v1', metadata=dict(
        reset_kind='cold_authored_reset', seed=17, split=split,
        generator_sha256='1' * 64, source_sha256='2' * 64,
        physics_sha256='3' * 64, camera_sha256='4' * 64,
        template_sha256='5' * 64, validation_status='authored_unverified'), trials=rows)


def encoded(payload):
    return json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf-8')


def fixture_build(seed=116507, n=2):
    raw = encoded(source_payload(n))
    return api()._build_manifest(raw, expected_source_sha256=hashlib.sha256(raw).hexdigest(),
                                 seed=seed, _test_min_trials_per_style=n)


def test_same_seed_deterministic_and_distinct_seed_disjoint():
    first, proof = fixture_build()
    again, again_proof = fixture_build()
    other, _ = fixture_build(seed=116508)
    assert first == again
    assert proof == again_proof
    old = {row['state_sha256'] for row in source_payload()['trials']}
    hashes = {row['state_sha256'] for row in first['trials']}
    assert len(hashes) == len(first['trials'])
    assert hashes.isdisjoint(old)
    assert hashes.isdisjoint({row['state_sha256'] for row in other['trials']})
    assert all('116507' in row['trial_id'] for row in first['trials'])


def test_only_wire_xy_and_world_yaw_change_all_other_fields_exact():
    output, proof = fixture_build()
    source = source_payload()
    for before, after, perturbation in zip(source['trials'], output['trials'], proof['perturbations']):
        expected = deepcopy(before['physical_state'])
        old_pose = expected.pop('wire_pose')
        actual = deepcopy(after['physical_state'])
        pose = actual.pop('wire_pose')
        assert actual == expected
        assert after['environment_origin_m'] == before['environment_origin_m']
        assert after['grasp_style'] == before['grasp_style']
        assert pose[2] == old_pose[2]
        dx, dy, yaw = perturbation['dx_m'], perturbation['dy_m'], perturbation['world_z_yaw_rad']
        assert abs(dx) <= .00025 and abs(dy) <= .00025 and abs(yaw) <= .002
        assert pose[:2] == [old_pose[0] + dx, old_pose[1] + dy]
        # Independent wxyz world-Z multiply oracle for original X rotation.
        c, s = math.cos(yaw / 2), math.sin(yaw / 2)
        w, x, y, z = old_pose[3:]
        expected_q = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
        norm = math.hypot(*expected_q)
        assert pose[3:] == pytest.approx([v/norm for v in expected_q], abs=1e-15)
        assert state_sha256(after['physical_state']) == after['state_sha256']
    validate_manifest(output, min_trials_per_style=2)
    assert output['metadata']['physics_sha256'] == source['metadata']['physics_sha256']
    assert output['metadata']['camera_sha256'] == source['metadata']['camera_sha256']


def test_source_template_and_actual_generator_hashes_no_native_claims():
    output, proof = fixture_build()
    source_sha = hashlib.sha256(encoded(source_payload())).hexdigest()
    assert output['metadata']['source_sha256'] == source_sha
    assert output['metadata']['template_sha256'] == source_sha
    assert output['metadata']['generator_sha256'] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert output['metadata']['validation_status'] == 'authored_unverified'
    assert proof['native_trials'] == proof['model_derivations'] == proof['training_updates'] == 0
    assert proof['native_runtime_verified'] is False
    assert proof['counts_as_native_validation'] is False
    assert proof['scope'] == 'same_reset_family_wire_pose_microperturbation'
    assert proof['generalization_claims'] == []
    assert proof['distribution']['expected_mean'] == 0.


@pytest.mark.parametrize('source_split', ['validation', 'test'])
def test_production_write_source_read_only_exclusive_and_valid(tmp_path, source_split):
    module = api()
    source = tmp_path / 'source.json'
    raw = encoded(source_payload(100, source_split))
    source.write_bytes(raw)
    output = tmp_path / 'new.json'
    result = module.generate_manifest(source, output,
        expected_source_sha256=hashlib.sha256(raw).hexdigest(), seed=116507)
    assert source.read_bytes() == raw
    manifest = json.loads(output.read_bytes())
    validate_manifest(manifest)
    assert len(manifest['trials']) == 200
    assert manifest['metadata']['split'] == 'validation'
    sidecar = Path(result['provenance_path'])
    provenance = json.loads(sidecar.read_bytes())
    assert provenance['output_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    old_output, old_sidecar = output.read_bytes(), sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        module.generate_manifest(source, output,
            expected_source_sha256=hashlib.sha256(raw).hexdigest(), seed=116508)
    assert output.read_bytes() == old_output and sidecar.read_bytes() == old_sidecar
    assert source.read_bytes() == raw


@pytest.mark.parametrize('case', ['wrong_sha', 'fixture_as_production', 'source_output_alias', 'sidecar_exists'])
def test_fail_closed_no_output_on_invalid_input_or_preexisting_sidecar(tmp_path, case):
    module = api()
    source = tmp_path / 'source.json'
    raw = encoded(source_payload(2, 'test') if case == 'fixture_as_production'
                  else source_payload(100, 'validation'))
    source.write_bytes(raw)
    output = source if case == 'source_output_alias' else tmp_path / 'new.json'
    sidecar = Path(str(output) + '.provenance.json')
    if case == 'sidecar_exists':
        sidecar.write_bytes(b'original')
    expected = '0' * 64 if case == 'wrong_sha' else hashlib.sha256(raw).hexdigest()
    with pytest.raises((ValueError, FileExistsError)):
        module.generate_manifest(source, output, expected_source_sha256=expected, seed=116507)
    assert source.read_bytes() == raw
    if output != source:
        assert not output.exists()
    if case == 'sidecar_exists':
        assert sidecar.read_bytes() == b'original'


def test_cli_has_no_fixture_minimum_and_requires_source_hash(tmp_path):
    parser = api().parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--input', str(tmp_path / 'in'), '--output', str(tmp_path / 'out')])
    with pytest.raises(SystemExit):
        parser.parse_args(['--input', 'in', '--output', 'out', '--source-sha256', '0'*64,
                           '--test-min-trials-per-style', '1'])


def test_bounded_resampling_cannot_admit_float32_identical_states(monkeypatch):
    module = api()
    class NoPerturbation:
        def __init__(self, _seed):
            pass
        def uniform(self, _lo, _hi):
            return 0.
    monkeypatch.setattr(module.random, 'Random', NoPerturbation)
    # Fixture orientation is normalized already. With no perturbation the
    # native float32 state stays identical despite JSON renormalization.
    raw = encoded(source_payload())
    with pytest.raises(ValueError, match='resampling'):
        module._build_manifest(raw, expected_source_sha256=hashlib.sha256(raw).hexdigest(),
                               seed=116507, _test_min_trials_per_style=2)
