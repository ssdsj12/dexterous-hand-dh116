"""Phase 0 read-only audit. CPU synthetic state, never Isaac/native evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from phase0_state_contract_fixture import cpu_task, observation_context, sensor_types


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/contracts/stereo_privileged_state_contract.json'
DOC = ROOT / 'docs/stereo_privileged_state_contract.md'


def manifest():
    return json.loads(CONTRACT.read_text(encoding='utf-8'))


def test_required_document_and_machine_contract_exist():
    assert DOC.is_file(), 'Phase 0 requires the human-readable 371-row audit'
    assert CONTRACT.is_file(), 'Phase 0 requires the executable per-dimension audit'


def test_schema_covers_every_dimension_once_and_preserves_na():
    c = manifest()
    rows = c['dimensions']
    assert c['size'] == len(rows) == 371
    assert [r['index'] for r in rows] == list(range(371))
    assert len({r['semantic_name'] for r in rows}) == 371
    required = {'index', 'semantic_name', 'source_expression', 'source_module',
                'unit', 'coordinate_frame', 'normalization', 'validity_mask',
                'student_source', 'estimator_head', 'deployment_allowed',
                'deployment_condition', 'loss_type', 'notes', 'current_source',
                'supervision_status'}
    allowed = {'RGB/depth', 'proprio', 'history', 'known_config', 'unobservable', 'history_inferred'}
    for row in rows:
        assert required <= row.keys()
        assert row['student_source'] in allowed
        assert type(row['deployment_allowed']) is bool
        assert row['validity_mask'] and row['current_source'] and row['notes']
        if row['supervision_status'] == 'NA':
            assert row['loss_type'] == 'none'
            assert row['deployment_allowed'] is False
            assert row['estimator_head'] == 'none'
        if row['deployment_allowed']:
            assert row['deployment_condition'] != 'unconditional'
    assert c['missing_value_policy'] == 'NA_is_not_zero;mask_required;no_simulator_fallback'
    assert '| 370 |' in DOC.read_text(encoding='utf-8')


def test_source_byte_hashes_and_ast_ranges_are_bound_to_checkout():
    c = manifest()
    for path, evidence in c['sources'].items():
        source = ROOT / path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == evidence['sha256'], path
        if path.endswith('.py'):
            tree = ast.parse(source.read_text(encoding='utf-8-sig'))
            for bound in evidence.get('symbols', []):
                nodes = [n for n in ast.walk(tree)
                         if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == bound['name'] and n.lineno == bound['line_start']]
                assert len(nodes) == 1, (path, bound)
                assert nodes[0].end_lineno == bound['line_end']
                digest = hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest()
                assert digest == bound['ast_sha256']


def test_real_observe_matches_all_documented_block_slices():
    c = manifest()
    t = cpu_task()
    observation = t.observe()
    assert observation.shape == (2, 371)
    assert observation.dtype == torch.float32 and observation.device.type == 'cpu'
    covered = []
    context = observation_context(t)
    for block in c['blocks']:
        begin, end = block['slice']
        expected = eval(block['expression'], {'torch': torch, '__builtins__': {}}, context)
        assert expected.shape == (2, end - begin), block['name']
        torch.testing.assert_close(observation[:, begin:end], expected.float(), msg=block['name'])
        covered.extend(range(begin, end))
        for row in c['dimensions'][begin:end]:
            expected_column = eval(row['source_expression'], {'torch': torch, '__builtins__': {}}, context)
            torch.testing.assert_close(observation[:, row['index']], expected_column.float())
    assert covered == list(range(371))
    assert sum(b['slice'][1] - b['slice'][0] for b in c['blocks']) == 371


def test_observe_dimensionality_is_configuration_dependent():
    t = cpu_task()
    del t.surface_feedback
    assert t.observe().shape == (2, 351)
    t.free_grasp_lift = False
    assert t.observe().shape == (2, 355)  # Four task phases are absent in canonical 371.
    t.free_grasp_lift = True
    t.force_reward_start = 'episode_start'
    assert t.observe().shape == (2, 350)


def test_joint_and_quaternion_normalization_use_actual_observe():
    t = cpu_task()
    t.m['q'][0] = t.low
    t.m['q'][1] = t.high
    o = t.observe()
    torch.testing.assert_close(o[0, :12], -torch.ones(12))
    torch.testing.assert_close(o[1, :12], torch.ones(12))
    # Current task observation copies the raw wxyz; it does not unit-normalize.
    t.m['object_quat'] *= 2.
    torch.testing.assert_close(t.observe()[:, 64:68], t.m['object_quat'].float())


def test_rotation_groups_and_fail_closed_mask_rules_are_explicit():
    c = manifest()
    assert [g['slice'] for g in c['rotation_groups']] == [[64, 68], [156, 160], [163, 167]]
    assert all(g['loss'] == 'SO3_geodesic' for g in c['rotation_groups'])
    assert c['runtime_mask_contract']['default'] == 'fail_closed'
    assert c['runtime_mask_contract']['no_mask_in_existing_flat_tensor'] is True
    for row in c['dimensions']:
        assert {'source_valid', 'identity_aligned', 'timestamp_valid'} <= set(row['required_validity_keys'])
        if row['supervision_status'] == 'NA':
            assert 'supervision_forbidden' in row['required_validity_keys']
    from go2_pvcnn.control.dh116.tabletop_surface_contract import feedback_schema
    expected = feedback_schema()
    actual = [b for b in c['blocks'] if b['name'].startswith('interval_')]
    assert [(b['name'][len('interval_'):], b['slice']) for b in actual] == [(b['name'], b['slice']) for b in expected]


def test_no_contact_zero_and_reset_feedback_do_not_certify_valid_labels():
    from go2_pvcnn.control.dh116.tabletop_surface_feedback import SurfaceFeedback
    t = cpu_task()
    t.m['contact_valid'].zero_()
    t.m['contact_center_relative_m'].fill_(float('nan'))
    o = t.observe()
    assert torch.isfinite(o).all()
    assert not o[:, 147:149].any()  # Legacy placeholder for missing radial centroids.
    t.surface_feedback = SurfaceFeedback(t.n, 'cpu')
    o = t.observe()
    assert not t.surface_feedback.valid.any()
    assert not o[:, 351:371].any()
    c = manifest()
    assert all(r['loss_type'] == 'none' and r['supervision_status'] == 'NA'
               for r in c['dimensions'][351:353])
    assert all('feedback.valid' in r['validity_mask'] for r in c['dimensions'][351:])


def test_actual_end_step_snapshots_previous_command_and_reuses_last_finite_row():
    t = cpu_task()
    t.nonfinite = torch.tensor([True, False])
    t.last_observation = t.observe().clone()
    prior = t.last_observation.clone()
    t.command += .001
    t.reward = torch.zeros(t.n)
    t.terms = {}
    t.used = torch.zeros(t.n)
    observation, _, _, _ = t.end_step()
    torch.testing.assert_close(t.previous, t.command)
    torch.testing.assert_close(observation[0], prior[0])
    assert not torch.equal(observation[1, 24:36], prior[1, 24:36])


def test_actual_sensor_history_and_adapter_keep_privileged_state_separate():
    SensorHistory, adapter, TensorDictFixture = sensor_types()
    t = cpu_task()
    n = t.n
    from types import SimpleNamespace
    world = SimpleNamespace(n=n, w=SimpleNamespace(ids=[list(range(6)), list(range(6))]),
                            q=t.m['q'].reshape(n, 2, 6), qd=t.m['qd'].reshape(n, 2, 6),
                            roots=t.m['wrist_pose'].clone(), origins=torch.zeros(n, 3))
    packet = dict(points=torch.arange(n * 2 * 128 * 3).reshape(n, 2, 128, 3).float() / 1000,
                  point_valid=torch.ones(n, 128, dtype=torch.bool),
                  image_valid=torch.ones(n, dtype=torch.bool), object_valid=torch.ones(n, dtype=torch.bool))
    issued = torch.zeros(n, 24)
    a = adapter(packet, world, issued, SensorHistory(n), t.observe())
    b = adapter(packet, world, issued, SensorHistory(n), t.observe() * 100.)
    assert isinstance(a, TensorDictFixture)
    assert a['policy'].shape == (n, 3844) and a['critic'].shape == (n, 371)
    torch.testing.assert_close(a['policy'], b['policy'])
    assert not torch.equal(a['critic'], b['critic'])
    newest = a['policy'].reshape(n, 4, 961)[:, -1]
    torch.testing.assert_close(newest[:, 896:908], t.m['q'].float())
    torch.testing.assert_close(newest[:, 908:920], t.m['qd'].float())
    torch.testing.assert_close(newest[:, 920:934], world.roots.flatten(1).float())
    torch.testing.assert_close(newest[:, 934:958], issued)
    with pytest.raises(RuntimeError, match='critic'):
        adapter(packet, world, issued, SensorHistory(n), torch.zeros(n, 370))
    invalid = copy.deepcopy(packet)
    invalid['image_valid'][0] = False
    with pytest.raises(RuntimeError, match='no privileged fallback'):
        adapter(invalid, world, issued, SensorHistory(n), t.observe())


def test_audit_never_imports_isaac_or_allocates_cuda():
    import sys
    assert not any(k == 'isaaclab' or k.startswith(('isaaclab.', 'omni.', 'isaacsim.')) for k in sys.modules)
    assert all(value.device.type == 'cpu' for value in cpu_task().m.values())
