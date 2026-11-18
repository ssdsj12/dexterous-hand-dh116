"""Historical cache labels and new backend evidence stay distinct offline."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS


TOOL = Path(__file__).resolve().parents[1]/'tools/audit_bimanual_joint_tracking.py'
spec = importlib.util.spec_from_file_location('target_analysis', TOOL)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def packet(version=4, backend=False):
    target = .05
    row = dict(schema=f'dh116-bimanual-joint-tracking-substep-v{version}',
        identity=dict(env_id=1, episode_uid='run:1:0', policy_step=0, physics_step=1, timestamp_s=.1),
        terminal=True, live=True, failure_code=0, joint_mapping_valid=True,
        right_finger12=dict(position_rad=.04, velocity_rad_s=0., sdk_target_rad=target,
            target_minus_actual_rad=.01, target_within_nominal_limits=True,
            nominal_lower_rad=0., nominal_upper_rad=1., physical_normalized_velocity_command=0.),
        torque=dict(applied_estimate_nm=[[.1]*6, [.1]*6]), passive_coupling=[[], []],
        contacts=dict(effective_pad_measurement_valid=False),
        active_joint_names=[list(ACTIVE), list(ACTIVE)], full_joint_names=[list(JOINTS), list(JOINTS)],
        sdk_active_target_rad=[[target]*6, [target]*6],
        wrist_world_translation_action=[[0.]*3, [0.]*3],
        wrist_intrinsic_xyz_rotation_action=[[0.]*3, [0.]*3])
    if version >= 2:
        row['right_finger12']['sdk_target_source'] = ('physx_backend_get_dof_position_targets' if backend
            else 'native_articulation_data_joint_pos_target' if version < 4
            else 'isaaclab_user_command_cache_joint_pos_target')
        row['sdk_target_sources'] = [row['right_finger12']['sdk_target_source']]*2
        row['sdk_target_native_readback_valid'] = [backend or version < 4]*2
    if version == 4:
        row['physx_backend_active_target_rad'] = [[target]*6]*2 if backend else [None, None]
        row['isaaclab_command_cache_active_target_rad'] = [[.06]*6]*2
        row['float32_submission_reconstruction_rad'] = [[.07]*6]*2
        row['joint_target_evidence'] = [dict(joint_names=list(JOINTS),
            backend=dict(source='physx_backend_get_dof_position_targets', available=backend,
                         finite_env=True if backend else None, values=[target]*11 if backend else None)) for _ in range(2)]
    return row


def summarize(tmp_path, row):
    path = tmp_path/'trace.jsonl'
    path.write_text(json.dumps(row)+'\n')
    return audit.summarize(path)


@pytest.mark.parametrize('version', [1, 2, 3])
def test_legacy_target_reference_is_not_reclassified_as_native_backend(tmp_path, version):
    row = packet(version)
    result = summarize(tmp_path, row)
    episode = result['episodes'][0]
    assert episode['backend_target_readback']['right']['valid_samples'] == 0
    assert episode['all_sdk_targets_within_nominal_limits'] is None
    assert episode['all_selected_target_references_within_nominal_limits'] is True
    selected = episode['last_eight_samples'][0]['target_provenance']
    assert selected['backend_readback_valid'] is False
    if version in (2, 3):
        assert selected['recorded_source_label'] == 'native_articulation_data_joint_pos_target'
        assert selected['interpreted_source'] == 'isaaclab_user_command_cache_joint_pos_target'
    else:
        assert selected['interpreted_source'] == 'legacy_v1_integrated_command_before_float32'


def test_v4_backend_requires_bound_values_and_propagates_native_error_statistics(tmp_path):
    result = summarize(tmp_path, packet(backend=True))
    episode = result['episodes'][0]
    assert episode['backend_target_readback']['right']['valid_samples'] == 1
    assert episode['all_sdk_targets_within_nominal_limits'] is True
    assert episode['backend_abs_tracking_error_rad']['max'] == pytest.approx(.01)
    provenance = episode['last_eight_samples'][0]['target_provenance']
    assert provenance['backend_target_rad'] == .05
    assert provenance['command_cache_target_rad'] == .06
    assert provenance['reconstructed_submission_target_rad'] == .07


@pytest.mark.parametrize('misleading_flag', [False, True])
def test_v4_missing_backend_with_available_cache_never_passes_native_check(tmp_path, misleading_flag):
    row = packet()
    row['sdk_target_native_readback_valid'] = [misleading_flag]*2
    result = summarize(tmp_path, row)
    episode = result['episodes'][0]
    assert episode['backend_target_readback']['right']['valid_samples'] == 0
    assert episode['all_sdk_targets_within_nominal_limits'] is None
    assert episode['backend_abs_tracking_error_rad']['valid_count'] == 0


def test_v4_inconsistent_backend_target_cannot_be_used_as_valid_evidence(tmp_path):
    row = packet(backend=True)
    row['joint_target_evidence'][1]['backend']['values'][list(JOINTS).index('finger12')] = .08
    result = summarize(tmp_path, row)
    assert result['episodes'][0]['backend_target_readback']['right']['valid_samples'] == 0
    assert result['episodes'][0]['all_sdk_targets_within_nominal_limits'] is None


def test_old_cli_remains_usable_with_historical_provenance(tmp_path):
    path = tmp_path/'old.jsonl'
    output = tmp_path/'result.json'
    path.write_text(json.dumps(packet(3))+'\n')
    result = subprocess.run([sys.executable, str(TOOL), str(path), '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_text())
    assert value['episodes'][0]['backend_target_readback']['right']['valid_samples'] == 0
