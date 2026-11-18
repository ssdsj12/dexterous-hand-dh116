"""CPU file/command tests, never native trials or qualified demonstrations."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def api():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_stage6_workflow')


def test_file_identity_binds_actual_source_path_and_refuses_changed_bytes(tmp_path):
    source = tmp_path/'actual-file.bin'
    source.write_bytes(b'CPU identity fixture, not a model')
    manifest = tmp_path/'inputs.json'
    payload = dict(version='dh116-stage6-inputs-v1', actor_dim=3844, critic_dim=371, action_dim=24,
                   source_checkpoint=dict(path=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    manifest.write_text(json.dumps(payload))
    checked = api()._checked_identity(payload['source_checkpoint'], manifest.parent)
    assert checked['path'] == str(source.resolve())
    with pytest.raises(ValueError, match='complete bound'):
        api().load_stage6_inputs(manifest)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        api().load_stage6_inputs(manifest)


def test_native_command_changes_scripts_only_and_preserves_bound_data_paths(tmp_path):
    old = Path('/frozen/source')
    template = dict(cwd=str(old), command=['bash', str(old/'scripts/python.sh'),
        str(old/'scripts/train_dh116_tabletop_gpu_ppo.py'), '--output', '/old-output',
        '--vision-teacher-source', str(old/'model-data.pt'), '--seed', '42',
        '--vision-teacher-mpc', '--bimanual-teacher-physics-audit', '--physics-hz', '480'])
    command = api().build_stage6_native_command(template, tmp_path/'new_source', tmp_path/'native_output',
                                                mode='teacher-context')
    assert command[1] == str(tmp_path/'new_source/scripts/python.sh')
    assert command[2] == str(tmp_path/'new_source/scripts/train_dh116_tabletop_gpu_ppo.py')
    assert command[command.index('--vision-teacher-source')+1] == str(old/'model-data.pt')
    assert command[command.index('--seed')+1] == '42'
    assert command[command.index('--physics-hz')+1] == '480'
    assert '--vision-sequence-context-sidecar' in command
    assert '--bimanual-teacher-physics-audit' not in command


def test_source_runner_cannot_keep_teacher_or_m2_flags(tmp_path):
    template = dict(cwd='/old',command=['bash','/old/scripts/python.sh',
        '/old/scripts/train_dh116_tabletop_gpu_ppo.py','--vision-teacher-source','/expert.pt',
        '--vision-teacher-mpc','--vision-teacher-mpc-profile','pressure_geometry_constrained',
        '--vision-mirror-preprocessing','m2_full_mirror','--vision-evaluation-checkpoint','/old.pt',
        '--vision-evaluation-manifest','/old.json'])
    command=api().build_stage6_native_command(template,tmp_path/'src',tmp_path/'run',mode='source',
                                             native_config=tmp_path/'mode.json')
    assert '--vision-teacher-source' not in command and '--vision-teacher-mpc' not in command
    assert '--vision-evaluation-checkpoint' not in command and '--vision-evaluation-manifest' not in command
    assert command[command.index('--vision-mirror-preprocessing')+1]=='m0_raw'
    assert command[command.index('--vision-stage6-native-config')+1]==str(tmp_path/'mode.json')


def test_source_projection_strips_real_diagnostics_template_evaluation_only_flags(tmp_path):
    template=dict(command=['bash','/old/scripts/python.sh','/old/scripts/train_dh116_tabletop_gpu_ppo.py',
        '--vision-evaluation-checkpoint','/actual-c58.pt','--vision-evaluation-manifest','/regression200.json',
        '--vision-evaluation-physics-audit','--bimanual-checkpoint-adapter'])
    command=api().build_stage6_native_command(template,tmp_path/'src',tmp_path/'out',mode='teacher-replay',
                                             native_config=tmp_path/'replay.json')
    assert '--vision-evaluation-physics-audit' not in command
    assert '--bimanual-checkpoint-adapter' not in command
    assert '--vision-evaluation-checkpoint' not in command


@pytest.mark.parametrize('mode', ['source', 'branch', 'teacher-replay'])
def test_student_native_projection_drops_teacher_perturbation_reset_contract(tmp_path, mode):
    template = dict(command=['bash', '/old/scripts/python.sh', '/old/scripts/train_dh116_tabletop_gpu_ppo.py',
        '--vision-teacher-source', '/expert.pt', '--vision-teacher-perturbation-manifest', '/teacher-tier1.json'])
    command = api().build_stage6_native_command(template, tmp_path/'src', tmp_path/'out',
        mode=mode, native_config=tmp_path/'native.json')
    assert '--vision-teacher-perturbation-manifest' not in command
    assert '/teacher-tier1.json' not in command


def test_teacher_context_preserves_explicit_perturbation_identity(tmp_path):
    template = dict(command=['bash', '/old/scripts/python.sh', '/old/scripts/train_dh116_tabletop_gpu_ppo.py',
        '--vision-teacher-source', '/expert.pt', '--vision-teacher-perturbation-manifest', '/teacher-tier1.json'])
    command = api().build_stage6_native_command(template, tmp_path/'src', tmp_path/'out', mode='teacher-context')
    assert command[command.index('--vision-teacher-perturbation-manifest')+1] == '/teacher-tier1.json'


def test_partial_identity_with_extra_keys_is_rejected_as_validation_error(tmp_path):
    payload = dict(version='dh116-stage6-inputs-v1', actor_dim=3844, critic_dim=371, action_dim=24,
                   source_checkpoint=dict(path='missing', extra='not a SHA'))
    manifest=tmp_path/'inputs.json';manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='SHA'):
        api().load_stage6_inputs(manifest)


def test_command_training_requires_native_backend_and_teacher_shadow_gates():
    with pytest.raises(ValueError,match='shadow'):
        api().require_native_shadow_parity(dict(passed=False))
    with pytest.raises(ValueError,match='backend|readback'):
        api().require_native_shadow_parity(dict(passed=True,native_sdk_readback_required=False))
    parity=dict(passed=True,native_sdk_readback_required=True,
                native_sdk_readback={'backend':dict(passed=True),'command_cache':dict(passed=True)})
    assert api().require_native_shadow_parity(parity)==parity


def test_main_workflow_does_not_silently_inherit_ablation_activation():
    api().require_main_preprocessing({'point_activation':'relu','mirror':'m0_raw'})
    with pytest.raises(ValueError,match='relu|activation'):
        api().require_main_preprocessing({'point_activation':'leaky_relu_001','mirror':'m0_raw'})


def test_native_requested_mode_cannot_disagree_with_config():
    api().require_native_mode('source', {'mode':'source'})
    api().require_native_mode('branch', {'mode':'branch'})
    api().require_native_mode('teacher-replay', {'mode':'teacher-replay'})
    with pytest.raises(ValueError,match='mode'):
        api().require_native_mode('source', {'mode':'branch'})
    with pytest.raises(ValueError,match='mode'):
        api().require_native_mode('branch', {'mode':'source'})


def test_matched_exposure_gate_keeps_padding_explicit():
    api().require_matched_exposures({'sequence_length':16,'min_valid_steps':16,
                                    'max_valid_steps':16,'padded_window_count':0})
    with pytest.raises(ValueError,match='exposure'):
        api().require_matched_exposures({'sequence_length':16,'min_valid_steps':7,
                                        'max_valid_steps':16,'padded_window_count':1})


def test_config_is_consumed_and_unknown_control_changes_rejected(tmp_path):
    config=api().load_stage6_config()
    assert config['training']['updates']==3000 and config['sequence']['length']==16
    config['loss']['finger_scale_rad']=.1
    path=tmp_path/'bad.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='scale|Stage6'):
        api().load_stage6_config(path)


def test_freezing_inside_source_is_rejected_before_any_creation_or_copy(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir();(source/'owned.py').write_text('x=1')
    target=source/'recursive_snapshot'
    def unsafe_copy(*args,**kwargs):
        pytest.fail('must reject nested snapshot before copying')
    monkeypatch.setattr(api().shutil,'copyfile',unsafe_copy)
    with pytest.raises(ValueError,match='outside source'):
        api().freeze_source(source,target)
    assert not target.exists()


def test_resolving_inputs_cannot_promote_unverified_collection(tmp_path):
    report=tmp_path/'teacher.json';report.write_text(json.dumps({'native_runtime_verified':False}))
    output=tmp_path/'resolved.json'
    with pytest.raises(ValueError,match='native'):
        api().resolve_stage6_inputs(tmp_path/'discovery.json',report,tmp_path/'heldout.json',output)
    assert not output.exists()
