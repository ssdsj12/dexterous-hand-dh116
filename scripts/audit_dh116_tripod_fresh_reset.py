#!/usr/bin/env python3
"""Read-only source-bound audit of an exited fresh17 Tripod Reset.

No App, live native view, policy or solver is created. Publication bookkeeping is
verified from its exact frozen Git blob; current physical sources remain strict.
"""
import argparse
import re

from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.run_dh116_work_packages import debugger_status
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile,profile_reader_scope,checked_file
from go2_pvcnn.control.dh116.tabletop_supported_mesh_reset import validate_supported_readback
from go2_pvcnn.control.dh116.tabletop_mesh_reset_numeric_state import validate_numeric_reset_audit
from go2_pvcnn.control.dh116.tabletop_power_supported_task import guard_hand_supports



def _require(value,message):
    if not value:
        raise ValueError(message)


def audit_run(run_dir, *, receipt, expected_sha256):
    """Verify an exited Reset and return component evidence, never grasp qualification."""
    root = Path(__file__).resolve().parents[1]
    out = Path(run_dir).resolve(strict=True)
    run = out / 'native'
    raw_outer = checked_file(receipt, expected_sha256)
    outer = json.loads(raw_outer)
    _require(type(outer.get('exit_code')) is int and outer['exit_code'] == 0
             and type(outer.get('debugger_inferior_exit_code')) is int
             and outer['debugger_inferior_exit_code'] == 0 and (outer.get('debugger_status_verified') is True) and (outer.get('debugger_inferior_signal') is None) and (outer.get('forced_exit') is False) and (outer.get('exceeded_budget') is False), 'natural exit zero without signal or forced termination required')
    plan = json.loads((run / 'frozen_plan.json').read_bytes())
    _require(
        plan['version'] == 'dh116-fresh-supported-tripod-reset-capture-v1' and plan['style'] == 'tripod',
        'fresh17 Tripod plan version/style differs',
    )
    _require(
        plan['budget']['sdk_steps'] == plan['budget']['sdk_setup_steps'] == 2,
        'exact two-step SDK setup budget required',
    )
    _require(
        plan['budget']['episode_resets'] == plan['budget']['num_envs'] == 1,
        'single environment and one Reset required',
    )
    _require(
        all((plan['budget'][key] == 0 for key in ('control_steps', 'solver_steps', 'ppo_updates', 'teacher_queries', 'model_loads', 'auto_retries'))),
        'zero control/solver/PPO/retry budget required',
    )
    required = {'frozen_plan.json', 'result.json', 'reset_capture.json', 'episode_reset.json', 'mesh_reset_state.json', 'mesh_reset_numeric_state.json', 'pre_action_common.json', 'reset_geometry.json', 'cleanup_stages.json', 'app_close_returned.json', 'stop_commit.json'}
    _require(required <= set(outer['artifact_sha256']), 'required native artifact inventory incomplete')
    _require(
        all((isinstance(name, str) and Path(name).name == name for name in outer['artifact_sha256'])),
        'native artifact paths must be direct files',
    )
    _require(plan == json.loads((out / 'plan.json').read_bytes()), 'inner and outer frozen plans differ')
    _require(
        digest(plan) == outer['frozen_sha256'] and plan['output'] == str(run),
        'frozen fingerprint/output identity differs',
    )
    _require(
        isinstance(outer['head'], str) and re.fullmatch('[0-9a-f]{40}', outer['head']),
        'full frozen source commit required',
    )
    _require(
        outer['exit_code'] == outer['debugger_inferior_exit_code'] == 0,
        'actual debugger and outer exit codes differ',
    )
    _require(
        outer['debugger_status_verified'] is True and outer['debugger_inferior_signal'] is None,
        'independent debugger signal verification missing',
    )
    _require(
        outer['forced_exit'] is False and outer['exceeded_budget'] is False,
        'forced or timed-out run cannot qualify',
    )
    _require(
        0 < outer['elapsed_s'] <= plan['budget']['soft_wall_seconds'] == 180,
        'run exceeded the explicit180-second wall budget',
    )
    _require(
        hashlib.sha256((out / 'native.log').read_bytes()).hexdigest() == outer['native_log_sha256'],
        'native debugger log SHA differs',
    )
    for (key, value) in debugger_status(out / 'native.log').items():
        _require(outer[key] == value, 'actual debugger log status differs from outer receipt')
    for (name, sha) in outer['artifact_sha256'].items():
        checked_file(run / name, sha)
    publication_metadata = {str(root / rel): rel for rel in ('docs/contracts/dh116_work_package_registry.json', 'docs/contracts/dh116_release_manifest.json')}
    for (path, sha) in plan['frozen_files'].items():
        if path in publication_metadata:
            blob = subprocess.check_output(['git', '-c', 'safe.directory=' + str(root), 'show', outer['head'] + ':' + publication_metadata[path]])
            _require(
                hashlib.sha256(blob).hexdigest() == sha,
                'frozen Git publication bookkeeping SHA differs',
            )
        else:
            checked_file(path, sha)
    life = json.loads((run / 'result.json').read_bytes())
    _require(
        life['failure'] is None and life['close_mode'] == 'normal',
        'natural lifecycle completion required',
    )
    _require(
        life['lifetime_released_before_close'] is True and life['play_simulations'] is False,
        'resource release and frozen player required',
    )
    _require(
        life['physics_steps'] == 2 and life['measurement_steps'] == life['rod_steps'] == life['ppo_updates'] == 0,
        'actual SDK/control/solver/PPO counters differ',
    )
    close = json.loads((run / 'app_close_returned.json').read_bytes())
    _require(close['app_close_returned'] is True, 'normal App.close return missing')
    stages = json.loads((run / 'cleanup_stages.json').read_bytes())
    _require(stages['failure'] is None, 'lifecycle cleanup failed')
    for (name, row) in stages['stages'].items():
        if row['attempted']:
            _require(row['returned'] and row['error'] is None, name)
    result = json.loads((run / 'reset_capture.json').read_bytes())
    _require(
        result == life['acquisition'] and result['sdk_step'] == 2 and (result['actual_rod_tick'] == 0),
        'acquisition and lifecycle clock/source identities differ',
    )
    _require(
        result['mesh_count'] == 17 and result['pair_capacity'] == 8364 and (result['clearance_passed'] is True),
        'actual17 slots and8364 pair capacity required',
    )
    _require(
        all((result[k] == 0 for k in ('new_control_steps', 'physics_rows', 'solver_steps', 'ppo_updates'))),
        'Reset acquisition advanced control/solver/PPO',
    )
    data = decode_owned(json.loads((run / 'mesh_reset_state.json').read_bytes()))
    packet = data['readback']
    completed = packet.pop('completed_reset_receipt')
    reset_event = decode_owned(json.loads((run / 'episode_reset.json').read_bytes()))
    reset = reset_event['reset']
    _require('mesh_state_readback' not in completed, 'callback receipt must precede readback attachment')
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_order import _digest
    _require(
        _digest(reset['mesh_state_readback']) == _digest(packet),
        'returned Reset mesh readback differs from durable callback',
    )
    _require(
        _digest({k: v for (k, v) in reset.items() if k != 'mesh_state_readback'}) == _digest(completed),
        'Reset callback and returned receipt differ',
    )
    profile = BackendProfile.load(root)
    _require(profile.digest == plan['backend_profile_sha256'], 'current physical backend profile differs')
    for item in profile.inventory:
        _require(
            plan['frozen_files'][str(Path(item.path).resolve(strict=True))] == item.sha256,
            'complete frozen profile source inventory differs',
        )
    for rel in ('scripts/record_dh116_tripod_supported_reset.py', 'src/go2_pvcnn/control/dh116/tabletop_fresh_supported_builder.py'):
        _require(str(root / rel) in plan['frozen_files'], 'native worker/constructor source pins missing')
    with profile_reader_scope(profile):
        mesh_audit = validate_supported_readback(packet, reset_receipt=reset, mesh_plan=data['mesh_plan'])
        numeric_audit = validate_numeric_reset_audit(json.loads((run / 'mesh_reset_numeric_state.json').read_bytes()), reset_receipt=reset)
    common = decode_owned(json.loads((run / 'pre_action_common.json').read_bytes()))
    (values, valid) = (np.asarray(common['values']), np.asarray(common['valid']))
    from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import validate_mounted_observation
    common_audit = validate_mounted_observation(common)
    _require(
        values.shape == valid.shape == (371,) and valid.dtype == np.bool_ and (valid.sum() == 66),
        'original371D shape and66 measured fields required',
    )
    _require(
        common_audit['valid_channels'] == 66 and common_audit['unknown_channels'] == 305,
        'measured66/unknown305 mask differs',
    )
    _require(
        np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all(),
        'common valid values must be finite and unknown fields NaN',
    )
    for (key, value) in reset['after_clock'].items():
        _require(common['identity'][key] == value, 'common pre-action and Reset clocks differ')
    _require(
        common['identity']['phase'] == 'pre_action_before_target_write' and common['identity']['policy_step'] == 0,
        'same pre-action phase and policy step0 required',
    )
    geometry = decode_owned(json.loads((run / 'reset_geometry.json').read_bytes()))
    _require(
        [row['side'] for row in geometry] == ['left', 'right'],
        'independent left/right geometry order required',
    )
    hulls = {}
    for row in geometry:
        _require(
            row['kind'] == 'initial_geometry' and row['evidence']['initial_geometry_passed'] is True,
            'initial native geometry failed',
        )
        for group in ('source_colliders', 'locked_pads'):
            _require(
                len(row['evidence'][group]) == (17 if group == 'source_colliders' else 5),
                'all original source/cooked shapes required',
            )
            for (name, shape) in row['evidence'][group].items():
                hulls[row['side'] + '.' + group + '.' + name] = shape['world_vertices']
    clearance = guard_hand_supports(hulls, json.loads(Path(plan['task_config']['path']).read_bytes())['support_scene'])
    _require(
        clearance['minimum_gap_m'] > 0 and len(hulls) == 44,
        'all44 envelopes must clear the original supports',
    )
    report = dict(
        version='dh116-fresh17-tripod-native-audit-v1',
        source_head=outer['head'],
        frozen_sha256=outer['frozen_sha256'],
        outer_sha256=hashlib.sha256(raw_outer).hexdigest(),
        natural_exit_code=0,
        signal=None,
        forced_exit=False,
        wall_s=outer['elapsed_s'],
        sdk_steps=2,
        control_steps=0,
        solver_steps=0,
        ppo_updates=0,
        episode_resets=1,
        mesh_audit=mesh_audit,
        numeric_audit=numeric_audit,
        common_valid_dimensions=66,
        common_unknown_dimensions=305,
        clearance_m=np.asarray(reset_event['clearance_m']).tolist(),
        minimum_support_gap_m=clearance['minimum_gap_m'],
        source_shapes=34,
        cooked_envelopes=10,
        mesh_plan_sha256=data['mesh_plan']['sha256'],
        reset_seed_sha256=result['reset_seed_sha256'],
        artifact_sha256=outer['artifact_sha256'],
        publication_metadata_source='exact frozen Git blob for registry/release only; physical files rehashed current',
        native_reset_component_verified=True,
        teacher_admission=False,
        grasp_qualified=False,
    )
    _require(result['reset_seed_sha256'] == reset['seed']['sha256'], 'Reset seed SHA binding differs')
    _require(
        result['mesh_plan_sha256'] == data['mesh_plan']['sha256'],
        'actual mesh plan SHA binding differs',
    )
    _require(
        reset_event['clearance_floor_m'] == 0.0021 and reset_event['clearance_passed'] is True,
        'original2.1mm wrist protection changed',
    )
    _require(
        (np.asarray(reset_event['clearance_m']) >= 0.0021).all(),
        'measured Reset table clearance below original floor',
    )
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--receipt-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        raise ValueError("audit evidence output must remain outside the project")
    report=audit_run(args.run,receipt=args.receipt,expected_sha256=args.receipt_sha256)
    from scripts.diagnose_dh116_teacher_initialization import write_new
    write_new(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!="artifact_sha256"},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
