"""Preflight or explicitly run a bounded, independent MPC capture on the Ubuntu host."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import (
    PreActionCapture, capture_environment, file_identity, finish_capture,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/train_dh116_tabletop_gpu_ppo.py'
NATIVE_PYTHON = Path('/mnt/mydisk/xk/envs/dh116/bin/python')
INPUTS = {
    '--pose-summary', '--grasp-initialization', '--mixed-grasp-initialization',
    '--load-profile', '--depth-camera-config', '--vision-teacher-source',
}
FLAGS = {
    '--adaptive-grasp', '--force-feedback', '--full-hand-contact', '--headless',
    '--direct-training', '--tensor-pipeline', '--training-episode-resets',
    '--pressure-relief', '--skip-hold', '--close-downward-penalty',
    '--grasp-shaping', '--vertical-force-reward', '--force-region-control',
    '--phase-force-reward', '--free-grasp-lift', '--continuous-motion',
    '--free-wrist-grasp', '--surface-force-sharing', '--native-sensor-batch',
    '--vision-student', '--enable_cameras', '--surface-feedback',
    '--vision-teacher-mpc', '--vision-sequence-context-sidecar', '--vision-mpc-state-capture',
}
VALUE_OPTIONS = INPUTS | {
    '--object-kind', '--object-material', '--hand-material', '--num-envs',
    '--rollout-steps', '--max-iterations', '--physics-hz', '--mimic-frequency',
    '--solver-type', '--position-iterations', '--velocity-iterations',
    '--reset-height-jitter-m', '--cpu-threads', '--device', '--grasp-controller',
    '--palm-transport', '--palm-exploration-axis', '--passive-velocity-limit-rad-s',
    '--contact-reward-window', '--grasp-readiness', '--vertical-force-reward-mode',
    '--force-reward-start', '--end-support-clearance-m', '--surface-diagnostic-steps',
    '--output', '--kit_args', '--physics-clone-mode', '--vision-width',
    '--vision-height', '--vision-camera-batch-size', '--vision-ppo-microbatch-size',
    '--vision-inference-batch-size', '--vision-visibility-mode', '--vision-camera-mode',
    '--surface-noise-mode', '--surface-reward-profile', '--vision-action-distribution',
    '--seed', '--vision-teacher-max-control-steps', '--vision-teacher-raw-chunk-frames',
    '--vision-teacher-mpc-profile', '--vision-teacher-hold-tail-control-steps',
}
REQUIRED = {
    '--object-kind': 'solid_cylinder', '--physics-hz': '480',
    '--vision-camera-mode': 'fixed', '--vision-width': '224', '--vision-height': '96',
    '--vision-action-distribution': 'tanh_gaussian',
    '--vision-teacher-mpc-profile': 'pressure_geometry_constrained',
    '--max-iterations': '1', '--device': 'cuda:0',
}


def baseline_arguments(tokens, bound_inputs, output):
    options = {}
    it = iter(tokens)
    for token in it:
        option, equals, inline = token.partition('=')
        if option in FLAGS and not equals:
            value = None
        elif option in VALUE_OPTIONS:
            value = inline if equals else next(it, '')
            if not value or (not equals and value.startswith('--')):
                raise ValueError(f'missing baseline option value: {option}')
        else:
            raise ValueError(f'unsupported baseline option: {option}')
        if option in options and options[option] != value:
            raise ValueError(f'conflicting baseline options: {option}')
        options[option] = value
    if not FLAGS <= options.keys() or any(options.get(name) != value for name, value in REQUIRED.items()):
        raise ValueError('baseline must use the original rigid 480Hz constrained MPC capture path')
    if set(bound_inputs) != INPUTS or any(options.get(k) != v['path'] for k, v in bound_inputs.items()):
        raise ValueError('all original native inputs must be explicitly bound by path and SHA')
    options.update({'--output': str(output), '--num-envs': '2', '--vision-teacher-max-control-steps': '32'})
    arguments = []
    for option, value in options.items():
        arguments.extend([option] if value is None else [f'{option}={value}'])
    return arguments


def check_inputs(inputs):
    for value in inputs:
        if file_identity(value['path'])['sha256'] != value['sha256']:
            raise ValueError(f'baseline input SHA differs: {value["path"]}')


def preflight(plan_path, expected_sha, output):
    identity = file_identity(plan_path)
    if identity['sha256'] != expected_sha:
        raise ValueError('reviewed baseline plan SHA differs')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    command = plan['command']
    if (plan['version'] != 'dh116-stage7-private-native-state-launch-v1'
            or command[0] != 'bash' or Path(command[1]).name != 'python.sh'
            or Path(command[2]).name != LAUNCHER.name):
        raise ValueError('reviewed Stage7 native capture plan required')
    inputs = [*plan['bound_inputs'].values(), *plan['bound_asset_dependencies']]
    arguments = baseline_arguments(command[3:], plan['bound_inputs'], output)
    check_inputs(inputs)
    return arguments, inputs, dict(baseline_plan=identity, launcher=file_identity(LAUNCHER),
        arguments=arguments, inputs=inputs, physics_hz=480, policy_hz=60, environments=2,
        max_policy_steps=32, qualified_for_bc=False)


def run_native(arguments, inputs, plan):
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_capture import _clock
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from go2_pvcnn.control.dh116.tabletop_surface_contract import validate_contract

    original = collection.run_teacher_collection
    result = None

    def observed_collection(world, device_world, task, env, args, contract, **kwargs):
        nonlocal result
        validate_contract(contract)
        recorder = PreActionCapture(args.num_envs, args.vision_teacher_max_control_steps)
        with capture_environment(env, recorder, lambda: _clock(device_world)):
            report = original(world, device_world, task, env, args, contract, **kwargs)
        check_inputs(inputs)
        result = finish_capture(args.output, recorder, contract, run_plan=plan)
        return dict(report, mpc_privileged_pre_action_capture=result)

    with native_instance_lock():
        original_argv, original_cwd = sys.argv, Path.cwd()
        collection.run_teacher_collection = observed_collection
        try:
            sys.argv = [str(LAUNCHER), *arguments]
            os.chdir(ROOT)
            try:
                runpy.run_path(str(LAUNCHER), run_name='__main__')
            except SystemExit as error:
                if error.code is not None and (type(error.code) is not int or error.code != 0):
                    raise
        finally:
            collection.run_teacher_collection = original
            sys.argv = original_argv
            os.chdir(original_cwd)
    if result is None:
        raise RuntimeError('native legacy collector did not finish the independent capture')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-plan', type=Path, required=True)
    parser.add_argument('--baseline-plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.output.is_symlink() or output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory under an existing experiment parent')
    arguments, inputs, plan = preflight(args.baseline_plan, args.baseline_plan_sha256, output)
    if args.execute_native:
        if Path(sys.executable) != NATIVE_PYTHON or os.environ.get('DH116_PYTHON') != str(NATIVE_PYTHON):
            raise RuntimeError('native capture requires scripts/python.sh with the project Ubuntu interpreter')
        print(json.dumps(run_native(arguments, inputs, plan), allow_nan=False))
    else:
        print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
