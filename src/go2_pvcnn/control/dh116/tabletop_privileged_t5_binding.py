"""Derive the Teacher's policy-only entry from the verified original T5 task."""
import argparse
import json
from pathlib import Path

from .tabletop_privileged_warm_start import _checked_bytes
from .tabletop_stereo_contract import STYLES, file_sha256, new_run, write_json_new
from .tabletop_stereo_native import native_task_identity
from .tabletop_surface_contract import semantic_sha256, validate_contract
from .tabletop_vision_evaluation_contract import evaluation_physics_sha256


T5_CONTRACT_RAW_SHA256 = 'eda0b7e316a493997192ffec9abd8b5648c826a265d817774dd9117c734ec551'
T5_CONTRACT_SEMANTIC_SHA256 = 'cc7d9f525f6ebeb04ff97a2ad0ba9afac02767f897ff3f14558c33c1ae56a0d6'
T5_PHYSICS_SHA256 = '7128e10e341316dd76132eec5ee1d5af6f769691eecbe281d99a26b974c00612'
VERSION = 'dh116-privileged-teacher-t5-binding-v1'
_DYNAMIC = {'--num-envs', '--seed', '--rollout-steps', '--max-iterations', '--output'}
_EXPERT_EXECUTION = {'--vision-teacher-source', '--vision-teacher-mpc', '--vision-teacher-mpc-profile',
    '--vision-teacher-max-control-steps', '--vision-teacher-hold-tail-control-steps',
    '--vision-teacher-raw-chunk-frames', '--vision-sequence-context-sidecar', '--vision-mpc-state-capture'}


def _policy_launch_arguments(arguments):
    required = {'--object-kind=solid_cylinder', '--physics-hz=480', '--grasp-controller=policy',
        '--vision-student', '--surface-feedback', '--native-sensor-batch',
        '--surface-reward-profile=support_deficit'}
    if (not required.issubset(arguments)
            or len({token.split('=', 1)[0] for token in arguments}) != len(arguments)):
        raise ValueError('original T5 controller, batched measurements and native reward are required')
    return [token for token in arguments if token.split('=', 1)[0] not in (_DYNAMIC | _EXPERT_EXECUTION)]


def verify_runtime_contract(contract, spec):
    if 'teacher_t5_binding' not in spec:
        return
    binding = spec['teacher_t5_binding']
    if (binding.get('version') != VERSION or binding.get('source_contract_sha256') != T5_CONTRACT_RAW_SHA256
            or binding.get('expected_semantic_sha256') != T5_CONTRACT_SEMANTIC_SHA256
            or binding.get('expected_physics_sha256') != T5_PHYSICS_SHA256
            or _policy_launch_arguments(spec['arguments']) != spec['arguments']):
        raise ValueError('invalid T5 policy-only task binding')
    if semantic_sha256(contract) != T5_CONTRACT_SEMANTIC_SHA256 or evaluation_physics_sha256(contract) != T5_PHYSICS_SHA256:
        raise ValueError('actual native contract differs from the preserved T5 physical and R0 reward contract')


def derive_task(plan, plan_sha256, t5_launch, original_contract, output):
    # Use the exact same pinned archived command, input/asset path check and
    # reviewed option whitelist as full T5 MPC. This path never calls run_native.
    from scripts.collect_dh116_mpc_privileged import preflight
    contract = json.loads(_checked_bytes(original_contract, T5_CONTRACT_RAW_SHA256, 'observed T5 tensor contract'))
    validate_contract(contract)
    output = Path(output).absolute()
    arguments, inputs, capture = preflight(plan, plan_sha256, output / 'unused_capture',
        t5_first_episodes=t5_launch)
    selected = _policy_launch_arguments(arguments)
    spec = dict(version='dh116-stereo-native-bootstrap-v1', task_kind='rigid_proxy',
        physical_object_kind='solid_cylinder', grasp_styles=list(STYLES), arguments=selected,
        teacher_t5_binding=dict(version=VERSION, source_contract_sha256=T5_CONTRACT_RAW_SHA256,
            expected_semantic_sha256=T5_CONTRACT_SEMANTIC_SHA256, expected_physics_sha256=T5_PHYSICS_SHA256,
            launch_sha256=capture['t5_launch']['sha256'], baseline_plan_sha256=capture['baseline_plan']['sha256']))
    verify_runtime_contract(contract, spec)
    task_identity = native_task_identity(spec)
    output = new_run(output)
    task_path = output / 'native_task.json'
    write_json_new(task_path, spec)
    contract_path = output / 'reviewed_t5_contract_binding.json'
    write_json_new(contract_path, dict(inherited_contract=contract, task_identity=task_identity,
        task_identity_sha256=task_identity['task_identity_sha256'],
        inherited_launcher_sha256=capture['launcher']['sha256'],
        source_scope='original_actual_t5_tensor_contract_plus_derived_policy_task_identity',
        source_contract_sha256=T5_CONTRACT_RAW_SHA256, derived_native_task_sha256=file_sha256(task_path),
        native_execution_performed=False, native_grasp_qualification=False))
    receipt = dict(version=VERSION, derived_task=str(task_path), derived_task_sha256=file_sha256(task_path),
        reviewed_contract=str(contract_path), reviewed_contract_sha256=file_sha256(contract_path),
        inherited_inputs=inputs, omitted_options=sorted(token.split('=', 1)[0] for token in arguments if token not in selected),
        optimizer_reward='teacher_r0_existing_passthrough_of_original_T5_native_support_deficit',
        native_execution_performed=False, native_grasp_qualification=False)
    write_json_new(output / 'derivation.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-plan', 't5-launch', 't5-tensor-contract', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--baseline-plan-sha256', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(derive_task(args.baseline_plan, args.baseline_plan_sha256, args.t5_launch,
        args.t5_tensor_contract, args.output), indent=2))
    return 0
