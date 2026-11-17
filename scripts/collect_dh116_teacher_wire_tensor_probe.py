"""Bounded original GPU Wire/contact Tensor readback with no force qualification."""
import argparse
import gzip
import json
import os
from pathlib import Path

import torch

from go2_pvcnn.control.dh116.tabletop_privileged_cli import (
    collect_or_evaluate, load_config, write_native_callback_completion,
)
from go2_pvcnn.control.dh116.tabletop_stereo_contract import ROOT, file_sha256, new_run, write_json_new
from go2_pvcnn.control.dh116.tabletop_stereo_native import launch_native
from go2_pvcnn.control.dh116.tabletop_teacher_wire_tensor_probe import WireRootForceTap, WireTensorProbe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-task-config', type=Path,
        default=ROOT/'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--policy-steps', type=int, default=16)
    parser.add_argument('--capacity-per-wire', type=int, default=256)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--enable-experiment', action='store_true')
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.policy_steps <= 32 or not 64 <= args.capacity_per_wire <= 1024:
        parser.error('bounded 1..32 controls and 64..1024 contact capacity per Wire required')
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        parser.error('frozen checkpoint SHA differs')
    if args.output.exists() or args.output.is_symlink():
        parser.error('independent new output required')
    config, ppo = load_config(args.config)
    plan = dict(environments=2, policy_steps=args.policy_steps, physics_hz=480, control_hz=60,
        capacity_per_wire=args.capacity_per_wire, native_task_config_sha256=file_sha256(args.native_task_config),
        checkpoint_sha256=args.checkpoint_sha256, training=False, force_qualification=False,
        full_report_settings_modified=False)
    if not args.execute_native:
        print(json.dumps(dict(plan, native_execution_performed=False), indent=2))
        return 0
    if not args.enable_experiment:
        parser.error('explicit --enable-experiment required')
    run = new_run(args.output)
    write_json_new(run/'wire_tensor_probe_plan.json', plan)
    torch.manual_seed(args.seed)

    def collect(bridge):
        import carb
        import omni.usd

        from go2_pvcnn.control.dh116.tabletop_privileged_t5_binding import verify_runtime_contract
        from go2_pvcnn.control.dh116.tabletop_stereo_camera import StereoCamera

        verify_runtime_contract(bridge.contract, json.loads(args.native_task_config.read_text(encoding='utf-8')))
        bridge.bind_camera(StereoCamera.from_native(bridge.world.sim, bridge.device_world.origins,
            ROOT/'configs/dh116_training/stereo_camera_640x480_v2.json', device=bridge.device))
        probe = WireTensorProbe(bridge, omni.usd.get_context().get_stage(), carb.settings.get_settings(),
            capacity_per_wire=args.capacity_per_wire)
        inventory = run/'wire_tensor_probe_inventory.json'
        write_json_new(inventory, probe.inventory)
        observer = bridge.env.evaluation_telemetry
        original = observer.callback
        path = run/'wire_tensor_probe_480hz.jsonl.gz'
        rows = valid = associated = unmatched_older = 0
        previous = None
        errors = {}

        def clock():
            sim = bridge.world.sim
            return (int(bridge.device_world._physics_steps), int(sim.current_time_step_index), float(sim.current_time))

        tap = WireRootForceTap(bridge.device_world.wire.root_physx_view, clock)
        unmatched_path = run/'wire_root_sdk_force_unassociated.json'
        with gzip.open(path, 'xt', encoding='utf-8') as stream, tap.install():
            def capture(packet):
                nonlocal rows, valid, previous, associated, unmatched_older
                world_step, sdk_step, time = clock()
                if previous is not None and (world_step != previous[0]+1 or sdk_step != previous[1]+1
                        or abs(time-previous[2]-1/480) > 1e-7):
                    raise RuntimeError('native probe clock hole or duplicate')
                row = dict(run_uuid=bridge.identities.run_uuid, policy_tick=bridge.policy_tick,
                    physics_step=world_step, sdk_step=sdk_step, timestamp=time,
                    envs=[dict(env_id=i, episode_uid=bridge.identities.uid(i), terminal=bool(packet['done'][i]))
                          for i, live in enumerate(packet['live'].tolist()) if live],
                    force_qualification=False, external_support_qualification=False)
                if bridge.device_world.wire.root_physx_view is not tap.view:
                    raise RuntimeError('actual Wire root view changed during diagnostic SDK write tap')
                row['wire_root_sdk_write_audit'] = tap.observed_step((world_step, sdk_step, time))
                associated += len(row['wire_root_sdk_write_audit']['associated_calls'])
                unmatched_older += len(row['wire_root_sdk_write_audit']['unmatched_older_calls'])
                try:
                    row['readback'] = probe.sample()
                    valid += 1
                except Exception as error:
                    message = type(error).__name__ + ": " + str(error)
                    row['readback_error'] = message
                    errors[message] = errors.get(message, 0) + 1
                if clock() != (world_step, sdk_step, time):
                    raise RuntimeError('contact readback altered the native clock')
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                original(packet)
                rows += 1
                previous = (world_step, sdk_step, time)

            observer.callback = capture
            try:
                outcome = collect_or_evaluate(bridge, config, ppo, checkpoint=args.checkpoint,
                    output=run, max_steps=args.policy_steps)
            finally:
                observer.callback = original
                write_json_new(unmatched_path, dict(unassociated_calls=tap.unassociated_calls,
                    observed_calls=tap.count, external_wrench_absence_proven=False))
                with unmatched_path.open('rb') as receipt:
                    os.fsync(receipt.fileno())
        if tap.count != associated + unmatched_older + len(tap.unassociated_calls):
            raise RuntimeError('Wire root SDK audit write count did not reconcile')
        result = dict(outcome, native_tensor_observation_only=True, tensor_probe_physical_steps=rows,
            tensor_probe_complete_steps=valid, tensor_probe_errors=errors, force_qualification=False,
            external_support_qualification=False, native_grasp_qualification=False,
            wire_root_sdk_force_calls=tap.count, wire_root_sdk_force_associated=associated,
            wire_root_sdk_force_unmatched_older=unmatched_older,
            wire_root_sdk_force_unassociated=len(tap.unassociated_calls))
        write_native_callback_completion(run, result, [run/'wire_tensor_probe_plan.json', inventory,
            path, unmatched_path, run/'teacher_evaluation.json', bridge.output/'native_physics.jsonl',
            bridge.output/'native_task_contract.json'])
        return result

    launch_native(collect, native_arguments=args.native_task_config, output=run/'native', num_envs=2,
        seed=args.seed, rollout_steps=config['rollout_steps'], enabled=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
