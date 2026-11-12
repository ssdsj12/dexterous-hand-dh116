"""Bounded 2×32 native contact report probe on the existing Teacher task."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from go2_pvcnn.control.dh116.tabletop_privileged_cli import collect_or_evaluate, load_config
from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import configure_report_processing, decode_contact_report
from go2_pvcnn.control.dh116.tabletop_stereo_contract import ROOT, file_sha256, new_run, write_json_new
from go2_pvcnn.control.dh116.tabletop_stereo_native import launch_native
from go2_pvcnn.control.dh116.tabletop_teacher_task_contact import (
    install_probe_world, probe_inventory, route_object_reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args(argv)
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        parser.error('frozen Teacher checkpoint SHA does not match')
    config, ppo = load_config(args.config)
    if args.output.exists():
        parser.error('independent new output required')
    plan = dict(environments=2, policy_steps=32, physics_hz=480,
                checkpoint_sha256=args.checkpoint_sha256, independent_probe_only=True,
                training=False, force_qualification=False)
    if not args.execute_native:
        print(json.dumps(plan, indent=2))
        return
    run = new_run(args.output)
    write_json_new(run/'contact_probe_plan.json', plan)
    torch.manual_seed(116)

    def collect(bridge):
        import carb
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

        from go2_pvcnn.control.dh116.tabletop_stereo_camera import StereoCamera
        camera_path = ROOT/'configs/dh116_training/stereo_camera_640x480_v2.json'
        bridge.bind_camera(StereoCamera.from_native(bridge.world.sim, bridge.device_world.origins,
            camera_path, device=bridge.device))
        settings = carb.settings.get_settings()
        inventory = probe_inventory(bridge, omni.usd.get_context().get_stage(), UsdPhysics, PhysxSchema)
        inventory['reporting_before_first_reset'] = bridge.world.contact_probe_before_reset
        inventory['reporting_after_reset'] = {
            'disable_contact_processing': settings.get('/physics/disableContactProcessing'),
            'suppress_readback': settings.get('/physics/suppressReadback'),
        }
        if inventory['reporting_after_reset']['suppress_readback'] is not False:
            raise RuntimeError('initial simulation reset suppressed contact reports')
        write_json_new(run/'contact_probe_inventory.json', inventory)
        physx = get_physx_simulation_interface()
        observer = bridge.env.evaluation_telemetry
        original = observer.callback
        rows = 0
        last_step = None
        with (run/'contact_probe_480hz.jsonl').open('x', encoding='utf-8') as stream:
            def capture(packet):
                nonlocal rows, last_step
                step = int(bridge.device_world._physics_steps)
                sim = bridge.world.sim
                sdk_step, time = int(sim.current_time_step_index), float(sim.current_time)
                if last_step is not None and step != last_step + 1:
                    raise RuntimeError('object contact probe lost a native physical step')
                if settings.get('/physics/suppressReadback') is not False:
                    raise RuntimeError('native readback changed during contact probe')
                raw = decode_contact_report(physx.get_full_contact_report(),
                    path_decoder=PhysicsSchemaTools.intToSdfPath)
                if int(sim.current_time_step_index) != sdk_step or float(sim.current_time) != time:
                    raise RuntimeError('native contact getter changed the sample clock')
                pairs = route_object_reports(raw, objects=inventory['objects'], digits=inventory['digits'])
                live = [i for i, value in enumerate(packet['live'].tolist()) if value]
                row = dict(physics_step=step, sdk_step=sdk_step, timestamp=time,
                    run_uuid=bridge.identities.run_uuid, policy_tick=bridge.policy_tick,
                    episode_uid={i:bridge.identities.uid(i) for i in live},
                    pairs=pairs, full_report=raw, force_qualification=False)
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                original(packet)
                last_step = step
                rows += 1
            observer.callback = capture
            try:
                result = collect_or_evaluate(bridge, config, ppo, checkpoint=args.checkpoint,
                    output=run, max_steps=32)
            finally:
                observer.callback = original
                stream.flush()
                os.fsync(stream.fileno())
        result = dict(**result, contact_probe_physics_rows=rows,
            contact_probe_sha256=file_sha256(run/'contact_probe_480hz.jsonl'),
            force_qualification=False, actual_process_exit_verified=False)
        # The protected launcher may close Kit before returning to this script.
        with (run/'contact_probe_capture_complete.json').open('x', encoding='utf-8') as receipt:
            json.dump(result, receipt, indent=2, allow_nan=False)
            receipt.flush()
            os.fsync(receipt.fileno())
        return result

    import isaaclab.app as app_module

    with ExitStack() as stack:
        original_launcher = app_module.AppLauncher

        class ContactProbeLauncher(original_launcher):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                import carb
                from go2_pvcnn.control.dh116 import tabletop_gpu_isaac

                install_probe_world(stack, tabletop_gpu_isaac, lambda: configure_report_processing(
                    carb.settings.get_settings(), enable=True, enable_readback=True))

        stack.enter_context(patch.object(app_module, 'AppLauncher', ContactProbeLauncher))
        launch_native(collect, native_arguments=ROOT/'configs/dh116_training/stereo_native_task_v1.json',
            output=run/'native', num_envs=2, seed=116, rollout_steps=32, enabled=True)


if __name__ == '__main__':
    main()
