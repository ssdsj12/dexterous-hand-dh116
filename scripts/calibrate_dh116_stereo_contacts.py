"""Explicit isolated thumb-link/rigid-probe source calibration, not a grasp task.

Uses original manufacturer thumb mesh, fixed mount and gravity-only probe.
This is NOT a full articulated hand or validation of R2/Force Proxy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _install_sigterm_monitor(output):
    def interrupted(signum, _frame):
        _write(output/'external_signal.json', dict(pid=os.getpid(), signal=signal.Signals(signum).name))
        raise SystemExit(128+signum)

    previous = signal.signal(signal.SIGTERM, interrupted)
    _write(output/'sigterm_monitor.json', dict(pid=os.getpid(), installed_after_app_launch=True))
    return previous


def _launch_calibration_app(launcher):
    launch=argparse.Namespace(headless=True, device='cuda:0', enable_cameras=False,
        kit_args='--/renderer/activeGpu=3 --/renderer/multiGpu/enabled=false '
                 '--/renderer/multiGpu/autoEnable=false --/physics/cudaDevice=0 '
                 '--/plugins/carb.tasking.plugin/threadCount=32')
    # Isaac Sim 4.5 fast shutdown can exit before close() returns to Python.
    return launcher(launch, fast_shutdown=False).app


def _record_shutdown_mode(output, settings):
    setting='/app/fastShutdown'
    effective=settings.get(setting)
    if effective is not False:
        raise RuntimeError(f'{setting} must read back as False after AppLauncher, got {effective!r}')
    _write(output/'shutdown_mode.json', dict(pid=os.getpid(), setting=setting,
        effective_fast_shutdown=effective))


def run(args):
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import ROOT
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import (
        capture_tensor_pair, read_binary_stl, summarize_pair_calibration, configure_report_processing,
        decode_contact_report, close_calibration_app, ContactReportReceiptBuffer, calibration_episode_uid)
    import numpy as np
    with native_instance_lock():
        from isaaclab.app import AppLauncher
        app=_launch_calibration_app(AppLauncher)
        sim=None
        full_subscription=None
        ordinary_subscription=None
        previous_sigterm=None
        try:
            if args.monitor_sigterm:
                previous_sigterm=_install_sigterm_monitor(args.output)
            import isaaclab.sim as sim_utils
            import omni.usd
            import omni.physics.tensors as tensors
            from omni.physx import get_physx_simulation_interface
            from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Vt, PhysicsSchemaTools
            import carb
            settings=carb.settings.get_settings()
            _record_shutdown_mode(args.output, settings)
            sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/480,device='cuda:0',
                gravity=(0.,0.,-9.81),physx=sim_utils.PhysxCfg(solver_type=0)))
            reporting=configure_report_processing(settings,enable=args.enable_full_report_processing)
            stage=omni.usd.get_context().get_stage()
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            side='L' if args.side=='left' else 'R'
            mesh=ROOT/f'resources/manufacturer/DH116V2.0_URDF_MJCF_Files/DH116-V2.0-{side}/meshes/finger14_link.STL'
            triangles=read_binary_stl(mesh)
            vertices=triangles.reshape(-1,3)
            finger='/World/Finger14'
            obj='/World/CalibrationObject'
            body=UsdGeom.Xform.Define(stage,finger).GetPrim()
            rigid=UsdPhysics.RigidBodyAPI.Apply(body)
            rigid.CreateKinematicEnabledAttr(True)
            shape=UsdGeom.Mesh.Define(stage,finger+'/collision')
            shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
            shape.CreateFaceVertexCountsAttr([3]*len(triangles))
            shape.CreateFaceVertexIndicesAttr(list(range(len(vertices))))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
            # An isolated source-calibration geometry. Original task assets,
            # materials, actuators, controllers and checkpoints are not touched.
            sphere=UsdGeom.Sphere.Define(stage,obj)
            radius=.004
            center=(vertices.min(axis=0)+vertices.max(axis=0))/2
            center[2]=vertices[:,2].max()+radius+.003
            sphere.CreateRadiusAttr(radius)
            sphere.AddTranslateOp().Set(Gf.Vec3d(*map(float,center)))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            dynamic=UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
            dynamic.CreateVelocityAttr(Gf.Vec3f(.02,0.,0.))
            UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(.02)
            rigid_ext=PhysxSchema.PhysxRigidBodyAPI.Apply(sphere.GetPrim())
            rigid_ext.CreateLinearDampingAttr(0.)
            rigid_ext.CreateAngularDampingAttr(0.)
            for prim in (body,sphere.GetPrim()):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            usd=args.output/'calibration_scene.usda'
            stage.Export(str(usd))
            inventory=dict(scope='isolated_actual_thumb_mesh_fixed_mount_not_articulated_hand',
                episode_uid=calibration_episode_uid(args.output,args.side),
                reporting=reporting,
                callback_channels_requested=dict(full=args.full_contact_callback,ordinary=args.ordinary_contact_callback),
                side=args.side, object_collider=obj, finger_collider=finger+'/collision', mesh_path=str(mesh), mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
                stage_sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),
                meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage), dt_s=1/480,
                collisions=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)],
                joints=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)],
                authored_force_attributes=[str(a.GetPath()) for p in stage.Traverse() for a in p.GetAttributes()
                    if a.HasAuthoredValueOpinion() and any(s in a.GetName().lower() for s in ('force','attachment'))],
                gravity_m_s2=[0.,0.,-9.81], original_task_modified=False)
            _write(args.output/'inventory.json',inventory)
            source=get_physx_simulation_interface()
            def decode(raw):
                return decode_contact_report(raw,path_decoder=PhysicsSchemaTools.intToSdfPath)
            callbacks=ContactReportReceiptBuffer(
                clock_reader=lambda:(sim.current_time_step_index,sim.current_time),
                path_decoder=PhysicsSchemaTools.intToSdfPath)
            def full_callback(*raw):
                callbacks.record('full',raw)
            def ordinary_callback(*raw):
                callbacks.record('ordinary',raw)
            if args.full_contact_callback:
                full_subscription=source.subscribe_full_contact_report_events(full_callback)
            if args.ordinary_contact_callback:
                ordinary_subscription=source.subscribe_contact_report_events(ordinary_callback)
            sim.reset()
            view=tensors.create_simulation_view('torch')
            view.set_subspace_roots('/')
            contact=view.create_rigid_contact_view(obj,[finger],max_contact_data_count=64)
            reverse=view.create_rigid_contact_view(finger,[obj],max_contact_data_count=64)
            rigid_view=view.create_rigid_body_view(obj)
            mass=float(rigid_view.get_masses().detach().cpu().reshape(-1)[0])
            rows=[]
            with (args.output/'raw_contact_480hz.jsonl').open('x',encoding='utf-8') as stream:
                for step in range(1,args.steps+1):
                    before=rigid_view.get_velocities().detach().cpu().reshape(-1,6)[0,:3].tolist()
                    callbacks.begin_step(step)
                    try:
                        sim.step(render=False)
                    finally:
                        callbacks.end_step()
                    after=rigid_view.get_velocities().detach().cpu().reshape(-1,6)[0,:3].tolist()
                    row=dict(env_id=0,episode_uid=inventory['episode_uid'],policy_step=None,
                        physics_step=step,sdk_step=int(sim.current_time_step_index),timestamp=float(sim.current_time),
                        dt=1/480,mass_kg=mass,gravity_m_s2=[0.,0.,-9.81],velocity_before=before,velocity_after=after,
                        pose=rigid_view.get_transforms().detach().cpu().tolist(),
                        tensor=capture_tensor_pair(contact,dt=1/480),
                        reverse_tensor=capture_tensor_pair(reverse,dt=1/480),
                        full_report=decode(source.get_full_contact_report()),callback_receipts=callbacks.drain(),
                        report_processing_settings=dict(disable_contact_processing=carb.settings.get_settings().get('/physics/disableContactProcessing')))
                    rows.append(row)
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    stream.flush()
                os.fsync(stream.fileno())
            summary=summarize_pair_calibration(rows,object_path=obj,expected_partner=finger,inventory=inventory)
            summary['sampling_complete']=True
            summary['raw_fsync_complete']=True
            summary['process_exit_verified']=False
            summary['raw_sha256']=hashlib.sha256((args.output/'raw_contact_480hz.jsonl').read_bytes()).hexdigest()
            _write(args.output/'summary.json',summary)
        finally:
            try:
                subscribed = dict(full=full_subscription is not None, ordinary=ordinary_subscription is not None)
                full_subscription=None
                ordinary_subscription=None
                _write(args.output/'callbacks_released.json', subscribed)
            finally:
                try:
                    close_calibration_app(app,args.output,simulation=sim)
                finally:
                    if previous_sigterm is not None:
                        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-native',action='store_true')
    parser.add_argument('--enable-full-report-processing',action='store_true')
    parser.add_argument('--full-contact-callback',action='store_true')
    parser.add_argument('--ordinary-contact-callback',action='store_true')
    parser.add_argument('--monitor-sigterm',action='store_true')
    parser.add_argument('--side',choices=('left','right'),required=True)
    parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if not args.execute_native or not 1<=args.steps<=480:
        parser.error('explicit execution and bounded 1..480 physical steps required')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        run(args)
    except Exception as error:
        _write(args.output/'failed.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
