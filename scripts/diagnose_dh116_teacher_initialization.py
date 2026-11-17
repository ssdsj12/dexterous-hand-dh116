#!/usr/bin/env python3
"""Frozen initialization-only N=1 Pinch worker. Default is CPU plan creation.

No Teacher/model query, control action, solver step, or new Power SDF surface.
Native execution and natural process exit remain separate external evidence.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

VERSION='dh116-teacher-initialization-only-v1'
BINDINGS_SHA256='1dd699e41f8483a3b49d944b376940af28e5ea0ca44f886842ec74dff722092d'
EVIDENCE=Path('/home/xk/.local/share/dh116-autonomous-3c80166/evidence')
EXPERIENCE=Path('/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.kit')
BINDINGS=EVIDENCE/'power-surface-audit-20261117/native-binding-reports.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())




def write_factory_owner_ids(output,row,errors):
    """Record an ID-only receipt without interrupting runtime ownership handoff."""
    try:
        write_new(Path(output)/"factory_owner_ids.json",row)
    except Exception as error:
        message="factory owner IDs failed: "+type(error).__name__+": "+str(error)
        # Detach the caught diagnostic IO traceback; no frame is cleared.
        error.__traceback__=None
        errors.append(message)


def write_tensor_view_owners(output,view_owners,manager,phase,errors,*,runtime=None):
    """Diagnostic failures add text without changing the owned cleanup flow."""
    try:
        row=(view_owners.capture(runtime,manager) if runtime is not None
             else view_owners.observe(manager))
        write_new(Path(output)/(phase+".tensor_view_owners.json"),row)
    except Exception as error:
        message=phase+" tensor view owners failed: "+type(error).__name__+": "+str(error)
        error.__traceback__=None
        errors.append(message)

def write_owner_probe(output,reference,phase):
    """Write bounded scalar owners at an existing lifecycle boundary only."""
    from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import probe_simulation_owner_edges
    try:
        write_new(Path(output)/(phase+".owner_edges.json"),
            probe_simulation_owner_edges(reference,max_referrers=16,max_frame_depth=4))
    except Exception as error:
        # A diagnostic write failure must not bypass clear/STOP/app.close.
        # Only text escapes; the probe drops all temporary object/frame aliases.
        return phase+" owner probe failed: "+type(error).__name__+": "+str(error)
    return None

def interpreter_identity():
    executable=Path(sys.executable).resolve(strict=True)
    return dict(executable=sys.executable,executable_resolved=str(executable),
        executable_sha256=sha(executable),prefix=sys.prefix,version=sys.version,
        pythonpath=os.environ.get('PYTHONPATH'))


def validate_worker_path(root,*,worker_file=None):
    expected=Path(root).resolve()/'scripts/diagnose_dh116_teacher_initialization.py'
    actual=Path(__file__ if worker_file is None else worker_file).resolve()
    if actual!=expected:raise ValueError('worker file and project identity differ')


def validate_package_origins(root,*,find_spec=None):
    import importlib.util
    root=Path(root).resolve()
    lookup=importlib.util.find_spec if find_spec is None else find_spec
    for name,expected in (('go2_pvcnn',root/'src/go2_pvcnn'),('scripts',root/'scripts')):
        spec=lookup(name)
        locations=getattr(spec,'submodule_search_locations',None)
        actual=[Path(p).resolve() for p in locations] if locations is not None else []
        valid=(set(actual)=={expected}) if name=='go2_pvcnn' else bool(actual and actual[0]==expected)
        if not valid:raise ValueError('package origin differs for '+name)
        if find_spec is None and name in sys.modules:
            loaded=[Path(p).resolve() for p in getattr(sys.modules[name],'__path__',())]
            valid=(set(loaded)=={expected}) if name=='go2_pvcnn' else bool(loaded and loaded[0]==expected)
            if not valid:raise ValueError('loaded package origin differs for '+name)


def validate_loaded_project_modules(root,*,frozen_files=None,modules=None):
    """Check every loaded project helper, allowing unused SDK scripts namespaces."""
    root=Path(root).resolve()
    for name,module in list((sys.modules if modules is None else modules).items()):
        if name.startswith('go2_pvcnn.'):
            base=root/'src'/Path(*name.split('.'))
        elif name.startswith('scripts.'):
            base=root/Path(*name.split('.'))
        else:continue
        expected=base.with_suffix('.py')
        if not expected.is_file():expected=base/'__init__.py'
        if expected.is_file():
            pins=frozen_files if frozen_files is not None else {str(expected):sha(expected)}
            validate_native_module(module,expected,pins)
        elif name.startswith('go2_pvcnn.'):
            paths={Path(p).resolve() for p in getattr(module,'__path__',())}
            if paths!={base}:raise ValueError('loaded project module has no owned source: '+name)


def validate_execution_context(root):
    validate_worker_path(root)
    validate_package_origins(root)
    validate_loaded_project_modules(root)


def validate_native_module(module,expected,frozen_files):
    expected=Path(expected).resolve(strict=True)
    if Path(module.__file__).resolve(strict=True)!=expected:
        raise ValueError('native module origin differs: '+str(expected))
    if sha(expected)!=frozen_files.get(str(expected)):
        raise ValueError('native module frozen SHA differs: '+str(expected))


def write_initialization_row(output,row,*,encoder):
    encoded=encoder(row)
    encoded['nonfinite_encoding']='tagged_non_measurements'
    write_new(Path(output)/(row['event']+'.json'),encoded)

def budget():
    return dict(gpu=2,num_envs=1,mode='pinch',sdk_setup_steps=2,control_steps=0,
        solver_steps=0,teacher_queries=0,ppo_updates=0,model_loads=0,
        explicit_app_updates_for_stop=1,soft_wall_seconds=180,auto_retries=0,auto_kill=False,
        new_power_sdf_bakes=0,new_power_mesh_registrations=0)


def external_input_paths(root):
    """Known retained runtime inputs, pinned without importing any runtime module."""
    paths=[EXPERIENCE,EVIDENCE/'paired-pinch-reset-BaP0hx/result.json',
        EVIDENCE.parent/'protected-ubuntu.json']
    for folder in (EVIDENCE/'flex-f4-parity-prep-Vu9H0f',EVIDENCE/'flex-f4-sdf-register-rK7gw0/bake'):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.exit_code'))
    backend=Path('/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python/ropesim')
    paths.extend(p for p in backend.rglob('*') if p.is_file() and p.suffix in ('.py','.so'))
    isaac=Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim')
    paths.extend([
        isaac/'exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py',
        isaac/'exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py',
        Path('/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py'),
        isaac/'exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py',
        isaac/'exts/isaacsim.core.api/isaacsim/core/api/physics_context/physics_context.py',
        isaac/'extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py',
        Path('/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py'),
        Path('/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py'),
        Path('/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py'),
        Path('/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py'),
        Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/carb/events/_events.pyi'),
        isaac/'extsPhysics/omni.physx/omni/physx/__init__.py',
        isaac/'extsPhysics/omni.physx/omni/physx/bindings/_physx.pyi'])
    schema=isaac/'extsPhysics/omni.usd.schema.physx'
    paths.extend(schema/p for p in ('bin/libphysxSchema.so','pxr/PhysxSchema/__init__.py',
        'pxr/PhysxSchema/_physxSchema.so','plugins/PhysxSchema/resources/plugInfo.json',
        'plugins/PhysxSchema/resources/generatedSchema.usda','plugins/PhysxSchema/resources/schema.usda'))
    return sorted(set(paths))


def project_input_paths(root):
    paths=[]
    for folder,suffix in (('src','.py'),('scripts','.py'),('configs','.json'),('docs/contracts','.json')):
        paths.extend(p for p in (root/folder).rglob('*'+suffix) if p.is_file())
    paths.extend(p for p in (root/'assets/fixed_dual_panda_dh116_v1').rglob('*') if p.is_file())
    return sorted(set(paths))


def freeze_plan(root,output,binding_reports):
    root=Path(root).resolve(strict=True);output=Path(output).resolve()
    binding_reports=Path(binding_reports).resolve(strict=True)
    if sha(binding_reports)!=BINDINGS_SHA256:
        raise ValueError('compact binding reports SHA differs')
    if output==root or root in output.parents:
        raise ValueError('raw evidence output must remain outside Git project')
    source_paths=[]
    for report in json.loads(binding_reports.read_text()):
        pins=dict(report['layer_sha256']);pins[report['source_path']]=report['source_sha256']
        for name,expected in pins.items():
            path=Path(name).resolve(strict=True)
            if sha(path)!=expected:raise ValueError('original source asset SHA differs')
            source_paths.append(path)
    paths=sorted(set(project_input_paths(root)+external_input_paths(root)+source_paths+[binding_reports]))
    teacher=root/'scripts/train_dh116_gpu_teacher_ppo.py'
    if teacher not in paths:raise ValueError('original Teacher source missing')
    return dict(version=VERSION,project=str(root),output=str(output),budget=budget(),
        binding_reports=str(binding_reports),binding_reports_sha256=BINDINGS_SHA256,
        execution_context=interpreter_identity(),
        frozen_files={str(p.resolve(strict=True)):sha(p) for p in paths},
        teacher_source_sha256=sha(teacher),experience=str(EXPERIENCE),
        native_execution_performed=False,process_exit_verified=False,
        preserved_scope='original supported Pinch seed and existing13 assembly; no new Power surfaces',
        clock_adapter_status='sha_bound_separate_sdk_clock_adapter; original TensorWorld contract unchanged')


def verify_plan(plan,approved_sha):
    if approved_sha!=digest(plan):raise ValueError('exact approved frozen SHA required')
    if plan['version']!=VERSION or plan['budget']!=budget():raise ValueError('initialization budget identity differs')
    rebuilt=freeze_plan(plan['project'],plan['output'],plan['binding_reports'])
    if rebuilt!=plan:raise ValueError('frozen fingerprint or input set changed')
    if Path(plan['output']).exists():raise ValueError('fresh exclusive output directory required')
    return plan


@contextmanager
def observe_compiler(module):
    """Observe the unique existing sim.reset call; restore compiler on all paths."""
    original=module.compile_force_runtime
    def instrument(source,expected_sha):
        tree=original(source,expected_sha);factory=tree.body[0];out=[];count=0
        for node in factory.body:
            is_reset=(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='reset'
                and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='sim')
            if is_reset:
                out.extend(ast.parse('force_hook.before_reset(sim)').body);count+=1
            out.append(node)
            if is_reset:out.extend(ast.parse('force_hook.after_reset(sim)').body)
        if count!=1:raise ValueError('exact one existing sim.reset required')
        factory.body=out
        return ast.fix_missing_locations(tree)
    module.compile_force_runtime=instrument
    try:yield
    finally:module.compile_force_runtime=original


def clock(sim):
    return dict(sdk_step=int(sim.current_time_step_index),sdk_time_s=float(sim.current_time))


def host(value):
    import numpy as np
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.array(value,dtype=float,copy=True)


def pose_matrix_xyzw(pose):
    import numpy as np
    p=np.asarray(pose,dtype=float)
    if p.shape!=(7,) or not np.isfinite(p).all() or not np.isclose(np.linalg.norm(p[3:]),1.,atol=1e-5,rtol=0):
        raise ValueError('finite actual SDK XYZW link pose required')
    # The raw SDK quaternion stays unchanged in the saved snapshot.
    # Normalize only the derived rotation, AFTER the original near-unit guard.
    x,y,z,w=p[3:]/np.linalg.norm(p[3:]);mat=np.eye(4)
    mat[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    mat[:3,3]=p[:3]
    return mat.tolist()


def capture_once(runtime,bindings,cooking,*,collector,on_snapshot=None):
    """One actual SDK pose-buffer snapshot per hand; no clock advancement."""
    hands=views=poses=sim=None
    try:
        sim=runtime.sim;before=clock(sim)
        playing=sim.is_playing()
        if playing is not True:
            raise RuntimeError("actual simulation is not running for cooked geometry readback")
        if before['sdk_step']!=2 or not math.isclose(before['sdk_time_s'],2/480,abs_tol=2e-6):
            raise ValueError('actual setup SDK tick2 required')
        if float(runtime.rope.time)!=0.:raise ValueError('solver unexpectedly advanced')
        hands=(runtime.left_hands,runtime.right_hands)
        for binding in bindings:
            paths=hands[binding['hand_index']].root_physx_view.link_paths
            if len(paths)!=1 or paths[0][binding['body_id']]!=binding['actual_body_path']:
                raise ValueError('actual SDK body path identity differs')
        poses=[host(hand.root_physx_view.get_link_transforms()) for hand in hands]
        snapshot_clock=clock(sim)
        # Durable CPU-only evidence comes before conversion or protected preflight.
        # This is the same two SDK reads, not an extra simulation/pose sample.
        if on_snapshot is not None:
            on_snapshot(dict(schema='dh116-raw-sdk-body-poses-v1',env_id=0,episode_id=0,
                policy_step=0,control_step=0,sdk_clock_before=before,sdk_clock_after=snapshot_clock,
                quaternion_order='xyzw',position_unit='m',
                poses_xyzw_by_hand=[value.tolist() for value in poses],
                bindings=bindings,source='actual SDK articulation get_link_transforms once per hand',
                nonfinite_encoding='tagged_non_measurements',native_attested=False,
                process_exit_verified=False))
        if snapshot_clock!=before:raise RuntimeError('SDK clock changed during pose snapshot')
        source_poses={side:{} for side in ('left','right')}
        for binding in bindings:
            hi=binding['hand_index'];bi=binding['body_id']
            source_poses[binding['side']][binding['source_body_path']]=pose_matrix_xyzw(poses[hi][0,bi])
        snapshot=dict(env_id=0,episode_id=0,tensor_world_physics_steps=None,
            clock_source='isaac_sim_context_without_tensor_world',sdk_step_index=before['sdk_step'],
            control_step=0,control_substep=0,simulation_time_s=before['sdk_time_s'],
            source_body_poses_w=source_poses,provenance=dict(
                source='actual SDK articulation get_link_transforms once per hand',
                tensor_world_owner='not_instantiated',tensor_world_not_instantiated=True,
                native_attested=False,setup_reset_only=True))
        captured=collector(sim.stage,cooking,bindings,snapshot,simulation_running=playing)
        after=clock(sim)
        if after!=before:raise RuntimeError('SDK clock changed during read-only capture')
        return dict(capture=captured,source_body_poses_w=source_poses,
            snapshot_count_per_hand=1,sdk_clock_before=before,sdk_clock_after=after,
            clock_adapter_blocked=any('CLOCK_INTEGER_REQUIRED_tensor_world_physics_steps' in str(r)
                for r in captured.get('reasons',[])),process_exit_verified=False)
    finally:
        runtime=hands=views=poses=sim=cooking=bindings=None


def persist_capture(output,captured):
    """Keep raw failed getter evidence before refusing an incomplete capture."""
    write_new(Path(output)/'stored_cooked_capture.json',captured)
    if captured['capture'].get('complete') is not True:
        raise RuntimeError('incomplete 34-source cooked capture: '+
            json.dumps(captured['capture'].get('reasons',[]),allow_nan=False))

def shutdown_observation(reference,*,manager_released,phase):
    """Observe a weak SDK owner; unavailable clocks remain null, never zero."""
    target=reference() if reference is not None else None
    try:
        measured=clock(target) if target is not None else dict(sdk_step=None,sdk_time_s=None)
        consistent=(measured['sdk_step']==2 and math.isclose(measured['sdk_time_s'],2/480,abs_tol=2e-6,rel_tol=0)) if target is not None else None
        return dict(phase=phase,clock_available=target is not None,**measured,
                    clock_consistent_with_setup=consistent,manager_physics_view_released=manager_released)
    finally:target=None


def continue_within_budget(started,phase,operation,*,now=None):
    """Soft boundary check only: never interrupt or kill an in-flight native call."""
    elapsed=(time.monotonic() if now is None else now)-started
    if elapsed>180:
        raise RuntimeError('soft wall budget exceeded at '+phase+': '+format(elapsed,'.3f')+'s > 180s')
    return operation()


def soft_budget_completion_errors(elapsed):
    return ['soft wall budget exceeded at completion: '+format(elapsed,'.3f')+'s > 180s'] if elapsed>180 else []


def device_receipt(manager,settings_get):
    """Read actual fields; avoid the installed device getter's -1 -> 0 write."""
    ordinal=settings_get('/physics/cudaDevice')
    result=dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cuda_device_setting=ordinal,simulation_manager_reported_device=None,
        physical_gpu_identity_verified=False)
    if type(ordinal) is int and ordinal>=0:
        try:result['simulation_manager_reported_device']=manager.get_physics_sim_device()
        except Exception as error:result['device_read_error']=type(error).__name__+': '+str(error)
    else:
        result['device_read_unavailable_reason']='installed getter could author fallback ordinal; not called'
    return result

def result_record(*,capture_complete,close_returned,wall_elapsed,
                  manager_released=None,shutdown_clock_consistent=None):
    return dict(version=VERSION,capture_complete=bool(capture_complete),app_close_returned=bool(close_returned),
        process_exit_verified=False,external_exit_attestation_required=True,
        manager_physics_view_released=manager_released,shutdown_clock_consistent=shutdown_clock_consistent,
        native_lifecycle_qualified=False,grasp_qualified=False,teacher_admitted=False,
        actual_unresolved_native_views_qualified=False,budget=budget(),wall_elapsed_s=wall_elapsed,
        soft_wall_exceeded=wall_elapsed>180,budget_errors=soft_budget_completion_errors(wall_elapsed),
        auto_retry_performed=False)


def close_fallback_app(app,output,*,close_app):
    """A returned close call is only a close receipt, never process-exit proof."""
    close_app(app,output,simulation=None)
    return True

def execute_native(plan):
    """Already-approved exclusive process; no automatic retry or forced exit."""
    validate_execution_context(Path(plan['project']))
    started=time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES']='2';os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    with native_instance_lock():
        from isaaclab.app import AppLauncher
        if any(name=='pxr' or name.startswith('pxr.') for name in sys.modules):
            raise RuntimeError('USD imported before owned AppLauncher')
        output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
        write_new(output/'frozen_plan.json',plan)
        app=None;failure=None;fallback_close_returned=False
        try:
            app=AppLauncher(headless=True,device='cuda:0',enable_cameras=False,fast_shutdown=False,
                experience=plan['experience']).app
            return continue_within_budget(started,'AppLauncher returned',
                lambda:_execute_app_owned(app,plan,started))
        except BaseException as error:
            failure=traceback.format_exc()
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
        # End the original exception before attempting fallback shutdown.
        if app is not None and not (output/'shutdown_started.json').exists():
            try:
                from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app
                fallback_close_returned=close_fallback_app(app,output,close_app=close_calibration_app)
            except BaseException as error:
                failure+='\n'+traceback.format_exc()
                traceback.clear_frames(error.__traceback__);error.__traceback__=None
        app=None
        if not (output/'result.json').exists():
            result=result_record(capture_complete=False,close_returned=fallback_close_returned,wall_elapsed=time.monotonic()-started)
            result.update(errors=[failure],native_execution_performed=True)
            write_new(output/'result.json',result)
        raise RuntimeError(failure) from None


def _execute_app_owned(app,plan,started):
    # All USD, PhysX, solver, trace and runtime imports occur after AppLauncher.
    import carb.settings
    import torch
    import omni.timeline
    import weakref
    from isaacsim.core.simulation_manager import SimulationManager
    import isaaclab.sim as sim_utils
    from omni.physx import get_physx_cooking_interface
    from scripts.dh116_pinch_startup_v3 import StartupTrace
    from go2_pvcnn.control.dh116 import tabletop_teacher_force_runtime as force
    from go2_pvcnn.control.dh116 import tabletop_teacher_seed_initialization as seed_initialization
    from go2_pvcnn.control.dh116 import tabletop_tripod_cooked_geometry as protected_cooked
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import prepare_power_cooked_bindings
    from go2_pvcnn.control.dh116.tabletop_teacher_sdk_cooked_capture import collect_sdk_cooked_geometry
    from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import (
        simulation_source_contract,clear_known_simulation_instance,TensorViewOwners)
    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app
    for module,name in ((force,'tabletop_teacher_force_runtime.py'),
        (seed_initialization,'tabletop_teacher_seed_initialization.py'),
        (sys.modules[prepare_power_cooked_bindings.__module__],'tabletop_teacher_power_surface_audit.py'),
        (sys.modules[collect_sdk_cooked_geometry.__module__],'tabletop_teacher_sdk_cooked_capture.py'),
        (protected_cooked,'tabletop_tripod_cooked_geometry.py')):
        validate_native_module(module,Path(plan['project'])/'src/go2_pvcnn/control/dh116'/name,plan['frozen_files'])
    validate_loaded_project_modules(Path(plan['project']),frozen_files=plan['frozen_files'])
    output=Path(plan['output'])
    trace=StartupTrace(output/'ownership.jsonl')
    scope=dict(sim=None);acquired={};errors=[];close_returned=False
    shutdown={};sim_ref=None
    cleanup_contract=simulation_source_contract()
    view_owners=TensorViewOwners()
    class Hook(force.ForceContactHook):
        def before_reset(self,sim):
            scope['sim']=sim
            trace.watch('simulation',sim,owner='initialization lifecycle',release='clear before STOP',callbacks=True)
            observed=clock(sim)
            if observed['sdk_step']!=0:raise RuntimeError('reset initial SDK boundary is not0')
            row=dict(phase='before_reset',**observed)
            write_new(output/'before_reset.json',row)
            with (output/'initialization_boundaries.jsonl').open('x') as stream:
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        def after_reset(self,sim):
            observed=clock(sim)
            row=dict(phase='after_reset',**observed)
            write_new(output/'after_reset.json',row)
            with (output/'initialization_boundaries.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            if observed['sdk_step']!=2:raise RuntimeError('reset final SDK boundary is not2')
    def operation(owners):
        rt=hand=view=cooking=hook=None
        try:
            hook=Hook(grasp='pinch',run_uid=output.name,
                emit=lambda row: write_initialization_row(output,row,encoder=force._encode_nonfinite))
            args=SimpleNamespace(mode='pinch',num_envs=1,updates=0,eval_only=True,
                device='cuda:0',gpu=2,horizon=0,bc_epochs=0,checkpoint=None)
            with observe_compiler(force):
                rt=force.create_force_runtime(plan['project'],args,app,plan['teacher_source_sha256'],hook,
                    owner_observer=lambda row:write_factory_owner_ids(output,row,errors))
            continue_within_budget(started,'create_force_runtime returned',lambda:None)
            validate_loaded_project_modules(Path(plan['project']),frozen_files=plan['frozen_files'])
            write_new(output/'actual_device_identity.json',
                device_receipt(SimulationManager,carb.settings.get_settings().get))
            owners.keep('runtime',rt)
            write_tensor_view_owners(output,view_owners,SimulationManager,'registered',errors,runtime=rt)
            scope['sim']=rt.sim
            for side in ('left','right'):
                hand=getattr(rt,side+'_hands');view=hand.root_physx_view
                trace.watch(side+'_hand',hand,owner='runtime',release='runtime dict cleared',gpu_view=True)
                trace.watch(side+'_view',view,owner='runtime',release='runtime dict cleared',gpu_view=True)
            reports=json.loads(Path(plan['binding_reports']).read_text())
            roots=['/World/envs/env_0/'+side+'Hand' for side in ('left','right')]
            names=[list(rt.left_hands.body_names),list(rt.right_hands.body_names)]
            bindings=prepare_power_cooked_bindings(reports,roots,names)
            cooking=get_physx_cooking_interface()
            captured=capture_once(rt,bindings,cooking,collector=collect_sdk_cooked_geometry,
                on_snapshot=lambda row:write_new(output/'raw_sdk_body_poses.json',force._encode_nonfinite(row)))
            persist_capture(output,captured)
            return dict(capture_complete=captured['capture'].get('complete') is True,
                clock_adapter_blocked=captured['clock_adapter_blocked'],prepared_collision_bindings=len(bindings),
                sdk_clock=captured['sdk_clock_after'],solver_steps=0,control_steps=0,teacher_queries=0)
        finally:
            if rt is not None:rt.__dict__.clear()
            rt=hand=view=cooking=hook=None
            # run_owned_call owns the sole explicit gc.collect for this scope.
    try:
        if sim_utils.SimulationContext.instance() is not None:
            raise RuntimeError('unexpected existing SimulationContext before owned runtime')
        acquired=run_owned_call(operation,synchronize=lambda:torch.cuda.synchronize('cuda:0'),ownership=trace)
    except BaseException as error:
        errors.append(traceback.format_exc());traceback.clear_frames(error.__traceback__);error.__traceback__=None
    # No active acquisition exception/frame is retained across STOP.
    try:
        settings=carb.settings.get_settings()
        settings.set_bool('/app/player/playSimulations',False)
        if settings.get('/app/player/playSimulations') is not False:raise RuntimeError('physics freeze failed')
        sim=scope.pop('sim',None)
        if sim is None:sim=sim_utils.SimulationContext.instance()
        if sim is not None:
            sim_ref=weakref.ref(sim)
            write_tensor_view_owners(output,view_owners,SimulationManager,'before_clear',errors)
            owner_probe_error=write_owner_probe(output,sim_ref,'before_clear')
            if owner_probe_error:errors.append(owner_probe_error)
        shutdown['before_clear']=shutdown_observation(sim_ref,manager_released=None,phase='before_clear')
        if shutdown['before_clear']['clock_consistent_with_setup'] is not True:
            errors.append('actual SDK tick2 unavailable or changed before cleanup')
        if sim is not None:
            write_new(output/'known_simulation_cleanup.json',clear_known_simulation_instance(sim,contract=cleanup_contract))
            sim=None
        torch.cuda.synchronize('cuda:0')
        timeline=omni.timeline.get_timeline_interface();timeline.stop();timeline=None
        app.update()  # The ONLY explicit App.update: deliver STOP, frozen physics.
        if settings.get('/app/player/playSimulations') is not False:raise RuntimeError('physics resumed at STOP')
        manager_view=SimulationManager.get_physics_sim_view()
        manager_released=manager_view is None
        manager_view=None
        shutdown['after_stop']=shutdown_observation(sim_ref,manager_released=manager_released,phase='after_stop')
        trace.event('SDK STOP',shutdown_observation=shutdown['after_stop'])
        if not manager_released:errors.append('SimulationManager physics view remains alive after STOP')
        if shutdown['after_stop']['clock_consistent_with_setup'] is False:
            errors.append('actual SDK clock advanced during frozen STOP App.update')
        if sim_ref is not None:
            write_tensor_view_owners(output,view_owners,SimulationManager,'after_stop',errors)
            owner_probe_error=write_owner_probe(output,sim_ref,'after_stop')
            if owner_probe_error:errors.append(owner_probe_error)
        trace.checkpoint('after STOP before app close')
    except BaseException as error:
        errors.append(traceback.format_exc());traceback.clear_frames(error.__traceback__);error.__traceback__=None
    scope.clear();sim=None;timeline=None;settings=None
    write_new(output/'shutdown_observations.json',shutdown)
    try:
        close_calibration_app(app,output,simulation=None)
        close_returned=True
    except BaseException as error:
        errors.append(traceback.format_exc());traceback.clear_frames(error.__traceback__);error.__traceback__=None
    scope.clear();sim=None;timeline=None;settings=None
    elapsed=time.monotonic()-started
    errors.extend(soft_budget_completion_errors(elapsed))
    result=result_record(capture_complete=acquired.get('capture_complete',False),
        close_returned=close_returned,wall_elapsed=elapsed,
        manager_released=shutdown.get('after_stop',{}).get('manager_physics_view_released'),
        shutdown_clock_consistent=(shutdown.get('before_clear',{}).get('clock_consistent_with_setup') is True
            and shutdown.get('after_stop',{}).get('clock_consistent_with_setup') is not False)
            if 'after_stop' in shutdown else None)
    result.update(acquisition=acquired,shutdown_observations=shutdown,errors=errors,native_execution_performed=True)
    write_new(output/'result.json',result)
    if errors:raise RuntimeError('\n'.join(errors)) from None
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--binding-reports',type=Path,default=BINDINGS)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--execute-native',action='store_true')
    parser.add_argument('--approved-frozen-sha256')
    args=parser.parse_args(argv)
    validate_execution_context(args.project)
    if args.execute_native:
        plan=json.loads(args.plan.read_text())
        verify_plan(plan,args.approved_frozen_sha256)
        if (str(args.output.resolve())!=plan['output'] or str(args.project.resolve())!=plan['project']
            or str(args.binding_reports.resolve())!=plan['binding_reports']):
            raise ValueError('CLI and frozen native identity differ')
        execute_native(plan)
    else:
        plan=freeze_plan(args.project,args.output,args.binding_reports)
        write_new(args.plan,plan)
        print(json.dumps(dict(native_execution_performed=False,plan=str(args.plan),frozen_sha256=digest(plan))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
