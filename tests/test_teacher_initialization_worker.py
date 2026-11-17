"""CPU contracts only; SDK fixtures are never native qualification."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest


def api():
    from scripts import diagnose_dh116_teacher_initialization as m
    return m


def frozen(tmp_path, monkeypatch):
    m=api()
    root=tmp_path/'repo'; (root/'scripts').mkdir(parents=True)
    (root/'scripts/train_dh116_gpu_teacher_ppo.py').write_text('frozen teacher source')
    binding=tmp_path/'binding.json'; binding.write_text('[]')
    monkeypatch.setattr(m,'BINDINGS_SHA256',m.sha(binding))
    monkeypatch.setattr(m,'external_input_paths',lambda root: [])
    plan=m.freeze_plan(root,tmp_path/'output',binding)
    return m,root,binding,plan


def test_budget_is_exact_initialization_only():
    m=api(); b=m.budget()
    assert b==dict(gpu=2,num_envs=1,mode='pinch',sdk_setup_steps=2,control_steps=0,
        solver_steps=0,teacher_queries=0,ppo_updates=0,model_loads=0,
        explicit_app_updates_for_stop=1,soft_wall_seconds=180,auto_retries=0,auto_kill=False,
        new_power_sdf_bakes=0,new_power_mesh_registrations=0)


def test_default_freezes_only_without_importing_native_or_teacher(tmp_path,monkeypatch):
    m,root,binding,plan=frozen(tmp_path,monkeypatch)
    before=set(sys.modules)
    monkeypatch.setattr(m,'validate_execution_context',lambda root:None,raising=False)
    monkeypatch.setattr(m,'execute_native',lambda *a: pytest.fail('native invoked by preflight'))
    out=tmp_path/'plan.json'
    assert m.main(['--project',str(root),'--output',str(tmp_path/'output'),
                   '--binding-reports',str(binding),'--plan',str(out)])==0
    assert json.loads(out.read_text())['budget']==m.budget()
    assert not (set(sys.modules)-before)&{'torch','pxr','isaaclab','omni','ropesim'}
    assert not (tmp_path/'output').exists()


def test_approved_sha_and_every_input_checked_before_execution(tmp_path,monkeypatch):
    m,root,binding,plan=frozen(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='approved frozen SHA'):
        m.verify_plan(plan,'0'*64)
    m.verify_plan(plan,m.digest(plan))
    (root/'scripts/train_dh116_gpu_teacher_ppo.py').write_text('tampered')
    with pytest.raises(ValueError,match='fingerprint'):
        m.verify_plan(plan,m.digest(plan))


def test_budget_and_output_tamper_rejected(tmp_path,monkeypatch):
    m,root,binding,plan=frozen(tmp_path,monkeypatch)
    original=m.digest(plan); plan['budget']['teacher_queries']=1
    with pytest.raises(ValueError):m.verify_plan(plan,original)
    plan['budget']=m.budget(); plan['output']=str(root/'runs')
    with pytest.raises(ValueError):m.verify_plan(plan,original)


def test_capture_uses_one_snapshot_per_hand_at_same_actual_tick():
    m=api(); calls=[]
    class View:
        def __init__(self,side):
            self.side=side
            self.link_paths=[['/World/envs/env_0/'+side+'Hand/base_link']]
        def get_link_transforms(self):
            calls.append(self.side)
            return np.array([[[1.,2.,3.,0.,0.,0.,1.]]])
    hands=[NS(body_names=['base_link'],root_physx_view=View(side)) for side in ('left','right')]
    rt=NS(sim=NS(current_time_step_index=2,current_time=2/480,stage='actual_stage',is_playing=lambda:True),
          left_hands=hands[0],right_hands=hands[1],rope=NS(time=0.))
    bindings=[dict(side=s,hand_index=i,body_id=0,source_body_path='/Source/'+s+'/base_link',
                   actual_body_path='/World/envs/env_0/'+s+'Hand/base_link')
              for i,s in enumerate(('left','right'))]
    def collect(stage,cooking,received,snapshot,*,simulation_running):
        assert stage=='actual_stage' and cooking=='getter' and received==bindings
        assert simulation_running and snapshot['sdk_step_index']==2
        assert snapshot['control_step']==snapshot['control_substep']==0
        assert snapshot['source_body_poses_w']['left']['/Source/left/base_link'][0][3]==1.
        assert snapshot['provenance']['tensor_world_not_instantiated'] is True
        return {'complete':True,'native_attested':False}
    result=m.capture_once(rt,bindings,'getter',collector=collect)
    assert calls==['left','right']
    assert result['snapshot_count_per_hand']==1
    assert result['sdk_clock_before']==result['sdk_clock_after']=={'sdk_step':2,'sdk_time_s':2/480}
    assert result['process_exit_verified'] is False


def test_reset_instrumentation_records_only_real_zero_two_and_restores():
    m=api(); events=[]
    class Hook:
        def before_reset(self,sim):events.append(sim.tick)
        def after_reset(self,sim):events.append(sim.tick)
    class Sim:
        tick=0
        def reset(self):self.tick=2
    def compiler(source,sha):return ast.parse('def create_runtime(sim, force_hook):\n    sim.reset()\n')
    module=NS(compile_force_runtime=compiler)
    with m.observe_compiler(module):
        tree=module.compile_force_runtime('',''); namespace={};exec(compile(tree,'fixture','exec'),namespace)
        namespace['create_runtime'](Sim(),Hook())
    assert module.compile_force_runtime is compiler
    assert events==[0,2]
    with pytest.raises(RuntimeError):
        with m.observe_compiler(module):raise RuntimeError('fixture')
    assert module.compile_force_runtime is compiler


def test_completion_never_attests_natural_exit():
    m=api(); result=m.result_record(capture_complete=True,close_returned=True,wall_elapsed=3.)
    assert result['process_exit_verified'] is False
    assert result['native_lifecycle_qualified'] is False
    assert result['grasp_qualified'] is False
    assert result['external_exit_attestation_required'] is True


def test_sdk_clock_adapter_rejects_missing_scope_and_bad_tick_time():
    from go2_pvcnn.control.dh116.tabletop_teacher_sdk_cooked_capture import validate_sdk_clock
    s=dict(tensor_world_physics_steps=None,clock_source='isaac_sim_context_without_tensor_world',
        provenance={'tensor_world_owner':'not_instantiated'},sdk_step_index=2,simulation_time_s=2/480,
        env_id=0,control_step=0,control_substep=0)
    validate_sdk_clock(s)
    for key,value in (('clock_source','unknown'),('simulation_time_s',3/480),('sdk_step_index',3),
                      ('tensor_world_physics_steps',0),('env_id',1),('control_step',1)):
        bad=dict(s);bad[key]=value
        with pytest.raises(ValueError):validate_sdk_clock(bad)
    bad=dict(s);bad['provenance']={}
    with pytest.raises(ValueError):validate_sdk_clock(bad)


def test_sdk_adapter_pins_source_and_keeps_original_collector_unchanged(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_tripod_cooked_geometry as old
    from go2_pvcnn.control.dh116.tabletop_teacher_sdk_cooked_capture import build_sdk_collector
    before=old._preflight
    source=tmp_path/'wrong.py';source.write_text('tampered')
    with pytest.raises(ValueError,match='protected collector SHA'):
        build_sdk_collector(source_path=source)
    collector=build_sdk_collector()
    assert old._preflight is before
    assert collector.__globals__ is not old.__dict__
    assert collector.__globals__['_preflight'] is not before


def test_sdk_adapter_captures_real_cpu_fixture34_with_absent_tensorworld(tmp_path):
    from test_tripod_cooked_geometry import scene
    from pxr import Sdf,UsdGeom
    from go2_pvcnn.control.dh116 import tabletop_tripod_cooked_geometry as old
    from go2_pvcnn.control.dh116.tabletop_teacher_sdk_cooked_capture import build_sdk_collector
    stage,models,roots,names,snapshot,getter=scene(tmp_path)
    UsdGeom.Xform.Define(stage,'/World/envs/env_0')
    assert Sdf.CopySpec(stage.GetRootLayer(),'/World/envs/env_4',stage.GetRootLayer(),'/World/envs/env_0')
    roots=[r.replace('env_4','env_0') for r in roots]
    bindings=old.prepare_cooked_bindings(models,roots,names)
    snapshot.update(env_id=0,tensor_world_physics_steps=None,
        clock_source='isaac_sim_context_without_tensor_world',sdk_step_index=2,
        simulation_time_s=2/480,control_step=0,control_substep=0,
        provenance={'tensor_world_owner':'not_instantiated'})
    collector=build_sdk_collector()
    result=collector(stage,getter,bindings,snapshot,simulation_running=True)
    assert result['complete'],result['reasons']
    assert len(result['colliders'])==34
    assert result['schema']=='dh116-teacher-sdk-clock-stored-cooked-v1'
    assert result['capture_clock']['tensor_world_physics_steps'] is None
    assert result['native_attested'] is False
    assert result['live_actor_shape_attachment_proven'] is False
    prior=len(getter.calls)
    rejected=old.collect_cooked_geometry(stage,getter,bindings,snapshot,simulation_running=True)
    assert rejected['censored'] and len(getter.calls)==prior



def test_sdk_pose_identity_mismatch_blocks_before_snapshot_getter():
    m=api()
    class View:
        link_paths=[['/wrong/body']]
        def get_link_transforms(self):pytest.fail('pose queried with wrong identity')
    hand=NS(body_names=['base_link'],root_physx_view=View())
    runtime=NS(sim=NS(current_time_step_index=2,current_time=2/480,is_playing=lambda:True),
        rope=NS(time=0.),left_hands=hand,right_hands=hand)
    bindings=[dict(hand_index=0,body_id=0,actual_body_path='/World/envs/env_0/leftHand/base_link')]
    with pytest.raises(ValueError,match='SDK body path identity'):
        m.capture_once(runtime,bindings,'getter',collector=lambda *a,**k:pytest.fail('collector invoked'))


def test_original_asset_hash_from_binding_is_checked_during_freeze(tmp_path,monkeypatch):
    m,root,binding,plan=frozen(tmp_path,monkeypatch)
    asset=tmp_path/'original.usd';asset.write_bytes(b'original source bytes')
    binding.write_text(json.dumps([dict(source_path=str(asset),source_sha256=m.sha(asset),
        layer_sha256={str(asset):m.sha(asset)})]))
    monkeypatch.setattr(m,'BINDINGS_SHA256',m.sha(binding))
    plan=m.freeze_plan(root,tmp_path/'output',binding)
    assert str(asset) in plan['frozen_files']
    asset.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='original source asset'):
        m.freeze_plan(root,tmp_path/'output',binding)


def test_shutdown_observations_distinguish_live_clock_manager_and_gone_owner():
    import weakref
    m=api()
    class Sim:
        current_time_step_index=2
        current_time=2/480
    sim=Sim();reference=weakref.ref(sim)
    observed=m.shutdown_observation(reference,manager_released=False,phase='after_stop')
    assert observed['clock_available'] is True
    assert observed['sdk_step']==2 and observed['sdk_time_s']==2/480
    assert observed['clock_consistent_with_setup'] is True
    assert observed['manager_physics_view_released'] is False
    sim.current_time_step_index=3
    assert m.shutdown_observation(reference,manager_released=True,phase='after_stop')['clock_consistent_with_setup'] is False
    del sim
    gone=m.shutdown_observation(reference,manager_released=True,phase='after_stop')
    assert gone['clock_available'] is False
    assert gone['sdk_step'] is None and gone['sdk_time_s'] is None
    assert gone['clock_consistent_with_setup'] is None
    assert gone['manager_physics_view_released'] is True


def test_result_distinguishes_manager_release_close_return_and_natural_exit():
    m=api()
    result=m.result_record(capture_complete=True,close_returned=True,wall_elapsed=5.,
        manager_released=False,shutdown_clock_consistent=True)
    assert result['manager_physics_view_released'] is False
    assert result['shutdown_clock_consistent'] is True
    assert result['app_close_returned'] is True
    assert result['process_exit_verified'] is False
    assert result['native_lifecycle_qualified'] is False


def test_soft_wall_guard_blocks_continuation_after_native_call_returns():
    m=api();calls=[]
    assert m.continue_within_budget(10.,'AppLauncher returned',lambda:calls.append('init'),now=189.) is None
    assert calls==['init']
    for phase in ('AppLauncher returned','create_force_runtime returned','completion'):
        with pytest.raises(RuntimeError,match='soft wall budget exceeded'):
            m.continue_within_budget(10.,phase,lambda:calls.append('forbidden'),now=191.)
    assert calls==['init']


def test_completion_overrun_is_an_error_not_only_a_boolean():
    m=api()
    assert m.soft_budget_completion_errors(181.)==['soft wall budget exceeded at completion: 181.000s > 180s']
    assert m.soft_budget_completion_errors(180.)==[]


def test_actual_device_receipt_retains_unknown_instead_of_zero(monkeypatch):
    m=api();monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    class Manager:
        @staticmethod
        def get_physics_sim_device():return 'cuda:7'
    receipt=m.device_receipt(Manager,lambda key:7)
    assert receipt['cuda_visible_devices']=='2'
    assert receipt['simulation_manager_reported_device']=='cuda:7'
    class NoFallback:
        @staticmethod
        def get_physics_sim_device():pytest.fail('unsafe installed fallback queried')
    receipt=m.device_receipt(NoFallback,lambda key:-1)
    assert receipt['simulation_manager_reported_device'] is None
    assert receipt['cuda_device_setting']==-1
    assert receipt['physical_gpu_identity_verified'] is False


@pytest.mark.parametrize('complete',[False,None,True])
def test_capture_outcome_is_persisted_before_incomplete_capture_is_rejected(tmp_path,complete):
    m=api()
    packet=dict(capture=dict(complete=complete,censored=complete is not True,
        reasons=[] if complete is True else ['ONE_OR_MORE_COLLIDERS_CENSORED'],
        colliders=[{'raw':'actual getter payload or error'}]),process_exit_verified=False)
    if complete is True:
        m.persist_capture(tmp_path,packet)
    else:
        with pytest.raises(RuntimeError,match='incomplete 34-source cooked capture.*ONE_OR_MORE_COLLIDERS_CENSORED'):
            m.persist_capture(tmp_path,packet)
    assert json.loads((tmp_path/'stored_cooked_capture.json').read_text())==packet
    assert packet['process_exit_verified'] is False


def test_budget_expired_startup_reports_actual_successful_fallback_close(tmp_path):
    m=api()
    class App:
        closed=False
    app=App()
    def close(owned_app,output,*,simulation):
        assert owned_app is app and output==tmp_path and simulation is None
        owned_app.closed=True
    returned=m.close_fallback_app(app,tmp_path,close_app=close)
    result=m.result_record(capture_complete=False,close_returned=returned,wall_elapsed=181.)
    assert app.closed and result['app_close_returned'] is True
    assert result['soft_wall_exceeded'] and result['budget_errors']
    assert result['process_exit_verified'] is False
    assert result['native_lifecycle_qualified'] is False



def test_worker_rejects_project_b_for_worker_a(tmp_path):
    m=api();a=tmp_path/'a';b=tmp_path/'b'
    worker=a/'scripts/diagnose_dh116_teacher_initialization.py'
    worker.parent.mkdir(parents=True);worker.write_text('source')
    m.validate_worker_path(a,worker_file=worker)
    with pytest.raises(ValueError,match='worker file and project'):
        m.validate_worker_path(b,worker_file=worker)


def test_package_shadowing_and_loaded_native_module_path_are_rejected(tmp_path):
    m=api();root=tmp_path/'a';other=tmp_path/'b'
    def spec(name):
        return NS(submodule_search_locations=[str(root/('src/go2_pvcnn' if name=='go2_pvcnn' else 'scripts'))])
    m.validate_package_origins(root,find_spec=spec)
    def shadow(name):return NS(submodule_search_locations=[str(other/name)])
    with pytest.raises(ValueError,match='package origin'):
        m.validate_package_origins(root,find_spec=shadow)
    expected=root/'module.py';expected.parent.mkdir(parents=True);expected.write_text('same source')
    alternate=other/'module.py';alternate.parent.mkdir(parents=True);alternate.write_text('same source')
    with pytest.raises(ValueError,match='native module origin'):
        m.validate_native_module(NS(__file__=str(alternate)),expected,{str(expected):m.sha(expected)})
    m.validate_native_module(NS(__file__=str(expected)),expected,{str(expected):m.sha(expected)})


def test_interpreter_and_pythonpath_are_part_of_frozen_identity(tmp_path,monkeypatch):
    m,root,binding,plan=frozen(tmp_path,monkeypatch)
    identity=plan['execution_context']
    assert identity['executable_sha256']==m.sha(Path(sys.executable).resolve())
    assert identity['prefix']==sys.prefix and identity['version']==sys.version
    monkeypatch.setenv('PYTHONPATH','/changed/module/search/path')
    with pytest.raises(ValueError,match='fingerprint'):
        m.verify_plan(plan,m.digest(plan))


def test_invalid_initialization_nonfinite_measurements_are_saved_as_tags(tmp_path):
    m=api()
    from go2_pvcnn.control.dh116.tabletop_teacher_force_runtime import _encode_nonfinite
    row=dict(event='initialization_readback',passed=False,native_success=False,
        hands={'left':{'actual_q_rad':[float('nan'),float('inf')]}})
    m.write_initialization_row(tmp_path,row,encoder=_encode_nonfinite)
    saved=json.loads((tmp_path/'initialization_readback.json').read_text())
    assert saved['hands']['left']['actual_q_rad']==['NaN','+Infinity']
    assert saved['nonfinite_encoding']=='tagged_non_measurements'
    assert saved['passed'] is False and saved['native_success'] is False


def test_scripts_namespace_allows_sdk_tail_but_requires_project_first(tmp_path):
    m=api();root=tmp_path/'repo';usd=tmp_path/'usd/scripts'
    def valid(name):
        locations=[str(root/'src/go2_pvcnn')] if name=='go2_pvcnn' else [str(root/'scripts'),str(usd)]
        return NS(submodule_search_locations=locations)
    m.validate_package_origins(root,find_spec=valid)
    def wrong(name):
        locations=[str(root/'src/go2_pvcnn')] if name=='go2_pvcnn' else [str(usd),str(root/'scripts')]
        return NS(submodule_search_locations=locations)
    with pytest.raises(ValueError,match='package origin'):
        m.validate_package_origins(root,find_spec=wrong)


def test_already_loaded_project_script_shadow_is_rejected_before_app(tmp_path):
    m=api();root=tmp_path/'repo'
    expected=root/'scripts/dh116_pinch_startup_v3.py';expected.parent.mkdir(parents=True)
    expected.write_text('owned helper')
    shadow=tmp_path/'shadow.py';shadow.write_text('owned helper')
    with pytest.raises(ValueError,match='native module origin'):
        m.validate_loaded_project_modules(root,modules={'scripts.dh116_pinch_startup_v3':NS(__file__=str(shadow))})
    m.validate_loaded_project_modules(root,modules={'scripts.dh116_pinch_startup_v3':NS(__file__=str(expected))})

def test_stopped_simulation_cannot_be_claimed_running_to_cooked_getter():
    m=api();queries=[]
    class View:
        def __init__(self,side):self.link_paths=[['/World/envs/env_0/'+side+'Hand/base_link']]
        def get_link_transforms(self):
            queries.append('pose')
            return np.array([[[1.,2.,3.,0.,0.,0.,1.]]])
    rt=NS(sim=NS(current_time_step_index=2,current_time=2/480,stage='actual_stage',
        is_playing=lambda:False),rope=NS(time=0.),
        left_hands=NS(root_physx_view=View('left')),right_hands=NS(root_physx_view=View('right')))
    bindings=[dict(side=s,hand_index=i,body_id=0,source_body_path='/Source/'+s+'/base_link',
        actual_body_path='/World/envs/env_0/'+s+'Hand/base_link') for i,s in enumerate(('left','right'))]
    def collector(*a,**kw):
        queries.append('cooked');return dict(complete=True)
    with pytest.raises(RuntimeError,match='running'):
        m.capture_once(rt,bindings,'getter',collector=collector)
    assert queries==[]
