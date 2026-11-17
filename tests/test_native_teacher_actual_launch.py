"""CPU launch contracts. Synthetic approval never authorizes native execution."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import pytest
from scripts import run_dh116_work_packages as scheduler
from test_dh116_work_packages import repo, write_json, git
from test_native_teacher_worker_contract import recipe, pin_source_closure

MODULE = 'go2_pvcnn.control.dh116.tabletop_native_teacher_actual_launch'
PROVIDER = 'src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_provider.py'
FILES = {'src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_launch.py',
    'src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_process.py',
    'scripts/train_dh116_native_teacher_worker.py',
    'tests/test_native_teacher_actual_launch.py','tests/test_native_teacher_actual_launch_process.py'}
ENVIRONMENT='configs/dh116_execution/actual-environment-v1.json'

def launch_module():
    assert importlib.util.find_spec(MODULE), 'actual native launch contract is missing'
    return importlib.import_module(MODULE)

def actual_fixture(repo, tmp_path):
    spec = recipe(tmp_path)
    spec['source_files'] = sorted(set(spec['source_files']) | FILES | {PROVIDER,ENVIRONMENT})
    for name in {'scripts/python.sh','scripts/run_dh116_work_packages.py',spec['entrypoint'],*spec['source_files']}:
        p=repo.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# CPU source fixture\n')
    spec_name='configs/dh116_execution/actual-core.json'
    package=dict(title='CPU actual fixture',status='pending',executor='native_bounded',dependencies=[],
        requires_approval=True,resource_class=scheduler.NATIVE_SCOPE,evidence_scope='cpu-only-fixture',
        retry_policy=scheduler.NATIVE_RETRY,command_spec=spec_name,tests=[])
    experience=tmp_path/'experience.kit';experience.write_text('# CPU experience bytes')
    write_json(repo.root/ENVIRONMENT,dict(schema='dh116-actual-native-environment-v1',
        experience=dict(path=str(experience),sha256=scheduler.file_hash(experience)),
        debugger=dict(path=scheduler.GDB,sha256=scheduler.file_hash(Path(scheduler.GDB)))))
    pin_source_closure(repo.root,spec)
    write_json(repo.root/spec_name,spec)
    core=repo.native_recipe(package)
    launch=dict(schema='dh116-actual-native-execution-v1',
        core_recipe_identity={k:core[k] for k in ('command_spec_sha256','code_config_sha256')},
        source_closure_sha256=spec['teacher_contract']['source_sha256'],role='a_fresh',
        device_plan=dict(physical_gpu_index=3,cuda_visible_devices='3',logical_device='cuda:0',
            experience_path=str(experience),experience_sha256=scheduler.file_hash(experience)),
        debugger='gdb_on_segv',provider=dict(path=PROVIDER,entrypoint='run_a',sha256=scheduler.file_hash(repo.root/PROVIDER)),checkpoint_origin=None)
    launch_name='configs/dh116_execution/actual-companion.json'
    write_json(repo.root/launch_name,launch)
    package['actual_launch']=dict(path=launch_name,sha256=scheduler.file_hash(repo.root/launch_name))
    registry=repo.registry();registry['packages']['actual']=package
    write_json(repo.root/scheduler.REGISTRY,registry)
    return package,spec,launch

def test_core_without_companion_retains_historical_identity(repo,tmp_path):
    package,_,_=actual_fixture(repo,tmp_path);package.pop('actual_launch')
    core=repo.native_recipe(package)
    assert repo.native_execution_identity(package)==dict(
        core_identity={k:core[k] for k in ('command_spec_sha256','code_config_sha256')},
        actual_identity=None,attempt_fingerprint=core['code_config_sha256'])

def test_companion_has_independent_identity_and_registered_fingerprint(repo,tmp_path):
    package,_,launch=actual_fixture(repo,tmp_path)
    mod=launch_module();identity=repo.native_execution_identity(package)
    assert identity['core_identity']==launch['core_recipe_identity']
    assert identity['actual_identity']==mod.actual_launch_identity(launch,companion_sha256=package['actual_launch']['sha256'])
    approval_identity={**identity['core_identity'],'actual_identity':identity['actual_identity']}
    approval=dict(version=1,batch_id='cpu-fixture',output_root=str(tmp_path/'output'),packages={'actual':approval_identity})
    assert repo.native_plan(scope='all',approval=approval)['native_ready']==['actual']
    intent=repo.register_native_attempt('actual',approval)
    raw=json.loads(intent.read_text())
    assert intent.parent.parent.name==identity['attempt_fingerprint']
    assert raw['core_identity']==identity['core_identity'] and raw['actual_identity']==identity['actual_identity']
    assert raw['attempt_fingerprint']==identity['attempt_fingerprint']
    assert raw['actual_launch']==package['actual_launch']

@pytest.mark.parametrize('damage',['unknown','bool','provider','provider_entry','logical','mask','gpu','source','origin','debugger'])
def test_companion_rejects_unbound_or_wider_contract(repo,tmp_path,damage):
    package,spec,launch=actual_fixture(repo,tmp_path);mod=launch_module()
    if damage=='unknown':launch['approved']=True
    elif damage=='bool':launch['device_plan']['physical_gpu_index']=True
    elif damage=='provider':launch['provider']['path']='src/arbitrary.py'
    elif damage=='provider_entry':launch['provider']['entrypoint']='run_b'
    elif damage=='logical':launch['device_plan']['logical_device']='cuda:3'
    elif damage=='mask':launch['device_plan']['cuda_visible_devices']='3,2'
    elif damage=='gpu':launch['device_plan']['physical_gpu_index']=2
    elif damage=='source':launch['source_closure_sha256']='f'*64
    elif damage=='origin':launch['checkpoint_origin']={}
    else:launch['debugger']=None
    with pytest.raises(ValueError):mod.validate_actual_launch(launch,repository=repo,package=package,recipe=repo.native_recipe(package))

def test_old_core_only_approval_cannot_register_actual(repo,tmp_path):
    package,_,launch=actual_fixture(repo,tmp_path)
    approval=dict(version=1,batch_id='cpu-fixture',output_root=str(tmp_path/'output'),packages={'actual':launch['core_recipe_identity']})
    with pytest.raises(ValueError,match='actual launch approval'):repo.start_native('actual',approval)

def test_companion_change_invalidates_approval_and_receipt_dispatch(repo,tmp_path):
    package,_,launch=actual_fixture(repo,tmp_path)
    first=repo.native_execution_identity(package)
    # Formatting alone is part of the independently frozen companion identity.
    p=repo.root/package['actual_launch']['path']
    p.write_text(json.dumps(launch,indent=4))
    with pytest.raises(ValueError):repo.native_execution_identity(package)
    package['actual_launch']['sha256']=scheduler.file_hash(repo.root/package['actual_launch']['path'])
    second=repo.native_execution_identity(package)
    assert first['attempt_fingerprint']!=second['attempt_fingerprint']
    registry=repo.registry();registry['packages']['actual']=package;write_json(repo.root/scheduler.REGISTRY,registry)
    with pytest.raises(ValueError,match='actual_artifacts_not_implemented'):
        repo._verified_native_receipt('actual','cpu-only-fixture',repo.native_recipe(package),'f'*64)

def test_l03_resume_explicitly_rejected(repo,tmp_path):
    package,_,launch=actual_fixture(repo,tmp_path)
    launch['role']='b_resume';launch['provider']['entrypoint']='run_b'
    launch['checkpoint_origin']=dict(package='a',receipt_sha256='f'*64)
    with pytest.raises(ValueError,match='actual_resume_not_implemented'):
        launch_module().validate_actual_launch(launch,repository=repo,package=package,recipe=repo.native_recipe(package))

def test_actual_device_checks_actual_mask_and_enumeration(repo,tmp_path):
    _,_,launch=actual_fixture(repo,tmp_path);mod=launch_module()
    assert mod.validate_actual_device(launch['device_plan'],cuda_visible_devices='3',physical_gpu_indices=[0,1,2,3],logical_device_count=1,
        physical_pci_bus_id='00000000:65:00.0',logical_pci_bus_id='0000:65:00.0')['logical_device']=='cuda:0'
    for mask,count,indices in [('',1,[3]),('3,2',2,[2,3]),('3',0,[]),('3',2,[3])]:
        with pytest.raises(ValueError):mod.validate_actual_device(launch['device_plan'],cuda_visible_devices=mask,physical_gpu_indices=indices,logical_device_count=count,
            physical_pci_bus_id='00000000:65:00.0',logical_pci_bus_id='0000:65:00.0')

def test_actual_device_requires_driver_logical_pci_match_to_physical_three(repo,tmp_path):
    _,_,launch=actual_fixture(repo,tmp_path)
    with pytest.raises(ValueError,match='actual_device_identity_unverified'):
        launch_module().validate_actual_device(launch['device_plan'],cuda_visible_devices='3',physical_gpu_indices=[0,1,2,3],
            logical_device_count=1,physical_pci_bus_id='00000000:65:00.0',logical_pci_bus_id='0000:17:00.0')

def test_isolated_worker_argv_cannot_authorize_itself(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_launch import main
    assert main(['--config',str(tmp_path/'no-config'),'--config-sha256','a'*64,'--additional-updates','1','--output',str(tmp_path/'out')])==2
    assert not (tmp_path/'out').exists()

def test_environment_change_cannot_live_only_in_companion(repo,tmp_path):
    package,_,launch=actual_fixture(repo,tmp_path)
    other=tmp_path/'other.kit';other.write_text('# altered experience')
    launch['device_plan'].update(experience_path=str(other),experience_sha256=scheduler.file_hash(other))
    with pytest.raises(ValueError,match='core environment'):
        launch_module().validate_actual_launch(launch,repository=repo,package=package)

def test_missing_provider_rejected_before_identity_or_registration(repo,tmp_path):
    package,spec,_=actual_fixture(repo,tmp_path)
    (repo.root/PROVIDER).unlink()
    with pytest.raises(ValueError,match='actual_provider_not_implemented'):
        repo.native_execution_identity(package)

def test_local_producer_dependency_must_be_in_core_inventory(repo,tmp_path):
    package,spec,launch=actual_fixture(repo,tmp_path)
    name='src/go2_pvcnn/control/dh116/actual_fixture_helper.py'
    (repo.root/name).write_text('# explicit CPU dependency\n')
    (repo.root/PROVIDER).write_text('from . import actual_fixture_helper\n')
    launch['provider']['sha256']=scheduler.file_hash(repo.root/PROVIDER)
    pin_source_closure(repo.root,spec);write_json(repo.root/package['command_spec'],spec)
    core=repo.native_recipe(package)
    launch['core_recipe_identity']={k:core[k] for k in ('command_spec_sha256','code_config_sha256')}
    launch['source_closure_sha256']=spec['teacher_contract']['source_sha256']
    with pytest.raises(ValueError,match='source closure incomplete'):
        launch_module().validate_actual_launch(launch,repository=repo,package=package,recipe=core)

def test_loaded_local_producer_must_be_frozen(repo,tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    package,_,_=actual_fixture(repo,tmp_path)
    module_path=repo.root/'src/undeclared_dynamic_fixture.py';module_path.write_text('# CPU fixture\n')
    monkeypatch.setitem(sys.modules,'undeclared_dynamic_fixture',SimpleNamespace(__file__=str(module_path)))
    with pytest.raises(ValueError,match='loaded producer source closure incomplete'):
        launch_module().audit_loaded_actual_sources(repo,package)

def test_package_initializers_are_part_of_transitive_source_closure(repo,tmp_path):
    package,_,_=actual_fixture(repo,tmp_path)
    (repo.root/'src/go2_pvcnn/__init__.py').write_text('# package import executes this CPU fixture\n')
    with pytest.raises(ValueError,match='source closure incomplete'):
        repo.actual_launch(package)

def test_actual_tests_cannot_be_omitted_from_source_identity(repo,tmp_path):
    package,spec,launch=actual_fixture(repo,tmp_path)
    spec['source_files'].remove('tests/test_native_teacher_actual_launch_process.py')
    pin_source_closure(repo.root,spec);write_json(repo.root/package['command_spec'],spec)
    core=repo.native_recipe(package)
    launch['core_recipe_identity']={k:core[k] for k in ('command_spec_sha256','code_config_sha256')}
    launch['source_closure_sha256']=spec['teacher_contract']['source_sha256']
    with pytest.raises(ValueError,match='source closure incomplete'):
        launch_module().validate_actual_launch(launch,repository=repo,package=package)

def test_scheduler_actual_branch_uses_original_gdb_and_budget_mask(repo,tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as process
    package,spec,launch=actual_fixture(repo,tmp_path)
    attempt=tmp_path/'attempt';attempt.mkdir()
    (repo.root/'scripts/python.sh').write_text('exec "$DH116_PYTHON" "$@"\n')
    (attempt/'intent.json').write_text('{}')
    captures=[]
    class CPUServer:
        def __init__(self,**kwargs):
            self.binding=dict(path=str(attempt/'CPU-binding.json'),sha256='f'*64)
        def __enter__(self):captures.append('enter');return self
        def serve_once(self,child):captures.append('serve')
        def __exit__(self,*args):captures.append('close')
    monkeypatch.setattr(process,'ActualLaunchServer',CPUServer)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    real=scheduler.subprocess.Popen
    def capture(*args,**kwargs):
        captures.append(kwargs['env'].copy())
        return real(*args,**kwargs)
    monkeypatch.setattr(scheduler.subprocess,'Popen',capture)
    result={}
    repo._native_child(attempt,spec,tmp_path/'output-parent'/'native',result,package=package,actual_launch=launch)
    env=next(item for item in captures if isinstance(item,dict))
    assert env['CUDA_VISIBLE_DEVICES']=='3'
    assert env['DH116_ACTUAL_LAUNCH_PATH']==str(attempt/'CPU-binding.json')
    assert Path(env['DH116_PYTHON'])==attempt/'debugger-python'
    assert result['debugger']=='gdb_on_segv'
    assert result['debugger_status_verified'] is True
    assert captures[-1]=='close'
    assert result.get('native_qualification') is not True

def test_actual_server_is_closed_when_popen_fails(repo,tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as process
    package,spec,launch=actual_fixture(repo,tmp_path)
    attempt=tmp_path/'attempt';attempt.mkdir();(attempt/'intent.json').write_text('{}')
    captures=[]
    class CPUServer:
        def __init__(self,**kwargs):self.binding=dict(path='/CPU-only',sha256='f'*64)
        def __enter__(self):captures.append('enter');return self
        def __exit__(self,*args):captures.append('close')
    monkeypatch.setattr(process,'ActualLaunchServer',CPUServer)
    def fail(*args,**kwargs):raise OSError('CPU Popen fixture failure')
    monkeypatch.setattr(scheduler.subprocess,'Popen',fail)
    with pytest.raises(OSError,match='CPU Popen fixture failure'):
        repo._native_child(attempt,spec,tmp_path/'output-parent'/'native',{},package=package,actual_launch=launch)
    assert captures==['enter','close']

def test_cli_main_keeps_canonical_repository_class(repo,tmp_path):
    import subprocess,sys,os
    source=Path(scheduler.__file__).resolve()
    (repo.root/'scripts').mkdir(exist_ok=True)
    (repo.root/'scripts/run_dh116_work_packages.py').write_bytes(source.read_bytes())
    (repo.root/'scripts/python.sh').write_bytes((source.parent/'python.sh').read_bytes())
    git(repo.root,'remote','set-url','origin',f'https://github.com/{scheduler.PROJECT}.git')
    env={**os.environ,'DH116_PYTHON':sys.executable,'CUDA_VISIBLE_DEVICES':''}
    result=subprocess.run(['bash','scripts/python.sh','scripts/run_dh116_work_packages.py','--plan',
        '--evidence-dir',str(tmp_path/'CPU-private')],
        cwd=repo.root,env=env,capture_output=True,text=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert 'native_ready' in json.loads(result.stdout)
    code="import runpy,sys; sys.argv=[sys.argv[1],'--help']; "
    code+="\ntry: runpy.run_path(sys.argv[0],run_name='__main__')\nexcept SystemExit: pass\n"
    code+="from scripts import run_dh116_work_packages as s; assert s.Repository.__module__=='__main__', s.Repository.__module__"
    result=subprocess.run([sys.executable,'-c',code,str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_entry_preserves_external_evidence_dir_from_bound_attempt(repo,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as process
    mod=launch_module();package,spec,launch=actual_fixture(repo,tmp_path)
    external=tmp_path/'private-evidence'
    binding=external/'native-attempts'/'actual'/('f'*64)/'attempt-cpu'/'actual-launch'/'binding.json'
    binding.parent.mkdir(parents=True);binding.write_text('{}')
    monkeypatch.setattr(mod,'__file__',str(repo.root/'src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_launch.py'))
    monkeypatch.setenv('DH116_ACTUAL_LAUNCH_PATH',str(binding));monkeypatch.setenv('DH116_ACTUAL_LAUNCH_SHA256',scheduler.file_hash(binding))
    seen=[]
    def static(binding,*,repository,package=None):
        seen.append(repository.private)
        raise ValueError('CPU stops after static repository selection')
    monkeypatch.setattr(process,'read_launch_binding',static)
    assert mod.main(spec['argv'][:-1]+[str(tmp_path/'never-app')])==2
    assert seen==[external]
