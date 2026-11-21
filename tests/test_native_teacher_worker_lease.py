"""Real isolated CPU child FLOCK tests. All JSON is synthetic, not authorization.

No production lock, GPU, App, native module or optimizer is touched.
"""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from scripts import run_dh116_work_packages as scheduler
from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config, authorization_config_digest
from test_native_teacher_worker_contract import recipe, pin_source_closure

MODULE="go2_pvcnn.control.dh116.tabletop_native_teacher_worker_lease"

CHILD = r"""
import fcntl, importlib, json, os, sys, time
from pathlib import Path
lock=Path(sys.argv[1]);stream=lock.open('a+b');fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
stat=lock.stat();text=Path('/proc/self/stat').read_text();start=int(text.rsplit(')',1)[1].split()[19])
print(json.dumps(dict(pid=os.getpid(),starttime_ticks=start,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    lock_device=stat.st_dev,lock_inode=stat.st_ino,issued_monotonic_s=time.monotonic())),flush=True)
lease=None
for line in sys.stdin:
    message=json.loads(line)
    try:
        if message['operation']=='unlock':fcntl.flock(stream.fileno(),fcntl.LOCK_UN);print(json.dumps({'ok':True}),flush=True);continue
        if message['operation']=='close':lease.close();print(json.dumps({'ok':True}),flush=True);continue
        if message['operation']=='revalidate':lease.revalidate();print(json.dumps({'ok':True}),flush=True);continue
        from scripts import run_dh116_work_packages as scheduler
        scheduler.NATIVE_LOCK=lock
        module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_native_teacher_worker_lease')
        repo=scheduler.Repository(message['root'])
        lease=module.verify_worker_lease(message['config'],repo=repo,package=message['package'],expected_semantic_sha256='1'*64)
        import pickle
        try:pickle.dumps(lease)
        except TypeError:nonserializable=True
        else:nonserializable=False
        print(json.dumps({'ok':True,'holder_pid':lease.holder_pid,'native_ready':False,'nonserializable':nonserializable}),flush=True)
    except Exception as error:print(json.dumps({'ok':False,'type':type(error).__name__,'error':str(error)}),flush=True)
"""


def write(path,value):path.write_text(json.dumps(value));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def holder(tmp_path):
    lock=tmp_path/'isolated-cpu.lock'
    child=subprocess.Popen([sys.executable,'-u','-c',CHILD,str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    metadata=json.loads(child.stdout.readline())
    try:yield child,metadata,lock
    finally:
        child.stdin.close();child.wait(timeout=5)
        assert child.returncode==0,child.stderr.read()


def fixture(tmp_path,holder):
    child,meta,lock=holder
    root=tmp_path/'repo';root.mkdir()
    pre=candidate_config();pre['budget']=dict(max_sdk_steps=258,max_resets=1,max_solver_calls=256,max_episode_count=1,wall_timeout_s=180)
    frozen=tmp_path/'frozen-training.json';write(frozen,dict(preflight=deepcopy(pre)))
    spec=recipe(tmp_path);binding=dict(path=str(frozen),sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())
    spec['teacher_contract']['training_config']=binding;spec['inputs']={binding['path']:binding['sha256']}
    spec['argv'][1]=binding['path'];spec['argv'][3]=binding['sha256']
    for name in [spec['entrypoint'],'scripts/python.sh','scripts/run_dh116_work_packages.py',*spec['source_files']]:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# structural CPU fixture only\n')
    command='configs/dh116_execution/cpu-fixture-v2.json';p=root/command;p.parent.mkdir(parents=True);pin_source_closure(root,spec);p.write_text(json.dumps(spec))
    package=dict(command_spec=command,dependencies=[],tests=[],executor='native_bounded',status='pending',
        title='CPU isolated child worker fixture',requires_approval=True,resource_class=scheduler.NATIVE_SCOPE,
        evidence_scope='cpu_fixture',dependency_scopes={},retry_policy=scheduler.NATIVE_RETRY)
    registry=root/scheduler.REGISTRY;registry.parent.mkdir(parents=True,exist_ok=True)
    write(registry,dict(version=2,packages={'cpu-native-worker':package}))
    identity=scheduler.Repository(root).native_recipe(package)
    hashes={k:identity[k] for k in ('command_spec_sha256','code_config_sha256')}
    auth=dict(schema='dh116-native-teacher-budget-authorization-v1',scope='new_native_teacher_ppo',
        profile=pre['profile'],style=pre['style'],budget=pre['budget'],max_ppo_updates=1,
        config_sha256=authorization_config_digest(pre),session_uid='synthetic-cpu-child',owner_pid=meta['pid'],
        owner_starttime_ticks=meta['starttime_ticks'],boot_id=meta['boot_id'],recipe_identity=hashes,
        semantic_sha256='1'*64,source_sha256=spec['teacher_contract']['source_sha256'],issued_monotonic_s=meta['issued_monotonic_s'],
        deadline_monotonic_s=meta['issued_monotonic_s']+180,evidence_scope='synthetic_cpu_fixture_not_native_authorization')
    auth_path=tmp_path/'SYNTHETIC_CPU_ONLY_authorization.json';pre['authorization']=write(auth_path,auth)
    resource=dict(schema='dh116-native-resource-reservation-v1',session_uid=auth['session_uid'],resource_class='isaac_native_singleton',
        authorization_sha256=pre['authorization']['sha256'],pid=meta['pid'],owner_starttime_ticks=meta['starttime_ticks'],
        boot_id=meta['boot_id'],lock_device=meta['lock_device'],lock_inode=meta['lock_inode'],recipe_identity=hashes)
    resource_path=tmp_path/'SYNTHETIC_CPU_ONLY_resource.json';pre['resource_lock']=write(resource_path,resource)
    return dict(operation='verify',root=str(root),config=pre,package=package),auth_path,resource_path


def ask(child,message):
    child.stdin.write(json.dumps(message)+'\n');child.stdin.flush();return json.loads(child.stdout.readline())


def test_real_child_lock_binding_revalidates_but_is_never_native_proof(tmp_path,holder):
    assert importlib.util.find_spec(MODULE) is not None,'source-bound worker lease verifier missing'
    message,auth,resource=fixture(tmp_path,holder);child,meta,lock=holder
    result=ask(child,message)
    assert result==dict(ok=True,holder_pid=meta['pid'],native_ready=False,nonserializable=True)
    assert ask(child,dict(operation='revalidate'))==dict(ok=True)
    ask(child,dict(operation='close'))
    result=ask(child,dict(operation='revalidate'));assert not result['ok'] and 'closed' in result['error']


@pytest.mark.parametrize('damage',['auth_only','pid','starttime','boot','inode','session','expired','source','recipe','config','updates'])
def test_cpu_fixture_cannot_bless_different_live_owner_or_source(tmp_path,holder,damage):
    message,auth_path,resource_path=fixture(tmp_path,holder);child,meta,lock=holder
    pre=message['config'];auth=json.loads(auth_path.read_text());resource=json.loads(resource_path.read_text())
    if damage=='auth_only':pre['resource_lock']=None
    elif damage=='pid':auth['owner_pid']=os.getpid();resource['pid']=os.getpid()
    elif damage=='starttime':auth['owner_starttime_ticks']+=1;resource['owner_starttime_ticks']+=1
    elif damage=='boot':auth['boot_id']='wrong-boot';resource['boot_id']='wrong-boot'
    elif damage=='inode':resource['lock_inode']+=1
    elif damage=='session':resource['session_uid']='other-session'
    elif damage=='expired':auth['issued_monotonic_s']-=1000;auth['deadline_monotonic_s']-=1000
    elif damage=='source':(Path(message['root'])/'scripts/python.sh').write_text('# altered source\n')
    elif damage=='recipe':auth['recipe_identity']['code_config_sha256']='9'*64
    elif damage=='config':pre['ppo']['gamma']=.9
    else:auth['max_ppo_updates']=2
    if damage!='auth_only':
        pre['authorization']=write(auth_path,auth);resource['authorization_sha256']=pre['authorization']['sha256']
        pre['resource_lock']=write(resource_path,resource)
    result=ask(child,message)
    assert result['ok'] is False,result
    assert result['type']=='ValueError',result


@pytest.mark.parametrize('change',['unlock','source','replace_inode','unregistered'])
def test_lease_is_rechecked_after_each_external_change(tmp_path,holder,change):
    message,auth_path,resource_path=fixture(tmp_path,holder);child,meta,lock=holder
    assert ask(child,message)['ok']
    if change=='unlock':ask(child,dict(operation='unlock'))
    elif change=='source':(Path(message['root'])/'scripts/python.sh').write_text('# changed source\n')
    elif change=='unregistered':write(Path(message['root'])/scheduler.REGISTRY,dict(version=2,packages={}))
    else:lock.rename(lock.with_suffix('.old'));lock.write_text('replacement inode')
    result=ask(child,dict(operation='revalidate'))
    assert result['ok'] is False,result


@pytest.mark.parametrize('damage',['foreign','unregistered','ambiguous'])
def test_real_child_lock_does_not_admit_caller_package_outside_unique_registry(tmp_path,holder,damage):
    message,auth_path,resource_path=fixture(tmp_path,holder)
    registry=Path(message['root'])/scheduler.REGISTRY
    if damage=='foreign':message['package']['title']='unregistered caller package'
    elif damage=='unregistered':write(registry,dict(version=2,packages={}))
    else:write(registry,dict(version=2,packages={'cpu-native-worker':message['package'],'duplicate':message['package']}))
    result=ask(holder[0],message)
    assert not result['ok'],result
    assert result['type']=='ValueError' and 'registry' in result['error'],result


def test_parent_cannot_borrow_live_child_lease(tmp_path,holder,monkeypatch):
    module=importlib.import_module(MODULE);message,auth,resource=fixture(tmp_path,holder)
    monkeypatch.setattr(scheduler,'NATIVE_LOCK',holder[2])
    with pytest.raises(ValueError,match='worker|holder|process'):
        module.verify_worker_lease(message['config'],repo=scheduler.Repository(message['root']),
            package=message['package'],expected_semantic_sha256='1'*64)


def test_json_or_constructor_cannot_create_a_lease(tmp_path,holder):
    module=importlib.import_module(MODULE)
    message,auth,resource=fixture(tmp_path,holder)
    with pytest.raises(TypeError,match='verification'):module.WorkerLease()
    for value in (True,{},dict(approved=True)):
        with pytest.raises(ValueError):
            module.verify_worker_lease(value,repo=scheduler.Repository(message['root']),
                package=message['package'],expected_semantic_sha256='1'*64)


@pytest.mark.parametrize('value',[True,[],None])
def test_malformed_auth_json_is_a_closed_gate(tmp_path,holder,value):
    message,auth_path,resource_path=fixture(tmp_path,holder)
    message['config']['authorization']=write(auth_path,value)
    result=ask(holder[0],message)
    assert result['ok'] is False
    assert result['type']=='ValueError',result


def test_recipe_grasp_cannot_disagree_with_the_actual_held_worker_config(tmp_path,holder):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
    message,auth_path,resource_path=fixture(tmp_path,holder)
    root=Path(message["root"])
    command=root/message["package"]["command_spec"]
    spec=json.loads(command.read_text())
    spec["teacher_contract"]["style"]="tripod"
    spec["source_files"]=sorted(set(spec["source_files"])|worker.TRIPOD_SOURCE_FILES)
    for name in worker.TRIPOD_SOURCE_FILES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("# explicit CPU source-closure fixture only\n")
    pin_source_closure(root,spec);write(command,spec)
    recipe=scheduler.Repository(root).native_recipe(message["package"])
    identity={k:recipe[k] for k in ("command_spec_sha256","code_config_sha256")}
    auth=json.loads(auth_path.read_bytes());resource=json.loads(resource_path.read_bytes())
    auth.update(recipe_identity=identity,source_sha256=spec["teacher_contract"]["source_sha256"])
    message["config"]["authorization"]=write(auth_path,auth)
    resource.update(recipe_identity=identity,authorization_sha256=message["config"]["authorization"]["sha256"])
    message["config"]["resource_lock"]=write(resource_path,resource)
    # Config/auth/template all still say Pinch; the frozen recipe says Tripod.
    # Every source digest and the real child FLOCK are otherwise valid.
    result=ask(holder[0],message)
    assert result["ok"] is False,result
    assert result["type"]=="ValueError" and any(word in result["error"] for word in ("style","grasp")),result
