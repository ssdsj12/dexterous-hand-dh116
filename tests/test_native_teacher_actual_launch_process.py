"""Real CPU peer/lock fixtures; never native qualification or production locks."""
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import pytest

MODULE = "go2_pvcnn.control.dh116.tabletop_native_teacher_actual_process"


def module():
    assert importlib.util.find_spec(MODULE) is not None, "parent-issued actual process handshake missing"
    return importlib.import_module(MODULE)


def test_process_identity_is_kernel_derived_and_stable():
    m = module()
    observed = m.process_identity(os.getpid())
    assert observed.pid == os.getpid()
    assert observed.uid == os.getuid()
    assert observed.gid == os.getgid()
    assert observed.starttime_ticks > 0
    assert observed == m.process_identity(os.getpid())
    assert len(observed.cmdline_sha256) == 64
    assert observed.exe_inode > 0


def test_peer_identity_uses_so_peercred_not_payload():
    m = module()
    with socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)[0] as left:
        # socketpair credentials were captured by the kernel at creation.
        assert m.peer_process_identity(left) == m.process_identity(os.getpid())


def test_public_constructor_cannot_create_actual_lease():
    with pytest.raises(TypeError, match="verified"):
        module().ActualWorkerLease()


def test_real_child_peer_is_derived_without_caller_pid(tmp_path):
    m = module()
    sock = tmp_path / "peer.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
        listener.bind(str(sock)); listener.listen(1); listener.settimeout(5)
        child = subprocess.Popen([sys.executable, "-c", "import socket,sys; s=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);s.connect(sys.argv[1]);s.send(b'pid=1');s.recv(1)",str(sock)])
        try:
            with listener.accept()[0] as connection:
                assert connection.recv(1024) == b"pid=1"
                identity = m.peer_process_identity(connection)
                assert identity.pid == child.pid
                assert identity == m.process_identity(child.pid)
                connection.send(b"x")
        finally:
            child.wait(timeout=5)
        assert child.returncode == 0


CPU_WORKER = r"""
import argparse,json,os,sys
from pathlib import Path
from scripts import run_dh116_work_packages as scheduler
from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as launch
parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--config-sha256');parser.add_argument('--additional-updates');parser.add_argument('--output');args=parser.parse_args()
root=Path(__file__).parents[1]
scheduler.NATIVE_LOCK=root/'CPU-ONLY.lock'
scheduler.NATIVE_SCHEDULER_LOCK=root/'CPU-SCHEDULER.lock'
repository=scheduler.Repository(root,evidence_dir=os.environ['CPU_EVIDENCE'])
binding=dict(path=os.environ['DH116_ACTUAL_LAUNCH_PATH'],sha256=os.environ['DH116_ACTUAL_LAUNCH_SHA256'])
try:
    envelope=launch.read_launch_binding(binding,repository=repository,package=None)
    package=repository.registry()['packages'][envelope.package_name]
    if os.environ.get('CPU_MODE')=='parent_lock':
        lease=launch.request_actual_launch(binding,repository=repository,package=package)
    else:
        import fcntl
        with scheduler.NATIVE_LOCK.open('a+b') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            if os.environ.get('CPU_MODE') in ('nonce','inode','pid_payload'):
                import socket
                with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as connection:
                    connection.connect(envelope.socket_path)
                    info=scheduler.NATIVE_LOCK.stat()
                    hello=dict(nonce=envelope.nonce,lock_device=info.st_dev,lock_inode=info.st_ino)
                    if os.environ['CPU_MODE']=='nonce':hello['nonce']='replayed'
                    elif os.environ['CPU_MODE']=='inode':hello['lock_inode']+=1
                    else:hello['pid']=os.getpid()
                    connection.send(json.dumps(hello).encode())
                    connection.recv(4096)
                raise ValueError('nonce replay rejected')
            lease=launch.request_actual_launch(binding,repository=repository,package=package)
            assert lease.role=='a_fresh' and lease.resume_origin is None
            assert lease.device_plan==envelope.device_plan and lease.launch_identity==envelope.actual_identity
            original=lease.device_plan;original['logical_device']='tampered'
            assert lease.device_plan['logical_device']=='cuda:0'
            original=lease.launch_identity;original['schema']='tampered'
            assert lease.launch_identity==envelope.actual_identity
            try:lease.role='b_resume'
            except AttributeError:pass
            else:raise AssertionError('mutable lease role')
            lease.revalidate()
            mode=os.environ.get('CPU_MODE')
            if mode in ('revalidate_process','revalidate_reply','revalidate_source','unlock'):
                if mode=='revalidate_process':Path(lease._reply.process_config['path']).write_text('{}')
                elif mode=='revalidate_reply':Path(lease._reply_binding['path']).write_text('{}')
                elif mode=='revalidate_source':(root/'scripts/python.sh').write_text('# changed CPU source')
                else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
                lease.revalidate()
                raise AssertionError('changed lease was not rejected')
            # Grant consumed the listener; revalidate above succeeds without IPC.
            import socket
            with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as extra:
                try:extra.connect(envelope.socket_path)
                except (ConnectionRefusedError,FileNotFoundError):pass
                else:raise AssertionError('grant listener allowed reconnect')
            import pickle
            try:pickle.dumps(lease)
            except TypeError:pass
            else:raise AssertionError('serializable lease')
            lease.close()
            # Closing a borrowed lease must not release the caller lock.
            launch.verify_kernel_lock(os.getpid(),scheduler.NATIVE_LOCK)
            print('CPU_RESULT '+json.dumps(dict(ok=True,holder_pid=os.getpid(),native_ready=False)),flush=True)
except Exception as error:
    print('CPU_RESULT '+json.dumps(dict(ok=False,error=str(error))),flush=True)
    sys.exit(2)
"""


CPU_PARENT = r"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from scripts import run_dh116_work_packages as scheduler
from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as launch
parser=argparse.ArgumentParser();parser.add_argument('--native-worker');parser.add_argument('--evidence-dir');args=parser.parse_args()
root=Path(__file__).parents[1]
scheduler.NATIVE_LOCK=root/'CPU-ONLY.lock';scheduler.NATIVE_SCHEDULER_LOCK=root/'CPU-SCHEDULER.lock'
repository=scheduler.Repository(root,evidence_dir=args.evidence_dir)
intent=Path(args.native_worker);binding=dict(path=str(intent),sha256=scheduler.file_hash(intent))
state=json.loads(intent.read_text());package=repository.registry()['packages'][state['package']]
companion=repository.actual_launch(package);spec=repository.native_recipe(package)['spec']
mode=os.environ.get('CPU_MODE','')
try:
    import contextlib
    with (scheduler.native_file_lock(intent.parent/'worker.lock') if mode!='missing_worker_lock' else contextlib.nullcontext()), (scheduler.native_file_lock(scheduler.NATIVE_SCHEDULER_LOCK) if mode!='missing_scheduler_lock' else contextlib.nullcontext()):
        with launch.ActualLaunchServer(repository=repository,package=package,intent_binding=binding,launch=companion,directory=intent.parent/'actual-launch',deadline=time.monotonic()+3) as server:
            env={**os.environ,'CPU_EVIDENCE':str(repository.private),'DH116_ACTUAL_LAUNCH_PATH':server.binding['path'],'DH116_ACTUAL_LAUNCH_SHA256':server.binding['sha256']}
            if mode in ('starttime','boot','pid'):
                binding_path=Path(server.binding['path']);value=json.loads(binding_path.read_text())
                if mode=='starttime':value['parent']['starttime_ticks']+=1
                elif mode=='boot':value['parent']['boot_id']='wrong-boot'
                else:value['parent']['pid']=1
                binding_path.write_bytes(launch._encode(value))
                env['DH116_ACTUAL_LAUNCH_SHA256']=scheduler.file_hash(binding_path)
            if mode=='symlink_binding':
                binding_path=Path(server.binding['path']);other=binding_path.with_suffix('.old');binding_path.rename(other);binding_path.symlink_to(other)
            if mode=='symlink_overlay':
                (intent.parent/'actual-launch/authorization.json').symlink_to(root/'forbidden-authorization.json')
            if mode=='intent':intent.write_text('{}')
            if mode=='expired':time.sleep(3.05)
            held=scheduler.NATIVE_LOCK.open('a+b')
            try:
                import fcntl
                if mode=='parent_lock':fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                interpreter=scheduler.native_debugger_wrapper(intent.parent,{}) if os.environ.get('CPU_DEBUGGER')=='1' else sys.executable
                argv=[interpreter,str(root/spec['entrypoint']),*(state['output'] if value=='{output}' else value for value in spec['argv'])]
                child=subprocess.Popen(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                spawn=child
                if mode=='foreign_root':spawn=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)'])
                try:
                    reply=server.serve_once(spawn)
                    print('CPU_SERVER '+json.dumps(dict(ok=True,reply=reply)),flush=True)
                except Exception as error:
                    print('CPU_SERVER '+json.dumps(dict(ok=False,error=str(error))),flush=True)
                out,err=child.communicate(timeout=10);print(out+err,flush=True)
                if spawn is not child:spawn.wait(timeout=5)
                code=child.returncode
            finally:held.close()
    sys.exit(code)
except Exception as error:
    print('CPU_SERVER '+json.dumps(dict(ok=False,error=str(error))),flush=True);sys.exit(2)
"""


def cpu_fixture(repo,tmp_path):
    from copy import deepcopy
    import hashlib,json
    from test_native_teacher_actual_launch import actual_fixture,PROVIDER
    from test_native_teacher_worker_contract import pin_source_closure
    from test_dh116_work_packages import write_json,git
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config
    package,spec,launch=actual_fixture(repo,tmp_path)
    worker=repo.root/spec['entrypoint'];worker.write_text(CPU_WORKER)
    (repo.root/'scripts/run_dh116_work_packages.py').write_text(CPU_PARENT)
    pre=candidate_config();pre['budget']=dict(max_sdk_steps=258,max_resets=1,max_solver_calls=256,max_episode_count=1,wall_timeout_s=180)
    config=Path(spec['teacher_contract']['training_config']['path'])
    write_json(config,dict(schema='dh116-native-teacher-training-cpu-v1',preflight=pre))
    config_binding=dict(path=str(config),sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    spec['teacher_contract']['training_config']=config_binding
    spec['argv'][1]=str(config);spec['argv'][3]=config_binding['sha256'];spec['inputs']={str(config):config_binding['sha256']}
    pin_source_closure(repo.root,spec);write_json(repo.root/package['command_spec'],spec)
    core=repo.native_recipe(package);launch['core_recipe_identity']={k:core[k] for k in ('command_spec_sha256','code_config_sha256')}
    launch['source_closure_sha256']=spec['teacher_contract']['source_sha256']
    write_json(repo.root/package['actual_launch']['path'],launch)
    package['actual_launch']['sha256']=hashlib.sha256((repo.root/package['actual_launch']['path']).read_bytes()).hexdigest()
    registry=repo.registry();registry['packages']['actual']=package;write_json(repo.root/scheduler.REGISTRY,registry)
    git(repo.root,'add','.');git(repo.root,'commit','-m','CPU actual source fixture')
    identity=repo.native_execution_identity(package)
    approval=dict(version=1,batch_id='cpu-only-fixture',output_root=str(tmp_path/'output'),packages={'actual':{**identity['core_identity'],'actual_identity':identity['actual_identity']}})
    intent=repo.register_native_attempt('actual',approval)
    return repo.registry()['packages']['actual'],spec,launch,dict(path=str(intent),sha256=hashlib.sha256(intent.read_bytes()).hexdigest())


from scripts import run_dh116_work_packages as scheduler
from test_dh116_work_packages import repo

def run_cpu_parent(repo,intent,*,mode='',debugger=False):
    env={**os.environ,'CPU_MODE':mode,'CPU_DEBUGGER':'1' if debugger else '0'}
    return subprocess.run([sys.executable,str(repo.root/'scripts/run_dh116_work_packages.py'),'--native-worker',intent['path'],'--evidence-dir',str(repo.private)],cwd=repo.root,env=env,capture_output=True,text=True,timeout=20)


@pytest.mark.parametrize('debugger',[False,True])
def test_real_cpu_direct_and_original_gdb_grant_natural_exit(repo,tmp_path,debugger):
    m=module();package,spec,launch,intent=cpu_fixture(repo,tmp_path)
    result=run_cpu_parent(repo,intent,debugger=debugger)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert '"ok": true' in result.stdout and '"native_ready": false' in result.stdout
    reply=Path(intent['path']).parent/'actual-launch/reply.json'
    assert reply.is_file()
    if debugger:
        log=tmp_path/'gdb-cpu.log';log.write_text(result.stdout+result.stderr)
        assert scheduler.debugger_status(log)['debugger_inferior_exit_code']==0


@pytest.mark.parametrize('damage',['parent_lock','nonce','inode','pid_payload','intent','expired','foreign_root','starttime','boot','pid','symlink_binding','symlink_overlay','missing_worker_lock','missing_scheduler_lock'])
def test_real_cpu_handshake_rejects_foreign_or_unapproved_grant(repo,tmp_path,damage):
    m=module();package,spec,launch,intent=cpu_fixture(repo,tmp_path)
    result=run_cpu_parent(repo,intent,mode=damage)
    assert result.returncode==2,(result.stdout,result.stderr)
    assert '"ok": false' in result.stdout


def test_pytest_parent_cannot_self_issue_actual_grant(repo,tmp_path):
    m=module();package,spec,launch,intent=cpu_fixture(repo,tmp_path)
    with pytest.raises(ValueError,match='scheduler'):
        with m.ActualLaunchServer(repository=repo,package=package,intent_binding=intent,launch=launch,directory=Path(intent['path']).parent/'actual-launch',deadline=__import__('time').monotonic()+5):
            pass


@pytest.mark.parametrize('damage',['revalidate_process','revalidate_reply','revalidate_source','unlock'])
def test_granted_lease_revalidates_original_files_and_kernel_lock_without_reconnect(repo,tmp_path,damage):
    m=module();package,spec,launch,intent=cpu_fixture(repo,tmp_path)
    result=run_cpu_parent(repo,intent,mode=damage)
    assert result.returncode==2,(result.stdout,result.stderr)
    assert 'CPU_SERVER {"ok": true' in result.stdout
    assert 'CPU_RESULT {"ok": false' in result.stdout


@pytest.mark.parametrize("damage", ["none", "receipt_sha", "intent", "binding", "reply",
                                    "process", "outer", "native_log", "artifacts", "promotion"])
def test_actual_diagnostic_receipt_rechecks_finished_cpu_grant(repo, tmp_path, monkeypatch, damage):
    import json
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_artifacts as artifacts

    package, _, _, intent_binding = cpu_fixture(repo, tmp_path)
    child = run_cpu_parent(repo, intent_binding, debugger=True)
    assert child.returncode == 0, child.stdout + child.stderr
    attempt = Path(intent_binding["path"]).parent
    intent = json.loads(Path(intent_binding["path"]).read_bytes())
    output = Path(intent["output"])
    output.mkdir(parents=True)
    native_log = attempt / "native.log"
    native_log.write_text(child.stdout + child.stderr)
    outer = attempt / "outer_exit.json"
    measured = dict(outer_exit_code=0, native_lock_observed=True, duration_seconds=.5,
                    log_sha256=scheduler.file_hash(native_log), **scheduler.debugger_status(native_log))
    outer.write_text(json.dumps(measured))
    files = ("intent.json", "actual-launch/binding.json", "actual-launch/reply.json", "outer_exit.json")
    evidence = {"native_qualification": False, "native_ready": False, "teacher_admission": False,
                "artifact_sha256": {"raw": "a" * 64}}
    receipt = dict(measured, **repo.native_execution_identity(package), **intent["core_identity"],
                   executor="native_bounded", native_execution_performed=True, exit_code=0,
                   tested_head=intent["head"], batch_id=intent["approval"]["batch_id"],
                   qualification_scope=package["evidence_scope"], output=str(output),
                   reason="actual_receipt_audit_pending", blocker_kind="failed_native",
                   actual_launch=package["actual_launch"],
                   launch_evidence_sha256={name: scheduler.file_hash(attempt / name) for name in files},
                   **evidence)

    def verify(_spec, path, *, repository, package, envelope):
        assert path == output and repository is repo and envelope.package_name == "actual"
        assert not Path(envelope.socket_path).exists()
        if damage == "artifacts":
            return dict(evidence, artifact_sha256={"raw": "b" * 64})
        return evidence

    monkeypatch.setattr(artifacts, "audit_actual_fresh_outputs", verify)
    files_to_damage = dict(intent="intent.json", binding="actual-launch/binding.json",
        reply="actual-launch/reply.json", process="actual-launch/process.json",
        outer="outer_exit.json", native_log="native.log")
    if damage in files_to_damage:
        (attempt / files_to_damage[damage]).write_text("{}")
    if damage == "promotion":
        receipt["native_qualification"] = True
    path = attempt / "receipt.json"
    path.write_text(json.dumps(receipt))
    sha = scheduler.file_hash(path) if damage != "receipt_sha" else "b" * 64
    # An offline readback must also work when later unrelated commits move HEAD.
    monkeypatch.setattr(repo, "head", lambda: "d" * 40)
    if damage == "none":
        result = repo.verify_actual_diagnostic("actual", sha)
        assert result["diagnostic_verified"] is True
        assert result["native_qualification"] is False and result["teacher_admission"] is False
        assert result["receipt_sha256"] == sha
    else:
        with pytest.raises(ValueError):
            repo.verify_actual_diagnostic("actual", sha)
