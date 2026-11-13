"""Dedicated-worker transport and qualification CPU tests, no native execution."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
WRAPPER="scripts/run_dh116_physx_lifetime_work_packages.py"
ENTRY="scripts/probe_dh116_physx_lifetime.py"
SOURCES=[WRAPPER,ENTRY,"scripts/dh116_physx_lifetime_acquisition.py",
 "scripts/dh116_physx_lifetime_debugger.py","scripts/validate_dh116_physx_lifetime.py"]
def wrapper():
    assert (ROOT/WRAPPER).exists(),"dedicated worker missing"
    return importlib.import_module("scripts.run_dh116_physx_lifetime_work_packages")
def spec():
    return dict(entrypoint=ENTRY,argv=["--cleanup","registry","--output","{output}"],
                source_files=SOURCES,debugger="gdb_on_segv",validator={"equals":{"case":"registry"}})

@pytest.mark.parametrize("generic,audit",[(True,True),(True,False),(False,True),(False,False)])
def test_postexit_gate_never_upgrades_failed_generic_and_always_preserves_diagnostic(tmp_path,monkeypatch,generic,audit):
    w=wrapper(); previous="prior fixture marker";monkeypatch.setenv(w.WORKER_ENV,previous)
    def child(self,path,spec,output,result,**kwargs):
        env=json.loads(os.environ[w.WORKER_ENV])
        assert env["case"]=="registry" and env["output"]==str(output.resolve())
        assert env["entry_sha256"]==w.scheduler.file_hash(ROOT/ENTRY)
        result["native_qualification"]=generic
    monkeypatch.setattr(w.scheduler.Repository,"_native_child",child)
    monkeypatch.setattr(w,"audit_outcome",lambda *a:{"lifecycle_qualified":audit,"cpu_fixture":True})
    result={};w.LifetimeRepository(ROOT)._native_child(tmp_path,spec(),tmp_path/"native",result)
    assert result["native_qualification"] is (generic and audit)
    assert os.environ[w.WORKER_ENV]==previous
    assert (tmp_path/w.AUDIT_NAME).exists()

def test_worker_scope_restores_marker_on_base_exception(tmp_path,monkeypatch):
    w=wrapper();monkeypatch.delenv(w.WORKER_ENV,raising=False)
    def fail(*a,**k):raise RuntimeError("CPU worker failed")
    monkeypatch.setattr(w.scheduler.Repository,"_native_child",fail)
    with pytest.raises(RuntimeError,match="CPU worker"):
        w.LifetimeRepository(ROOT)._native_child(tmp_path,spec(),tmp_path/"native",{})
    assert w.WORKER_ENV not in os.environ

def test_old_prerequisite_receipts_keep_their_original_validator(tmp_path,monkeypatch):
    w=wrapper();path=tmp_path/"old"
    monkeypatch.setattr(w.scheduler.Repository,"_verified_native_receipt",lambda *a:path)
    monkeypatch.setattr(w,"audit_outcome",lambda *a:pytest.fail("old receipt was reinterpreted"))
    assert w.LifetimeRepository(ROOT)._verified_native_receipt("old","old",{"spec":{"entrypoint":"old.py"}},"x")==path

def test_globals_restore_even_when_cli_raises(monkeypatch):
    w=wrapper();original=(w.scheduler.__file__,w.scheduler.Repository,w.scheduler.native_debugger_wrapper)
    def fail():
        assert w.scheduler.__file__==str(ROOT/WRAPPER)
        assert w.scheduler.Repository is w.LifetimeRepository
        raise RuntimeError("CPU CLI error")
    monkeypatch.setattr(w.scheduler,"main",fail)
    with pytest.raises(RuntimeError,match="CPU CLI"):w.main()
    assert (w.scheduler.__file__,w.scheduler.Repository,w.scheduler.native_debugger_wrapper)==original

@pytest.mark.parametrize("reject",[False,True])
def test_real_subprocess_reenters_wrapper_preserves_root_and_capture_and_audit(tmp_path,reject):
    w=wrapper();scripts=tmp_path/"scripts";scripts.mkdir();(scripts/"__init__.py").touch()
    for name in (WRAPPER,"scripts/dh116_physx_lifetime_debugger.py"):
        shutil.copyfile(ROOT/name,tmp_path/name)
    (scripts/"probe_dh116_physx_lifetime.py").write_text("VERSION='dh116-physx-lifetime-probe-v1'\n")
    (scripts/"dh116_physx_lifetime_acquisition.py").write_text("# CPU stub\n")
    (scripts/"validate_dh116_physx_lifetime.py").write_text("""
from pathlib import Path
def audit_outcome(output,outer,cleanup):
    assert not (Path(outer).parent/'receipt.json').exists()
    return {'lifecycle_qualified': not (Path(__file__).parents[1]/'reject').exists(),'cpu_fixture':True}
""")
    if reject:(tmp_path/"reject").touch()
    source=(ROOT/"scripts/run_dh116_work_packages.py").read_text()
    stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    output.mkdir(parents=True)
    (path/'launch_argv.json').write_text(json.dumps({'argv':['--output',str(output)]}))
    native_debugger_wrapper(path,result)
    capture=(path/'debugger.gdb').read_text()
    assert 'DH116_PHYSX_LIFETIME_GDB' in capture and str(output) in capture
    marker=json.loads(os.environ['DH116_PHYSX_LIFETIME_WORKER_V1'])
    assert marker['output']==str(output)
    result.update(native_qualification=True,native_execution_performed=False,
                  worker_root=str(self.root),worker_private=str(self.private),worker_pid=os.getpid(),
                  worker_cli_entry=str(Path(sys.argv[0]).resolve()))
Repository._native_child=_cpu_native
Repository.verify_local_target=lambda self:None
Repository.head=lambda self:'b'*40
Repository.registry=lambda self:json.loads((self.private/'registry.json').read_text())
Repository.native_recipe=lambda self,package:json.loads((self.private/'recipe.json').read_text())
Repository.native_plan=lambda self,**kwargs:{'native_qualified':[],'native_ready':['lifetime']}
Repository.native_resource_blocker=lambda *a,**k:None
NATIVE_SCHEDULER_LOCK=Path(__file__).resolve().parents[1]/'private'/'reservation.lock'
"""
    marker="if __name__ == '__main__':";assert source.count(marker)==1
    (scripts/"run_dh116_work_packages.py").write_text(source.replace(marker,stub+"\n"+marker))
    private=tmp_path/"private";private.mkdir()
    identity=dict(command_spec_sha256="c"*64,code_config_sha256="d"*64)
    recipe=dict(spec=dict(spec(),budget={"timeout_seconds":10}),**identity)
    (private/"recipe.json").write_text(json.dumps(recipe))
    package=dict(executor="native_bounded",evidence_scope="cpu_fixture_only",dependencies=[],dependency_scopes={})
    (private/"registry.json").write_text(json.dumps({"packages":{"lifetime":package}}))
    script=tmp_path/"parent.py"
    script.write_text("""
import json,sys
from pathlib import Path
from scripts import run_dh116_physx_lifetime_work_packages as w
root=Path(__file__).resolve().parent
repo=w.LifetimeRepository(root,evidence_dir=root/'private')
approval={'version':1,'batch_id':'cpu','output_root':str(root/'artifacts'/'output'),
          'packages':{'lifetime':{'command_spec_sha256':'c'*64,'code_config_sha256':'d'*64}}}
original=w.scheduler.__file__
w.scheduler.__file__=str(root/'scripts/run_dh116_physx_lifetime_work_packages.py')
try:
    result=repo.start_native('lifetime',approval)
finally:
    w.scheduler.__file__=original
print(json.dumps(result))
""")
    env={**os.environ,"PYTHONPATH":str(tmp_path),"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"}
    done=subprocess.run([sys.executable,str(script)],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=20)
    assert done.returncode==0,done.stdout+done.stderr
    child=json.loads(done.stdout.splitlines()[-1])
    assert child["worker_cli_entry"]==str(tmp_path/WRAPPER)
    assert child["worker_pid"]!=os.getpid()
    assert child["worker_root"]==str(tmp_path) and child["worker_private"]==str(private)
    assert child["native_execution_performed"] is False
    assert child["native_qualification"] is (not reject)
    receipt=next((private/"native-attempts").glob("*/*/*/receipt.json"))
    assert json.loads(receipt.read_text())["native_qualification"] is (not reject)


@pytest.mark.parametrize("case",["registry","explicit"])
def test_recipe_pins_case_budget_sources_and_installed_sdk_without_environment_bypass(case):
    w=wrapper()
    relative=f"configs/dh116_execution/physx_lifetime_{case}_v1.json"
    assert (ROOT/relative).exists(),"independent case recipe missing"
    recipe=w.LifetimeRepository(ROOT).native_recipe({"command_spec":relative,"tests":[]})
    spec=recipe["spec"]
    assert w.cleanup_case(spec)==case
    assert spec["budget"]["physics_steps"]==2 and spec["budget"]["updates"]==0
    assert spec["budget"]["timeout_seconds"]==180 and spec["budget"]["gpu_index"]==3
    assert spec["validator"]["line_counts"]=={"initialization_boundaries.jsonl":2}
    assert w.WORKER_ENV not in spec.get("environment",{})
    assert all(name in spec["source_files"] for name in SOURCES)
    assert any("libomni.physx.tensors.plugin.so" in name for name in spec["inputs"])
    assert any("simulation_manager.py" in name for name in spec["inputs"])
