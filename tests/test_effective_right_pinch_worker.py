"""CPU identity, scope restoration and scheduler gates; no native execution."""
from copy import deepcopy
import importlib
import sys
import json
from pathlib import Path
import pytest
from scripts import verify_dh116_effective_right_pinch as entry
from scripts import run_dh116_effective_right_work_packages as wrapper
from scripts import prepare_dh116_effective_right_native as operation
from scripts import prepare_dh116_pinch_margin_contact as margin
from scripts import dh116_pinch_margin_contact_audit as audit
from scripts import prepare_dh116_native_parameter_task as parameter
from scripts import prepare_dh116_effective_right_pinch as plan
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from test_dh116_pinch_v3 import cpu
from test_dh116_dual_pinch_closure_worker import BLOCKER
ROOT=Path(__file__).resolve().parents[1]
ENTRY,WRAPPER=entry.ENTRY,entry.WRAPPER
RECIPE="configs/dh116_execution/deformx_effective_right_pinch_contact_v1.json"

def test_scope_routes_all_old_local_imports_and_restores():
    before=(margin.VERSION,margin.action_schedule,margin.inspect_interval,audit.operation,audit.SCOPE,parameter.VERSION)
    old=margin.load_pair()
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    for fail in (False,True):
        try:
            with operation.scenario_scope(ROOT):
                assert margin.VERSION==before[0] and parameter.VERSION==operation.VERSION
                assert operation.action_schedule().equal(plan.action_schedule())
                assert margin.inspect_interval is before[2]
                assert audit.operation is operation and audit.SCOPE==operation.SCOPE
                assert builder.load_pair_proposal()==plan.load_pair(ROOT)
                assert margin.load_pair()==old
                if fail:raise RuntimeError("test")
        except RuntimeError:assert fail
        assert before==(margin.VERSION,margin.action_schedule,margin.inspect_interval,audit.operation,audit.SCOPE,parameter.VERSION)

def test_preflight_uses_right_identity_and_original_qualified_parent():
    c=entry.preflight(ROOT)
    assert c["version"]==operation.VERSION and c["physics_steps"]==234
    assert c["native_execution_performed"] is False and c["teacher_admission"] is False
    assert c["reset_seed_sha256"]==plan.load_pair()["reset_seed"]["sha256"]
    assert set(entry.FILES)<=set(c["sources"]["local_sources"])
    assert c["sources"]["external_sources"][str(plan.PROOF)]==plan.PROOF_SHA
    p,_=entry.load_parent()
    assert p["qualification_scope"]==entry.PARENT_SCOPE and p["native_qualification"] is True

def test_clean_preflight_no_native_import():
    done=cpu(BLOCKER+"""
from pathlib import Path
from scripts import verify_dh116_effective_right_pinch as entry
c=entry.preflight(Path.cwd())
entry.previous.prior.original.prior.assert_clean_import_boundary()
assert c["physics_steps"]==234
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("RIGHT_PREFLIGHT_CPU")
""")
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="RIGHT_PREFLIGHT_CPU"

def cleanup(spec):
    with bindings((wrapper.base,"ENTRY",entry.ENTRY),(wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(spec)

def test_frozen_recipe_and_code_match():
    spec=entry.read(ROOT/RECIPE);c=entry.preflight(ROOT)
    assert cleanup(spec)=="explicit"
    assert spec["entrypoint"]==entry.ENTRY
    assert spec["validator"]["equals"]["version"]==entry.VERSION
    assert spec["validator"]["equals"]["code_config_sha256"]==c["code_config_sha256"]
    assert c["sources"]["external_sources"].items()<=spec["inputs"].items()

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_budget_cannot_expand(key,value):
    spec=entry.read(ROOT/RECIPE);spec["budget"][key]=value
    with pytest.raises(ValueError):cleanup(spec)

@pytest.mark.parametrize("marker",[False,True])
def test_real_cli_marker_no_app(tmp_path,marker):
    done=cpu(BLOCKER+"""
import os,json
from pathlib import Path
from scripts import verify_dh116_effective_right_pinch as entry
out=Path(OUTPUT);c=entry.preflight(Path.cwd());called=[]
os.environ.pop(entry.WORKER_ENV,None)
if ALLOW:
    os.environ[entry.WORKER_ENV]=json.dumps(dict(version=entry.VERSION,case="explicit",output=str(out.resolve()),
        entry_sha256=entry.sha(Path.cwd()/entry.ENTRY),wrapper_sha256=entry.sha(Path.cwd()/entry.WRAPPER)))
os.environ["CUDA_VISIBLE_DEVICES"]="3";os.environ["DH116_PHYSX_LIFETIME_GDB"]="1"
entry.execute=lambda *a:called.append(True)
sys.argv=[entry.ENTRY,"--cleanup","explicit","--output",str(out),"--headless","--execute-native","--steps","234",
    "--approved-code-config-sha256",c["code_config_sha256"],"--left-scheduler-receipt",str(entry.PARENT_RECEIPT),
    "--left-receipt-sha256",entry.PARENT_SHA]
try:entry.main()
except (ValueError,RuntimeError,SystemExit):
    assert not ALLOW and not called and not out.exists()
else:assert ALLOW and called==[True]
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("RIGHT_CLI_CPU")
""".replace("OUTPUT",repr(str(tmp_path/"native"))).replace("ALLOW",repr(marker)))
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="RIGHT_CLI_CPU"

def test_postexit_routes_new_operation_before_auditing(monkeypatch,tmp_path):
    from scripts import validate_dh116_native_parameter_task as validator
    seen=[]
    def checked(*a,**k):
        assert audit.operation is operation
        from scripts import validate_dh116_dual_pinch_closure_interval as closure_audit
        from scripts import validate_dh116_effective_right_contact as own_audit
        assert closure_audit.validate_artifacts is own_audit.validate_artifacts
        assert margin.VERSION!="dh116-effective-right-pinch-contact-v1"
        assert operation.action_schedule().equal(plan.action_schedule())
        seen.append(True)
        return dict(lifecycle_qualified=False,reason="CPU injected failure")
    monkeypatch.setattr(validator,"audit_outcome",checked)
    r=wrapper.audit_outcome(tmp_path,tmp_path/"outer.json","explicit")
    assert seen==[True] and r["lifecycle_qualified"] is False
    assert r["qualification_scope"]==operation.SCOPE and r["teacher_admission"] is False


def test_validator_body_differs_only_in_explicit_operation_import():
    import ast,inspect
    from scripts import validate_dh116_effective_right_contact as new
    a=ast.parse(inspect.getsource(audit.validate_artifacts))
    b=ast.parse(inspect.getsource(new.validate_artifacts))
    a.body[0].body[0].names[0].name="prepare_dh116_effective_right_native"
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

@pytest.mark.parametrize("reject",[False,True])
def test_real_cpu_scheduler_child_reenters_director_wrapper_and_audits_before_receipt(tmp_path,reject):
    import os,shutil,subprocess
    w=importlib.import_module("scripts.run_dh116_effective_right_work_packages")
    scripts=tmp_path/"scripts";scripts.mkdir();(scripts/"__init__.py").touch()
    shutil.copytree(ROOT/"scripts",scripts,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    for name in set(w.AUDIT_SOURCES):
        (tmp_path/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,tmp_path/name)
    (scripts/"validate_dh116_native_parameter_task.py").write_text((ROOT/"scripts/validate_dh116_native_parameter_task.py").read_text()+"""
from pathlib import Path
VERSION='cpu-fixture-old-validator'
def audit_outcome(output,outer,cleanup,*,profile_sha256):
    assert not (Path(outer).parent/'receipt.json').exists()
    return {'lifecycle_qualified':not (Path(__file__).parents[1]/'reject').exists(),
            'teacher_admission':False,'grasp_qualified':False,'cpu_fixture':True}
""")
    if reject:(tmp_path/"reject").touch()
    source=(ROOT/"scripts/run_dh116_work_packages.py").read_text()
    stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    from types import SimpleNamespace
    from scripts import verify_dh116_effective_right_pinch as entry
    output.mkdir(parents=True)
    (path/'launch_argv.json').write_text(json.dumps({'argv':['--output',str(output)]}))
    native_debugger_wrapper(path,result)
    capture=(path/'debugger.gdb').read_text()
    assert 'set non-stop off' in capture and 'stopped_thread_count' in capture and str(output) in capture
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    os.environ['DH116_PHYSX_LIFETIME_GDB']='1' # CPU substitute for actual debugger environment.
    entry.require_worker(SimpleNamespace(cleanup='explicit',output=output),self.root)
    (output/'gdb_finalize.json').write_text(json.dumps({'stopped_thread_count':2}))
    result.update(native_qualification=True,native_execution_performed=False,
        worker_root=str(self.root),worker_private=str(self.private),worker_pid=os.getpid(),
        worker_cli_entry=str(Path(sys.argv[0]).resolve()))
Repository._native_child=_cpu_native
Repository.verify_local_target=lambda self:None
Repository.head=lambda self:'b'*40
Repository.registry=lambda self:json.loads((self.private/'registry.json').read_text())
Repository.native_recipe=lambda self,package:json.loads((self.private/'recipe.json').read_text())
Repository.native_plan=lambda self,**kwargs:{'native_qualified':[],'native_ready':['frame']}
Repository.native_resource_blocker=lambda *a,**k:None
# Only child reentry/audit ordering is exercised here; exact real parent bytes are tested above.
Repository.resolve_native_dependency_receipt=lambda self,binding,packages:dict(
    **binding,path=str(self.private/'cpu-parent.json'),sha256='e'*64)
NATIVE_SCHEDULER_LOCK=Path(__file__).resolve().parents[1]/'private'/'reservation.lock'
"""
    marker="if __name__ == '__main__':";assert source.count(marker)==1
    (scripts/"run_dh116_work_packages.py").write_text(source.replace(marker,stub+"\n"+marker))
    private=tmp_path/"private";private.mkdir()
    identity=dict(command_spec_sha256="c"*64,code_config_sha256="d"*64)
    spec=dict(entrypoint=ENTRY,argv=["--cleanup","explicit","--output","{output}",
        "--approved-code-config-sha256","a"*64],
        dependency_receipt=dict(package=w.entry.PARENT_PACKAGE,qualification_scope=w.entry.PARENT_SCOPE),
        source_files=list(w.AUDIT_SOURCES),debugger="gdb_on_segv",budget={"timeout_seconds":180,"physics_steps":234,"gpu_index":3,"environments":1,"updates":0},
        validator={"equals":{"case":"explicit","version":w.entry.VERSION,"code_config_sha256":"a"*64}})
    (private/"recipe.json").write_text(json.dumps(dict(spec=spec,**identity)))
    package=dict(executor="native_bounded",evidence_scope="cpu_fixture_only",dependencies=[],dependency_scopes={})
    (private/"registry.json").write_text(json.dumps({"packages":{"frame":package}}))
    parent="""
import json
from pathlib import Path
from scripts import run_dh116_effective_right_work_packages as w
root=Path.cwd()
def parent_main():
    repo=w.base.LifetimeRepository(root,evidence_dir=root/'private')
    approval={'version':1,'batch_id':'cpu','output_root':str(root/'artifacts'/'output'),
       'packages':{'frame':{'command_spec_sha256':'c'*64,'code_config_sha256':'d'*64}}}
    print(json.dumps(repo.start_native('frame',approval)))
    return 0
w.base.scheduler.main=parent_main
raise SystemExit(w.main())
"""
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(tmp_path),str(ROOT/"src"))),
        "CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"}
    done=subprocess.run([sys.executable,"-c",parent],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=25)
    assert done.returncode==0,done.stdout+done.stderr
    child=json.loads(done.stdout.splitlines()[-1])
    assert child["worker_cli_entry"]==str(tmp_path/WRAPPER)
    assert child["worker_root"]==str(tmp_path) and child["worker_private"]==str(private)
    assert child["native_execution_performed"] is False
    assert child["native_qualification"] is (not reject)
    receipt=next((private/"native-attempts").glob("*/*/*/receipt.json"))
    assert json.loads(receipt.read_text())["native_qualification"] is (not reject)


def test_actual_scheduler_recipe_loads_and_keeps_assets_out_of_publish_scope(tmp_path):
    from scripts import run_dh116_work_packages as scheduler
    repo=scheduler.Repository(ROOT,evidence_dir=tmp_path)
    package=dict(command_spec=RECIPE,dependencies=[entry.PARENT_PACKAGE],
        dependency_scopes={entry.PARENT_PACKAGE:dict(scope=entry.PARENT_SCOPE,native=True)})
    f=repo.native_recipe(package)
    assert f["spec"]["budget"]["physics_steps"]==234
    assert not any(p.startswith("assets/") for p in f["spec"]["source_files"])
    assert len(f["command_spec_sha256"])==len(f["code_config_sha256"])==64
