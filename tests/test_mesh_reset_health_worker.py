"""CPU-only worker identity, budget and source binding for the new mesh owner."""
from pathlib import Path
from copy import deepcopy
import sys,importlib,json
import pytest
from test_dh116_pinch_v3 import cpu
from test_dh116_dual_pinch_closure_worker import BLOCKER

from scripts import verify_dh116_mesh_reset_health_projected_pinch as entry
from scripts import run_dh116_mesh_reset_health_work_packages as wrapper
from scripts import prepare_dh116_mesh_reset_health_projected_pinch as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
ROOT=Path(__file__).resolve().parents[1]
ENTRY,WRAPPER=entry.ENTRY,entry.WRAPPER

def test_preflight_uses_new_backend_and_keeps_old_source_default():
    before=source._parameters.BACKEND_SHA
    c=entry.preflight(ROOT)
    assert c["parameter_source_basis"]["backend_sha256"]==source.BACKEND_SHA
    assert c["mesh_observation_source"]["backend_sha256"]==source.BACKEND_SHA
    assert source._parameters.BACKEND_SHA==before==source.OLD_BACKEND_SHA
    assert c["physics_steps"]==234 and c["teacher_admission"] is False
    assert c["native_execution_performed"] is False
    assert c["reset_mesh_qualified"] is False and c["substep_contact_frame_qualified"] is False
    assert set(entry.FILES)<=set(c["sources"]["local_sources"])
    assert c["sources"]["external_sources"][str(source.BACKEND)]==source.BACKEND_SHA
    assert c["sources"]["external_sources"][str(entry.PROJECTED_RECEIPT)]==entry.PROJECTED_SHA

def spec():
    return dict(entrypoint=entry.ENTRY,argv=["--cleanup","explicit","--steps","234",
        "--approved-code-config-sha256","a"*64],
        budget=dict(physics_steps=234,updates=0,environments=1,gpu_index=3,timeout_seconds=180),
        dependency_receipt=dict(package=entry.PARENT_PACKAGE,qualification_scope=entry.PARENT_SCOPE),
        source_files=list(wrapper.AUDIT_SOURCES),debugger="gdb_on_segv",
        validator=dict(equals=dict(case="explicit",version=entry.VERSION,code_config_sha256="a"*64)))
def cleanup(value):
    with bindings((wrapper.base,"ENTRY",entry.ENTRY),(wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(value)

def test_original_exact_budget_kept():
    assert cleanup(spec())=="explicit"

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_no_budget_expansion(key,value):
    value_spec=spec();value_spec["budget"][key]=value
    with pytest.raises(ValueError):cleanup(value_spec)

def test_old_consumed_entry_not_accepted():
    value=spec();value["entrypoint"]="scripts/verify_dh116_projected_pinch.py"
    with pytest.raises(ValueError):cleanup(value)

def test_raw_reset_pair_audit_required_before_qualification(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(wrapper.previous,"audit_outcome",lambda *a:dict(
        lifecycle_qualified=True,teacher_admission=False))
    def inspect(path,*,complete=True,parent_lifecycle_qualified=False):
        assert parent_lifecycle_qualified is True
        seen.append((path,complete))
        return dict(reset_mesh_qualified=True,substep_contact_frame_qualified=True,
            same_clock_health_getter_qualified=True,teacher_admission=False)
    monkeypatch.setattr(operation,"inspect_observations",inspect)
    result=wrapper.audit_outcome(tmp_path,tmp_path/"outer.json","explicit")
    assert seen==[(tmp_path,True)]
    assert result["reset_mesh_qualified"] is True
    assert result["teacher_admission"] is False

def test_failed_parent_stays_failure_without_fake_reset_success(monkeypatch,tmp_path):
    monkeypatch.setattr(wrapper.previous,"audit_outcome",lambda *a:dict(lifecycle_qualified=False))
    def inspect(*a,**k):
        assert k==dict(complete=False,parent_lifecycle_qualified=False)
        raise ValueError("missing reset prefix")
    monkeypatch.setattr(operation,"inspect_observations",inspect)
    result=wrapper.audit_outcome(tmp_path,tmp_path/"outer.json","explicit")
    assert result["reset_mesh_qualified"] is False and result["lifecycle_qualified"] is False
    assert result["mesh_observation_prefix_error"]=="ValueError: missing reset prefix"

def test_observation_failure_cannot_leave_pass_flag(monkeypatch,tmp_path):
    monkeypatch.setattr(wrapper.previous,"audit_outcome",lambda *a:dict(lifecycle_qualified=True))
    monkeypatch.setattr(operation,"inspect_observations",lambda *a,**k:(_ for _ in ()).throw(ValueError("missing raw")))
    with pytest.raises(ValueError,match="missing raw"):
        wrapper.audit_outcome(tmp_path,tmp_path/"outer.json","explicit")

def test_old_health_blocker_unchanged():
    from scripts import verify_dh116_health_projected_pinch as old
    with pytest.raises(RuntimeError,match="deferred_mesh_reset"):
        old.execute(None,ROOT,{})

def test_foreign_source_scope_not_silently_overridden(monkeypatch):
    monkeypatch.setattr(source._parameters,"BACKEND_SHA","f"*64)
    with pytest.raises(ValueError,match="binding"):
        entry.preflight(ROOT)

@pytest.mark.parametrize("reject",[False,True])
def test_real_cpu_scheduler_child_reenters_director_wrapper_and_audits_before_receipt(tmp_path,reject):
    import os,shutil,subprocess
    w=importlib.import_module("scripts.run_dh116_mesh_reset_health_work_packages")
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
    (scripts/"validate_dh116_health_projected_pinch.py").write_text((ROOT/"scripts/validate_dh116_health_projected_pinch.py").read_text()+"""
def audit_outcome(output,parent):
    assert parent.get('cpu_fixture') is True
    from pathlib import Path
    reject=(Path(__file__).parents[1]/'reject').exists()
    return dict(parent,lifecycle_qualified=parent['lifecycle_qualified'] and not reject,
                same_clock_health_getter_qualified=not reject,teacher_admission=False)
""")
    observed=scripts/"prepare_dh116_mesh_reset_health_projected_pinch.py"
    observed.write_text(observed.read_text()+"""
def inspect_observations(output,*,complete=True,parent_lifecycle_qualified=False):
    assert complete==parent_lifecycle_qualified
    return dict(reset_mesh_qualified=complete,substep_contact_frame_qualified=complete,
                same_clock_health_getter_qualified=complete,teacher_admission=False)
""")
    if reject:(tmp_path/"reject").touch()
    source=(ROOT/"scripts/run_dh116_work_packages.py").read_text()
    stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    from types import SimpleNamespace
    from scripts import verify_dh116_mesh_reset_health_projected_pinch as entry
    output.mkdir(parents=True)
    (path/'launch_argv.json').write_text(json.dumps({'argv':['--output',str(output)]}))
    native_debugger_wrapper(path,result)
    capture=(path/'debugger.gdb').read_text()
    assert 'set non-stop off' in capture and 'stopped_thread_count' in capture and str(output) in capture
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    os.environ['DH116_PHYSX_LIFETIME_GDB']='1' # CPU substitute for actual debugger environment.
    entry.require_worker(SimpleNamespace(cleanup='explicit',output=output),self.root)
    (output/'gdb_finalize.json').write_text(json.dumps({'stopped_thread_count':2}))
    for marker_name in ('left_pinch_approach_scope.json','projected_pinch_scope.json'):
        (output/marker_name).write_text(json.dumps(entry.operation.scope_marker()))
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
from scripts import run_dh116_mesh_reset_health_work_packages as w
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

def test_frozen_recipe_matches_all_final_source_bytes(tmp_path):
    from scripts import run_dh116_work_packages as scheduler
    c=entry.preflight(ROOT)
    name="configs/dh116_execution/deformx_mesh_reset_health_projected_pinch_v1.json"
    value=entry.read(ROOT/name)
    assert cleanup(value)=="explicit"
    assert value["validator"]["equals"]["code_config_sha256"]==c["code_config_sha256"]
    assert c["sources"]["external_sources"].items()<=value["inputs"].items()
    assert set(entry.FILES)<=set(value["source_files"])
    repo=scheduler.Repository(ROOT,evidence_dir=tmp_path)
    recipe=repo.native_recipe(dict(command_spec=name,dependencies=[entry.PARENT_PACKAGE],
        dependency_scopes={entry.PARENT_PACKAGE:dict(scope=entry.PARENT_SCOPE,native=True)}))
    assert recipe["spec"]["budget"]["physics_steps"]==234
    assert not any(p.startswith("assets/") for p in recipe["spec"]["source_files"])
    assert entry.WORKER_ENV not in value["environment"]

def test_clean_preflight_no_native_import():
    done=cpu(BLOCKER+"""
from pathlib import Path
from scripts import verify_dh116_mesh_reset_health_projected_pinch as entry
c=entry.preflight(Path.cwd())
entry.assert_clean_import_boundary()
assert c["physics_steps"]==234
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("RIGHT_PREFLIGHT_CPU")
""")
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="RIGHT_PREFLIGHT_CPU"

@pytest.mark.parametrize("marker",[False,True])
def test_real_cli_marker_no_app(tmp_path,marker):
    done=cpu(BLOCKER+"""
import os,json
from pathlib import Path
from scripts import verify_dh116_mesh_reset_health_projected_pinch as entry
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
