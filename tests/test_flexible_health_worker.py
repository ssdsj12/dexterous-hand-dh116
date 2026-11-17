"""CPU only: actual subprocess reentry and scope/fingerprint gates."""
from copy import deepcopy
import importlib,sys,json
from pathlib import Path
import pytest
from scripts import verify_dh116_health_projected_pinch as entry
from scripts import run_dh116_health_projected_pinch_work_packages as wrapper
from scripts import prepare_dh116_health_projected_pinch as operation
from scripts import prepare_dh116_left_pinch_approach as plan
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from test_dh116_pinch_v3 import cpu
from test_dh116_dual_pinch_closure_worker import BLOCKER
ROOT=Path(__file__).resolve().parents[1]
ENTRY,WRAPPER=entry.ENTRY,entry.WRAPPER
RECIPE="configs/dh116_execution/deformx_health_projected_pinch_v1.json"
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
from scripts import verify_dh116_health_projected_pinch as entry
c=entry.preflight(Path.cwd())
entry.assert_clean_import_boundary()
assert c["physics_steps"]==234
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("RIGHT_PREFLIGHT_CPU")
""")
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="RIGHT_PREFLIGHT_CPU"

def cleanup(spec):
    with bindings((wrapper.base,"ENTRY",entry.ENTRY),(wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(spec)

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_budget_cannot_expand(key,value):
    spec=entry.read(ROOT/RECIPE);spec["budget"][key]=value
    with pytest.raises(ValueError):cleanup(spec)

@pytest.mark.parametrize("flag",["main","execute"])
def test_unresolved_reset_blocks_before_preflight_mkdir_or_app(monkeypatch,tmp_path,flag):
    from types import SimpleNamespace
    called=[]
    monkeypatch.setattr(entry,"preflight",lambda *a,**k:called.append("preflight"))
    monkeypatch.setattr(sys,"argv",[entry.ENTRY,"--execute-native","--output",str(tmp_path/"native")])
    with pytest.raises(RuntimeError,match="deferred_mesh_reset"):
        if flag=="main":entry.main()
        else:entry.execute(SimpleNamespace(output=tmp_path/"native"),ROOT,{})
    assert called==[] and not (tmp_path/"native").exists()

@pytest.mark.parametrize("reject",[False,True])
def test_real_cpu_scheduler_child_reenters_director_wrapper_and_audits_before_receipt(tmp_path,reject):
    import os,shutil,subprocess
    w=importlib.import_module("scripts.run_dh116_health_projected_pinch_work_packages")
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
    if reject:(tmp_path/"reject").touch()
    source=(ROOT/"scripts/run_dh116_work_packages.py").read_text()
    stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    from types import SimpleNamespace
    from scripts import verify_dh116_health_projected_pinch as entry
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
from scripts import run_dh116_health_projected_pinch_work_packages as w
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

def test_exact_health_sources_and_projected_receipt_are_bound():
    c=entry.preflight(ROOT)
    assert c["projected_source_receipt_sha256"]==entry.PROJECTED_SHA
    assert c["sources"]["external_sources"][str(entry.PROJECTED_RECEIPT)]==entry.PROJECTED_SHA
    for suffix in ("tabletop_health_getter_evidence.py","tabletop_r0_health_separation.py",
                   "validate_dh116_health_projected_pinch.py"):
        assert any(name.endswith(suffix) for name in c["sources"]["local_sources"])
    assert c["live_r0_computed"] is False
    assert c["native_execution_allowed"] is False and c["mesh_reset_qualified"] is False
    assert c["blocked_reasons"]==[entry.NATIVE_BLOCKER]

def test_worker_source_tamper_changes_preflight_without_editing_real_files(monkeypatch):
    a=entry.preflight(ROOT)
    from scripts import verify_dh116_effective_right_pinch as core
    sha=core.sha
    monkeypatch.setattr(core,"sha",lambda p:"f"*64 if str(p).endswith("tabletop_health_getter_evidence.py") else sha(p))
    b=entry.preflight(ROOT)
    assert a["code_config_sha256"]!=b["code_config_sha256"]

def test_actual_recipe_matches_final_preflight():
    c=entry.preflight(ROOT);spec=entry.read(ROOT/RECIPE)
    assert cleanup(spec)=="explicit"
    assert spec["entrypoint"]==entry.ENTRY
    assert set(wrapper.AUDIT_SOURCES)<=set(spec["source_files"])
    assert spec["validator"]["equals"]["code_config_sha256"]==c["code_config_sha256"]
    assert spec["argv"][spec["argv"].index("--approved-code-config-sha256")+1]==c["code_config_sha256"]
    assert c["sources"]["external_sources"].items()<=spec["inputs"].items()
    assert entry.WORKER_ENV not in spec["environment"]
