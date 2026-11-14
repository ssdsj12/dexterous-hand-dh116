"""CPU gates and actual scheduler subprocess reentry for director acquisition."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import pytest

from scripts import verify_dh116_pinch_margin_contact as entry
from scripts import run_dh116_pinch_margin_contact_work_packages as wrapper
from test_dh116_pinch_v3 import cpu
from test_dh116_dual_pinch_closure_worker import BLOCKER

ROOT=Path(__file__).resolve().parents[1]
ENTRY,WRAPPER=entry.ENTRY,entry.WRAPPER
RECIPE="configs/dh116_execution/deformx_pinch_margin_contact_v1.json"


def test_exact_qualified_closure_parent_and_source_basis():
    parent,inputs=entry.load_parent()
    assert parent["native_qualification"] is True and parent["outer_exit_code"]==0
    assert parent["qualification_scope"]==entry.PARENT_SCOPE
    assert str(entry.PARENT_RECEIPT) in inputs
    c=entry.preflight(ROOT)
    assert c["parameter_endpoint_count"]==233 and c["physics_steps"]==234
    assert c["material_slip_qualified"] is False
    assert set(entry.FILES)<=set(c["sources"]["local_sources"])
    basis=c["parameter_source_basis"]
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader
    assert {k:v for k,v in basis["sources"].items() if k!=str(reader.ROOT/reader.CONFIG)}.items()<=c["sources"]["external_sources"].items()
    assert c["sources"]["local_sources"][reader.CONFIG]==reader.CONFIG_SHA
    assert basis["backend_sha256"]=="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"


@pytest.mark.parametrize("damage",["path","sha","receipt","ledger"])
def test_other_parent_or_changed_bytes_rejected(monkeypatch,damage):
    if damage in ("path","sha"):
        with pytest.raises(ValueError,match="exact published"):
            entry.load_parent(entry.PARENT_RECEIPT.with_name("other") if damage=="path" else entry.PARENT_RECEIPT,
                "0"*64 if damage=="sha" else entry.PARENT_SHA)
    else:
        parent=entry.read(entry.PARENT_RECEIPT)
        target={"receipt":entry.PARENT_RECEIPT,
            "audit":entry.PARENT_RECEIPT.with_name("native-parameter-task-post-exit-audit.json"),
            "ledger":Path(parent["output"])/"physical.jsonl"}[damage]
        actual=entry.previous.prior.sha
        old_locked=entry.operation.read_locked
        def locked(p,h):
            if Path(p)==target:raise ValueError("bytes differ")
            return old_locked(p,h)
        monkeypatch.setattr(entry.operation,"read_locked",locked)
        monkeypatch.setattr(entry,"sha",lambda p:"0"*64 if Path(p)==target else actual(p))
        with pytest.raises(ValueError,match="differ"):entry.load_parent()


def test_clean_preflight_never_imports_native():
    done=cpu(BLOCKER+"""
from pathlib import Path
from scripts import verify_dh116_pinch_margin_contact as entry
value=entry.preflight(Path.cwd())
entry.previous.prior.original.prior.assert_clean_import_boundary()
assert value['physics_steps']==234 and value['native_execution_performed'] is False
assert not forbidden.intersection({x.split('.')[0] for x in sys.modules})
print('DIRECTOR_CPU_PREFLIGHT')
""")
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="DIRECTOR_CPU_PREFLIGHT"


@pytest.mark.parametrize("marker",[False,True])
def test_real_cli_refuses_generic_entry_before_native_and_accepts_dedicated_marker(tmp_path,marker):
    done=cpu(BLOCKER+"""
import os,json
from pathlib import Path
from scripts import verify_dh116_pinch_margin_contact as entry
output=Path(OUTPUT);contract=entry.preflight(Path.cwd());called=[]
os.environ.pop(entry.WORKER_ENV,None)
if MARKER:
    os.environ[entry.WORKER_ENV]=json.dumps(dict(version=entry.VERSION,case='explicit',
        output=str(output.resolve()),entry_sha256=entry.sha(Path.cwd()/entry.ENTRY),
        wrapper_sha256=entry.sha(Path.cwd()/entry.WRAPPER)))
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ['DH116_PHYSX_LIFETIME_GDB']='1'
entry.execute=lambda *args:called.append(True)
sys.argv=[entry.ENTRY,'--cleanup','explicit','--output',str(output),'--headless',
    '--execute-native','--steps','234','--approved-code-config-sha256',contract['code_config_sha256'],
    '--left-scheduler-receipt',str(entry.PARENT_RECEIPT),'--left-receipt-sha256',entry.PARENT_SHA]
try:entry.main()
except (ValueError,RuntimeError,SystemExit):
    assert not MARKER and not called and not output.exists()
else:assert MARKER and called==[True]
assert not forbidden.intersection({x.split('.')[0] for x in sys.modules})
print('CLI_MARKER_CHECKED')
""".replace("OUTPUT",repr(str(tmp_path/"native"))).replace("MARKER",repr(marker)))
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="CLI_"+str(marker)+"_CHECKED"


def cleanup(spec):
    with entry.bindings((wrapper.base,"ENTRY",entry.ENTRY),(wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(spec)


def test_frozen_recipe_source_budget_and_parent_match_current_preflight():
    spec=entry.read(ROOT/RECIPE);c=entry.preflight(ROOT)
    assert cleanup(spec)=="explicit"
    code=spec["argv"][spec["argv"].index("--approved-code-config-sha256")+1]
    assert code==spec["validator"]["equals"]["code_config_sha256"]==c["code_config_sha256"]
    assert spec["entrypoint"]==ENTRY
    assert set(wrapper.AUDIT_SOURCES)|set(entry.FILES)<=set(spec["source_files"])
    assert c["sources"]["external_sources"].items()<=spec["inputs"].items()
    assert entry.WORKER_ENV not in spec["environment"]
    assert spec["dependency_receipt"]==dict(package=entry.PARENT_PACKAGE,qualification_scope=entry.PARENT_SCOPE)
    assert "director_reset.json" in spec["validator"]["artifacts"]


@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_frozen_budget_cannot_expand(key,value):
    spec=entry.read(ROOT/RECIPE);spec["budget"][key]=value
    with pytest.raises(ValueError):cleanup(spec)

@pytest.mark.parametrize("reject",[False,True])
def test_real_cpu_scheduler_child_reenters_director_wrapper_and_audits_before_receipt(tmp_path,reject):
    import os,shutil,subprocess
    w=importlib.import_module("scripts.run_dh116_pinch_margin_contact_work_packages")
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
    from scripts import verify_dh116_pinch_margin_contact as entry
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
from scripts import run_dh116_pinch_margin_contact_work_packages as w
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

@pytest.mark.parametrize("unpublishable_asset",[False,True])
def test_actual_scheduler_loads_frozen_recipe_without_publishing_assets(monkeypatch,tmp_path,unpublishable_asset):
    from scripts import run_dh116_work_packages as scheduler
    recipe=ROOT/RECIPE
    if unpublishable_asset:
        bad=json.loads(recipe.read_bytes())
        bad["source_files"].append("assets/fixed_dual_panda_dh116_v1/dependencies/3046ad2ed855_DH116_right.usd")
        original=Path.read_bytes
        monkeypatch.setattr(Path,"read_bytes",lambda path:json.dumps(bad).encode() if path==recipe else original(path))
    repo=scheduler.Repository(ROOT,evidence_dir=tmp_path)
    p=dict(command_spec=RECIPE,dependencies=[entry.PARENT_PACKAGE],
        dependency_scopes={entry.PARENT_PACKAGE:dict(scope=entry.PARENT_SCOPE,native=True)})
    if unpublishable_asset:
        with pytest.raises(ValueError,match="not a publishable"):repo.native_recipe(p)
    else:
        frozen=repo.native_recipe(p)
        assert len(frozen["command_spec_sha256"])==len(frozen["code_config_sha256"])==64
        assert frozen["spec"]["budget"]["physics_steps"]==234

def test_preflight_fingerprint_is_independent_of_checkout_location(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader
    a=entry.preflight(ROOT)
    integration=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/integration")
    # Both real checkouts carry the same frozen configuration bytes.
    other=integration if ROOT!=integration else Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/native-parameter-task-20261113")
    assert entry.sha(ROOT/reader.CONFIG)==entry.sha(other/reader.CONFIG)
    monkeypatch.setattr(reader,"ROOT",other)
    b=entry.preflight(ROOT)
    assert a["code_config_sha256"]==b["code_config_sha256"]
    assert a["sources"]["local_sources"][reader.CONFIG]==reader.CONFIG_SHA
