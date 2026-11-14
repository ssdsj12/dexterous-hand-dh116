"""CPU process boundary tests. No Isaac or CUDA runtime is launched."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
ENTRY="scripts/verify_dh116_pinch_candidate_v3.py"
WRAPPER="scripts/run_dh116_pinch_work_packages_v3.py"
BLOCKER="""
import sys,importlib.abc,traceback
class BlockEarlyUsd(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('pxr','PhysxSchema'):
            raise RuntimeError('CPU_GUARD_EARLY_USD:'+fullname)
sys.meta_path.insert(0,BlockEarlyUsd())
"""

def cpu(code,timeout=40):
    return subprocess.run([sys.executable,"-c",code],cwd=ROOT,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},
        capture_output=True,text=True,timeout=timeout)

def test_v2_reproduces_early_usd_import_in_real_cpu_process():
    done=cpu(BLOCKER+"""
from pathlib import Path
from scripts.verify_dh116_pinch_candidate_v2 import preflight
try:preflight(Path.cwd())
except RuntimeError as error:
    assert str(error).startswith('CPU_GUARD_EARLY_USD:pxr')
    print('REPRODUCED')
else:raise AssertionError('expected actual v2 early import')
""")
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="REPRODUCED"

def test_v3_preflight_and_wrapper_import_do_not_import_usd():
    done=cpu(BLOCKER+"""
from pathlib import Path
from scripts import verify_dh116_pinch_candidate_v3 as entry
from scripts import run_dh116_pinch_work_packages_v3 as wrapper
contract,cfg=entry.preflight(Path.cwd())
entry.assert_clean_import_boundary()
assert 'ropesim' not in sys.modules and 'isaaclab' not in sys.modules
assert contract['version']==entry.VERSION
assert cfg['initial_pose']['version']=='dh116-pinch-pose-candidate-v2'
print('BYTE_ONLY_PREFLIGHT')
""")
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="BYTE_ONLY_PREFLIGHT"

@pytest.mark.parametrize("wrong_marker",[False,True])
def test_real_native_cli_reaches_app_constructor_without_early_usd(tmp_path,wrong_marker):
    # Real entry, real byte preflight, worker gate, real v2 execute function.
    # Only the lock, CUDA module and AppLauncher are CPU substitutes.
    code=BLOCKER+"""
import contextlib,json,runpy,types
from pathlib import Path
from scripts import verify_dh116_pinch_candidate_v3 as entry
from go2_pvcnn.control.dh116 import tabletop_stereo_native as native_lock
native_lock.native_instance_lock=contextlib.nullcontext
contract,cfg=entry.preflight(Path.cwd())
output=Path(OUTPUT)
marker=dict(version=entry.VERSION,case='explicit',output=str(output.resolve()),
            entry_sha256=entry.sha(Path.cwd()/entry.ENTRY),
            wrapper_sha256=entry.sha(Path.cwd()/entry.WRAPPER))
if WRONG:marker['version']='consumed-v2-marker'
os.environ[entry.WORKER_ENV]=json.dumps(marker)
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['DH116_PHYSX_LIFETIME_GDB']='1'
solver=types.ModuleType('ropesim._ropesim')
solver.__file__=str(entry.PAIR_ROOT/'ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so')
package=types.ModuleType('ropesim');package._ropesim=solver
sys.modules.update({'ropesim':package,'ropesim._ropesim':solver})
class StopAtApp(RuntimeError):pass
class CpuLauncher:
    def __init__(self,**kwargs):
        entry.assert_clean_import_boundary()
        assert kwargs['enable_cameras'] is False and kwargs['headless'] is True
        print('APP_CONSTRUCTOR_CPU_SENTINEL')
        raise StopAtApp('no native app constructed')
app=types.ModuleType('isaaclab.app');app.AppLauncher=CpuLauncher
lab=types.ModuleType('isaaclab');lab.__path__=[]
sys.modules.update({'isaaclab':lab,'isaaclab.app':app})
sys.argv=[entry.ENTRY,'--cleanup','explicit','--output',str(output),'--headless',
          '--execute-native','--approved-code-config-sha256',contract['code_config_sha256']]
try:runpy.run_module('scripts.verify_dh116_pinch_candidate_v3',run_name='__main__')
except StopAtApp:
    assert not WRONG and output.is_dir()
except ValueError as error:
    assert WRONG and 'worker' in str(error) and not output.exists()
else:raise AssertionError('native must not execute')
"""
    code="import os\n"+code.replace("OUTPUT",repr(str(tmp_path/"native"))).replace("WRONG",repr(wrong_marker))
    done=cpu(code)
    assert done.returncode==0,done.stdout+done.stderr
    assert ("APP_CONSTRUCTOR_CPU_SENTINEL" in done.stdout) is (not wrong_marker)

def test_byte_source_identity_matches_independent_full_cpu_geometry():
    from scripts import verify_dh116_pinch_candidate_v3 as entry
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    contract,cfg=entry.old_preflight(entry.default_arguments(ROOT),ROOT)
    assert entry.pose_source_identity(ROOT,contract)==load_geometry(ROOT,side="left")["source_sha256"]

@pytest.mark.parametrize("name",["pxr","pxr.Usd","pxr.PhysxSchema","PhysxSchema"])
def test_already_imported_usd_rejected(name,monkeypatch):
    entry=importlib.import_module("scripts.verify_dh116_pinch_candidate_v3")
    monkeypatch.setitem(sys.modules,name,SimpleNamespace())
    with pytest.raises(RuntimeError,match="before AppLauncher"):
        entry.assert_clean_import_boundary()

def test_empty_owners_preserves_primary_failure_and_actual_app_close(tmp_path):
    from scripts import dh116_pinch_startup_v3 as startup
    from scripts import dh116_pinch_acquisition_v2 as old
    closed=[]
    primary="CPU SimulationContext constructor failure"
    def make_sim():raise RuntimeError(primary)
    before=(old.VERSION,old.LifetimeTrace,old.snapshot_weakrefs)
    with pytest.raises(RuntimeError,match=primary) as caught:
        startup.run_lifecycle(SimpleNamespace(close=lambda:closed.append(True),update=lambda:None),
            make_sim=make_sim,acquire=lambda *a:pytest.fail("no acquired simulation"),
            synchronize=lambda:None,clear_sim=lambda sim:None,stop=lambda:None,
            manager_is_none=lambda:True,freeze=lambda:False,output=tmp_path,cleanup="explicit")
    assert closed==[True]
    assert "diagnostic bounds" not in str(caught.value)
    result=json.loads((tmp_path/"result.json").read_text())
    assert result["version"]==startup.VERSION
    assert result["physics_steps"] is None and result["lifetime_released_before_close"] is False
    assert result["failure"].count(primary)==1
    assert (old.VERSION,old.LifetimeTrace,old.snapshot_weakrefs)==before
    for phase in ("before_app_close","after_app_close"):
        data=json.loads((tmp_path/(phase+".refs.json")).read_text())
        assert data["resources"]=={} and data["native_release_verified"] is False
        assert data["observation"]=="no_registered_owners"
        assert (tmp_path/(phase+".maps")).stat().st_size>0

def test_nonempty_snapshot_keeps_old_strict_bounds():
    from scripts import dh116_pinch_startup_v3 as startup
    with pytest.raises(ValueError,match="bounds"):
        startup.snapshot_for_startup({str(i):None for i in range(33)})
    result=startup.snapshot_for_startup({"unsupported":None})
    assert result["resources"]["unsupported"]["alive"] is None

def test_v3_recipe_scope_and_identity():
    from scripts import verify_dh116_pinch_candidate_v3 as entry
    from scripts import run_dh116_pinch_work_packages_v3 as wrapper
    spec=json.loads((ROOT/"configs/dh116_execution/deformx_pinch_left_candidate_v3.json").read_text())
    contract,cfg=entry.preflight(ROOT)
    assert spec["entrypoint"]==entry.ENTRY
    with wrapper.bindings((wrapper.base,"ENTRY",entry.ENTRY),
                          (wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        assert wrapper.cleanup_case(spec)=="explicit"
    recipe=wrapper.base.scheduler.Repository(ROOT).native_recipe({
        "command_spec":"configs/dh116_execution/deformx_pinch_left_candidate_v3.json",
        "tests":["tests/test_dh116_pinch_v3.py","tests/test_dh116_pinch_v2.py","tests/test_pinch_initial_geometry_v2.py"]})
    assert len(recipe["code_config_sha256"])==64
    assert spec["argv"][spec["argv"].index("--approved-code-config-sha256")+1]==contract["code_config_sha256"]
    assert set(wrapper.AUDIT_SOURCES)<=set(spec["source_files"])
    assert spec["budget"]==json.loads((ROOT/"configs/dh116_execution/deformx_pinch_left_candidate_v2.json").read_text())["budget"]
    assert entry.WORKER_ENV not in spec["environment"]

def test_v3_wrapper_restores_bindings_on_failure(monkeypatch):
    from scripts import run_dh116_pinch_work_packages_v3 as w
    fields=("__file__","ENTRY","WRAPPER","VERSION","WORKER_ENV","AUDIT_NAME","AUDIT_SOURCES",
            "cleanup_case","render_gdb_commands","audit_outcome")
    before={n:getattr(w.base,n) for n in fields}
    def fail():
        assert w.base.VERSION==w.entry.VERSION
        assert w.base.__file__==w.__file__
        raise RuntimeError("CPU wrapper failure")
    monkeypatch.setattr(w.base,"main",fail)
    with pytest.raises(RuntimeError,match="CPU wrapper failure"):w.main()
    assert before=={n:getattr(w.base,n) for n in fields}

@pytest.mark.parametrize("reject",[False,True])
def test_real_cpu_worker_reentry_keeps_v3_marker_stopped_capture_and_postexit_gate(tmp_path,reject):
    import os,shutil,subprocess
    w=importlib.import_module("scripts.run_dh116_pinch_work_packages_v3")
    scripts=tmp_path/"scripts";scripts.mkdir();(scripts/"__init__.py").touch()
    shutil.copytree(ROOT/"scripts",scripts,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    for name in set(w.AUDIT_SOURCES):
        (tmp_path/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,tmp_path/name)
    (scripts/"validate_dh116_pinch_v3.py").write_text("""
from pathlib import Path
VERSION='cpu-fixture-old-validator'
def audit_outcome(output,outer,cleanup):
    assert not (Path(outer).parent/'receipt.json').exists()
    return {'lifecycle_qualified':not (Path(__file__).parents[1]/'reject').exists(),
            'teacher_admission':False,'grasp_qualified':False,'cpu_fixture':True}
""")
    if reject:(tmp_path/"reject").touch()
    source=(ROOT/"scripts/run_dh116_work_packages.py").read_text()
    stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    from types import SimpleNamespace
    from scripts import verify_dh116_pinch_candidate_v3 as entry
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
NATIVE_SCHEDULER_LOCK=Path(__file__).resolve().parents[1]/'private'/'reservation.lock'
"""
    marker="if __name__ == '__main__':";assert source.count(marker)==1
    (scripts/"run_dh116_work_packages.py").write_text(source.replace(marker,stub+"\n"+marker))
    private=tmp_path/"private";private.mkdir()
    identity=dict(command_spec_sha256="c"*64,code_config_sha256="d"*64)
    spec=dict(entrypoint=ENTRY,argv=["--cleanup","explicit","--output","{output}",
        "--approved-code-config-sha256","a"*64],
        source_files=list(w.AUDIT_SOURCES),debugger="gdb_on_segv",budget={"timeout_seconds":180,"physics_steps":240,"gpu_index":3,"environments":1,"updates":0},
        validator={"equals":{"case":"explicit","version":w.entry.VERSION,"code_config_sha256":"a"*64}})
    (private/"recipe.json").write_text(json.dumps(dict(spec=spec,**identity)))
    package=dict(executor="native_bounded",evidence_scope="cpu_fixture_only",dependencies=[],dependency_scopes={})
    (private/"registry.json").write_text(json.dumps({"packages":{"frame":package}}))
    parent="""
import json
from pathlib import Path
from scripts import run_dh116_pinch_work_packages_v3 as w
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


@pytest.mark.parametrize("damage",[None,"missing_boundary","early_usd","old_version","empty_owners","outer","clock","live"])
def test_v3_full_disk_gate_keeps_prior_requirements(tmp_path,monkeypatch,damage):
    from tests.test_dh116_pinch_v2 import test_independent_disk_audit_accepts_complete_cpu_fixture_and_rejects_damage
    from scripts import validate_dh116_pinch_v3 as audit
    from scripts import dh116_pinch_startup_v3 as startup
    from scripts import validate_dh116_pinch_v2 as prior
    test_independent_disk_audit_accepts_complete_cpu_fixture_and_rejects_damage(tmp_path,monkeypatch,None)
    output=tmp_path/"native"
    def read(name):return json.loads((output/name).read_text())
    def write(name,value):(output/name).write_text(json.dumps(value))
    result=read("result.json");result["version"]=startup.VERSION
    if damage=="old_version":result["version"]="dh116-single-left-pinch-v2"
    if damage=="clock":result["clocks"][-1]["sdk_step"]=241
    write("result.json",result)
    boundary=dict(version=startup.BOUNDARY_VERSION,phase="before_isaaclab_app_import",
        loaded_usd_modules=[],native_execution_performed=False)
    if damage=="early_usd":boundary["loaded_usd_modules"]=["pxr"]
    if damage!="missing_boundary":write("before_app_launcher.json",boundary)
    if damage=="empty_owners":write("before_app_close.refs.json",startup.snapshot_for_startup({}))
    if damage=="live":
        refs=read("before_app_close.refs.json");refs["resources"]["contact_view"]["alive"]=True
        write("before_app_close.refs.json",refs)
    if damage=="outer":
        outer=tmp_path/"outer_exit.json";value=json.loads(outer.read_text())
        value["outer_exit_code"]=1;outer.write_text(json.dumps(value))
    before=prior.VERSION
    outcome=audit.audit_outcome(output,tmp_path/"outer_exit.json","explicit")
    assert prior.VERSION==before
    assert outcome["lifecycle_qualified"] is (damage is None),outcome
    assert outcome["grasp_qualified"] is outcome["teacher_admission"] is False

def test_pose_source_damage_is_rejected_before_geometry(monkeypatch):
    from scripts import verify_dh116_pinch_candidate_v3 as entry
    original=entry.pose_source_identity
    def damaged(project,contract):
        result=original(project,contract);result["usd"]="0"*64
        return result
    monkeypatch.setattr(entry,"pose_source_identity",damaged)
    with pytest.raises(ValueError,match="locked sources"):
        entry.preflight(ROOT)
