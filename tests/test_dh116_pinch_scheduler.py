"""CPU-only scheduler transport tests; stubs are never native evidence."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER="scripts/run_dh116_pinch_work_packages.py"


def wrapper():
    assert (ROOT/WRAPPER).exists(), "dedicated Pinch receipt qualification wrapper is missing"
    return importlib.import_module("scripts.run_dh116_pinch_work_packages")


def spec():
    return {"entrypoint":"scripts/verify_dh116_deformx_pinch_contact.py",
            "argv":["--approved-code-config-sha256","a"*64],
            "source_files":[WRAPPER,"scripts/verify_dh116_deformx_pinch_contact.py",
                            "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py"],
            "validator":{"equals":{"code_config_sha256":"a"*64}}}


@pytest.mark.parametrize("reject",[False,True])
def test_audit_runs_before_receipt_and_any_exception_withdraws_qualification(tmp_path,monkeypatch,reject):
    w=wrapper()
    monkeypatch.setattr(w.scheduler.Repository,"_native_child",
                        lambda self,path,spec,output,result,**kwargs:result.update(native_qualification=True))
    def audit(output,outer,sha):
        assert not (tmp_path/"receipt.json").exists()
        assert sha=="a"*64
        if reject: raise RuntimeError("injected pair-force failure")
        return {"grasp_qualified":False,"teacher_admission":False}
    monkeypatch.setattr(w,"validate_pinch_artifacts",audit)
    repo=w.PinchRepository(ROOT,evidence_dir=tmp_path)
    result={}
    repo._native_child(tmp_path,spec(),tmp_path/"output",result)
    assert result["native_qualification"] is (not reject)
    if reject:
        assert result["blocker_kind"]=="failed_native"
        assert "injected pair-force failure" in result["reason"]
        assert "pinch_post_exit_audit_sha256" not in result
    else:
        assert result["pinch_post_exit_audit_sha256"]==w.scheduler.file_hash(tmp_path/w.AUDIT_NAME)
        assert set(result["pinch_validator_source_sha256"])==set(spec()["source_files"])


def test_wrapper_restores_globals_after_scheduler_error(monkeypatch):
    w=wrapper(); original_file=w.scheduler.__file__; original_repo=w.scheduler.Repository
    def fail():
        assert w.scheduler.__file__==str((ROOT/WRAPPER).resolve())
        assert w.scheduler.Repository is w.PinchRepository
        raise RuntimeError("scheduler failure")
    monkeypatch.setattr(w.scheduler,"main",fail)
    with pytest.raises(RuntimeError,match="scheduler failure"): w.main()
    assert w.scheduler.__file__==original_file
    assert w.scheduler.Repository is original_repo


def test_same_bytes_worktree_has_identical_candidate_fingerprint(tmp_path):
    from scripts.verify_dh116_deformx_pinch_contact import preflight,default_arguments
    reference,_=preflight(default_arguments(ROOT),ROOT)
    relative=set(reference["sources"]["local_sources"])
    relative.add("configs/dh116_execution/deformx_thumb_left_lifecycle_v2.json")
    # _sources also verifies the original other-hand asset read-only.
    manifest=json.loads((ROOT/"assets/fixed_dual_panda_dh116_v1/manifest.json").read_text())
    for row in manifest["sources"]:
        if row["kind"]=="dh116":
            relative.add("assets/fixed_dual_panda_dh116_v1/dependencies/"+Path(row["path"]).name)
    for name in relative:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    moved,_=preflight(default_arguments(tmp_path),tmp_path)
    assert moved["code_config_sha256"]==reference["code_config_sha256"]
    assert not any(str(ROOT) in key for key in moved["sources"]["external_sources"])


@pytest.mark.parametrize("reject",[False,True])
def test_real_start_native_worker_reenters_wrapper_and_audits_before_receipt(tmp_path,reject):
    w=wrapper()
    scripts=tmp_path/"scripts";scripts.mkdir();(scripts/"__init__.py").touch()
    shutil.copyfile(ROOT/WRAPPER,scripts/Path(WRAPPER).name)
    # Genuine worker CLI/start_native/registered-worker receipt write are retained.
    # Only repository guards and the native body are stubbed in this private tree.
    source=(ROOT/"scripts/run_dh116_work_packages.py").read_text()
    stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    output.mkdir(parents=True)
    (path/'native.log').write_text('CPU fixture; no native execution')
    (path/'outer_exit.json').write_text('{}')
    result.update(native_qualification=True,native_execution_performed=False,
                  worker_root=str(self.root),worker_private=str(self.private),
                  worker_pid=os.getpid(),worker_cli_entry=str(Path(sys.argv[0]).resolve()))
Repository._native_child=_cpu_native
Repository.verify_local_target=lambda self:None
Repository.head=lambda self:'b'*40
Repository.registry=lambda self:json.loads((self.private/'registry.json').read_text())
Repository.native_recipe=lambda self,package:json.loads((self.private/'recipe.json').read_text())
Repository.native_plan=lambda self,**kwargs:{'native_qualified':[]}
Repository.native_resource_blocker=lambda *a,**k:None
NATIVE_SCHEDULER_LOCK=Path(__file__).resolve().parents[1]/'private'/'reservation.lock'
"""
    marker="if __name__ == '__main__':"
    assert source.count(marker)==1
    (scripts/"run_dh116_work_packages.py").write_text(source.replace(marker,stub+"\n"+marker))
    (scripts/"verify_dh116_deformx_pinch_contact.py").write_text("""
import json
from pathlib import Path
def validate_pinch_artifacts(output,outer,expected):
    attempt=Path(outer).parent
    assert not (attempt/'receipt.json').exists(), 'qualification was already persisted'
    (attempt/'audit_called.json').write_text(json.dumps({'output':str(output),'sha':expected}))
    if (Path(__file__).parents[1]/'reject').exists():
        raise RuntimeError('CPU fixture rejected multi-pad identity')
    return {'grasp_qualified':False,'teacher_admission':False,'cpu_fixture':True}
""")
    runtime=tmp_path/spec()["source_files"][2];runtime.parent.mkdir(parents=True)
    runtime.write_text("# CPU fixture; no native execution\n")
    if reject:(tmp_path/"reject").touch()
    private=tmp_path/"private";private.mkdir()
    identity={"command_spec_sha256":"c"*64,"code_config_sha256":"d"*64}
    recipe={"spec":dict(spec(),budget={"timeout_seconds":10}),**identity}
    (private/"recipe.json").write_text(json.dumps(recipe))
    package={"executor":"native_bounded","evidence_scope":"cpu_fixture_only","dependencies":[],
             "dependency_scopes":{}}
    (private/"registry.json").write_text(json.dumps({"packages":{"pinch":package}}))
    attempt=private/"native-attempts"/"pinch"/identity["code_config_sha256"]/"attempt-cpu"
    attempt.mkdir(parents=True)
    output_root=tmp_path/"artifacts"/"outputs"
    approval={"version":1,"batch_id":"cpu-fixture","output_root":str(output_root),
              "packages":{"pinch":identity}}
    intent=attempt/"intent.json"
    output=output_root/"pinch"/attempt.name/"native"
    intent.write_text(json.dumps({"package":"pinch","head":"b"*40,"output":str(output),
                                 "approval":approval,**identity}))
    parent="""
import json,os
from pathlib import Path
from scripts import run_dh116_pinch_work_packages as w
repo=w.PinchRepository(Path.cwd(),evidence_dir=Path.cwd()/'private')
intent=next(repo.private.glob('native-attempts/*/*/*/intent.json'))
repo.register_native_attempt=lambda *a:intent
def parent_main():
    result=repo.start_native('pinch',{})
    print(json.dumps({'parent_pid':os.getpid(),'child':result}))
    return 0
w.scheduler.main=parent_main
raise SystemExit(w.main())
"""
    env={**os.environ,"PYTHONPATH":str(tmp_path),"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"}
    result=subprocess.run([sys.executable,"-c",parent],cwd=tmp_path,env=env,
                          check=True,capture_output=True,text=True,timeout=20)
    payload=json.loads(result.stdout);child=payload["child"]
    assert child is not None, (attempt/"worker.log").read_text()
    assert child["worker_pid"]!=payload["parent_pid"]
    assert child["worker_cli_entry"]==str(tmp_path/WRAPPER)
    assert child["worker_root"]==str(tmp_path) and child["worker_private"]==str(private)
    assert child["native_execution_performed"] is False
    assert (attempt/"audit_called.json").is_file()
    assert child["native_qualification"] is (not reject)
    persisted=json.loads((attempt/"receipt.json").read_text())
    assert persisted["native_qualification"] is (not reject)
    if reject:
        assert "CPU fixture rejected multi-pad identity" in persisted["reason"]
        assert "pinch_post_exit_audit_sha256" not in persisted
    else:
        assert child["pinch_post_exit_audit_sha256"]==w.scheduler.file_hash(attempt/w.AUDIT_NAME)


def test_qualified_receipt_reopens_audit_and_rejects_missing_or_changed_evidence(tmp_path,monkeypatch):
    w=wrapper()
    monkeypatch.setattr(w.scheduler.Repository,"_native_child",
                        lambda self,path,spec,output,result,**kwargs:result.update(native_qualification=True))
    monkeypatch.setattr(w,"validate_pinch_artifacts",lambda *a:{"grasp_qualified":False,"teacher_admission":False})
    repo=w.PinchRepository(ROOT,evidence_dir=tmp_path)
    result={"output":str(tmp_path/"output")}
    repo._native_child(tmp_path,spec(),tmp_path/"output",result)
    receipt=tmp_path/"receipt.json";receipt.write_text(json.dumps(result))
    monkeypatch.setattr(w.scheduler.Repository,"_verified_native_receipt",lambda *a:receipt)
    assert repo._verified_native_receipt("pinch","candidate",{"spec":spec()},"b"*64)==receipt
    monkeypatch.setattr(w,"validate_pinch_artifacts",lambda *a:{"grasp_qualified":False,"teacher_admission":True})
    with pytest.raises(ValueError,match="evidence changed"):
        repo._verified_native_receipt("pinch","candidate",{"spec":spec()},"b"*64)
    result.pop("pinch_post_exit_audit_sha256")
    receipt.write_text(json.dumps(result))
    with pytest.raises(ValueError,match="audit"):
        repo._verified_native_receipt("pinch","candidate",{"spec":spec()},"b"*64)


def test_audit_source_must_be_frozen_before_any_native_child(tmp_path,monkeypatch):
    w=wrapper()
    monkeypatch.setattr(w.scheduler.Repository,"_native_child",lambda *a,**k:pytest.fail("native called"))
    recipe=spec();recipe["source_files"].remove(WRAPPER)
    with pytest.raises(ValueError,match="freeze"):
        w.PinchRepository(ROOT)._native_child(tmp_path,recipe,tmp_path/"output",{})


def test_failed_native_child_never_attempts_audit(tmp_path,monkeypatch):
    w=wrapper()
    monkeypatch.setattr(w.scheduler.Repository,"_native_child",
                        lambda self,path,spec,output,result,**kwargs:result.update(native_qualification=False))
    monkeypatch.setattr(w,"validate_pinch_artifacts",lambda *a:pytest.fail("audit called after native failure"))
    result={}
    w.PinchRepository(ROOT)._native_child(tmp_path,spec(),tmp_path/"output",result)
    assert result["native_qualification"] is False


def test_frozen_recipe_binds_wrapper_and_current_candidate_identity():
    from scripts.verify_dh116_deformx_pinch_contact import preflight,default_arguments
    w=wrapper()
    spec_path="configs/dh116_execution/deformx_pinch_left_candidate_v1.json"
    recipe=json.loads((ROOT/spec_path).read_text())
    report,_=preflight(default_arguments(ROOT),ROOT)
    assert w.candidate_sha(recipe)==report["code_config_sha256"]
    assert "DH116_PINCH_WORKER_V1" not in recipe.get("environment",{})
    assert "tests/test_dh116_pinch_scheduler.py" in recipe["source_files"]
    bound=w.PinchRepository(ROOT).native_recipe({"command_spec":spec_path,"tests":[]})
    assert len(bound["command_spec_sha256"])==64


@pytest.mark.parametrize("dedicated",[False,True])
def test_real_native_cli_requires_dedicated_worker_before_sdk_import(tmp_path,monkeypatch,dedicated):
    w=wrapper()
    monkeypatch.delenv("DH116_PINCH_WORKER_V1",raising=False)
    monkeypatch.setenv("PYTHONPATH",os.pathsep.join((str(ROOT/"src"),str(ROOT),os.environ.get("PYTHONPATH",""))))
    scripts=tmp_path/"scripts";scripts.mkdir()
    shutil.copyfile(ROOT/WRAPPER,tmp_path/WRAPPER)
    native_source=(ROOT/w.ENTRY).read_text()
    # Keep the production main/approval/worker-marker gate. Replace only source
    # preflight and SDK launch/body with CPU stubs inside this private test tree.
    stub="""
import types,contextlib
def _cpu_preflight(args,project):
    return {'code_config_sha256':'a'*64},{}
preflight=_cpu_preflight
class _CpuAppLauncher:
    def __init__(self,**kwargs): self.app=object()
sys.modules['isaaclab']=types.ModuleType('isaaclab')
app_module=types.ModuleType('isaaclab.app')
app_module.AppLauncher=_CpuAppLauncher
sys.modules['isaaclab.app']=app_module
from go2_pvcnn.control.dh116 import tabletop_stereo_native as _native
_native.native_instance_lock=lambda:contextlib.nullcontext()
def _cpu_run(app,args,contract,cfg,ownership):
    (args.output/'sdk-boundary.json').write_text(json.dumps({'cpu_fixture':True}))
    (args.output/'result.json').write_text(json.dumps({'cpu_fixture':True}))
    (args.output/'physics.jsonl').write_text('{}\\n')
run_app=_cpu_run
"""
    marker='if __name__=="__main__": main()'
    assert native_source.count(marker)==1
    (tmp_path/w.ENTRY).write_text(native_source.replace(marker,stub+"\n"+marker))
    (scripts/"python.sh").write_text('exec "$DH116_PYTHON" "$@"\n')
    runtime=tmp_path/w.RUNTIME;runtime.parent.mkdir(parents=True)
    shutil.copyfile(ROOT/w.RUNTIME,runtime)
    attempt=tmp_path/"attempt";attempt.mkdir()
    output=tmp_path/"private"/"native"
    recipe=spec()
    recipe["argv"]+=["--output","{output}","--headless","--execute-native"]
    recipe["budget"]={"gpu_index":3,"timeout_seconds":10,"physics_steps":1}
    recipe["validator"]={"result_file":"result.json","equals":{"cpu_fixture":True,"code_config_sha256":"a"*64},
                         "artifacts":["result.json","physics.jsonl"],"line_counts":{"physics.jsonl":1}}
    # The generic result lacks candidate semantics deliberately: this checks
    # transport to the SDK boundary, never claims a physical result.
    native_source=(tmp_path/w.ENTRY).read_text().replace("json.dumps({'cpu_fixture':True}))",
                      "json.dumps({'cpu_fixture':True,'code_config_sha256':'a'*64}))")
    (tmp_path/w.ENTRY).write_text(native_source)
    monkeypatch.setattr(w.scheduler.Repository,"native_lock_observed",lambda *a:True)
    monkeypatch.setattr(w,"validate_pinch_artifacts",lambda *a:{"cpu_fixture":True,"teacher_admission":False})
    repo=(w.PinchRepository if dedicated else w.scheduler.Repository)(tmp_path)
    result={}
    repo._native_child(attempt,recipe,output,result)
    assert "DH116_PINCH_WORKER_V1" not in os.environ
    if dedicated:
        assert result["outer_exit_code"]==0, (attempt/"native.log").read_text()
        assert json.loads((output/"sdk-boundary.json").read_text())["cpu_fixture"] is True
    else:
        assert result["outer_exit_code"]!=0
        assert not (output/"sdk-boundary.json").exists()
        assert "dedicated Pinch worker" in (attempt/"native.log").read_text()


def test_worker_marker_is_scoped_even_when_base_child_raises(tmp_path,monkeypatch):
    w=wrapper()
    monkeypatch.setenv(w.WORKER_ENV,"original-parent-value")
    def fail(self,path,spec,output,result,**kwargs):
        marker=json.loads(os.environ[w.WORKER_ENV])
        assert marker["wrapper_sha256"]==w.scheduler.file_hash(ROOT/WRAPPER)
        assert marker["output"]==str(output.resolve())
        raise RuntimeError("base child failed")
    monkeypatch.setattr(w.scheduler.Repository,"_native_child",fail)
    with pytest.raises(RuntimeError,match="base child failed"):
        w.PinchRepository(ROOT)._native_child(tmp_path,spec(),tmp_path/"out",{})
    assert os.environ[w.WORKER_ENV]=="original-parent-value"
