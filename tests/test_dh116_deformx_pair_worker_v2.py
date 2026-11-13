"""Real registered-shape CPU subprocess gates; calibration is always stubbed."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from scripts import run_dh116_work_packages as scheduler

ROOT=Path(__file__).resolve().parents[1]
ENTRY="scripts/verify_dh116_deformx_pair_capture_v2.py"
WRAPPER="scripts/run_dh116_deformx_pair_work_packages_v2.py"
OLD_ENTRY="scripts/verify_dh116_deformx_pair_capture.py"
OLD_WRAPPER="scripts/run_dh116_deformx_pair_work_packages.py"
ENV="DH116_DEFORMX_PAIR_WORKER_V2"

def module():
    assert (ROOT/ENTRY).exists(),"corrected independently versioned pair entry missing"
    return importlib.import_module("scripts.verify_dh116_deformx_pair_capture_v2")

def fixture_repo(tmp_path, *, legacy=False, worker=False, bad_error=False):
    module()
    root=tmp_path/"repo";root.mkdir()
    scripts=root/"scripts";scripts.mkdir();(scripts/"__init__.py").touch()
    for name in (ENTRY,WRAPPER,OLD_ENTRY,OLD_WRAPPER,"scripts/run_dh116_work_packages.py","scripts/python.sh"):
        shutil.copyfile(ROOT/name,root/name)
    if worker:
        source=(scripts/"run_dh116_work_packages.py").read_text()
        stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    from types import SimpleNamespace
    from scripts import verify_dh116_deformx_pair_capture_v2 as entry
    import contextlib
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    entry.original.native_instance_lock=lambda:contextlib.nullcontext()
    def construct(extension,sha):
        (output/'cpu_construct.json').write_text(json.dumps({'cpu_fixture':True,'gpu_constructed':False}))
        raw={'substeps':12,'mesh_count':1,'capacity':36,'attempted':1,'overflow':False,
             'indices':[[0,0,0]],'vectors':[[0,0,1,0,0,1,0,0,12,0,0,0]],
             'node_forces_on_rod_world_n':[[0,0,1],[0,0,0],[0,0,0]],
             'mesh_wrenches_on_mesh_world':[[0,0,-1,0,0,0]]}
        report=entry.original.validate_pair_capture(raw,dt_s=1/480,
            node_forces=raw['node_forces_on_rod_world_n'],mesh_wrenches=raw['mesh_wrenches_on_mesh_world'])
        report.update(solver_clock_before_s=0.,solver_clock_after_s=1/480,stale_reset_rejected=True,solver_last_error=0)
        return report,raw
    entry.run_calibration=construct
    sys.argv=[str(self.root/spec['entrypoint']),*[str(output) if a=='{output}' else a for a in spec['argv']]]
    entry.main()
    result.update(native_qualification=True,native_execution_performed=False,cpu_fixture=True,
        worker_root=str(self.root),worker_private=str(self.private),worker_pid=os.getpid(),
        worker_cli_entry=str(Path(__file__).parent/'run_dh116_deformx_pair_work_packages_v2.py'))
Repository._native_child=_cpu_native
Repository.verify_local_target=lambda self:None
Repository.native_plan=lambda self,**kwargs:{'native_ready':['pair-v2'],'native_qualified':[]}
Repository.native_resource_blocker=lambda *a,**k:None
NATIVE_SCHEDULER_LOCK=Path(__file__).resolve().parents[1]/'cpu-reservation.lock'
"""
        if bad_error:
            stub=stub.replace("stale_reset_rejected=True,solver_last_error=0)",
                              "stale_reset_rejected=True,solver_last_error=7)")
        marker="if __name__ == '__main__':";assert source.count(marker)==1
        (scripts/"run_dh116_work_packages.py").write_text(source.replace(marker,stub+"\n"+marker))
    extension=tmp_path/"build"/"ropesim"/"_ropesim.cpython-310-cpu-fixture.so"
    extension.parent.mkdir(parents=True);extension.write_bytes(b"CPU fake extension; never imported")
    selected=OLD_ENTRY if legacy else ENTRY
    argv=["--output","{output}","--extension",str(extension),"--extension-sha256",
          scheduler.file_hash(extension),"--execute-native"]
    spec=dict(version=1,entrypoint=selected,argv=argv,
        source_files=[ENTRY,WRAPPER,OLD_ENTRY,OLD_WRAPPER],
        budget=dict(environments=1,gpu_index=3,min_disk_free_bytes=1073741824,
            min_gpu_free_mib=4096,physics_steps=0,timeout_seconds=90,updates=0),
        inputs={str(extension):scheduler.file_hash(extension)},
        validator=dict(result_file="result.json",equals={"reconciled_pairs":True},
                       artifacts=["result.json","pairs.json","raw_pairs.json","plan.json"],line_counts={}))
    recipe_path="configs/dh116_execution/cpu_pair_v2.json"
    dest=root/recipe_path;dest.parent.mkdir(parents=True);dest.write_text(json.dumps(spec))
    package=dict(title="CPU fixture only",status="pending",executor="native_bounded",dependencies=[],
        dependency_scopes={},requires_approval=True,resource_class=scheduler.NATIVE_SCOPE,
        evidence_scope="cpu_fixture_only",retry_policy=scheduler.NATIVE_RETRY,
        command_spec=recipe_path,tests=[])
    registry=root/scheduler.REGISTRY;registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"version":2,"packages":{"pair-v2":package}}))
    git=["git","-c",f"safe.directory={root}","-c","user.name=CPUFixture",
         "-c","user.email=cpu@example.invalid","-C",str(root)]
    for args in (["init","-q"],["add","."],["commit","-qm","CPU fixture only"]):
        subprocess.run(git+args,check=True,capture_output=True)
    private=tmp_path/"evidence"
    repo=scheduler.Repository(root,evidence_dir=private)
    recipe=repo.native_recipe(package)
    identity={k:recipe[k] for k in ("command_spec_sha256","code_config_sha256")}
    output_root=tmp_path/"outputs"
    attempt=private/"native-attempts"/"pair-v2"/identity["code_config_sha256"]/"attempt-cpu"
    attempt.mkdir(parents=True)
    output=output_root/"pair-v2"/attempt.name/"native"
    approval=dict(version=1,batch_id="cpu-only",output_root=str(output_root),
                  packages={"pair-v2":identity})
    intent=dict(package="pair-v2",head=repo.head(),output=str(output),approval=approval,**identity)
    path=attempt/"intent.json";path.write_text(json.dumps(intent))
    return root,private,path,output,extension,spec,approval

def run_gate(tmp_path, *, legacy=False, damage=None):
    root,private,intent_path,output,extension,spec,approval=fixture_repo(tmp_path,legacy=legacy)
    intent=json.loads(intent_path.read_text())
    if damage=="head":intent["head"]="a"*40
    if damage=="code":intent["code_config_sha256"]="f"*64
    if damage=="approval":intent["approval"]["packages"]["pair-v2"]["command_spec_sha256"]="e"*64
    if damage=="output":intent["output"]=str(tmp_path/"wrong")
    if damage=="output_root":
        output=tmp_path/"tampered-output"
        intent["output"]=str(output)
    if damage=="hierarchy":
        intent_path=tmp_path/"evidence"/"wrong-attempt-root"/"pair-v2"/intent["code_config_sha256"]/"attempt-cpu"/"intent.json"
        intent_path.parent.mkdir(parents=True)
    intent_path.write_text(json.dumps(intent))
    if damage=="entry":
        p=root/"configs/dh116_execution/cpu_pair_v2.json"
        value=json.loads(p.read_text());value["entrypoint"]=OLD_ENTRY;p.write_text(json.dumps(value))
    module_name="verify_dh116_deformx_pair_capture" if legacy else "verify_dh116_deformx_pair_capture_v2"
    runner="""
import contextlib,json,sys
from pathlib import Path
from scripts import MODULE_NAME as entry
original=entry if MODULE_NAME_TEXT.endswith('_capture') else entry.original
original.native_instance_lock=lambda:contextlib.nullcontext()
def construct(extension,sha):
    (Path(sys.argv[sys.argv.index('--output')+1])/'cpu_construct.json').write_text(json.dumps(
        {'cpu_fixture':True,'gpu_constructed':False}))
    return {'pairs':[],'stale_reset_rejected':True},{}
entry.run_calibration=construct
entry.main()
""".replace("MODULE_NAME_TEXT",repr(module_name)).replace("MODULE_NAME",module_name)
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(root),str(ROOT/"src"))),
         "PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":"2" if damage=="gpu" else "3"}
    for key in ("DH116_DEFORMX_PAIR_WORKER_V1",ENV):
        env.pop(key,None)
    if damage!="missing":
        env["DH116_DEFORMX_PAIR_WORKER_V1" if legacy else ENV]=str(intent_path)
    args=[sys.executable,"-c",runner,*[str(output) if a=="{output}" else a for a in spec["argv"]]]
    done=subprocess.run(args,cwd=root,env=env,text=True,capture_output=True,timeout=20)
    return done,output,private,intent_path

def test_actual_shaped_v1_root_bug_is_reproduced_without_cuda(tmp_path):
    done,output,private,path=run_gate(tmp_path,legacy=True)
    assert path.parents[3]==private/"native-attempts"
    assert path.parents[4]==private
    assert done.returncode!=0 and "registered approval" in done.stderr
    assert not output.exists()

@pytest.mark.parametrize("damage",[None,"head","code","approval","output","output_root","hierarchy","entry","gpu","missing"])
def test_real_cpu_native_child_gate_reaches_stub_only_for_exact_authorized_identity(tmp_path,damage):
    done,output,private,path=run_gate(tmp_path,damage=damage)
    if damage is None:
        assert done.returncode==0,done.stdout+done.stderr
        assert json.loads((output/"cpu_construct.json").read_text())=={
            "cpu_fixture":True,"gpu_constructed":False}
    else:
        assert done.returncode!=0 and not output.exists()

@pytest.mark.parametrize("bad_error",[False,True])
def test_real_scheduler_child_reentry_preserves_evidence_root_and_actual_pair_gate(tmp_path,bad_error):
    root,private,path,output,extension,spec,approval=fixture_repo(tmp_path,worker=True,bad_error=bad_error)
    parent="""
import json,os
from pathlib import Path
from scripts import run_dh116_deformx_pair_work_packages_v2 as wrapper
root=Path.cwd()
def parent_main():
    repo=wrapper.original.PairRepository(root,evidence_dir=Path(EVIDENCE_ROOT))
    intent=Path(INTENT_PATH)
    repo.register_native_attempt=lambda *a:intent
    value=repo.start_native('pair-v2',json.loads(APPROVAL))
    print(json.dumps({'parent_pid':os.getpid(),'child':value}))
    return 0
wrapper.original.scheduler.main=parent_main
raise SystemExit(wrapper.main())
""".replace("EVIDENCE_ROOT",repr(str(private))).replace("INTENT_PATH",repr(str(path))).replace("APPROVAL",repr(json.dumps(approval)))
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(root),str(ROOT/"src"))),
        "PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":""}
    done=subprocess.run([sys.executable,"-c",parent],cwd=root,env=env,text=True,capture_output=True,timeout=25)
    assert done.returncode==0,done.stdout+done.stderr
    payload=json.loads(done.stdout);child=payload["child"]
    assert child is not None,(path.parent/"worker.log").read_text()
    assert child["worker_pid"]!=payload["parent_pid"]
    assert child["worker_root"]==str(root) and child["worker_private"]==str(private)
    assert child["worker_cli_entry"]==str(root/WRAPPER)
    assert child["native_execution_performed"] is False
    assert child["native_qualification"] is (not bad_error)
    if bad_error:
        assert child["blocker_kind"]=="failed_native" and "zero solver last_error" in child["reason"]
    assert (output/"cpu_construct.json").is_file()


@pytest.mark.parametrize("error",[0,7])
def test_v2_calibration_uses_current_all_free_api_and_checks_error_before_packet(tmp_path,monkeypatch,error):
    from types import SimpleNamespace
    import numpy as np
    entry=module()
    extension=tmp_path/"ropesim"/"_ropesim.cpython-310-cpu-fixture.so"
    extension.parent.mkdir();extension.write_bytes(b"CPU stub, never loaded")
    events=[]
    class Rod:
        time=0.
        reset=False
        @classmethod
        def from_positions(cls,nodes,**kwargs):
            assert nodes.shape==(1,3,3) and nodes.dtype==np.float32
            np.testing.assert_array_equal(kwargs["pinned"],[False,False,False])
            assert kwargs["fix_root_orientation"] is False
            assert kwargs["max_h"]==1/480/12 and kwargs["num_itr"]==8
            assert kwargs["density"]==.065 and kwargs["k_stretch"]==800. and kwargs["k_bend"]==.018
            events.append("construct")
            return cls()
        def set_gravity(self,value):np.testing.assert_allclose(value,[0,0,-9.81])
        def set_contact(self,**kwargs):pass
        def add_mesh(self,*args,**kwargs):return 0
        def enable_contact_forces(self,on):assert on is True
        def enable_contact_pair_capture(self,capacity):assert capacity==36
        def step_free(self,dt,substeps):
            assert dt==1/480 and substeps==12
            events.append("step");self.time=dt
        @property
        def last_error(self):
            events.append("last_error")
            return error
        def contact_pair_samples(self):
            events.append("packet")
            if self.reset:raise ValueError("stale")
            return {"substeps":12,"mesh_count":1,"capacity":36,"attempted":1,"overflow":False,
                "indices":[[0,0,0]],"vectors":[[0,0,1,0,0,1,0,0,12,0,0,0]],
                "node_forces_on_rod_world_n":[[0,0,1],[0,0,0],[0,0,0]],
                "mesh_wrenches_on_mesh_world":[[0,0,-1,0,0,0]]}
        def reset_envs(self,ids):
            assert ids==[0]
            self.reset=True
    fake=SimpleNamespace(RopeBatch=Rod,_ropesim=SimpleNamespace(__file__=str(extension)),
        sdf=SimpleNamespace(from_function=lambda *args:SimpleNamespace(as_args=lambda:())))
    monkeypatch.setitem(sys.modules,"ropesim",fake)
    monkeypatch.setattr(sys,"path",list(sys.path))
    if error:
        with pytest.raises(ValueError,match="solver last_error"):
            entry.run_calibration(extension,scheduler.file_hash(extension))
        assert events==["construct","step","last_error"]
    else:
        report,raw=entry.run_calibration(extension,scheduler.file_hash(extension))
        assert report["solver_last_error"]==0
        assert events==["construct","step","last_error","packet","packet"]

def test_v2_calibration_has_no_unrelated_physics_changes():
    import ast
    old=ast.parse((ROOT/OLD_ENTRY).read_text())
    new=ast.parse((ROOT/ENTRY).read_text())
    before=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=="run_calibration")
    after=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=="run_calibration")
    last_error=next(i for i,x in enumerate(after.body)
        if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="solver_last_error" for t in x.targets))
    assert ast.unparse(after.body[last_error-1])=="rod.step_free(DT, 12)"
    assert ast.unparse(after.body[last_error+1]).startswith("if solver_last_error != 0:")
    del after.body[last_error:last_error+2]
    final=after.body[-1].value.elts[0]
    pos=next(i for i,key in enumerate(final.keys) if isinstance(key,ast.Constant) and key.value=="solver_last_error")
    del final.keys[pos];del final.values[pos]
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)

@pytest.mark.parametrize("kind",["entry","wrapper"])
def test_scoped_module_bindings_restore_on_exception(monkeypatch,kind):
    selected=module() if kind=="entry" else importlib.import_module("scripts.run_dh116_deformx_pair_work_packages_v2")
    names=("WORKER_ENV","verify_worker","run_calibration") if kind=="entry" else (
        "__file__","ENTRY","WRAPPER","WORKER_ENV","validate_artifacts")
    saved={name:getattr(selected.original,name) for name in names}
    def fail():raise RuntimeError("CPU failure")
    monkeypatch.setattr(selected.original,"main",fail)
    with pytest.raises(RuntimeError,match="CPU failure"):
        selected.main()
    assert {name:getattr(selected.original,name) for name in names}==saved

@pytest.mark.parametrize("value",[None,True,1,0.0,0])
def test_v2_artifact_validator_requires_actual_integer_zero(tmp_path,monkeypatch,value):
    entry=module()
    monkeypatch.setattr(entry.original,"validate_artifacts",lambda output:None)
    (tmp_path/"pairs.json").write_text(json.dumps({"solver_last_error":value}))
    if type(value) is int and value==0:
        entry.validate_artifacts(tmp_path)
    else:
        with pytest.raises(ValueError,match="zero solver last_error"):
            entry.validate_artifacts(tmp_path)

def test_new_recipe_preserves_consumed_calibration_budget_and_pins_both_layers():
    path=ROOT/"configs/dh116_execution/deformx_mesh_pair_capture_calibration_v2.json"
    assert path.exists(),"new versioned recipe is required"
    old=json.loads((ROOT/"configs/dh116_execution/deformx_mesh_pair_capture_calibration_v1.json").read_text())
    new=json.loads(path.read_text())
    for field in ("argv","budget","environment","inputs","validator"):
        assert new[field]==old[field]
    assert new["entrypoint"]==ENTRY
    assert set(old["source_files"]).issubset(new["source_files"])
    assert {OLD_ENTRY,ENTRY,WRAPPER,OLD_WRAPPER,"tests/test_dh116_deformx_pair_worker_v2.py",
            "tests/test_dh116_deformx_pair_capture.py"}.issubset(new["source_files"])
    assert ENV not in new["environment"]
