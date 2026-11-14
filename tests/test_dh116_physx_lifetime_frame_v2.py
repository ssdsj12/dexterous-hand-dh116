"""CPU retained-builder-frame reproduction; all synthetic owners are fake."""
import ast
from copy import deepcopy
import gc
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import weakref

import pytest

ROOT=Path(__file__).resolve().parents[1]
ACQ="scripts/dh116_physx_lifetime_frame_acquisition_v2.py"
ENTRY="scripts/probe_dh116_physx_lifetime_frame_v2.py"
WRAPPER="scripts/run_dh116_physx_lifetime_frame_work_packages_v2.py"

def acquisition():
    assert (ROOT/ACQ).exists(),"new explicit builder ownership scope missing"
    return importlib.import_module("scripts.dh116_physx_lifetime_frame_acquisition_v2")

def fixture_builder(detached,frames,refs):
    module=acquisition()
    tree=ast.parse((ROOT/ACQ).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="build_native_detached")
    # Exercise the EXACT production finally suite with SDK-shaped fake locals;
    # tests compile this fixture only, never production native logic dynamically.
    body=ast.parse("""
hand,view,composer,runtime,provider,asset_sim_view=Resource(),Resource(),Resource(),Resource(),Resource(),Resource()
manager_sim_view=asset_sim_view
hand.data=SimpleNamespace(joint_pos=np.zeros((1,6)),joint_vel=np.zeros((1,6)),body_link_pos_w=np.zeros((1,5,3)))
full,materials,grids=Resource(),[Resource()],{"pad":Resource()}
runtime.rope=Resource()
provider.runtime=runtime
hand.composer=composer
composer.asset=hand
hand.view=view
view.getter=lambda:hand
snapshot=lambda:hand
mesh_poses=lambda:view
refs.update(hand=weakref.ref(hand),view=weakref.ref(view),composer=weakref.ref(composer),
    runtime=weakref.ref(runtime),provider=weakref.ref(provider),asset_sim_view=weakref.ref(asset_sim_view),
    manager_sim_view=weakref.ref(manager_sim_view),full=weakref.ref(full))
refs["rope"]=weakref.ref(runtime.rope)
refs["material"]=weakref.ref(materials[0])
refs["grid"]=weakref.ref(grids["pad"])
frames.append(sys._getframe())
if args.fail:
    raise RuntimeError("CPU builder failed after acquisition")
return SimpleNamespace(hand=hand,hand_view=view,composer=composer,runtime=runtime,
    provider=provider,rope=runtime.rope,asset_sim_view=asset_sim_view,
    manager_sim_view=manager_sim_view,metadata={})
""").body
    guarded=next(n for n in function.body if isinstance(n,ast.Try))
    if detached:
        guarded.body=body
    else:
        function.body=body
    namespace=dict(module.__dict__,Resource=type("Resource",(),{"reset":lambda self:None}),frames=frames,refs=refs,
                   SimpleNamespace=SimpleNamespace,sys=sys,weakref=weakref,np=__import__("numpy"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 "<CPU-retained-build-native-frame>","exec"),namespace)
    return namespace["build_native_detached"]

@pytest.mark.parametrize("detached",[False,True])
@pytest.mark.parametrize("failure",[False,True])
def test_retained_actual_shaped_builder_frame_clears_native_locals_before_close(tmp_path,detached,failure):
    frames,refs=[],{};build=fixture_builder(detached,frames,refs)
    args=SimpleNamespace(output=tmp_path,fail=failure)
    result=None
    try:
        result=build(object(),args,{}, {},ROOT)
    except RuntimeError as error:
        assert failure and "CPU builder failed" in str(error)
    # The caller consumes then drops the ordinary returned bundle as v1 explicit
    # does. A deliberately retained completed frame remains alive in both cases.
    result=None
    gc.collect()
    assert len(frames)==1
    assert all(ref() is None for ref in refs.values()) is detached
    if detached:
        audit=json.loads((tmp_path/"build_frame_detached.json").read_text())
        assert audit["clearing_completed"] is True and audit["operation_raised"] is failure
        assert all(frames[0].f_locals.get(name) is None for name in acquisition().NATIVE_LOCALS)
    frames.clear();gc.collect()
    assert all(ref() is None for ref in refs.values())

def test_new_native_builder_try_body_is_semantically_identical_to_frozen_original():
    m=acquisition()
    original=ast.parse((ROOT/"scripts/dh116_physx_lifetime_acquisition.py").read_text())
    old=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="build_native")
    new=next(n for n in ast.parse((ROOT/ACQ).read_text()).body if isinstance(n,ast.FunctionDef)
             and n.name=="build_native_detached")
    guarded=next(n for n in new.body if isinstance(n,ast.Try))
    assert ast.dump(ast.Module(body=guarded.body,type_ignores=[]))==ast.dump(ast.Module(body=old.body,type_ignores=[]))
    assert ast.dump(new.args)==ast.dump(old.args)
    cleared={target.id for statement in guarded.finalbody if isinstance(statement,ast.Assign)
             for target in statement.targets if isinstance(target,ast.Name)}
    assert cleared==set(m.NATIVE_LOCALS)
    methods={n.func.attr for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert "collect" not in methods and not {"setprofile","settrace"} & methods
    assert not any(isinstance(n,ast.Attribute) and n.attr=="f_locals" for n in ast.walk(new))

def test_original_builder_file_remains_exactly_pinned():
    import hashlib
    m=acquisition()
    assert hashlib.sha256((ROOT/"scripts/dh116_physx_lifetime_acquisition.py").read_bytes()).hexdigest()==m.ORIGINAL_SOURCE_SHA256


def entry():
    assert (ROOT/ENTRY).exists(),"new gated v2 entry missing"
    return importlib.import_module("scripts.probe_dh116_physx_lifetime_frame_v2")

def test_v2_entry_restores_frozen_module_bindings_even_on_error(monkeypatch):
    m=entry()
    before=(m.probe.VERSION,m.probe.CASES,m.probe.require_worker,m.original_acquisition.build_native)
    def fail():
        assert m.probe.VERSION==m.VERSION and m.probe.CASES==("explicit",)
        assert m.original_acquisition.build_native is acquisition().build_native_detached
        raise RuntimeError("CPU entry boundary")
    monkeypatch.setattr(m.probe,"main",fail)
    with pytest.raises(RuntimeError,match="CPU entry"):m.main()
    assert (m.probe.VERSION,m.probe.CASES,m.probe.require_worker,m.original_acquisition.build_native)==before

def test_native_v2_cli_requires_its_own_worker_before_mkdir_or_sdk(tmp_path):
    import os,subprocess
    m=entry();output=tmp_path/"native"
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"}
    env.pop(m.WORKER_ENV,None)
    result=subprocess.run([sys.executable,"-m","scripts.probe_dh116_physx_lifetime_frame_v2",
        "--cleanup","explicit","--headless","--output",str(output),"--execute-native"],
        cwd=ROOT,env=env,text=True,capture_output=True,timeout=15)
    assert result.returncode!=0 and "dedicated" in result.stderr and not output.exists()

@pytest.mark.parametrize("damage",[None,"missing","raised","local","stopped_threads"])
def test_v2_audit_requires_explicit_frame_clear_and_new_stopped_capture(tmp_path,damage):
    import importlib.util
    m=entry()
    spec=importlib.util.spec_from_file_location("cpu_evidence",ROOT/"tests/test_dh116_physx_lifetime_validation.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    output=fixture.evidence(tmp_path)
    result=json.loads((output/"result.json").read_text());result["version"]=m.VERSION
    result["case"]="explicit";result["acquisition"]["case"]="explicit"
    (output/"result.json").write_text(json.dumps(result))
    for name in ("before_app_close.json","app_close_returned.json"):
        value=json.loads((output/name).read_text());value["case"]="explicit"
        (output/name).write_text(json.dumps(value))
    capture=json.loads((output/"gdb_finalize.json").read_text());capture["stopped_thread_count"]=1
    (output/"gdb_finalize.json").write_text(json.dumps(capture))
    acquisition().record_detachment(output,False)
    if damage=="missing":(output/"build_frame_detached.json").unlink()
    if damage in ("raised","local"):
        packet=json.loads((output/"build_frame_detached.json").read_text())
        if damage=="raised":packet["operation_raised"]=True
        if damage=="local":packet["cleared_locals"].remove("provider")
        (output/"build_frame_detached.json").write_text(json.dumps(packet))
    if damage=="stopped_threads":
        capture["stopped_thread_count"]=0
        (output/"gdb_finalize.json").write_text(json.dumps(capture))
    audit=m.audit_outcome(output,tmp_path/"outer_exit.json","explicit")
    assert audit["lifecycle_qualified"] is (damage is None)
    assert audit["teacher_admission"] is audit["grasp_qualified"] is False


def wrapper():
    assert (ROOT/WRAPPER).exists(),"frame-v2 worker missing"
    return importlib.import_module("scripts.run_dh116_physx_lifetime_frame_work_packages_v2")

def test_v2_wrapper_restores_v1_transport_and_selects_stopped_debugger(monkeypatch):
    w=wrapper();before=(w.base.__file__,w.base.ENTRY,w.base.VERSION,w.base.render_gdb_commands,w.base.audit_outcome)
    def fail():
        assert w.base.__file__==str(ROOT/WRAPPER)
        assert w.base.ENTRY==ENTRY and w.base.VERSION==entry().VERSION
        assert "set non-stop off" in w.base.render_gdb_commands(Path("/tmp/cpu-fixture"))
        raise RuntimeError("CPU worker wrapper failure")
    monkeypatch.setattr(w.base,"main",fail)
    with pytest.raises(RuntimeError,match="CPU worker"):w.main()
    assert (w.base.__file__,w.base.ENTRY,w.base.VERSION,w.base.render_gdb_commands,w.base.audit_outcome)==before

@pytest.mark.parametrize("reject",[False,True])
def test_real_cpu_worker_reentry_keeps_v2_marker_stopped_capture_and_postexit_gate(tmp_path,reject):
    import os,shutil,subprocess
    w=wrapper()
    scripts=tmp_path/"scripts";scripts.mkdir();(scripts/"__init__.py").touch()
    for name in set(w.AUDIT_SOURCES)|{WRAPPER,ENTRY,ACQ}:
        shutil.copyfile(ROOT/name,tmp_path/name)
    (scripts/"validate_dh116_physx_lifetime.py").write_text("""
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
    from scripts import probe_dh116_physx_lifetime_frame_v2 as entry
    from scripts.dh116_physx_lifetime_frame_acquisition_v2 import record_detachment
    output.mkdir(parents=True)
    (path/'launch_argv.json').write_text(json.dumps({'argv':['--output',str(output)]}))
    native_debugger_wrapper(path,result)
    capture=(path/'debugger.gdb').read_text()
    assert 'set non-stop off' in capture and 'stopped_thread_count' in capture and str(output) in capture
    os.environ['DH116_PHYSX_LIFETIME_GDB']='1' # CPU substitute for actual debugger environment.
    entry.require_worker(SimpleNamespace(cleanup='explicit',output=output),self.root)
    record_detachment(output,False)
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
    spec=dict(entrypoint=ENTRY,argv=["--cleanup","explicit","--output","{output}"],
        source_files=list(w.AUDIT_SOURCES),debugger="gdb_on_segv",budget={"timeout_seconds":10},
        validator={"equals":{"case":"explicit","version":entry().VERSION}})
    (private/"recipe.json").write_text(json.dumps(dict(spec=spec,**identity)))
    package=dict(executor="native_bounded",evidence_scope="cpu_fixture_only",dependencies=[],dependency_scopes={})
    (private/"registry.json").write_text(json.dumps({"packages":{"frame":package}}))
    parent="""
import json
from pathlib import Path
from scripts import run_dh116_physx_lifetime_frame_work_packages_v2 as w
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

def test_single_recipe_preserves_budget_and_pins_new_scope_and_stopped_gdb():
    w=wrapper();relative="configs/dh116_execution/physx_lifetime_frame_explicit_v2.json"
    assert (ROOT/relative).exists(),"new independently approved recipe missing"
    recipe=w.base.scheduler.Repository(ROOT).native_recipe({"command_spec":relative,"tests":[]})
    spec=recipe["spec"]
    with w.configured():
        assert w.base.cleanup_case(spec)=="explicit"
    assert spec["budget"]["physics_steps"]==2 and spec["budget"]["updates"]==0
    assert spec["budget"]["environments"]==1 and spec["budget"]["gpu_index"]==3
    assert spec["budget"]["timeout_seconds"]==180
    assert "build_frame_detached.json" in spec["validator"]["artifacts"]
    assert entry().WORKER_ENV not in spec.get("environment",{})


@pytest.mark.parametrize("failure",[False,True])
def test_actual_lifecycle_reaches_close_with_detached_retained_builder_frame(tmp_path,failure):
    m=entry();frames,refs=[],{}
    build=fixture_builder(True,frames,refs)
    args=SimpleNamespace(output=tmp_path,fail=failure)
    class Sim:
        current_time_step_index=0
        current_time=0.
        def reset(self):
            self.current_time_step_index=2
            self.current_time=2/480
    sim=Sim();events=[]
    def acquire(sim,owners,trace,cleanup):
        m.probe.record_reset(sim,tmp_path)
        return m.probe.acquire_graph(sim,owners,trace,cleanup,
            lambda value:build(value,args,{}, {},ROOT))
    def close():
        assert len(frames)==1 and all(ref() is None for ref in refs.values())
        assert json.loads((tmp_path/"build_frame_detached.json").read_text())["clearing_completed"] is True
        events.append("app.close")
    app=SimpleNamespace(update=lambda:events.append("app.update"),close=close)
    kwargs=dict(make_sim=lambda:sim,acquire=acquire,synchronize=lambda:events.append("sync"),
        clear_sim=lambda _:events.append("clear"),stop=lambda:events.append("STOP"),
        manager_is_none=lambda:True,freeze=lambda:False,output=tmp_path,cleanup="explicit")
    if failure:
        with pytest.raises(RuntimeError,match="CPU builder failed"):m.probe.run_lifecycle(app,**kwargs)
    else:
        result=m.probe.run_lifecycle(app,**kwargs)
        assert result["lifetime_released_before_close"] is True
    assert events.index("sync")<events.index("STOP")<events.index("app.close")
    assert events.count("app.update")==events.count("app.close")==1
