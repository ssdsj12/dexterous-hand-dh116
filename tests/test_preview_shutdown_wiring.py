"""Run actual acquire function on CPU collaborators; never a native trial."""
import ast
from pathlib import Path
from types import SimpleNamespace
import json
import uuid
import pytest
from test_preview_retained_frames import hold_completed

ROOT=Path(__file__).resolve().parents[1]


def acquire_function(output,events,*,fail=False):
    source=ROOT/'scripts/preview_dh116_physical_webrtc.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='acquire')
    class Render:
        count=7
        def __init__(self,*a):pass
        def install(self):events.append('render-install')
        def close(self):events.append('render-close')
    class Journal:
        def __init__(self,*a,**kw):pass
        def close(self):events.append('journal-close')
    def run(*a,**kw):
        events.append('scene')
        if fail:raise RuntimeError('scene failure')
        return dict(control_intervals=0,physics_rows=0,ppo_updates=0)
    namespace=dict(RenderBudget=Render,DualHandEvidenceJournal=Journal,output=output,
        BUDGET=dict(max_render_calls=162,total_sdk_steps=130),run_owned_scene=run,
        builder=SimpleNamespace(build_reset_owned=object()),NativeParameterEndpoints=object(),
        backend=object(),ROOT=ROOT,diagnostic_reset_owners=lambda:[],uuid=uuid,
        app=SimpleNamespace(stop_presentation=lambda:events.append('stream-stop')))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['acquire']


def test_actual_acquire_stops_stream_and_persists_count_before_native_close(tmp_path):
    events=[]
    acquire=acquire_function(tmp_path,events)
    result=acquire(object(),object(),object(),'explicit')
    assert events==['render-install','scene','stream-stop','render-close','journal-close']
    assert result['render_calls']==7
    saved=json.loads((tmp_path/'preview_acquire_result.json').read_text())
    assert saved['render_calls']==7
    assert saved['native_qualification'] is False


def test_actual_acquire_exception_still_quiesces_and_keeps_failure(tmp_path):
    events=[]
    acquire=acquire_function(tmp_path,events,fail=True)
    with pytest.raises(RuntimeError,match='scene failure'):
        acquire(object(),object(),object(),'explicit')
    assert events==['render-install','scene','stream-stop','render-close','journal-close']
    assert not (tmp_path/'preview_acquire_result.json').exists()


def test_actual_acquire_journal_constructor_failure_releases_borrowed_arguments(tmp_path):
    import gc,weakref
    from test_preview_retained_frames import Borrowed
    events=[];refs=[]
    acquire=acquire_function(tmp_path,events)
    class BrokenJournal:
        def __init__(self,*a,**kw):raise OSError('journal open failed')
    acquire.__globals__['DualHandEvidenceJournal']=BrokenJournal
    def invoke():
        sim=Borrowed();owners=Borrowed();trace=Borrowed()
        refs.extend([weakref.ref(sim),weakref.ref(owners),weakref.ref(trace)])
        try:
            with pytest.raises(RuntimeError,match='journal open failed'):
                acquire(sim,owners,trace,'explicit')
        finally:sim=owners=trace=None
    with hold_completed(acquire) as frames:
        invoke();gc.collect()
        assert events==['stream-stop','render-close']
        assert frames and all(ref() is None for ref in refs)


def test_actual_acquire_preserves_scene_and_cleanup_failure_together(tmp_path):
    import traceback
    events=[]
    acquire=acquire_function(tmp_path,events,fail=True)
    def stop():
        assert 'scene failure' in json.loads((tmp_path/'preview_acquire_error.json').read_text())['error']
        events.append('stream-stop');raise RuntimeError('stream shutdown failure')
    acquire.__globals__['app']=SimpleNamespace(stop_presentation=stop)
    with pytest.raises(RuntimeError) as caught:
        acquire(object(),object(),object(),'explicit')
    text=''.join(traceback.format_exception(caught.type,caught.value,caught.tb))
    assert 'scene failure' in text and 'stream shutdown failure' in text
    assert events==['render-install','scene','stream-stop','render-close','journal-close']
