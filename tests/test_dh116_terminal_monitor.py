"""Terminal status must survive initialization failures without losing progress."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]


def test_error_publishes_terminal_status_and_keeps_latest_progress(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_run_status import publish_terminal_monitor
    path = tmp_path/'monitor.json'
    path.write_text(json.dumps(dict(status='training',iteration=75,timesteps=7372800,
                                   safety_stop='force',per_env=[{'env_index':545}])))
    publish_terminal_monitor(path,dict(status='error',error='test traceback',timesteps=0),2)
    result=json.loads(path.read_text())
    assert result == dict(status='error',error='test traceback',process_exit_code=2,
                         iteration=75,timesteps=7372800,safety_stop='force',per_env=[{'env_index':545}])


def test_initialization_failure_and_interrupt_are_terminal(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_run_status import publish_terminal_monitor
    path=tmp_path/'monitor.json'
    for existing in (None,'broken json',json.dumps(dict(status='initializing',num_envs=4096))):
        if existing is not None:path.write_text(existing)
        publish_terminal_monitor(path,dict(status='error',error='launch failed'),2)
        assert json.loads(path.read_text())['status']=='error'
    publish_terminal_monitor(path,dict(status='interrupted',interrupted=True),2)
    assert json.loads(path.read_text())['interrupted'] is True


def test_cli_launcher_failure_writes_terminal_monitor(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
    import traceback
    tree=ast.parse((ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    guarded=next((node for node in tree.body if isinstance(node,ast.Try) and any(
        isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='AppLauncher'
        for statement in node.body for n in ast.walk(statement))),None)
    assert guarded is not None, 'AppLauncher must be inside the terminal-status exception guard'
    def launch(_):raise RuntimeError('synthetic launcher failure')
    def write(path,value):path.write_text(json.dumps(value))
    ns=dict(app=None,vec=None,sim=None,report=dict(status='initializing'),args=SimpleNamespace(output=tmp_path),
            AppLauncher=launch,write_json=write,finalize_run=finalize_run,
            traceback=traceback,json=json)
    exec(compile(ast.Module(body=[guarded],type_ignores=[]),'<guarded launcher>','exec'),ns)
    monitor=json.loads((tmp_path/'monitor.json').read_text())
    assert monitor['status']=='error' and monitor['process_exit_code']==2
    assert 'synthetic launcher failure' in monitor['error']


def test_cleanup_failure_still_cleans_app_and_reports_failure(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
    calls=[]
    def close_vec():
        calls.append('vec')
        raise OSError('episode flush failed')
    report=dict(status='training_budget_complete')
    code=finalize_run(tmp_path,report,[close_vec,lambda:calls.append('app')])
    assert calls==['vec','app'] and code==2
    monitor=json.loads((tmp_path/'monitor.json').read_text())
    assert monitor['status']=='error' and 'episode flush failed' in monitor['error']


def test_publish_failure_does_not_skip_cleanup(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_run_status as status
    calls=[]
    def broken(*_):raise OSError('monitor disk failure')
    monkeypatch.setattr(status,'publish_terminal_monitor',broken)
    code=status.finalize_run(tmp_path,dict(status='training_budget_complete'),[lambda:calls.append('app')])
    assert code==2 and calls==['app']
    assert json.loads((tmp_path/'summary.json').read_text())['process_exit_code']==2


def test_cli_nonzero_system_exit_is_not_success(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
    import traceback
    tree=ast.parse((ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    guarded=next(node for node in tree.body if isinstance(node,ast.Try) and any(
        isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='AppLauncher'
        for statement in node.body for n in ast.walk(statement)))
    def launch(_):raise SystemExit(7)
    ns=dict(app=None,vec=None,sim=None,report=dict(status='initializing'),args=SimpleNamespace(output=tmp_path),
            AppLauncher=launch,finalize_run=finalize_run,traceback=traceback,json=json)
    exec(compile(ast.Module(body=[guarded],type_ignores=[]),'<guarded launcher>','exec'),ns)
    monitor=json.loads((tmp_path/'monitor.json').read_text())
    assert monitor['status']=='error' and monitor['process_exit_code']!=0
    assert 'SystemExit: 7' in monitor['error']


def test_summary_failure_corrects_previously_published_success(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
    replace=Path.replace
    def fail_summary(self,target):
        if Path(target).name=='summary.json':raise OSError('summary disk failure')
        return replace(self,target)
    monkeypatch.setattr(Path,'replace',fail_summary)
    assert finalize_run(tmp_path,dict(status='training_budget_complete'),[])==2
    monitor=json.loads((tmp_path/'monitor.json').read_text())
    assert monitor['status']=='error' and monitor['process_exit_code']==2


def test_success_keeps_training_counters(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
    (tmp_path/'monitor.json').write_text(json.dumps(dict(status='training',iteration=75,episodes=100)))
    assert finalize_run(tmp_path,dict(status='training_budget_complete'),[])==0
    monitor=json.loads((tmp_path/'monitor.json').read_text())
    assert monitor==dict(status='training_budget_complete',iteration=75,episodes=100,process_exit_code=0)


def test_cli_publishes_before_nonreturning_isaac_shutdown(tmp_path,monkeypatch):
    import sys
    import types
    import traceback
    import pytest
    from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
    observed=[]
    def close(**_):
        monitor=json.loads((tmp_path/'monitor.json').read_text())
        assert monitor['status']=='error' and monitor['process_exit_code']==2
        assert (tmp_path/'summary.json').exists()
        observed.append('shutdown')
        raise SystemExit(2)
    def flush():raise OSError('flush failed')
    omni=types.ModuleType('omni');kit=types.ModuleType('omni.kit');kit_app=types.ModuleType('omni.kit.app')
    kit_app.get_app=lambda:SimpleNamespace(post_quit=lambda code:observed.append(code))
    omni.kit=kit;kit.app=kit_app
    for name,module in [('omni',omni),('omni.kit',kit),('omni.kit.app',kit_app)]:monkeypatch.setitem(sys.modules,name,module)
    tree=ast.parse((ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    guarded=next(node for node in tree.body if isinstance(node,ast.Try) and any(
        isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='AppLauncher'
        for statement in node.body for n in ast.walk(statement)))
    ns=dict(app=SimpleNamespace(close=close),vec=SimpleNamespace(close=flush),sim=None,
            report=dict(status='training_budget_complete'),args=SimpleNamespace(output=tmp_path),
            finalize_run=finalize_run,traceback=traceback,json=json)
    with pytest.raises(SystemExit,match='2'):
        exec(compile(ast.Module(body=guarded.finalbody,type_ignores=[]),'<final shutdown>','exec'),ns)
    assert observed==[2,'shutdown']


def test_cli_releases_timeline_stop_callback_before_application_shutdown(tmp_path,monkeypatch):
    import sys
    import types
    import traceback
    from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
    calls=[]
    subscription={'active':True}
    def clear_instance():
        subscription['active']=False
        calls.append('unsubscribe')
    def close(**_):
        # Installed IsaacLab stop callback renders forever while the stopped
        # timeline remains subscribed. Model that failure without hanging pytest.
        if subscription['active']:
            raise RuntimeError('timeline stop callback would render forever')
        calls.append('app_close')
    def post_quit(code):
        assert not subscription['active']
        calls.append(('quit',code))
    omni=types.ModuleType('omni');kit=types.ModuleType('omni.kit');kit_app=types.ModuleType('omni.kit.app')
    kit_app.get_app=lambda:SimpleNamespace(post_quit=post_quit)
    omni.kit=kit;kit.app=kit_app
    for name,module in [('omni',omni),('omni.kit',kit),('omni.kit.app',kit_app)]:monkeypatch.setitem(sys.modules,name,module)
    tree=ast.parse((ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    guarded=next(node for node in tree.body if isinstance(node,ast.Try) and any(
        isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='AppLauncher'
        for statement in node.body for n in ast.walk(statement)))
    ns=dict(app=SimpleNamespace(close=close),vec=SimpleNamespace(close=lambda:calls.append('vec')),
            sim=SimpleNamespace(clear_all_callbacks=lambda:calls.append('callbacks'),clear_instance=clear_instance),
            report=dict(status='training_budget_complete'),args=SimpleNamespace(output=tmp_path),
            finalize_run=finalize_run,traceback=traceback,json=json)
    exec(compile(ast.Module(body=guarded.finalbody,type_ignores=[]),'<final shutdown>','exec'),ns)
    assert calls==['vec','callbacks','unsubscribe',('quit',0),'app_close']
    assert json.loads((tmp_path/'summary.json').read_text())['process_exit_code']==0
