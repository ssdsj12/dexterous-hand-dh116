import json
import os
import pytest
from test_fault_safe_lifecycle_v3 import fixture, api

def test_acquire_error_is_durable_before_app_close(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_preview_diagnostics import StageJournal
    f=fixture(tmp_path,monkeypatch,damage="acquire")
    path=tmp_path/"stages.jsonl"
    closed=[]
    def close():
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert any(x["stage"]=="acquire" and x["event"]=="error" and "injected acquire" in x["error"] for x in rows)
        closed.append(True)
        raise RuntimeError("close failed afterwards")
    f.app.close=close
    with StageJournal(path) as journal:
        with pytest.raises(RuntimeError,match="injected acquire"):
            api().run_lifecycle(f.app,stage_observer=journal.emit,**f.kwargs)
    assert closed == [True]

def test_stage_journal_fsyncs_each_event(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_preview_diagnostics import StageJournal
    calls=[]
    original=os.fsync
    def sync(fd):
        calls.append(fd);original(fd)
    monkeypatch.setattr(os,"fsync",sync)
    with StageJournal(tmp_path/"stages.jsonl") as journal:
        journal.emit(dict(stage="acquire",event="start"))
        journal.emit(dict(stage="acquire",event="error",error="original"))
    assert len(calls)>=2

def test_broken_observer_does_not_skip_original_cleanup(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch,damage="acquire")
    def broken(row): raise OSError("diagnostic sink unavailable")
    with pytest.raises(RuntimeError) as caught:
        api().run_lifecycle(f.app,stage_observer=broken,**f.kwargs)
    assert "injected acquire" in str(caught.value)
    assert "diagnostic sink unavailable" in str(caught.value)
    assert "close" in f.events

def test_render_budget_close_releases_sim_even_when_budget_object_retained():
    import weakref
    from go2_pvcnn.control.dh116.tabletop_preview_contract import RenderBudget
    class Sim:
        def render(self):return None
    sim=Sim();ref=weakref.ref(sim)
    budget=RenderBudget(sim,2);budget.install();budget.close()
    del sim
    assert ref() is None
    with pytest.raises(RuntimeError,match="closed"):budget.render()
    budget.close()
