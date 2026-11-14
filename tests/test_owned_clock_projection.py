"""Clock validation still delegates to the original exact contract."""
import importlib,importlib.util
from copy import deepcopy
from types import SimpleNamespace
import sys
import pytest
from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts as original
MODULE="go2_pvcnn.control.dh116.tabletop_owned_clock_projection"

def api():
    assert importlib.util.find_spec(MODULE) is not None,"owned clock projection missing"
    return importlib.import_module(MODULE)

def good():
    return dict(run_uuid="run",env_id=0,episode_uid="ep",reset_id=1,rod_uid="rod",
        physics_step=2,rod_tick=0,timestamp_s=2/480,solver_time_s=0.,clock_valid=True)

class CopyCounter:
    calls=0
    def __deepcopy__(self,memo):
        type(self).calls+=1
        return self

def test_original_validation_without_copying_non_clock_payload():
    m=api();row=good();row["large_non_clock_payload"]=CopyCounter();CopyCounter.calls=0
    calls=[]
    result=m.project_clock(lambda:(calls.append(1),row)[1])
    assert result==original._clock(lambda:good()) and len(calls)==1
    assert CopyCounter.calls==0
    original._clock(lambda:row)
    assert CopyCounter.calls==1 # test demonstrably detects original unnecessary copy

@pytest.mark.parametrize("key,value",[("env_id",True),("reset_id",-1),("physics_step",2.),("rod_tick",-1),
    ("run_uuid",""),("episode_uid",None),("rod_uid",1),("clock_valid",1),
    ("timestamp_s",float("nan")),("timestamp_s",.2),("solver_time_s",float("inf"))])
def test_same_invalid_clock_rejections(key,value):
    m=api();row=good();row[key]=value
    for func in (m.project_clock,original._clock):
        with pytest.raises(ValueError):func(lambda:row)

@pytest.mark.parametrize("row",[None,{},[],{"env_id":0}])
def test_missing_clock_contract_still_rejected(row):
    m=api()
    with pytest.raises(ValueError):m.project_clock(lambda:row)

@pytest.mark.parametrize("fail",[False,True])
def test_scope_restores_source_and_imported_alias_on_all_exits(monkeypatch,fail):
    m=api()
    from scripts import validate_dh116_native_parameter_task as target
    old=original._clock;alias=target._clock
    def run():
        with m.clock_projection_scope():
            assert original._clock is m.project_clock and target._clock is m.project_clock
            assert target._clock(lambda:good())==good()
            if fail:raise RuntimeError("owner failed")
    if fail:
        with pytest.raises(RuntimeError):run()
    else:run()
    assert original._clock is old and target._clock is alias

def test_unexpected_alias_does_not_get_overwritten(monkeypatch):
    m=api()
    from scripts import validate_dh116_native_parameter_task as target
    other=lambda _:None
    monkeypatch.setattr(target,"_clock",other)
    before=original._clock
    with pytest.raises(RuntimeError):
        with m.clock_projection_scope():pass
    assert target._clock is other and original._clock is before

def test_nested_scope_rejected_and_outer_restored():
    m=api();before=original._clock
    with m.clock_projection_scope():
        with pytest.raises(RuntimeError,match="nest"):
            with m.clock_projection_scope():pass
        assert original._clock is m.project_clock
    assert original._clock is before

def test_late_import_alias_restored(monkeypatch):
    m=api();name="go2_pvcnn.control.dh116.tabletop_material_director_readback"
    monkeypatch.delitem(sys.modules,name,raising=False)
    with m.clock_projection_scope():
        late=SimpleNamespace(shared_clock=original._clock)
        monkeypatch.setitem(sys.modules,name,late)
    assert late.shared_clock is original._clock

def test_custom_container_retains_original_copy():
    m=api()
    class Custom(dict):
        copies=0
        def __deepcopy__(self,memo):
            type(self).copies+=1
            return dict(self)
    row=Custom(good())
    assert m.project_clock(lambda:row)==good()
    assert Custom.copies==1

def test_reader_exception_is_not_swallowed():
    m=api()
    def read():raise LookupError("actual clock failed")
    with pytest.raises(LookupError):m.project_clock(read)

def test_clock_snapshot_not_cached_across_reads():
    m=api();row=good()
    first=m.project_clock(lambda:row)
    row["physics_step"]+=1;row["timestamp_s"]=row["physics_step"]/480
    second=m.project_clock(lambda:row)
    assert first["physics_step"]==2 and second["physics_step"]==3
    assert first is not second
