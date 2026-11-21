"""CPU ownership regressions through the real worker lifecycle; no native qualification."""
import gc
import sys
import weakref
import pytest
from scripts import dh116_pinch_acquisition_v2 as old
from scripts import dh116_fault_safe_lifecycle_v3 as lifecycle
from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_provider as provider
from test_fault_safe_lifecycle_v3 import fixture,read


@pytest.mark.parametrize("failure",[None,"acquire","cache","both"])
def test_worker_releases_completed_constructor_cache_on_success_and_fault(tmp_path,monkeypatch,failure):
    f=fixture(tmp_path,monkeypatch)
    cache=[];calls=[];simulation=[None];completed_end=[None]
    original_make=f.kwargs["make_sim"]
    original_owned=old.run_owned_call
    class CachedResource:
        def __init__(self,sim):
            self.sim=sim
            cache.append(sys._getframe())
    def make():
        sim=original_make()
        simulation[0]=weakref.ref(sim)
        return sim
    def acquire(sim,owners,trace,cleanup):
        resource=None
        try:
            f.frames.append(sys._getframe())
            resource=owners.keep("runtime",CachedResource(sim))
            trace.watch("runtime",resource)
            if failure in ("acquire","both"):
                raise RuntimeError("original PPO source fault")
            completed_end[0]=2
            return {"physics_rows":0,"code_config_sha256":"a"*64}
        finally:
            resource=sim=None
    def release(sim,*,expected_sdk_step):
        # This is only the unavailable Kit cache I/O boundary; original lifecycle,
        # real failing-frame cleanup, GC and worker scope execute unmodified.
        assert sim is simulation[0]() and expected_sdk_step==2
        assert f.events[-1]=="synchronize"
        if failure in ("acquire","both"):
            assert f.frames[-1].f_locals=={}
        calls.append(expected_sdk_step)
        cache.clear()
        if failure in ("cache","both"):
            raise RuntimeError("cache source rejection")
    f.kwargs.update(make_sim=make,acquire=acquire)
    try:
        if failure:
            with pytest.raises(RuntimeError,match="original PPO source fault" if failure in ("acquire","both") else "cache source rejection") as raised:
                provider._run_lifecycle_with_stat_cache(f.app,lifecycle=lifecycle.run_lifecycle,
                    cache_release=release,read_sim=lambda:simulation[0]() if simulation[0] else None,
                    read_completed_sdk_step=lambda:completed_end[0],max_sdk_step=66,**f.kwargs)
        else:
            result=provider._run_lifecycle_with_stat_cache(f.app,lifecycle=lifecycle.run_lifecycle,
                cache_release=release,read_sim=lambda:simulation[0]() if simulation[0] else None,
                read_completed_sdk_step=lambda:completed_end[0],max_sdk_step=66,**f.kwargs)
            assert result["failure"] is None
            assert result["native_qualification"] is result["teacher_admission"] is False
        if failure=="both":
            assert "cache source rejection" in str(raised.value)
        assert calls==[2]
        assert all(reference() is None for reference in f.refs.values())
        assert f.events.count("stop")==f.events.count("close")==1
        assert old.run_owned_call is original_owned
        assert read(tmp_path,"result.json")["teacher_admission"] is False
    finally:
        cache.clear()
        gc.collect()


@pytest.mark.parametrize("damage",["step_zero","over_budget","completed_mismatch","missing_sim","make_sim"])
def test_worker_rejects_unbound_cache_clock_without_faking_release(tmp_path,monkeypatch,damage):
    f=fixture(tmp_path,monkeypatch,damage="make_sim" if damage=="make_sim" else None)
    original_make=f.kwargs["make_sim"];original_owned=old.run_owned_call
    simref=[None];calls=[]
    def make():
        sim=original_make()
        sim.current_time_step_index={"step_zero":0,"over_budget":67}.get(damage,2)
        sim.current_time=sim.current_time_step_index/480
        simref[0]=weakref.ref(sim)
        return sim
    f.kwargs["make_sim"]=make
    def read_sim():
        return None if damage=="missing_sim" or simref[0] is None else simref[0]()
    expected=("injected make_sim" if damage=="make_sim" else
              "live worker SDK clock owner" if damage=="missing_sim" else "actual SDK clock")
    with pytest.raises(RuntimeError,match=expected):
        provider._run_lifecycle_with_stat_cache(f.app,lifecycle=lifecycle.run_lifecycle,
            cache_release=lambda *a,**k:calls.append(k),read_sim=read_sim,
            read_completed_sdk_step=lambda:3 if damage=="completed_mismatch" else None,
            max_sdk_step=66,**f.kwargs)
    assert calls==[] and old.run_owned_call is original_owned
    assert f.events.count("close")==1
    assert f.events.count("stop")==1
    assert read(tmp_path,"result.json")["teacher_admission"] is False


def test_worker_scope_refuses_foreign_pre_gc_callback_and_restores_binding():
    from types import SimpleNamespace
    original=old.run_owned_call;calls=[]
    def foreign_lifecycle(app,**kwargs):
        return old.run_owned_call(lambda owners:None,synchronize=lambda:None,
            ownership=SimpleNamespace(checkpoint=lambda _:None),
            before_collect=lambda:calls.append("foreign"))
    with pytest.raises(ValueError,match="owns the source-bound"):
        provider._run_lifecycle_with_stat_cache(object(),lifecycle=foreign_lifecycle,
            cache_release=lambda *a,**k:calls.append("cache"),read_sim=lambda:None,
            read_completed_sdk_step=lambda:None,max_sdk_step=66)
    assert calls==[] and old.run_owned_call is original
