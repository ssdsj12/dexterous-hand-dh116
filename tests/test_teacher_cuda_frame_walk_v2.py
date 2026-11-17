"""CPU-only bounded frame graph walk, independent of native success."""
import gc
import inspect
import json
from pathlib import Path
import sys
import weakref

import pytest
from go2_pvcnn.control.dh116 import tabletop_simulation_owner_cleanup as owners
from scripts import diagnose_dh116_teacher_initialization as worker


def frame_case():
    external={}
    def registerOp():
        external["held_registration_frame"]=sys._getframe()
    def _register_triton_kernels():
        registerOp()
    def _lazy_init():
        _register_triton_kernels()
    class Scene:
        def __init__(self):_lazy_init()
    def factory():
        scene=Scene()
        return weakref.ref(scene)
    return external,factory()


def api():
    fn=owners.probe_simulation_owner_edges
    assert "cuda_frame_owner_budget" in inspect.signature(fn).parameters, "bounded frame walk missing"
    return fn


def packets(row):
    return [p for edge in row["selected_owner_edges"] for p in edge.get("upstream_frame_owners",[])]


def test_walk_finds_terminal_holder_and_does_not_retain_runtime():
    external,reference=frame_case()
    row=api()(reference,trace_cuda_init_owners=True,cuda_frame_owner_budget=8)
    found=packets(row)
    assert [p["frame"]["function"] for p in found]==["_lazy_init","_register_triton_kernels","registerOp"]
    assert any("held_registration_frame" in p.get("matching_keys",[]) for p in found[-1]["owners"])
    stats=row["cuda_frame_owner_trace"]
    assert stats["frames_followed"]==3 and stats["root_frames_followed"]==1
    assert stats["max_frames"]==8 and stats["frame_walk_truncated"] is False
    assert row["max_hops"]==10 and row["root_cause_proven"] is False
    assert all(p["native_owner_qualified"] is False for p in found)
    assert json.loads(json.dumps(row))==row
    external.clear();gc.collect()
    assert reference() is None, "bounded walk retained the simulated scene"


@pytest.mark.parametrize("budget",[0,9,True,None])
def test_frame_budget_rejects_invalid_or_boolean_values(budget):
    external,reference=frame_case()
    with pytest.raises(ValueError,match="1..8"):
        api()(reference,trace_cuda_init_owners=True,cuda_frame_owner_budget=budget)
    external.clear()


def test_walk_stops_at_budget_and_reports_unfollowed_frame(monkeypatch):
    external,reference=frame_case()
    leaf=external["held_registration_frame"];registration=leaf.f_back
    lazy=registration.f_back;init=lazy.f_back;target=reference();calls=[]
    def query(value):
        if value is target:return [init]
        if value is init:return [lazy]
        if value is lazy:calls.append(id(value));return [registration]
        if value is registration:calls.append(id(value));return [leaf]
        if value is leaf:raise AssertionError("walk exceeded frame budget")
        raise AssertionError("unexpected traversal")
    monkeypatch.setattr(owners.gc,"get_referrers",query)
    row=api()(reference,trace_cuda_init_owners=True,cuda_frame_owner_budget=2)
    stats=row["cuda_frame_owner_trace"]
    assert calls==[id(lazy),id(registration)] and stats["frames_followed"]==2
    assert stats["frame_walk_truncated"] is True
    assert stats["pending_frame_candidates_at_limit"]==1
    external.clear()


def test_walk_handles_duplicate_edges_and_cycles_once(monkeypatch):
    external,reference=frame_case()
    leaf=external["held_registration_frame"];registration=leaf.f_back
    lazy=registration.f_back;init=lazy.f_back;target=reference();calls=[]
    def query(value):
        if value is target:return [init]
        if value is init:return [lazy]
        if value is lazy:calls.append(id(value));return [registration,registration]
        if value is registration:calls.append(id(value));return [lazy]
        raise AssertionError("unexpected traversal")
    monkeypatch.setattr(owners.gc,"get_referrers",query)
    row=api()(reference,trace_cuda_init_owners=True,cuda_frame_owner_budget=8)
    stats=row["cuda_frame_owner_trace"]
    assert calls==[id(lazy),id(registration)]
    assert stats["frame_walk_duplicates_skipped"]==2
    assert stats["frame_walk_truncated"] is False and stats["frames_followed"]==2
    external.clear()


def test_existing_one_frame_opt_in_budget_remains_default():
    external,reference=frame_case()
    row=api()(reference,trace_cuda_init_owners=True)
    stats=row["cuda_frame_owner_trace"]
    assert stats["max_frames"]==1 and stats["frames_followed"]==1 and row["max_hops"]==3
    assert stats["frame_walk_truncated"] is True
    external.clear()


def test_worker_uses_eight_frame_budget_at_existing_boundary(tmp_path,monkeypatch):
    external,reference=frame_case();calls=[]
    def probe(ref,**kwargs):
        calls.append(kwargs);return {"native_release_verified":False}
    monkeypatch.setattr(owners,"probe_simulation_owner_edges",probe)
    worker.write_owner_probe(tmp_path,reference,"after_stop")
    assert calls[0].get("cuda_frame_owner_budget")==8
    external.clear()


def test_actual_registration_sources_are_frozen():
    root=Path(worker.__file__).resolve().parents[1];paths=worker.external_input_paths(root)
    base=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/torch")
    assert base/"__init__.py" in paths
    assert base/"library.py" in paths


def test_walk_error_does_not_leave_frame_aliases_in_worker(tmp_path,monkeypatch):
    external,reference=frame_case()
    leaf=external["held_registration_frame"];registration=leaf.f_back
    lazy=registration.f_back;init=lazy.f_back;target=reference()
    def query(value):
        if value is target:return [init]
        if value is init:return [lazy]
        if value is lazy:return [registration]
        if value is registration:raise RuntimeError("CPU walk fault")
        raise AssertionError("unexpected traversal")
    monkeypatch.setattr(owners.gc,"get_referrers",query)
    assert "cuda_frame_owner_budget" in inspect.signature(owners.probe_simulation_owner_edges).parameters
    error=worker.write_owner_probe(tmp_path,reference,"after_stop")
    assert "CPU walk fault" in error and not (tmp_path/"after_stop.owner_edges.json").exists()
    external.clear();leaf=registration=lazy=init=target=None;gc.collect()
    assert reference() is None
