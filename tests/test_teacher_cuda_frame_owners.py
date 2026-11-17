"""CPU third-hop tracing; fake CUDA frames never qualify native release."""
import ast
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
    def _lazy_init():
        external["held_cuda_init"]=sys._getframe()
    class Model:
        def __init__(self):
            _lazy_init()
    def factory():
        sim=Model()
        return weakref.ref(sim)
    return external,factory()


def api():
    fn=owners.probe_simulation_owner_edges
    assert "trace_cuda_init_owners" in inspect.signature(fn).parameters, "third-hop opt-in missing"
    return fn


def test_trace_finds_holder_above_cuda_frame_without_retaining_objects():
    external,reference=frame_case()
    row=api()(reference,trace_cuda_init_owners=True)
    followups=[f for e in row["selected_owner_edges"] for f in e.get("upstream_frame_owners",[])]
    assert len(followups)==1
    packet=followups[0]
    assert packet["frame"]["function"]=="_lazy_init"
    assert any("held_cuda_init" in p.get("matching_keys",[]) for p in packet["owners"])
    assert row["max_hops"]==3 and row["cuda_frame_owner_trace"]["frames_followed"]==1
    assert row["root_cause_proven"] is False and row["native_release_verified"] is False
    assert packet["native_owner_qualified"] is False
    assert json.loads(json.dumps(row))==row
    external.clear();gc.collect()
    assert reference() is None, "scalar third-hop result retained the simulated runtime"


def test_default_two_hop_trace_does_not_scan_cuda_frame_owners(monkeypatch):
    external,reference=frame_case()
    frame=external["held_cuda_init"];calls=[]
    original=owners.gc.get_referrers
    def query(value):
        if value is frame:calls.append("third")
        return original(value)
    monkeypatch.setattr(owners.gc,"get_referrers",query)
    row=api()(reference)
    assert calls==[] and row["max_hops"]==2
    assert not any(e.get("upstream_frame_owners") for e in row["selected_owner_edges"])
    row=api()(reference,trace_cuda_init_owners=True)
    assert calls==["third"] and row["cuda_frame_owner_trace"]["frames_followed"]==1
    external.clear();frame=None


@pytest.mark.parametrize("flag",[1,None,"yes"])
def test_trace_flag_requires_boolean(flag):
    external,reference=frame_case()
    with pytest.raises(ValueError,match="boolean"):
        api()(reference,trace_cuda_init_owners=flag)
    external.clear()


def test_third_hop_scan_is_bounded_and_empty_python_owners_remain_unknown(monkeypatch):
    external,reference=frame_case()
    lazy=external["held_cuda_init"];init=lazy.f_back
    target=reference();mode={"count":100}
    def query(value):
        if value is target:return [init]
        if value is init:return [lazy]
        if value is lazy:return [{"retained":lazy} for _ in range(mode["count"])]
        raise AssertionError("unexpected graph traversal")
    monkeypatch.setattr(owners.gc,"get_referrers",query)
    row=api()(reference,max_referrers=3,trace_cuda_init_owners=True)
    packet=row["selected_owner_edges"][0]["upstream_frame_owners"][0]
    assert packet["enumerated_count"]==100 and packet["scanned_count"]==3
    assert len(packet["owners"])==3 and packet["truncated"] is True
    mode["count"]=0
    row=api()(reference,max_referrers=3,trace_cuda_init_owners=True)
    packet=row["selected_owner_edges"][0]["upstream_frame_owners"][0]
    assert packet["owners"]==[] and packet["observation"]=="no_visible_python_owner"
    assert packet["native_owner_qualified"] is False and row["native_release_verified"] is False
    external.clear()


def test_only_one_cuda_frame_is_followed_per_probe(monkeypatch):
    external,reference=frame_case()
    lazy=external["held_cuda_init"];init=lazy.f_back;target=reference();calls=[]
    def query(value):
        if value is target:return [init,init]
        if value is init:return [lazy,lazy]
        if value is lazy:calls.append(id(value));return [external]
        raise AssertionError("unexpected traversal")
    monkeypatch.setattr(owners.gc,"get_referrers",query)
    row=api()(reference,trace_cuda_init_owners=True)
    assert calls==[id(lazy)] and row["cuda_frame_owner_trace"]["frames_followed"]==1
    assert row["cuda_frame_owner_trace"]["duplicate_candidates_skipped"]==3
    external.clear()


def test_third_hop_helpers_do_not_read_clear_or_collect_frames():
    fn=getattr(owners,"_probe_cuda_frame_owners",None)
    assert callable(fn), "bounded frame-owner helper missing"
    for f in (fn,owners.probe_simulation_owner_edges):
        tree=ast.parse(inspect.getsource(f))
        for n in ast.walk(tree):
            if isinstance(n,ast.Attribute):
                assert n.attr not in ("f_locals","clear","collect"), n.attr


def test_worker_requests_cuda_frame_trace_at_existing_boundaries(tmp_path,monkeypatch):
    calls=[];external,reference=frame_case()
    def probe(ref,**kwargs):
        assert ref is reference
        calls.append(kwargs)
        return {"native_release_verified":False}
    monkeypatch.setattr(owners,"probe_simulation_owner_edges",probe)
    worker.write_owner_probe(tmp_path,reference,"after_stop")
    assert calls[0].get("trace_cuda_init_owners") is True
    external.clear()


def test_freeze_pins_actual_torch_cuda_source():
    root=Path(worker.__file__).resolve().parents[1]
    paths=worker.external_input_paths(root)
    expected=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/torch/cuda/__init__.py")
    assert expected in paths, "actual upstream CUDA frame source is not frozen"
