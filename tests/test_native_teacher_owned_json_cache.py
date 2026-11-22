"""Owned JSON deserialization reuse; no native physics or qualification."""
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import numpy as np
import pytest


def module(monkeypatch, *, eligible=True):
    m=importlib.import_module("go2_pvcnn.control.dh116.tabletop_fast_owned_codec")
    assert callable(getattr(m,"decode_owned_json",None)), "bounded owned JSON decoder missing"
    monkeypatch.setattr(m,"_JSON_CACHE",OrderedDict())
    if eligible:
        monkeypatch.setattr(m,"_JSON_CACHE_MIN_RAW_BYTES",1)
    return m


def parse_counter(monkeypatch,m):
    original=json.loads
    calls=Counter()
    def parse(raw,*args,**kwargs):
        calls[raw]+=1
        return original(raw,*args,**kwargs)
    monkeypatch.setattr(m.json,"loads",parse)
    return calls


def test_cached_first_and_hit_results_are_independent_and_preserve_numeric_schema(monkeypatch):
    m=module(monkeypatch)
    calls=parse_counter(monkeypatch,m)
    raw=json.dumps(dict(array={"__dh116_array__":[[-0.0,1.25],[2.,3.]],"dtype":"float32"},
        meta={"items":[1,2]},integer=2**80,missing={"__dh116_nonfinite__":"nan"},valid=False)).encode()
    first=m.decode_owned_json(raw)
    first["array"][0,0]=9
    first["meta"]["items"].append(99)
    second=m.decode_owned_json(raw)
    assert second["array"].dtype==np.dtype("float32")
    assert np.signbit(second["array"][0,0])
    assert second["array"][0,1]==1.25 and second["meta"]["items"]==[1,2]
    assert type(second["integer"]) is int and second["integer"]==2**80
    assert np.isnan(second["missing"]) and second["valid"] is False
    assert not np.shares_memory(first["array"],second["array"])
    second["array"][1,1]=100
    second["meta"]["items"].clear()
    third=m.decode_owned_json(raw)
    assert third["array"][1,1]==3 and third["meta"]["items"]==[1,2]
    assert not np.shares_memory(second["array"],third["array"])
    assert calls[raw]==1


def test_cache_lru_evicts_old_content_without_reusing_changed_bytes(monkeypatch):
    m=module(monkeypatch)
    calls=parse_counter(monkeypatch,m)
    a,b,c=(json.dumps({"value":i}).encode() for i in (1,2,3))
    for raw in (a,b,a,c,b):
        assert m.decode_owned_json(raw)["value"] in (1,2,3)
    assert calls==Counter({a:1,b:2,c:1})


def test_cache_total_decoded_bytes_evicts_even_below_entry_limit(monkeypatch):
    m=module(monkeypatch)
    monkeypatch.setattr(m,"_JSON_CACHE_MAX_BYTES",6*1024**2)
    calls=parse_counter(monkeypatch,m)
    a=json.dumps({"array":{"__dh116_array__":[0]*(512*1024),"dtype":"float64"}}).encode()
    b=json.dumps({"array":{"__dh116_array__":[1]*(512*1024),"dtype":"float64"}}).encode()
    for raw,value in ((a,0),(b,1),(a,0)):
        decoded=m.decode_owned_json(raw)
        assert decoded["array"].shape==(512*1024,) and np.all(decoded["array"]==value)
    assert calls==Counter({a:2,b:1})


def test_decoded_oversize_bypasses_retention(monkeypatch):
    m=module(monkeypatch)
    monkeypatch.setattr(m,"_JSON_CACHE_MAX_BYTES",1)
    calls=parse_counter(monkeypatch,m)
    raw=b'{"array":{"__dh116_array__":[1,2],"dtype":"int64"}}'
    for _ in range(2):
        assert m.decode_owned_json(raw)["array"].tolist()==[1,2]
    assert calls[raw]==2


def test_default_small_documents_keep_original_parse_behavior(monkeypatch):
    m=module(monkeypatch,eligible=False)
    calls=parse_counter(monkeypatch,m)
    raw=b'{"value":1}'
    assert m.decode_owned_json(raw)==m.decode_owned_json(raw)=={"value":1}
    assert calls[raw]==2


@pytest.mark.parametrize("raw,error",[
    (b'{"unfinished":',json.JSONDecodeError),
    (b'{"__dh116_nonfinite__":"invalid"}',ValueError),
])
def test_decode_errors_are_not_cached_or_changed(monkeypatch,raw,error):
    m=module(monkeypatch)
    calls=parse_counter(monkeypatch,m)
    for _ in range(2):
        with pytest.raises(error):
            m.decode_owned_json(raw)
    assert calls[raw]==2


def test_concurrent_same_content_returns_owned_trees_from_one_parse(monkeypatch):
    m=module(monkeypatch)
    calls=parse_counter(monkeypatch,m)
    raw=b'{"array":{"__dh116_array__":[1,2],"dtype":"int64"},"items":[3]}'
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(m.decode_owned_json,[raw]*8))
    assert calls[raw]==1
    assert len({id(value["array"]) for value in values})==8
    assert len({id(value["items"]) for value in values})==8
    values[0]["array"][0]=100
    assert all(value["array"].tolist()==[1,2] for value in values[1:])


def test_cache_does_not_keep_replaced_parser_instrumentation_owner_alive(monkeypatch):
    import gc
    import weakref
    m=module(monkeypatch)
    original=json.loads
    class Owner:
        pass
    owner=Owner()
    ref=weakref.ref(owner)
    def parse(raw,_owner=owner):
        return original(raw)
    with monkeypatch.context() as patch:
        patch.setattr(m.json,"loads",parse)
        assert m.decode_owned_json(b'{"value":1}')=={"value":1}
    del parse,owner
    gc.collect()
    assert ref() is None
