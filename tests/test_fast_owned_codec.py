"""Byte-exact independent codec path; no synthetic native qualification."""
import hashlib,json
import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116 import tabletop_deformable_r0_replay as original
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope

def bytes_for(value):
    return json.dumps(original.encode_owned(value),allow_nan=False,sort_keys=True,separators=(",",":")).encode()

@pytest.mark.parametrize("dtype",[np.bool_,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,
    np.uint32,np.uint64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize("shape",[(),(0,),(2,0,3),(2,3)])
def test_numeric_shapes_and_dtypes_are_byte_and_digest_exact(dtype,shape):
    a=np.arange(max(1,int(np.prod(shape))),dtype=np.int64)[:int(np.prod(shape))].astype(dtype).reshape(shape)
    packet=dict(array=a,mask=np.ones(shape,dtype=bool),nested=[a.T,{"scalar":np.float32(.1)}])
    before=bytes_for(packet)
    with codec_scope():
        after=bytes_for(packet)
        assert hashlib.sha256(after).digest()==hashlib.sha256(before).digest()
    assert after==before

@pytest.mark.parametrize("a",[
    np.array([np.nan,np.inf,-np.inf,-0.,0.,1.]),
    np.arange(48,dtype=np.float64).reshape(6,8)[::-2,::3],
    np.array([0,2**64-1],dtype=np.uint64),
    np.array([-2**63,2**63-1],dtype=np.int64),
    np.array([-0.,0.,np.nextafter(0.,1.),1e308],dtype=np.float64),
    np.array([1.,2.],dtype=">f8"),
    torch.tensor([1.,float("nan"),-0.],requires_grad=True),
])
def test_special_values_masks_tensors_strides_and_endian_exact(a):
    v=dict(value=a,invalid=np.array([False,True]),metadata=None)
    expected=bytes_for(v)
    with codec_scope():actual=bytes_for(v)
    assert actual==expected

def test_snapshot_remains_owned_and_unknowns_are_not_zero_filled():
    a=np.array([1.,np.nan]);mask=np.array([True,False])
    with codec_scope():copy=original._copy(dict(a=a,valid=mask))
    a[0]=99;mask[0]=False
    assert copy["a"][0]==1 and np.isnan(copy["a"][1])
    assert copy["valid"].tolist()==[True,False]

@pytest.mark.parametrize("bad",[object(),np.array([object()],dtype=object),np.array([1j]),{1:"bad"}])
def test_arbitrary_native_handles_and_invalid_types_rejected(bad):
    with pytest.raises(ValueError):bytes_for(bad)
    with codec_scope(),pytest.raises(ValueError):bytes_for(bad)

def test_nested_scope_and_exception_restore_original_callable():
    before=original.encode_owned
    with pytest.raises(RuntimeError):
        with codec_scope():
            fast=original.encode_owned
            assert fast is not before
            with codec_scope():assert original.encode_owned is fast
            assert original.encode_owned is fast
            raise RuntimeError("fixture")
    assert original.encode_owned is before

def test_actual_cpu_owner_replay_receipts_unchanged(monkeypatch,tmp_path):
    from test_deformable_r0_replay import setup,advance
    ep,rod,hands,sim,rows,obs=setup(monkeypatch,tmp_path,mode="empty")
    advance(ep,rows,obs,aux=True)
    bundle=obs.bundle()
    expected=bytes_for(original.replay_diagnostic_bundle(bundle))
    with codec_scope():actual=bytes_for(original.replay_diagnostic_bundle(bundle))
    assert actual==expected

def test_finite_builtin_payload_does_not_repeat_legacy_scalar_dispatch():
    # Operation-count guard, avoiding flaky wall-clock assertions on CI hosts.
    import cProfile
    packet={"values":[float(i)/3 for i in range(2048)],"ids":list(range(2048))}
    expected=bytes_for(packet)
    trace=cProfile.Profile()
    with codec_scope():
        trace.enable()
        actual=bytes_for(packet)
        trace.disable()
    legacy=original.encode_owned.__code__
    legacy_calls=sum(entry.callcount for entry in trace.getstats() if entry.code is legacy)
    assert actual==expected
    assert legacy_calls<=1, "recursive finite scalar fallback defeats the fast path"

@pytest.mark.parametrize("value",[None,True,False,0,2**150,-2**150,0.,-0.,1e308,
    float("nan"),float("inf"),float("-inf"),"wire",[],{},(1,2)])
def test_builtin_values_retain_exact_legacy_bytes(value):
    expected=bytes_for(value)
    with codec_scope():assert bytes_for(value)==expected
