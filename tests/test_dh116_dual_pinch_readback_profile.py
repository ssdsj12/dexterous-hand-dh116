import numpy as np
import pytest
from scripts import prepare_dh116_dual_pinch_readback_profile as m


def test_exact_one_nonzero_interval_and_no_wrist_action():
    a=m.action()
    assert np.flatnonzero(a).tolist()==[1,2,7,8]
    np.testing.assert_array_equal(a[[1,2,7,8]],[.1,.1,.05,.1])
    assert not a[12:].any()
    p=m.profile("a"*64)
    assert p["setup_sdk_steps"]==2 and p["total_sdk_steps"]==18
    assert [x["sdk_steps"] for x in p["phases"]]==[8,8]
    assert p["reset_seed_sha256"]==m.SEED_SHA
    assert p["native_qualification"] is False


@pytest.mark.parametrize("n",[0,1,7,9,241,248,8.0,True])
def test_partial_or_unbounded_intervals_refused(n):
    with pytest.raises(ValueError):m.interval_count(n)


@pytest.mark.parametrize("n",[8,16,240])
def test_whole_bounded_interval_accounting(n):
    assert m.interval_count(n)==n//8


def test_actual_kernels_integrate_distinct_side_velocities():
    q=np.tile([.3,.4,.6,1.,1.,1.],2)
    rows=m.targets(q,q)
    assert len(rows)==9
    np.testing.assert_allclose(rows[-1]-q,m.action()[:12]*.3*8/480,atol=3e-7)
    assert rows[-1][7]-q[7]==pytest.approx((rows[-1][1]-q[1])/2,abs=2e-7)


def test_original_lead_guard_still_applies():
    q=np.tile([.3,.4,.6,1.,1.,1.],2);command=q.copy();command[1]+=.1
    rows=m.targets(command,q)
    assert rows[-1][1]<=q[1]+.1+3e-8


def test_full_shape_guard_rejects_missing_or_subfloor():
    with pytest.raises(ValueError):m.table_guard(None)
    assert m.table_guard([.0021,.0022])
    assert not m.table_guard([.002099,.003])


def test_profile_denies_unbound_result():
    for v in ("", "z"*64, None):
        with pytest.raises(ValueError):m.profile(v)


def test_stale_or_tampered_native_ledger_rejected(tmp_path):
    p=tmp_path/"bad";p.write_text("{}")
    with pytest.raises(ValueError):m.load_native(ledger=p)


def test_exclusive_output(tmp_path):
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])


def test_each_next_target_starts_from_actual_float32_backend_roundtrip():
    q=np.tile([.3,.4,.6,1.,1.,1.],2).astype(np.float32).astype(float)
    rows=m.targets(q,q);expected=q.copy()
    for i in range(1,9):
        expected=(expected+m.action()[:12]*.3/480).astype(np.float32).astype(float)
        np.testing.assert_array_equal(rows[i],expected)
