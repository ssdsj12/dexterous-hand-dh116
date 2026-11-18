from types import SimpleNamespace
import pytest
from go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader

def reader_for(reference,times):
    reader=object.__new__(OwnedNativeRGBPairReader)
    reader.reference=SimpleNamespace(get_data=lambda:reference)
    reader.times=SimpleNamespace(get_data=lambda:times)
    reader.core=SimpleNamespace(get_sim_time_at_swh_frame=lambda _:1.)
    return reader

def valid_times():
    return dict(swhFrameNumber=1,frameNumber=2,simulationTime=1.,
        rationalTimeOfSimNumerator=2,rationalTimeOfSimDenominator=30)

def test_reference_zero_over_zero_is_unready_before_first_valid_clock():
    assert reader_for(dict(referenceTimeNumerator=0,referenceTimeDenominator=0),{}).read_clock() is None

def test_zero_isaac_times_are_unready_before_first_valid_clock():
    times=dict(swhFrameNumber=0,frameNumber=0,simulationTime=0.,
        rationalTimeOfSimNumerator=0,rationalTimeOfSimDenominator=0)
    assert reader_for(dict(referenceTimeNumerator=2,referenceTimeDenominator=30),times).read_clock() is None

def test_zero_sentinel_is_rejected_after_valid_clock():
    reference=dict(referenceTimeNumerator=2,referenceTimeDenominator=30);times=valid_times()
    reader=reader_for(reference,times)
    assert reader.read_clock()["frame_id"]==2
    reference.update(referenceTimeNumerator=0,referenceTimeDenominator=0)
    with pytest.raises(ValueError):reader.read_clock()

def test_nonzero_zero_denominator_is_still_corrupt():
    with pytest.raises(ValueError):
        reader_for(dict(referenceTimeNumerator=2,referenceTimeDenominator=0),{}).read_clock()
