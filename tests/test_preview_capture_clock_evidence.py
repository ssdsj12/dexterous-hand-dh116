"""A failed timestamp check must preserve scalar evidence, not admit the frame."""
import json
import pytest
from go2_pvcnn.control.dh116.tabletop_stereo_rgb_bridge import RGBPairBridge
from test_dh116_stereo_rgb_bridge import Reader,Sim,RenderSetting,calibration,state

def rejected_capture():
    c=calibration();reader=Reader(c)
    read=reader.read
    reader.last_clock_diagnostic=dict(reference="raw-reference",times="raw-times",startup=False)
    def mismatched():
        packet=read()
        packet.update(left_simulation_time=0.,right_simulation_time=0.)
        packet["render_timing"]["simulation_time"]=0.
        return packet
    reader.read=mismatched
    bridge=RGBPairBridge(Sim(),reader,c,lambda:state(c),render_setting=RenderSetting())
    return bridge

def test_rejected_capture_preserves_raw_and_expected_clocks_without_accepting_rgb():
    bridge=rejected_capture()
    with pytest.raises(ValueError,match="render frame and physical timestamp mismatch") as caught:
        bridge.capture()
    text=str(caught.value)
    assert "sdk_timestamp_s" in text and "render_timestamp_s" in text
    assert "raw-reference" in text and "raw-times" in text
    d=bridge.last_capture_diagnostic
    assert d["sdk_timestamp_s"]==1. and d["sdk_step"]==4
    assert d["render_timestamp_s"]==0.
    assert d["absolute_time_error_s"]==1.
    assert d["timestamp_tolerance_s"]==1e-6
    assert bridge.last_frame==-1 and bridge.last_epoch is None
    json.dumps(d)  # metadata contains no RGB/GPU/native-handle objects

def test_raw_reader_evidence_is_bounded_and_no_frame_is_relabelled():
    bridge=rejected_capture()
    bridge.reader.last_clock_diagnostic={"times":"x"*100000}
    with pytest.raises(ValueError) as caught:bridge.capture()
    assert len(str(caught.value))<6000
    assert "reader_clock_diagnostic" in str(caught.value)
    assert bridge.last_frame==-1
