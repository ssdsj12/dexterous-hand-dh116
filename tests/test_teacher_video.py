import asyncio
from pathlib import Path
import pytest
from go2_pvcnn.control.dh116.tabletop_teacher_video import frame_indices, capture_frame, encode_frames

def test_full_sequence_no_interpolation():
    assert frame_indices(481,60,30)==list(range(0,481,2))
    with pytest.raises(ValueError):frame_indices(480,60,30)
    with pytest.raises(ValueError):frame_indices(481,60,25)

def test_capture_waits_without_physics(tmp_path):
    clock=[17,1.2];p=tmp_path/"f.png";ticks=[]
    async def write():
        await asyncio.sleep(0);p.write_bytes(b"native-test")
        return True
    def update():ticks.append(1)
    capture_frame(lambda:write(),update,lambda:tuple(clock),p,max_updates=10)
    assert ticks and p.exists()

def test_capture_rejects_clock_change(tmp_path):
    clock=[17,1.2];p=tmp_path/"f.png"
    async def write():await asyncio.sleep(0)
    def update():clock[0]+=1
    with pytest.raises(RuntimeError,match="physics"):
        capture_frame(lambda:write(),update,lambda:tuple(clock),p,max_updates=10)

def test_capture_bound(tmp_path):
    async def stuck():
        for _ in range(99):await asyncio.sleep(0)
    with pytest.raises(RuntimeError,match="budget"):
        capture_frame(lambda:stuck(),lambda:None,lambda:(1,.1),tmp_path/"x",max_updates=2)

def test_encoder_real_cpu_fixture(tmp_path):
    from PIL import Image
    import imageio_ffmpeg
    for i in range(3):Image.new("RGB",(64,48),(i*70,0,0)).save(tmp_path/f"frame_{i:05d}.png")
    out=tmp_path/"cpu_synthetic_test.mp4"
    result=encode_frames(tmp_path,out,fps=30,expected_count=3,ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
    assert out.stat().st_size>500 and result["frame_count"]==3
    assert len(result["sha256"])==64
    with pytest.raises(FileExistsError):
        encode_frames(tmp_path,out,fps=30,expected_count=3,ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())

def test_encoder_missing_frame_fails(tmp_path):
    with pytest.raises(ValueError,match="sequence"):
        encode_frames(tmp_path,tmp_path/"no.mp4",fps=30,expected_count=3,ffmpeg="no-execution")
