"""Native frame capture and lossless sequence provenance; no simulation calls."""
import asyncio
import hashlib
from pathlib import Path
import subprocess
import time

def frame_indices(count,source_fps,target_fps):
    if (not all(isinstance(v,int) and v>0 for v in (count,source_fps,target_fps))
            or source_fps%target_fps or (count-1)%(source_fps//target_fps)):
        raise ValueError("integer decimation must include initial and final frames")
    return list(range(0,count,source_fps//target_fps))

def capture_frame(schedule,update,read_clock,path,*,max_updates=12,write_timeout_s=3.0):
    """Pump render-only application updates; any native clock change aborts."""
    before=read_clock()
    loop=asyncio.get_event_loop()
    future=asyncio.ensure_future(schedule(),loop=loop)
    try:
        for _ in range(max_updates):
            if future.done() and Path(path).is_file() and Path(path).stat().st_size:
                break
            update()
            if read_clock()!=before:raise RuntimeError("capture advanced physics clock")
            if not loop.is_running():loop.run_until_complete(asyncio.sleep(0))
        if not future.done():raise RuntimeError("capture update budget exhausted")
        future.result()
        if read_clock()!=before:raise RuntimeError("capture advanced physics clock")
        deadline=time.monotonic()+write_timeout_s
        while not Path(path).is_file() or not Path(path).stat().st_size:
            if time.monotonic()>=deadline:raise RuntimeError("viewport file-write deadline exhausted")
            if read_clock()!=before:raise RuntimeError("file wait advanced physics clock")
            time.sleep(.01)
        if read_clock()!=before:raise RuntimeError("file wait advanced physics clock")
    finally:
        if not future.done():
            future.cancel()
            if not loop.is_running():loop.run_until_complete(asyncio.sleep(0))

def encode_frames(directory,output,*,fps,expected_count,ffmpeg):
    """Encode exact sequential PNGs; refuse incomplete sequences and overwrites."""
    directory,output=Path(directory),Path(output)
    if output.exists():raise FileExistsError(output)
    expected=[directory/f"frame_{i:05d}.png" for i in range(expected_count)]
    actual=sorted(directory.glob("frame_*.png"))
    if (not expected_count or actual!=expected or any(p.stat().st_size==0 for p in expected)
            or not isinstance(fps,int) or fps<=0):
        raise ValueError("incomplete or unexpected frame sequence")
    command=[str(ffmpeg),"-nostdin","-n","-loglevel","error","-framerate",str(fps),
        "-i",str(directory/"frame_%05d.png"),"-frames:v",str(expected_count),
        "-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-preset","veryfast",
        "-movflags","+faststart",str(output)]
    completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    if completed.returncode or not output.is_file():
        raise RuntimeError("video encode failed: "+completed.stderr[-4000:])
    return dict(frame_count=expected_count,fps=fps,duration_s=expected_count/fps,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source="recorded_native_replay",interpolated=False,counts_as_success=False)
