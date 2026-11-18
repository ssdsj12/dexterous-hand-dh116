"""Phase-level allocation telemetry; reads metadata and never changes physics."""
import json
from pathlib import Path
import time
import torch


def memory_snapshot(args,label,world=None):
    row=dict(phase=label,wall_unix_s=time.time(),num_envs=args.num_envs)
    if torch.cuda.is_available() and str(args.device).startswith('cuda'):
        torch.cuda.synchronize()
        free,total=torch.cuda.mem_get_info()
        row.update(free_gib=free/1024**3,total_gib=total/1024**3,
            driver_used_gib=(total-free)/1024**3,
            torch_allocated_gib=torch.cuda.memory_allocated()/1024**3,
            torch_reserved_gib=torch.cuda.memory_reserved()/1024**3,
            torch_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        try:
            import carb
            settings=carb.settings.get_settings()
            row['sensor_aa_op']=settings.get_as_int('/rtx/post/aa/op')
            row['sensor_dlssg_enabled']=settings.get_as_bool('/rtx-transient/dlssg/enabled')
        except ImportError:
            row['sensor_renderer_settings_unavailable']=True
    if world is not None:
        views=[]
        for sensor in getattr(world,'sensors',()):
            try:
                view=sensor.contact_physx_view
                if view is not None:
                    views.append(dict(count=len(view.sensor_paths),filter_count=int(view.filter_count)))
            except (AttributeError,RuntimeError,AssertionError):
                pass
        row['contact_views']=views
    with (Path(args.output)/'vision_memory.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')
    print('VISION_MEMORY',json.dumps(row),flush=True)
    return row
